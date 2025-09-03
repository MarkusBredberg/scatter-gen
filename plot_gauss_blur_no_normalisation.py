#!/usr/bin/env python3

from utils.data_loader2 import load_galaxies
from utils.data_loader2 import apply_formatting
from utils.calc_tools import check_tensor
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from astropy.io import fits
from astropy.convolution import convolve_fft, Gaussian2DKernel
import os, glob
import torch


print("Loading scatter_galaxies/plot_gauss_blur_grid_editing.py...")

ARCSEC = np.deg2rad(1/3600)
crop_size = (3, 1, 512, 512)  # (T, C, H, W) for 3 channels, 512x512 images

def _stretch_from_p(arr, p_lo, p_hi, clip=True):
    if arr is None:
        return None
    y = (arr - p_lo) / (p_hi - p_lo + 1e-6)
    return np.clip(y, 0, 1) if clip else y

def _nanpct(arr, lo=60, hi=95):
    """Percentiles that ignore NaNs."""
    return tuple(np.nanpercentile(arr, [lo, hi]))

def _nanrms(a, b):
    """RMS difference between two arrays, ignoring NaNs."""
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return np.nan
    d = a[m] - b[m]
    return float(np.sqrt(np.mean(d*d)))

def _fits_path_triplet(base_dir, real_base):
    raw_path  = f"{base_dir}/{real_base}.fits"
    t25_path  = _first(f"{base_dir}/{real_base}T25kpc*.fits")
    t50_path  = _first(f"{base_dir}/{real_base}T50kpc*.fits")
    t100_path = _first(f"{base_dir}/{real_base}T100kpc*.fits")
    return raw_path, t25_path, t50_path, t100_path

def _load_fits_arrays_scaled(name, crop_ch=1, out_hw=(128,128)):
    base_dir = _first(f"{ROOT}/fits/{name}*") or f"{ROOT}/fits/{name}"
    raw_path = _first(f"{base_dir}/{os.path.basename(base_dir)}*.fits")
    if raw_path is None:
        raise FileNotFoundError(f"RAW FITS not found under {base_dir}")
    real_base = os.path.splitext(os.path.basename(raw_path))[0]
    raw_path, t25_path, t50_path, t100_path = _fits_path_triplet(base_dir, real_base)

    raw_hdr  = fits.getheader(raw_path)
    t25_hdr  = fits.getheader(t25_path)  if t25_path  else None
    t50_hdr  = fits.getheader(t50_path)  if t50_path  else None
    t100_hdr = fits.getheader(t100_path) if t100_path else None

    pix_native = _pixscale_arcsec(raw_hdr)

    raw_arr  = np.squeeze(fits.getdata(raw_path)).astype(float)
    t25_arr  = np.squeeze(fits.getdata(t25_path)).astype(float)   if t25_path  else None
    t50_arr  = np.squeeze(fits.getdata(t50_path)).astype(float)   if t50_path  else None
    t100_arr = np.squeeze(fits.getdata(t100_path)).astype(float)  if t100_path else None

    ch, Hc_raw, Wc_raw = crop_ch, crop_size[-2], crop_size[-1]
    outH, outW = out_hw

    def _fmt(arr, ver_hdr):
        if ver_hdr is None or arr is None:
            return None
        scale = abs(raw_hdr['CDELT1'] / ver_hdr['CDELT1'])
        Hc = int(round(Hc_raw * scale))
        Wc = int(round(Wc_raw * scale))
        ten = torch.from_numpy(arr).unsqueeze(0).float()
        ten = apply_formatting(ten, (ch, Hc, Wc), (ch, outH, outW)).squeeze(0).numpy()
        return ten

    raw_cut  = _fmt(raw_arr,  raw_hdr)
    t25_cut  = _fmt(t25_arr,  t25_hdr)
    t50_cut  = _fmt(t50_arr,  t50_hdr)
    t100_cut = _fmt(t100_arr, t100_hdr)

    ds_factor = (int(round(Hc_raw)) / outH)
    pix_eff_arcsec = pix_native * ds_factor

    return raw_cut, t25_cut, t50_cut, t100_cut, raw_hdr, t25_hdr, t50_hdr, t100_hdr, pix_eff_arcsec

def fwhm_to_sigma_rad(fwhm_arcsec):
    return (fwhm_arcsec*ARCSEC) / (2*np.sqrt(2*np.log(2)))

def beam_cov_matrix(bmaj_as, bmin_as, pa_deg):
    # sigmas in radians
    sx = fwhm_to_sigma_rad(bmaj_as)
    sy = fwhm_to_sigma_rad(bmin_as)
    th = np.deg2rad(pa_deg)
    R = np.array([[np.cos(th), -np.sin(th)],
                  [np.sin(th),  np.cos(th)]], dtype=float)
    S = np.diag([sx**2, sy**2])
    return R @ S @ R.T

def kernel_from_beams(raw_hdr, targ_hdr, pixscale_arcsec):
    # beams (arcsec) and PAs (deg)
    bmaj_r = raw_hdr['BMAJ']*3600.0; bmin_r = raw_hdr['BMIN']*3600.0; pa_r = raw_hdr.get('BPA', 0.0)
    bmaj_t = targ_hdr['BMAJ']*3600.0; bmin_t = targ_hdr['BMIN']*3600.0; pa_t = targ_hdr.get('BPA', pa_r)

    C_raw = beam_cov_matrix(bmaj_r, bmin_r, pa_r)
    C_tgt = beam_cov_matrix(bmaj_t, bmin_t, pa_t)

    # kernel covariance in **radians^2**
    C_ker = C_tgt - C_raw
    # numerical guard: clip tiny negatives to zero
    w, V = np.linalg.eigh(C_ker)
    w = np.clip(w, 0.0, None)
    C_ker = (V * w) @ V.T

    # convert to pixel units
    pixrad = (pixscale_arcsec*ARCSEC)
    C_pix = C_ker / (pixrad**2)

    # turn back into Gaussian2DKernel params
    w_pix, V_pix = np.linalg.eigh(C_pix)
    sx_pix, sy_pix = np.sqrt(np.maximum(w_pix[1], 0.0)), np.sqrt(np.maximum(w_pix[0], 0.0))  # w[1]≥w[0]
    theta = np.arctan2(V_pix[1,1], V_pix[0,1])  # PA of major axis

    # if target <= native in any direction, sx_pix or sy_pix can be 0; kernel becomes delta (no blur)
    return Gaussian2DKernel(x_stddev=sx_pix, y_stddev=sy_pix, theta=theta)

def convolve_to_target(raw_img, raw_hdr, target_hdr, pixscale_arcsec):
    ker = kernel_from_beams(raw_hdr, target_hdr, pixscale_arcsec)
    return convolve_fft(raw_img, ker, boundary='fill',
                        fill_value=np.nan, nan_treatment='interpolate',
                        normalize_kernel=True)

def _first(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None

def _pixscale_arcsec(hdr):
    if 'CDELT1' in hdr:
        return abs(hdr['CDELT1']) * 3600.0
    # fallback if using CD matrix
    cd11 = hdr.get('CD1_1'); cd12 = hdr.get('CD1_2', 0.0)
    if cd11 is not None:
        pix_deg = np.hypot(cd11, cd12)
        return abs(pix_deg) * 3600.0
    raise KeyError("No CDELT* or CD* keywords in FITS header")

ROOT = "/users/mbredber/scratch/data/PSZ2"
def load_headers(base, ds_factor=1.0):
    base_dir = _first(f"{ROOT}/fits/{base}*") or f"{ROOT}/fits/{base}"
    raw_path = _first(f"{base_dir}/{os.path.basename(base_dir)}*.fits")
    if raw_path is None:
        raise FileNotFoundError(f"RAW FITS not found under {base_dir}")

    real_base = os.path.splitext(os.path.basename(raw_path))[0]
    t50_path  = _first(f"{base_dir}/{real_base}T50kpc*.fits")
    t100_path = _first(f"{base_dir}/{real_base}T100kpc*.fits")

    raw_hdr  = fits.getheader(raw_path)
    t50_hdr  = fits.getheader(t50_path)  if t50_path  else None
    t100_hdr = fits.getheader(t100_path) if t100_path else None

    # native pixel scale (arcsec/pixel)
    pix_native = _pixscale_arcsec(raw_hdr)
    # effective pixel scale of the **downsampled** arrays you display
    pix_eff = pix_native * ds_factor

    return raw_hdr, t50_hdr, t100_hdr, pix_eff


def plot_galaxy_grid(images, filenames, labels, lo=60, hi=95, RES_PCT=100):
    """
    Layout (9 columns):
      T25, RAW→25, Residual 25,  T50, RAW→50, Residual 50,  T100, RAW→100, Residual 100

    Normalisation:
      • One GLOBAL percentile clip (lo,hi) computed from ALL T and RAW→T arrays.
      • All non-residual panels share the same Normalize(0,1).
      • Residuals use a single symmetric range [-R,+R] with viridis (R from RES_PCT of |res|).
    """
    # pick 3 per class (DE first)
    de_idx  = [i for i, y in enumerate(labels) if int(y) == 50][:3]
    nde_idx = [i for i, y in enumerate(labels) if int(y) == 51][:3]
    order   = de_idx + nde_idx
    if len(de_idx) < 3 or len(nde_idx) < 3:
        raise ValueError(f"Need ≥3 of each class; got {len(de_idx)} DE and {len(nde_idx)} NDE.")

    images     = images[order]
    filenames  = [filenames[i] for i in order]

    # grid
    n_each, gap = 3, 1
    n_rows, n_cols = n_each*2 + gap, 9
    cell = 1.6
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols*cell, n_rows*cell*0.98),
        gridspec_kw=dict(
            left=0.04, right=0.995, top=0.92, bottom=0.05,
            wspace=0.04, hspace=0.04,
            height_ratios=[1,1,1,0.12,1,1,1]),
        constrained_layout=False
    )
    col_titles = [
        "T25 kpc", "RAW → 25 kpc", "│res│ 25",
        "T50 kpc", "RAW → 50 kpc", "│res│ 50",
        "T100 kpc", "RAW → 100 kpc", "│res│ 100",
    ]
    for ax, t in zip(axes[0], col_titles): ax.set_title(t, fontsize=12, pad=6)

    # map 6 sources to 7 grid rows (skip spacer row=3)
    row_map = [0,1,2,4,5,6]
    outH, outW = images.shape[-2], images.shape[-1]

    # ---------- PASS 1: load everything (no stretching here) ----------
    rows = []
    for i_src, grid_row in enumerate(row_map):
        name = Path(str(filenames[i_src])).stem
        raw_cut, t25_cut, t50_cut, t100_cut, raw_hdr, t25_hdr, t50_hdr, t100_hdr, pix_eff = \
            _load_fits_arrays_scaled(name, crop_ch=1, out_hw=(outH, outW))
            
        # Check the tensor mean and std for debugging
        check_tensor(f"raw_cut {name}", torch.tensor(raw_cut))
        check_tensor(f"t25_cut {name}", torch.tensor(t25_cut))
        check_tensor(f"t50_cut {name}", torch.tensor(t50_cut))
        check_tensor(f"t100_cut {name}", torch.tensor(t100_cut))

        r2_25  = convolve_to_target(raw_cut, raw_hdr, t25_hdr,  pix_eff) if t25_hdr  is not None else None
        r2_50  = convolve_to_target(raw_cut, raw_hdr, t50_hdr,  pix_eff) if t50_hdr  is not None else None
        r2_100 = convolve_to_target(raw_cut, raw_hdr, t100_hdr, pix_eff) if t100_hdr is not None else None
        
        # Check the tensor mean and std for debugging
        check_tensor(f"r2_25 {name}", torch.tensor(r2_25))
        check_tensor(f"r2_50 {name}", torch.tensor(r2_50))
        check_tensor(f"r2_100 {name}", torch.tensor(r2_100))

        rows.append(dict(name=name, grid_row=grid_row,
                         t25=t25_cut, t50=t50_cut, t100=t100_cut,
                         r25=r2_25,  r50=r2_50,  r100=r2_100))
        
    # Check tensors again for debugging
    for r in rows:
        for key in ('t25','t50','t100','r25','r50','r100'):
            arr = r[key]
            if arr is not None:
                check_tensor(f"Row {r['name']} key {key}", torch.tensor(arr))

    # ---------- GLOBAL lo/hi across ALL T and RAW→T arrays ----------
    "This part finds the global percentile clip values across all images."
    all_vals = []
    for r in rows:
        for k in ("t25","t50","t100","r25","r50","r100"):
            a = r[k]
            if a is not None:
                a = np.asarray(a, float)
                a = a[np.isfinite(a)]
                if a.size:
                    all_vals.append(a.ravel())
    if not all_vals:
        g_lo, g_hi = 0.0, 1.0
    else:
        v = np.concatenate(all_vals)
        g_lo, g_hi = np.nanpercentile(v, [lo, hi])
        if not np.isfinite(g_hi - g_lo) or g_hi <= g_lo:
            g_lo, g_hi = np.nanmin(v), np.nanmax(v)
    print(f"Global clip percentiles: {lo}→{hi} gives {g_lo:.3e} to {g_hi:.3e}")

    # ---------- PASS 2: apply the same stretch everywhere; build residuals ----------
    residual_vals = []
    for r in rows:
        # global-stretched images
        t25_s  = _stretch_from_p(r['t25'],  g_lo, g_hi) if r['t25']  is not None else None
        r25_s  = _stretch_from_p(r['r25'],  g_lo, g_hi) if r['r25']  is not None else None
        t50_s  = _stretch_from_p(r['t50'],  g_lo, g_hi) if r['t50']  is not None else None
        r50_s  = _stretch_from_p(r['r50'],  g_lo, g_hi) if r['r50']  is not None else None
        t100_s = _stretch_from_p(r['t100'], g_lo, g_hi) if r['t100'] is not None else None
        r100_s = _stretch_from_p(r['r100'], g_lo, g_hi) if r['r100'] is not None else None

        # signed residuals in the same normalised space
        res25  = (t25_s  - r25_s) if (t25_s  is not None and r25_s  is not None) else None
        res50  = (t50_s  - r50_s) if (t50_s  is not None and r50_s  is not None) else None
        res100 = (t100_s - r100_s) if (t100_s is not None and r100_s is not None) else None

        for rr in (res25, res50, res100):
            if rr is not None:
                a = np.asarray(rr, float)
                a = a[np.isfinite(a)]
                if a.size: residual_vals.append(np.abs(a).ravel())

        r['planes'] = [t25_s, r25_s, res25,  t50_s, r50_s, res50,  t100_s, r100_s, res100]
        
    # Check tensors again for debugging
    for r in rows:
        for i, arr in enumerate(r['planes']):
            if arr is not None:
                check_tensor(f"Row {r['name']} plane {i}", torch.tensor(arr))

    # shared norms
    img_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    R = np.nanpercentile(np.concatenate(residual_vals), RES_PCT) if residual_vals else 1.0
    #res_norm = mcolors.TwoSlopeNorm(vmin=-R, vcenter=0.0, vmax=R)   # viridis + symmetric range
    res_norm = mcolors.Normalize(vmin=0.0, vmax=R)
    
    # Check tensor once more for debugging
    for r in rows:
        for i, arr in enumerate(r['planes']):
            if arr is not None:
                check_tensor(f"Final Row {r['name']} plane {i}", torch.tensor(arr))

    RES_COLS = {2, 5, 8}

    # ---------- draw ----------
    for r in rows:
        gr = r['grid_row']
        for j, arr in enumerate(r['planes']):
            ax = axes[gr, j]
            if arr is not None:
                if j in RES_COLS:  # residual columns {2,5,8}
                    ax.imshow(np.abs(arr), cmap="viridis", norm=res_norm,
                            origin="lower", interpolation="nearest")
                else:
                    ax.imshow(arr, cmap="viridis", norm=img_norm,
                            origin="lower", interpolation="nearest")
            ax.set_axis_off()
        axes[gr, 0].text(-0.02, 0.5, r['name'], transform=axes[gr,0].transAxes,
                         rotation=90, va='center', ha='right', fontsize=8)

    # spacer + labels
    for j in range(n_cols):
        axes[3, j].axis('off'); axes[3, j].set_facecolor('white')
        axes[2, j].spines['bottom'].set_color('white'); axes[2, j].spines['bottom'].set_linewidth(3)
        axes[4, j].spines['top'].set_color('white');    axes[4, j].spines['top'].set_linewidth(3)
    axes[1, 0].text(-0.20, 0.5, "DE",  transform=axes[1,0].transAxes,
                    va='center', ha='right', fontsize=13, fontweight='bold')
    axes[5, 0].text(-0.20, 0.5, "NDE", transform=axes[5,0].transAxes,
                    va='center', ha='right', fontsize=13, fontweight='bold')

    outname = "psz2_psfmatch_pairs25_50_100_with_residuals_GLOBAL_NN.pdf"
    plt.savefig(outname, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {outname}")

if __name__ == "__main__": 
    result = load_galaxies(
        galaxy_classes=[50, 51],
        versions=['T25kpc', 'T50kpc', 'T100kpc'],   # or include 'RAW' if you want it in the cube
        fold=5,
        crop_size=(3, 1, 512, 512),                 # same physical FOV for all versions
        downsample_size=(3, 1, 128, 128),           # common output size
        sample_size=500,
        USE_GLOBAL_NORMALISATION=False,             # <- leave off
        NORMALISE=False, STRETCH=False,             # <- leave off
        NORMALISETOPM=False, AUGMENT=False,
        EXTRADATA=False, PRINTFILENAMES=True,
        train=False, DEBUG=True
    )
    eval_imgs   = result[2]
    eval_labels = result[3]  # eval_labels lives at index 3
    eval_fns    = result[5]
    plot_galaxy_grid(eval_imgs, eval_fns, eval_labels)
