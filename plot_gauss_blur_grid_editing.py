#!/usr/bin/env python3

from utils.data_loader2 import load_galaxies
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter 
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve_fft, Gaussian2DKernel
import os, glob
import torch
from utils.data_loader2 import apply_formatting


print("Loading scatter_galaxies/plot_gauss_blur_grid_editing.py...")

ARCSEC = np.deg2rad(1/3600)
crop_size = (3, 1, 512, 512)  # (T, C, H, W) for 3 channels, 512x512 images

def _residual(a, b):
    """Return a - b with NaNs where either side is NaN/None."""
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    out = np.full_like(a, np.nan, dtype=float)
    out[m] = a[m] - b[m]
    return out

def _imshow_residual(ax, res, q=99.0):
    """Show residual with symmetric limits and a diverging colormap."""
    if res is None or not np.isfinite(res).any():
        ax.set_axis_off()
        return
    finite = res[np.isfinite(res)]
    lim = np.percentile(np.abs(finite), q) if finite.size else 1.0
    if not np.isfinite(lim) or lim <= 0:
        lim = np.max(np.abs(finite)) if finite.size else 1.0
        if lim <= 0: lim = 1.0
    ax.imshow(res, cmap='RdBu_r', origin='lower', interpolation='nearest',
              vmin=-lim, vmax=lim)
    ax.set_axis_off()

def _stretch_from_p(arr, p_lo, p_hi):
    """Linear stretch based on given percentiles."""
    return (arr - p_lo) / (p_hi - p_lo + 1e-6)
 
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

def _load_fits_arrays_scaled(name, crop_ch=1, out_hw=(128,128), ROOT = "/users/mbredber/scratch/data/PSZ2"):
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

def _joint_pct(*arrs, lo=60, hi=95):
    """Percentiles from the union of finite pixels across all given arrays."""
    vals = []
    for a in arrs:
        if a is not None:
            aa = np.asarray(a, float)
            vals.append(aa[np.isfinite(aa)].ravel())
    if not vals:
        return 0.0, 1.0
    v = np.concatenate(vals)
    if v.size == 0:
        return 0.0, 1.0
    qlo, qhi = np.nanpercentile(v, [lo, hi])
    if not np.isfinite(qhi - qlo) or qhi <= qlo:
        qlo, qhi = np.nanmin(v), np.nanmax(v)
    return float(qlo), float(qhi)

def _stretch_from_p(arr, p_lo, p_hi, clip=True):
    """Linear stretch with optional clipping to [0,1]."""
    if arr is None:
        return None
    y = (arr - p_lo) / (p_hi - p_lo + 1e-6)
    return np.clip(y, 0, 1) if clip else y

def plot_galaxy_grid(images, filenames, labels, STRETCH=False, CLIP_NORM=False):
    """
    3 DE rows, a thin spacer row, then 3 NDE rows.
    Columns: T25, RAW→25, Residual, T50, RAW→50, Residual, T100, RAW→100, Residual
    """
    de_idx  = [i for i, y in enumerate(labels) if int(y) == 50][:3]
    nde_idx = [i for i, y in enumerate(labels) if int(y) == 51][:3]
    order   = de_idx + nde_idx
    if len(de_idx) < 3 or len(nde_idx) < 3:
        raise ValueError(f"Need ≥3 of each class; got {len(de_idx)} DE and {len(nde_idx)} NDE.")

    images     = images[order]
    filenames  = [filenames[i] for i in order]
    labels     = [labels[i]    for i in order]

    n_each  = 3
    gap     = 1
    n_rows  = n_each*2 + gap                   # 7 total
    n_cols  = 9                                # add three residual columns

    cell = 1.6
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols*cell, n_rows*cell*0.98),
        gridspec_kw=dict(
            left=0.04, right=0.995, top=0.92, bottom=0.05,
            wspace=0.04, hspace=0.04,
            height_ratios=[1,1,1,0.12,1,1,1]
        ),
        constrained_layout=False
    )

    col_titles = [
        "T25 kpc", "RAW → 25 kpc", "Residual 25",
        "T50 kpc", "RAW → 50 kpc", "Residual 50",
        "T100 kpc", "RAW → 100 kpc", "Residual 100",
    ]
    for ax, t in zip(axes[0], col_titles):
        ax.set_title(t, fontsize=12, pad=6)

    row_map = [0,1,2,4,5,6]

    for i_src, grid_row in enumerate(row_map):
        name = Path(str(filenames[i_src])).stem

        # Load pre-stretch arrays (all at the same output size)
        raw_cut, t25_cut, t50_cut, t100_cut, raw_hdr, t25_hdr, t50_hdr, t100_hdr, pix_eff = \
            _load_fits_arrays_scaled(name, crop_ch=1, out_hw=(images.shape[-2], images.shape[-1]))

        # PSF-match the pre-stretch RAW
        raw_to_25  = convolve_to_target(raw_cut, raw_hdr, t25_hdr,  pix_eff) if t25_hdr  is not None else None
        raw_to_50  = convolve_to_target(raw_cut, raw_hdr, t50_hdr,  pix_eff) if t50_hdr  is not None else None
        raw_to_100 = convolve_to_target(raw_cut, raw_hdr, t100_hdr, pix_eff) if t100_hdr is not None else None

        # --- HYBRID STRETCHING (for the image columns) ---
        # 25 kpc pair
        lo25, hi25 = _joint_pct(t25_cut,  raw_to_25,  lo=60, hi=95)  # one mapping for both
        t25_s      = _stretch_from_p(t25_cut,  lo25, hi25)
        r2_25_s    = _stretch_from_p(raw_to_25, lo25, hi25)

        # 50 kpc pair
        lo50, hi50 = _joint_pct(t50_cut,  raw_to_50,  lo=60, hi=95)
        t50_s      = _stretch_from_p(t50_cut,  lo50, hi50)
        r2_50_s    = _stretch_from_p(raw_to_50, lo50, hi50)

        # 100 kpc pair
        lo100, hi100 = _joint_pct(t100_cut, raw_to_100, lo=60, hi=95)
        t100_s       = _stretch_from_p(t100_cut,  lo100, hi100)
        r2_100_s     = _stretch_from_p(raw_to_100, lo100, hi100)


        r2_25_s  = _stretch_from_p(raw_to_25,  r_lo, r_hi)   if raw_to_25  is not None else None
        r2_50_s  = _stretch_from_p(raw_to_50,  r_lo, r_hi)   if raw_to_50  is not None else None
        r2_100_s = _stretch_from_p(raw_to_100, r_lo, r_hi)   if raw_to_100 is not None else None

        # --- Residuals in linear units (pre-stretch) ---
        res25  = _residual(t25_cut,  raw_to_25)
        res50  = _residual(t50_cut,  raw_to_50)
        res100 = _residual(t100_cut, raw_to_100)

        planes     = [t25_s, r2_25_s, res25,  t50_s, r2_50_s, res50,  t100_s, r2_100_s, res100]
        is_resid   = [False, False,  True,    False, False,  True,    False, False,   True]

        for j, (arr, rflag) in enumerate(zip(planes, is_resid)):
            ax = axes[grid_row, j]
            if arr is None:
                ax.set_axis_off()
                continue
            if rflag:
                _imshow_residual(ax, arr)  # diverging, symmetric limits
            else:
                ax.imshow(arr, cmap="viridis", origin="lower", interpolation="nearest")
                ax.set_axis_off()

        # row label
        ax0 = axes[grid_row, 0]
        ax0.text(-0.02, 0.5, name, transform=ax0.transAxes, rotation=90,
                 va='center', ha='right', fontsize=8)

    # spacer row & section labels
    for j in range(n_cols):
        axes[3, j].axis('off')
        axes[3, j].set_facecolor('white')

    axes[1, 0].text(-0.20, 0.5, "DE",  transform=axes[1,0].transAxes,
                    va='center', ha='right', fontsize=13, fontweight='bold')
    axes[5, 0].text(-0.20, 0.5, "NDE", transform=axes[5,0].transAxes,
                    va='center', ha='right', fontsize=13, fontweight='bold')

    for j in range(n_cols):
        axes[2, j].spines['bottom'].set_color('white')
        axes[2, j].spines['bottom'].set_linewidth(3)
        axes[4, j].spines['top'].set_color('white')
        axes[4, j].spines['top'].set_linewidth(3)

    outname = f"psz2_psfmatch_pairs25_50_100_with_residuals_cs{crop_size[-1]}_out{images.shape[-1]}_editing.pdf"
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
        train=False,
    )
    eval_imgs   = result[2]
    eval_labels = result[3]  # eval_labels lives at index 3
    eval_fns    = result[5]
    plot_galaxy_grid(eval_imgs, eval_fns, eval_labels)
