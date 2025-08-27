#!/usr/bin/env python3

crop_size = (3, 1, 512, 512)  # (T, C, H, W) for 3 channels, 512x512 images
from utils.data_loader2 import load_galaxies
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
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


print("Loading scatter_galaxies/plot_gauss_blur_grid2.py...")

ARCSEC = np.deg2rad(1/3600)

def _stretch_from_p(arr, p_lo, p_hi):
     y = (arr - p_lo) / (p_hi - p_lo + 1e-6)
     return np.clip(y, 0, 1)
 
def _nanpct(arr, lo=60, hi=95):
    """Percentiles that ignore NaNs."""
    return tuple(np.nanpercentile(arr, [lo, hi]))

def _nanrms(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return np.nan
    d = a[m] - b[m]
    return float(np.sqrt(np.mean(d*d)))

def _fits_path_triplet(base_dir, real_base):
    raw_path  = f"{base_dir}/{real_base}.fits"
    t50_path  = _first(f"{base_dir}/{real_base}T50kpc*.fits")
    t100_path = _first(f"{base_dir}/{real_base}T100kpc*.fits")
    return raw_path, t50_path, t100_path

def _load_fits_arrays_scaled(name, crop_ch=1, out_hw=(128,128)):
    """
    Load FITS arrays for a galaxy, crop them to the specified size, and scale them to the output size.
    """
    # resolve folder + real_base (same logic you already use)
    base_dir = _first(f"{ROOT}/fits/{name}*") or f"{ROOT}/fits/{name}"
    raw_path = _first(f"{base_dir}/{os.path.basename(base_dir)}*.fits")
    if raw_path is None:
        raise FileNotFoundError(f"RAW FITS not found under {base_dir}")
    real_base = os.path.splitext(os.path.basename(raw_path))[0]
    raw_path, t50_path, t100_path = _fits_path_triplet(base_dir, real_base)

    # headers
    raw_hdr  = fits.getheader(raw_path)
    t50_hdr  = fits.getheader(t50_path)  if t50_path  else None
    t100_hdr = fits.getheader(t100_path) if t100_path else None

    # native pix scale (arcsec / pix)
    pix_native = _pixscale_arcsec(raw_hdr)

    # read arrays
    raw_arr  = np.squeeze(fits.getdata(raw_path)).astype(float)
    t50_arr  = np.squeeze(fits.getdata(t50_path)).astype(float)   if t50_path  else None
    t100_arr = np.squeeze(fits.getdata(t100_path)).astype(float)  if t100_path else None

    # match sky area to RAW by scaling the crop size using CDELT ratio
    ch, Hc_raw, Wc_raw = crop_ch, crop_size[-2], crop_size[-1]
    outH, outW        = out_hw

    def _fmt(arr, ver_hdr):
        if ver_hdr is None or arr is None:
            return None
        scale = abs(raw_hdr['CDELT1'] / ver_hdr['CDELT1'])
        Hc = int(round(Hc_raw * scale))
        Wc = int(round(Wc_raw * scale))
        ten = torch.from_numpy(arr).unsqueeze(0).float()       # [1,H,W]
        ten = apply_formatting(ten, (ch, Hc, Wc), (ch, outH, outW)).squeeze(0).numpy()
        return ten

    raw_cut  = _fmt(raw_arr,  raw_hdr)
    t50_cut  = _fmt(t50_arr,  t50_hdr)
    t100_cut = _fmt(t100_arr, t100_hdr)

    # effective pixel scale for the *displayed* arrays (downsample factor)
    ds_factor = (int(round(Hc_raw)) / outH)   # 512 / 128 = 4 in your run
    pix_eff_arcsec = pix_native * ds_factor

    return raw_cut, t50_cut, t100_cut, raw_hdr, t50_hdr, t100_hdr, pix_eff_arcsec

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

def plot_galaxy_grid(images, filenames, labels, percentile_lo=60, percentile_hi=95, sigmas=(1.0, 3.0, 5.0)):
    """
    3 DE rows, a thin spacer row, then 3 NDE rows.
    Columns: RAW, T50 kpc, T100 kpc, then Gaussian blurs of RAW.
    """
    # pick 3 per class, DE first
    de_idx  = [i for i, y in enumerate(labels) if int(y) == 50][:3]
    nde_idx = [i for i, y in enumerate(labels) if int(y) == 51][:3]
    order   = de_idx + nde_idx
    if len(de_idx) < 3 or len(nde_idx) < 3:
        raise ValueError(f"Need ≥3 of each class; got {len(de_idx)} DE and {len(nde_idx)} NDE.")

    images     = images[order]
    filenames  = [filenames[i] for i in order]
    labels     = [labels[i]    for i in order]

    # layout
    n_each  = 3
    gap     = 1                 # spacer row
    n_rows  = n_each*2 + gap    # 7 total
    n_cols = 5  # RAW, T50, T100, RAW→50, RAW→100

    # compact figure with square axes; small wspace/hspace
    cell = 1.6                   # inches per tile (tweak if you want larger)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols*cell, n_rows*cell*0.98),
        gridspec_kw=dict(
            left=0.04, right=0.995, top=0.92, bottom=0.05,
            wspace=0.04, hspace=0.04,
            height_ratios=[1,1,1,0.12,1,1,1]  # thin spacer row
        ),
        constrained_layout=False
    )
    

    # column titles (set once, before the loop)       
    col_titles = ["RAW", "T50 kpc", "T100 kpc", "RAW → 50 kpc", "RAW → 100 kpc"]
    for ax, t in zip(axes[0], col_titles):
        ax.set_title(t, fontsize=12, pad=6)

    # map our 6 sources onto 7 grid-rows (skip spacer row index 3)
    row_map = [0,1,2,4,5,6]
    rms_50_list, rms_100_list = [], []


    for i_src, grid_row in enumerate(row_map):
        name = Path(str(filenames[i_src])).stem

        # === work with pre-stretch FITS arrays ===
        raw_cut, t50_cut, t100_cut, raw_hdr, t50_hdr, t100_hdr, pix_eff = \
            _load_fits_arrays_scaled(name, crop_ch=1, out_hw=(images.shape[-2], images.shape[-1]))

        # PSF-match the *pre-stretch* RAW
        raw_to_50  = convolve_to_target(raw_cut,  raw_hdr, t50_hdr,  pix_eff)
        raw_to_100 = convolve_to_target(raw_cut,  raw_hdr, t100_hdr, pix_eff)

        # --- HYBRID STRETCHING ---
        # RAW: use RAW's own percentiles
        r_lo, r_hi = _nanpct(raw_cut, 60, 95)
        raw_s      = _stretch_from_p(raw_cut, r_lo, r_hi)

        # T50: use T50's percentiles
        if t50_cut is not None and np.isfinite(t50_cut).any():
            p50_lo, p50_hi = _nanpct(t50_cut, 60, 95)
            t50_s   = _stretch_from_p(t50_cut,   p50_lo, p50_hi)
        else:
            t50_s   = None  # or _stretch_from_p(t50_cut, r_lo, r_hi) as a fallback

        # T100: use T100's percentiles
        if t100_cut is not None and np.isfinite(t100_cut).any():
            p100_lo, p100_hi = _nanpct(t100_cut, 60, 95)
            t100_s  = _stretch_from_p(t100_cut,  p100_lo, p100_hi)
        else:
            t100_s  = None  # or _stretch_from_p(t100_cut, r_lo, r_hi) as a fallback

        # Reproductions (RAW → 50/100): use RAW's percentiles (like screenshot 2)      
        rr50_lo,  rr50_hi  = _nanpct(raw_to_50,  percentile_lo, percentile_hi)
        rr100_lo, rr100_hi = _nanpct(raw_to_100, percentile_lo, percentile_hi)
        r2_50_s  = _stretch_from_p(raw_to_50,  rr50_lo,  rr50_hi)
        r2_100_s = _stretch_from_p(raw_to_100, rr100_lo, rr100_hi)
        
        if t50_s  is not None:  rms_50_list.append(_nanrms(r2_50_s,  t50_s))
        if t100_s is not None:  rms_100_list.append(_nanrms(r2_100_s, t100_s))



        planes = (raw_s, t50_s, t100_s, r2_50_s, r2_100_s)
        for j, arr in enumerate(planes):
            ax = axes[grid_row, j]
            ax.imshow(arr, cmap="viridis", origin="lower",
                    interpolation="nearest", vmin=0, vmax=1)
            ax.set_axis_off()


        # row label
        ax0 = axes[grid_row, 0]
        ax0.text(-0.02, 0.5, name, transform=ax0.transAxes, rotation=90,
                va='center', ha='right', fontsize=8)

    # spacer row: blank + labels “DE” and “NDE”
    for j in range(n_cols):
        axes[3, j].axis('off')
        axes[3, j].set_facecolor('white')
    # big tags at the left of the blocks
    axes[1, 0].text(-0.20, 0.5, "DE",  transform=axes[1,0].transAxes,
                    va='center', ha='right', fontsize=13, fontweight='bold')
    axes[5, 0].text(-0.20, 0.5, "NDE", transform=axes[5,0].transAxes,
                    va='center', ha='right', fontsize=13, fontweight='bold')
    # thin line across the spacer
    for j in range(n_cols):
        axes[3, j].set_facecolor('white')
    # draw lines on the axes above and below the spacer
    for j in range(n_cols):
        axes[2, j].spines['bottom'].set_color('white')
        axes[2, j].spines['bottom'].set_linewidth(3)
        axes[4, j].spines['top'].set_color('white')
        axes[4, j].spines['top'].set_linewidth(3)

    outname = f"psz2_psfmatch_repro_cs{crop_size[-1]}_out{images.shape[-1]}_p{percentile_lo}-{percentile_hi}.pdf"
    plt.savefig(outname, dpi=300, bbox_inches="tight", pad_inches=0.02)
    
    mean50  = np.nanmean(rms_50_list)  if len(rms_50_list)  else np.nan
    mean100 = np.nanmean(rms_100_list) if len(rms_100_list) else np.nan
    print(f"RMS(RAW→50 vs T50)={mean50:.4g}  RMS(RAW→100 vs T100)={mean100:.4g}  using percentiles {percentile_lo}-{percentile_hi}")
    plt.close(fig)
    print(f"Wrote {outname}")
    return mean50, mean100



if __name__ == "__main__": 
    result = load_galaxies(
        galaxy_class=[50, 51],
        fold=5,
        crop_size=crop_size,
        downsample_size=(3,1,128,128),
        sample_size=500,
        REMOVEOUTLIERS=True,
        BALANCE=False,
        FLUX_CLIPPING=False,
        STRETCH=False,
        percentile_lo=60,
        percentile_hi=95,
        AUGMENT=False,
        NORMALISE=False,
        NORMALISETOPM=False,
        EXTRADATA=True,
        train=False,
    )
    eval_imgs   = result[2]
    eval_labels = result[3]  # eval_labels lives at index 3
    eval_fns    = result[5]    
    P_LO, P_HI = 60, 95
    plot_galaxy_grid(eval_imgs, eval_fns, eval_labels, percentile_lo=P_LO, percentile_hi=P_HI, sigmas=(1.0, 5.0, 10.0))


