#!/usr/bin/env python3

from utils.data_loader2 import load_galaxies
from utils.data_loader2 import apply_formatting
from utils.calc_tools import check_tensor
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy.convolution import convolve_fft, Gaussian2DKernel
import os, glob
import torch
from astropy.wcs import WCS
try:
    from reproject import reproject_interp
    HAVE_REPROJECT = True
except Exception:
    HAVE_REPROJECT = False
print(f"HAVE_REPROJECT={HAVE_REPROJECT}")
    
print("Loading scatter_galaxies/plot_gauss_blur_grid_editing.py...")

ARCSEC = np.deg2rad(1/3600)
crop_size = (1, 512, 512)  # (T, C, H, W) for 3 channels, 512x512 images

def reproject_like(arr, src_hdr, dst_hdr):
    if arr is None or src_hdr is None or dst_hdr is None:
        return None

    # 2-D celestial WCS only
    try:
        w_src = WCS(src_hdr).celestial
        w_dst = WCS(dst_hdr).celestial
    except Exception:
        w_src = w_dst = None

    if HAVE_REPROJECT and (w_src is not None) and (w_dst is not None):
        ny_out = int(dst_hdr['NAXIS2'])
        nx_out = int(dst_hdr['NAXIS1'])
        reproj, _ = reproject_interp((arr, w_src), w_dst,
                                     shape_out=(ny_out, nx_out),
                                     order='bilinear')
        return reproj.astype(float)

    # Fallback: center-alignment translation
    from scipy.ndimage import shift as _shift
    if (w_src is None) or (w_dst is None):
        return arr.astype(float)

    ny, nx = arr.shape
    (ra, dec) = w_src.wcs_pix2world([[nx/2.0, ny/2.0]], 0)[0]
    (x_dst, y_dst) = w_dst.wcs_world2pix([[ra, dec]], 0)[0]
    dx = (float(dst_hdr['NAXIS1'])/2.0) - x_dst
    dy = (float(dst_hdr['NAXIS2'])/2.0) - y_dst
    return _shift(arr, shift=(dy, dx), order=1, mode="nearest").astype(float)


def _stretch_from_p(arr, p_lo, p_hi, clip=False):
    if arr is None:
        return None
    y = (arr - p_lo) / (p_hi - p_lo + 1e-6)
    return np.clip(y, 0, 1) if clip else y

def _summ(arr):
    a = np.asarray(arr, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(min=np.nan, max=np.nan, mean=np.nan, std=np.nan)
    return dict(min=float(np.min(a)), max=float(np.max(a)),
                mean=float(np.mean(a)), std=float(np.std(a)))

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

def xcorr_shift(a, b):
    """Integer-pixel shift (dy, dx) that best aligns a to b via FFT x-corr."""
    A = np.fft.fftn(np.nan_to_num(a))
    B = np.fft.fftn(np.nan_to_num(b))
    cc = np.fft.ifftn(A * np.conj(B))
    ij = np.unravel_index(np.argmax(np.abs(cc)), cc.shape)
    dy = ij[0] if ij[0] <= a.shape[0]//2 else ij[0] - a.shape[0]
    dx = ij[1] if ij[1] <= a.shape[1]//2 else ij[1] - a.shape[1]
    return float(dy), float(dx)


def _fits_path_triplet(base_dir, real_base):
    raw_path  = f"{base_dir}/{real_base}.fits"
    t25_path  = _first(f"{base_dir}/{real_base}T25kpc*.fits")
    t50_path  = _first(f"{base_dir}/{real_base}T50kpc*.fits")
    t100_path = _first(f"{base_dir}/{real_base}T100kpc*.fits")
    return raw_path, t25_path, t50_path, t100_path

def _load_fits_arrays_scaled_old(name, crop_ch=1, out_hw=(128,128)):
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

    # Downsampled (display) versions of the taper images
    raw_cut  = _fmt(raw_arr,  raw_hdr)
    t25_cut  = _fmt(t25_arr,  t25_hdr)
    t50_cut  = _fmt(t50_arr,  t50_hdr)
    t100_cut = _fmt(t100_arr, t100_hdr)

    # convolve RAW to target beams at the NATIVE grid
    r2_25_native  = convolve_to_target_native(raw_arr, raw_hdr, t25_hdr)  if t25_hdr  is not None else None
    r2_50_native  = convolve_to_target_native(raw_arr, raw_hdr, t50_hdr)  if t50_hdr  is not None else None
    r2_100_native = convolve_to_target_native(raw_arr, raw_hdr, t100_hdr) if t100_hdr is not None else None

    # Downsample the native-convolved images to the display grid.
    # IMPORTANT: use raw_hdr here because r2_*_native lives on the RAW grid.
    rt25_cut  = _fmt(r2_25_native,  raw_hdr)  if r2_25_native  is not None else None
    rt50_cut  = _fmt(r2_50_native,  raw_hdr)  if r2_50_native  is not None else None
    rt100_cut = _fmt(r2_100_native, raw_hdr)  if r2_100_native is not None else None

    ds_factor = (int(round(Hc_raw)) / outH)
    pix_eff_arcsec = pix_native * ds_factor

    # Now we return the RT (raw→target) cuts too
    return (raw_cut, t25_cut, t50_cut, t100_cut,
            rt25_cut, rt50_cut, rt100_cut,
            raw_hdr, t25_hdr, t50_hdr, t100_hdr, pix_eff_arcsec)


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

    # 1) Reproject each T image onto the RAW grid
    t25_on_raw  = reproject_like(t25_arr,  t25_hdr,  raw_hdr)  if t25_hdr  is not None else None
    t50_on_raw  = reproject_like(t50_arr,  t50_hdr,  raw_hdr)  if t50_hdr  is not None else None
    t100_on_raw = reproject_like(t100_arr, t100_hdr, raw_hdr)  if t100_hdr is not None else None

    # 2) Convolve RAW on its native grid and convert units to Jy/beam_target
    r2_25_native  = convolve_to_target_native(raw_arr, raw_hdr, t25_hdr)  if t25_hdr  is not None else None
    r2_50_native  = convolve_to_target_native(raw_arr, raw_hdr, t50_hdr)  if t50_hdr  is not None else None
    r2_100_native = convolve_to_target_native(raw_arr, raw_hdr, t100_hdr) if t100_hdr is not None else None

    # 3) Downsample everything from the SAME (RAW) grid to the display grid
    raw_cut   = _fmt(raw_arr,       raw_hdr)
    t25_cut   = _fmt(t25_on_raw,    raw_hdr)   if t25_on_raw   is not None else None
    t50_cut   = _fmt(t50_on_raw,    raw_hdr)   if t50_on_raw   is not None else None
    t100_cut  = _fmt(t100_on_raw,   raw_hdr)   if t100_on_raw  is not None else None
    rt25_cut  = _fmt(r2_25_native,  raw_hdr)   if r2_25_native is not None else None
    rt50_cut  = _fmt(r2_50_native,  raw_hdr)   if r2_50_native is not None else None
    rt100_cut = _fmt(r2_100_native, raw_hdr)   if r2_100_native is not None else None
    
    ds_factor = (int(round(Hc_raw)) / outH)
    pix_eff_arcsec = pix_native * ds_factor

    return (raw_cut, t25_cut, t50_cut, t100_cut,
            rt25_cut, rt50_cut, rt100_cut,
            raw_hdr, t25_hdr, t50_hdr, t100_hdr, pix_eff_arcsec)
    
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

    # convert to pixel units using the full WCS Jacobian (handles rotation & anisotropy)
    def _cd_matrix_rad(hdr):
        if 'CD1_1' in hdr:
            CD = np.array([[hdr['CD1_1'], hdr.get('CD1_2', 0.0)],
                        [hdr.get('CD2_1', 0.0), hdr['CD2_2']]], dtype=float)
        else:
            pc11 = hdr.get('PC1_1', 1.0); pc12 = hdr.get('PC1_2', 0.0)
            pc21 = hdr.get('PC2_1', 0.0); pc22 = hdr.get('PC2_2', 1.0)
            CDELT1 = hdr.get('CDELT1', 1.0); CDELT2 = hdr.get('CDELT2', 1.0)
            CD = np.array([[pc11, pc12],[pc21, pc22]], dtype=float) @ np.diag([CDELT1, CDELT2])
        return CD * (np.pi/180.0)  # radians per pixel

    J = _cd_matrix_rad(raw_hdr)           # d(world)/d(pixel)
    Jinv = np.linalg.inv(J)
    C_pix = Jinv @ C_ker @ Jinv.T         # covariance in pixel coords

    w_pix, V_pix = np.linalg.eigh(C_pix)  # eigenvalues sorted ascending
    w_pix = np.clip(w_pix, 0.0, None)
    sx_pix, sy_pix = np.sqrt(w_pix[1]), np.sqrt(w_pix[0])  # major then minor
    theta = np.arctan2(V_pix[1,1], V_pix[0,1])

    return Gaussian2DKernel(x_stddev=float(sx_pix), y_stddev=float(sy_pix), theta=float(theta))


def _maybe_draw_beam(ax, j, r):
    try:
        hdr = r.get('hdr25') if j in (0,1) else r.get('hdr50') if j in (3,4) else r.get('hdr100') if j in (6,7) else None
        if hdr is None or r.get('pix_eff') is None:
            return
        bmaj_as = float(hdr['BMAJ']) * 3600.0
        bmin_as = float(hdr['BMIN']) * 3600.0
        pa_deg  = float(hdr.get('BPA', 0.0))
        maj_px  = bmaj_as / float(r['pix_eff'])
        min_px  = bmin_as / float(r['pix_eff'])
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        cx = x0 + 0.12*(x1 - x0); cy = y0 + 0.12*(y1 - y0)
        e1 = Ellipse((cx, cy), width=maj_px, height=min_px, angle=-pa_deg, fill=False, lw=1.4, ec='w', alpha=0.9)
        e2 = Ellipse((cx, cy), width=maj_px, height=min_px, angle=-pa_deg, fill=False, lw=2.2, ec='k', alpha=0.3)
        ax.add_patch(e2); ax.add_patch(e1)
    except Exception:
        pass

def _beam_solid_angle_sr(hdr):
    """Return Gaussian beam solid angle in steradians from FITS header (BMAJ/BMIN in deg)."""
    bmaj_deg = float(hdr['BMAJ'])
    bmin_deg = float(hdr['BMIN'])
    bmaj_rad = abs(bmaj_deg) * np.pi / 180.0
    bmin_rad = abs(bmin_deg) * np.pi / 180.0
    return (np.pi / (4.0 * np.log(2.0))) * bmaj_rad * bmin_rad

# REPLACE your convolve_to_target() with this
def convolve_to_target(raw_img, raw_hdr, target_hdr, pixscale_arcsec):
    """
    Convolve RAW (in Jy/beam_native) to the target beam and convert units to Jy/beam_target
    by multiplying with Ω_target / Ω_native.
    """
    ker = kernel_from_beams(raw_hdr, target_hdr, pixscale_arcsec)
    out = convolve_fft(raw_img, ker, boundary='fill',
                       fill_value=np.nan, nan_treatment='interpolate',
                       normalize_kernel=True)

    # If we have beam info, rescale Jy/beam_native → Jy/beam_target
    try:
        omega_raw = _beam_solid_angle_sr(raw_hdr)
        omega_tgt = _beam_solid_angle_sr(target_hdr)
        scale = omega_tgt / omega_raw
        out = out * scale
    except Exception as e:
        # Fall back silently if headers are missing beam keywords
        pass

    return out

def convolve_to_target_native(raw_arr, raw_hdr, target_hdr):
    pix_as = _pixscale_arcsec(raw_hdr)
    ker = kernel_from_beams(raw_hdr, target_hdr, pix_as)
    out_native = convolve_fft(
        raw_arr, ker, boundary='fill', fill_value=np.nan,
        nan_treatment='interpolate', normalize_kernel=True
    )
    # Jy/beam_native → Jy/beam_target
    try:
        scale = _beam_solid_angle_sr(target_hdr) / _beam_solid_angle_sr(raw_hdr)
        out_native = out_native * scale
    except Exception:
        pass
    return out_native

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


def plot_galaxy_grid(images, filenames, labels,
                     lo=60, hi=95, RES_PCT=99,
                     RES_CMAP     = "RdBu_r",
                     SKIP_CLIP_NORM=False,
                     SCALE_SEPARATE=False):
    """
    Layout (9 columns):
      T25, RAW→25, res25,  T50, RAW→50, res50,  T100, RAW→100, res100

    When SKIP_CLIP_NORM=True:
      • No percentile clipping or normalisation anywhere.
      • Residuals are abs(T - RAW→T) in native units, each panel uses Matplotlib autoscale.

    When SKIP_CLIP_NORM=False (default):
      • Per-row percentile clip (lo,hi) shared by that row's T and RAW→T panels.
      • Residuals use a per-row max given by RES_PCT of res.
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
        "T25 kpc", "RAW → 25 kpc", "res 25 kpc",
        "T50 kpc", "RAW → 50 kpc", "res 50 kpc",
        "T100 kpc", "RAW → 100 kpc", "res 100 kpc",
    ]
    for ax, t in zip(axes[0], col_titles): ax.set_title(t, fontsize=12, pad=6)

    # map 6 sources to 7 grid rows (skip spacer row=3)
    row_map = [0,1,2,4,5,6]
    outH, outW = images.shape[-2], images.shape[-1]

    # ---------- PASS 1: load, convolve, and paired debug ----------
    rows = []
    for i_src, grid_row in enumerate(row_map):
        name = Path(str(filenames[i_src])).stem
        (raw_cut, t25_cut, t50_cut, t100_cut,
        rt25_cut, rt50_cut, rt100_cut,
        raw_hdr, t25_hdr, t50_hdr, t100_hdr, pix_eff) = \
            _load_fits_arrays_scaled(name, crop_ch=1, out_hw=(outH, outW))

        # Debug prints: T next to RT
        check_tensor(f"t25_cut {name}",  torch.tensor(t25_cut))
        check_tensor(f"rt25kpc {name}",   torch.tensor(rt25_cut))
        check_tensor(f"t50_cut {name}",  torch.tensor(t50_cut))
        check_tensor(f"rt50kpc {name}",   torch.tensor(rt50_cut))
        check_tensor(f"t100_cut {name}", torch.tensor(t100_cut))
        check_tensor(f"rt100kpc {name}",  torch.tensor(rt100_cut))

        rows.append(dict(
            name=name, grid_row=grid_row,
            t25=t25_cut, t50=t50_cut, t100=t100_cut,
            rt25=rt25_cut, rt50=rt50_cut, rt100=rt100_cut,
            hdr25=t25_hdr, hdr50=t50_hdr, hdr100=t100_hdr,
            pix_eff=pix_eff
        ))
        
        # --- Registration sanity check on the display grid (T vs RT) ---
        for tgt, T, RT in [(25, t25_cut, rt25_cut),
                        (50, t50_cut, rt50_cut),
                        (100, t100_cut, rt100_cut)]:
            if T is not None and RT is not None:
                dy, dx = xcorr_shift(T, RT)
                print(f"{name}  T{tgt} vs RT{tgt}: estimated shift dy={dy:.2f}, dx={dx:.2f} px")

        for r in rows:
            name = r['name']
            for tgt in (25, 50, 100):
                T  = r.get(f"t{tgt}")
                RT = r.get(f"rt{tgt}")   # <- rt, not r
                if T is None or RT is None:
                    continue
                sT, sR = _summ(T), _summ(RT)
                rms = _nanrms(T, RT)
                print(f"Row {name}  T{tgt}kpc vs RT{tgt}kpc  |  "
                    f"T[min={sT['min']:.3g}, max={sT['max']:.3g}, mean={sT['mean']:.3g}, std={sT['std']:.3g}]  ||  "
                    f"RT[min={sR['min']:.3g}, max={sR['max']:.3g}, mean={sR['mean']:.3g}, std={sR['std']:.3g}]  |  "
                    f"RMSΔ={rms:.3g}"
                )

    def describe(h):
        return f"{h['BMAJ']*3600:.2f}\"×{h['BMIN']*3600:.2f}\" @ PA={h.get('BPA',0):.1f}°"
    print("RAW:", describe(raw_hdr), "  T50:", describe(t50_hdr))

    # ---------- Optional: show global clip stats (for reference only) ----------
    if not SKIP_CLIP_NORM:
        all_vals = []
        for r in rows:
            for k in ("t25","t50","t100","rt25","rt50","rt100"):
                a = r[k]
                if a is not None:
                    a = np.asarray(a, float)
                    a = a[np.isfinite(a)]
                    if a.size:
                        all_vals.append(a.ravel())
        if all_vals:
            v = np.concatenate(all_vals)
            g_lo, g_hi = np.nanpercentile(v, [lo, hi])
            if not np.isfinite(g_hi - g_lo) or g_hi <= g_lo:
                g_lo, g_hi = np.nanmin(v), np.nanmax(v)
            print(f"Global clip percentiles: {lo}→{hi} gives {g_lo:.3e} to {g_hi:.3e}")
            
    # ---------- FAST PATH: no clipping/normalisation ----------
    if SKIP_CLIP_NORM:
        # Build planes with SIGNED residuals
        for r in rows:
            res25  = (r['t25']  - r['rt25'])  if (r['t25']  is not None and r['rt25']  is not None) else None
            res50  = (r['t50']  - r['rt50'])  if (r['t50']  is not None and r['rt50']  is not None) else None
            res100 = (r['t100'] - r['rt100']) if (r['t100'] is not None and r['rt100'] is not None) else None
            r['planes'] = [r['t25'], r['rt25'], res25,
                        r['t50'], r['rt50'], res50,
                        r['t100'], r['rt100'], res100]

        # one shared scale for T/RT (positive images)
        NONRES_COLS = {0,1,3,4,6,7}
        nonres_vals = [a for r in rows for j,a in enumerate(r['planes'])
                    if a is not None and j in NONRES_COLS]
        vmin = min(float(np.nanmin(a)) for a in nonres_vals) if nonres_vals else 0.0
        vmax = max(float(np.nanmax(a)) for a in nonres_vals) if nonres_vals else 1.0

        # symmetric robust scale for signed residuals
        RES_COLS = {2,5,8}
        resid_vals = [a for r in rows for j,a in enumerate(r['planes'])
                    if a is not None and j in RES_COLS]
        if resid_vals:
            rr = np.concatenate([np.abs(a[np.isfinite(a)]).ravel() for a in resid_vals])
            R = float(np.nanpercentile(rr, RES_PCT)) or 1.0
        else:
            R = 1.0
        resid_norm = TwoSlopeNorm(vmin=-R, vcenter=0.0, vmax=+R)

        # draw
        for r in rows:
            gr = r['grid_row']
            for j, arr in enumerate(r['planes']):
                ax = axes[gr, j]
                if arr is not None:
                    if j in RES_COLS:
                        ax.imshow(arr, cmap=RES_CMAP, norm=resid_norm, origin="lower", interpolation="nearest")
                    else:
                        ax.imshow(arr, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
                        _maybe_draw_beam(ax, j, r)
                ax.set_axis_off()

        # two stacked colorbars (non-resid, resid)
        sm_nonres = ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap="viridis")
        sm_resid  = ScalarMappable(norm=resid_norm, cmap=RES_CMAP)
        cax_nonres = fig.add_axes([0.996, 0.55, 0.012, 0.38])
        cax_resid  = fig.add_axes([0.996, 0.07, 0.012, 0.38])
        fig.colorbar(sm_nonres, cax=cax_nonres, label="Jy/beam")
        fig.colorbar(sm_resid,  cax=cax_resid,  label="Δ (Jy/beam)")

        # spacer + labels (unchanged)
        for j in range(n_cols):
            axes[3, j].axis('off'); axes[3, j].set_facecolor('white')
            axes[2, j].spines['bottom'].set_color('white'); axes[2, j].spines['bottom'].set_linewidth(3)
            axes[4, j].spines['top'].set_color('white');    axes[4, j].spines['top'].set_linewidth(3)
        axes[1, 0].text(-0.20, 0.5, "DE",  transform=axes[1,0].transAxes,
                        va='center', ha='right', fontsize=13, fontweight='bold')
        axes[5, 0].text(-0.20, 0.5, "NDE", transform=axes[5,0].transAxes,
                        va='center', ha='right', fontsize=13, fontweight='bold')

        outname = "psz2_psfmatch_pairs25_50_100_signedres_TWO_CBARS.pdf"
        plt.savefig(outname, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"Wrote {outname}")
        return

    # ---------- Build planes per row (SIGNED residuals, real clipping) ----------
    rows_planes = []
    R_all = []  # collect residuals to make a single residual colorbar later

    for r in rows:
        # Per-row percentiles, shared by that row's T and RT panels
        vals = []
        for k in ("t25","t50","t100","rt25","rt50","rt100"):
            a = r[k]
            if a is None: 
                continue
            a = np.asarray(a, float)
            a = a[np.isfinite(a)]
            if a.size: vals.append(a.ravel())
        if vals:
            v = np.concatenate(vals)
            lo_i, hi_i = np.nanpercentile(v, [lo, hi])
            if not np.isfinite(hi_i - lo_i) or hi_i <= lo_i:
                lo_i, hi_i = np.nanmin(v), np.nanmax(v)
        else:
            lo_i, hi_i = 0.0, 1.0

        def _stretch_panel(arr):
            if arr is None:
                return None
            plo, phi = _nanpct(arr, lo=lo, hi=hi)
            if not np.isfinite(phi - plo) or phi <= plo:
                plo, phi = np.nanmin(arr), np.nanmax(arr)
            return _stretch_from_p(arr, plo, phi, clip=True)

        if not SCALE_SEPARATE:
            t25_s   = _stretch_from_p(r['t25'],   lo_i, hi_i, clip=True)  if r['t25']   is not None else None
            rt25_s  = _stretch_from_p(r['rt25'],  lo_i, hi_i, clip=True)  if r['rt25']  is not None else None
            t50_s   = _stretch_from_p(r['t50'],   lo_i, hi_i, clip=True)  if r['t50']   is not None else None
            rt50_s  = _stretch_from_p(r['rt50'],  lo_i, hi_i, clip=True)  if r['rt50']  is not None else None
            t100_s  = _stretch_from_p(r['t100'],  lo_i, hi_i, clip=True)  if r['t100']  is not None else None
            rt100_s = _stretch_from_p(r['rt100'], lo_i, hi_i, clip=True)  if r['rt100'] is not None else None
        else:
            t25_s   = _stretch_panel(r['t25'])
            rt25_s  = _stretch_panel(r['rt25'])
            t50_s   = _stretch_panel(r['t50'])
            rt50_s  = _stretch_panel(r['rt50'])
            t100_s  = _stretch_panel(r['t100'])
            rt100_s = _stretch_panel(r['rt100'])


        # SIGNED residuals, in the same stretched space
        res25   = (t25_s  - rt25_s)   if (t25_s  is not None and rt25_s  is not None) else None
        res50   = (t50_s  - rt50_s)   if (t50_s  is not None and rt50_s  is not None) else None
        res100  = (t100_s - rt100_s)  if (t100_s is not None and rt100_s is not None) else None

        # Robust symmetric limit for residuals (per row)
        rr = [np.abs(x[np.isfinite(x)]).ravel() for x in (res25,res50,res100) if x is not None]
        R_i = float(np.nanpercentile(np.concatenate(rr), 99.5)) if rr else 1.0
        if not np.isfinite(R_i) or R_i == 0: R_i = 1.0

        r['planes']   = [t25_s, rt25_s, res25,  t50_s, rt50_s, res50,  t100_s, rt100_s, res100]
        r['res_norm'] = TwoSlopeNorm(vmin=-R_i, vcenter=0.0, vmax=+R_i)
        rows_planes.append(r)
        if rr: R_all.append(R_i)

    # norms/which columns
    img_norm  = mcolors.Normalize(vmin=0.0, vmax=1.0)  # stretched images live in [0,1]
    RES_COLS  = {2,5,8}

    # ---------- draw ----------
    for r in rows_planes:
        gr = r['grid_row']
        for j, arr in enumerate(r['planes']):
            ax = axes[gr, j]
            if arr is not None:
                if j in RES_COLS:
                    ax.imshow(arr, cmap=RES_CMAP, norm=r['res_norm'], origin="lower", interpolation="nearest")
                else:
                    ax.imshow(arr, cmap="viridis", norm=img_norm, origin="lower", interpolation="nearest")
                    _maybe_draw_beam(ax, j, r)

            ax.set_axis_off()
        axes[gr, 0].text(-0.02, 0.5, r['name'], transform=axes[gr,0].transAxes,
                        rotation=90, va='center', ha='right', fontsize=8)

    # two stacked colorbars (shared for all rows)
    cax_nonres = fig.add_axes([0.996, 0.55, 0.012, 0.38])
    cax_resid  = fig.add_axes([0.996, 0.07, 0.012, 0.38])
    fig.colorbar(ScalarMappable(norm=img_norm, cmap="viridis"), cax=cax_nonres,
                label="stretched intensity (0–1)")
    valid_R = [float(r) for r in R_all if np.isfinite(r) and r > 0]
    R_global = max(valid_R) if valid_R else 1.0
    res_norm_global = TwoSlopeNorm(vmin=-R_global, vcenter=0.0, vmax=R_global)
    fig.colorbar(ScalarMappable(norm=res_norm_global, cmap=RES_CMAP),
                cax=cax_resid, label="Δ (stretched units)")


    # spacer + labels
    for j in range(n_cols):
        axes[3, j].axis('off'); axes[3, j].set_facecolor('white')
        axes[2, j].spines['bottom'].set_color('white'); axes[2, j].spines['bottom'].set_linewidth(3)
        axes[4, j].spines['top'].set_color('white');    axes[4, j].spines['top'].set_linewidth(3)
    axes[1, 0].text(-0.20, 0.5, "DE",  transform=axes[1,0].transAxes,
                    va='center', ha='right', fontsize=13, fontweight='bold')
    axes[5, 0].text(-0.20, 0.5, "NDE", transform=axes[5,0].transAxes,
                    va='center', ha='right', fontsize=13, fontweight='bold')

    outname = f"psz2_psfmatch_pairs25_50_100_with_residuals_ROWCLIP.pdf"
    plt.savefig(outname, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {outname}")


if __name__ == "__main__": 
    result = load_galaxies(
        galaxy_classes=[50, 51],
        versions=['T25kpc', 'T50kpc', 'T100kpc'],   # or include 'RAW' if you want it in the cube
        fold=5,
        crop_size=(1, 512, 512),                 # same physical FOV for all versions
        downsample_size=(1, 128, 128),           # common output size
        sample_size=500,
        USE_GLOBAL_NORMALISATION=False,             
        NORMALISE=False, STRETCH=False,           
        NORMALISETOPM=False, AUGMENT=False,
        EXTRADATA=False, PRINTFILENAMES=True,
        train=False, DEBUG=True
    )
    eval_imgs   = result[2]
    eval_labels = result[3]  # eval_labels lives at index 3
    eval_fns    = result[5]
    plot_galaxy_grid(eval_imgs, eval_fns, eval_labels, lo=60, hi=95, SKIP_CLIP_NORM=False, SCALE_SEPARATE=False)