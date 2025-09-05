#!/usr/bin/env python3

from utils.data_loader2 import load_galaxies
from utils.data_loader2 import apply_formatting
from utils.calc_tools import check_tensor
import numpy as np
from scipy import signal
import torch
from astropy.wcs import WCS
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy.convolution import convolve_fft, Gaussian2DKernel
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os, glob
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
    """
    Reproject a 2-D image from src_hdr WCS to dst_hdr WCS.

    If the 'reproject' package is available and both headers contain a valid
    2-D celestial WCS, use bilinear interpolation. Otherwise fall back to a
    center-alignment translation (keeps shape, best-effort alignment).
    """
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

    # Fallback: simple center alignment with subpixel shift
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

def _load_fits_arrays_scaled(name, crop_ch=1, out_hw=(128,128)):
    """
    Load RAW + taper images, reproject each taper to the RAW grid, convolve RAW
    → each target on the RAW grid (and rescale to Jy/beam_target), then downsample
    EVERYTHING from the same grid to the display size 'out_hw'.

    Returns:
      (raw_cut, t25_cut, t50_cut, t100_cut,
       rt25_cut, rt50_cut, rt100_cut,
       raw_hdr, t25_hdr, t50_hdr, t100_hdr, pix_eff_arcsec)
    """
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

    # 1) Reproject tapers to the RAW grid
    t25_on_raw  = reproject_like(t25_arr,  t25_hdr,  raw_hdr)  if t25_hdr  is not None else None
    t50_on_raw  = reproject_like(t50_arr,  t50_hdr,  raw_hdr)  if t50_hdr  is not None else None
    t100_on_raw = reproject_like(t100_arr, t100_hdr, raw_hdr)  if t100_hdr is not None else None

    # 2) Convolve RAW on its native grid and convert to Jy/beam_target
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

def kernel_from_beams(raw_hdr, targ_hdr, pixscale_arcsec=None, fudge_scale=1.0):
    """
    Build a Gaussian2DKernel that turns the RAW restoring beam into the TARGET
    restoring beam on the RAW pixel grid.

    Steps:
      • form beam covariances in world coords (radians),
      • kernel covariance C_ker = C_tgt - C_raw (with optional broadening),
      • map to pixel coords using the full 2×2 WCS Jacobian (CD/PC),
      • make Gaussian2DKernel with those pixel stddevs/orientation.

    Notes
    -----
    - 'pixscale_arcsec' is unused (kept for backward compatibility).
    - Small negative eigenvalues (numerical noise) are clipped to zero.
    """
    def _fwhm_to_sigma_rad(fwhm_arcsec):
        return (fwhm_arcsec * ARCSEC) / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    def _beam_cov_radians(bmaj_as, bmin_as, pa_deg):
        sx = _fwhm_to_sigma_rad(bmaj_as)
        sy = _fwhm_to_sigma_rad(bmin_as)
        th = np.deg2rad(pa_deg)
        R  = np.array([[np.cos(th), -np.sin(th)],
                       [np.sin(th),  np.cos(th)]], dtype=float)
        S  = np.diag([sx**2, sy**2])
        return R @ S @ R.T  # world (rad^2)

    def _cd_matrix_rad(hdr):
        # 2×2 Jacobian: d(world)/d(pixel) in radians/pixel (handles rotation/anisotropy)
        if 'CD1_1' in hdr:
            CD = np.array([[hdr['CD1_1'], hdr.get('CD1_2', 0.0)],
                           [hdr.get('CD2_1', 0.0), hdr['CD2_2']]], dtype=float)
        else:
            pc11 = hdr.get('PC1_1', 1.0); pc12 = hdr.get('PC1_2', 0.0)
            pc21 = hdr.get('PC2_1', 0.0); pc22 = hdr.get('PC2_2', 1.0)
            cd1  = hdr.get('CDELT1', 1.0); cd2  = hdr.get('CDELT2', 1.0)
            CD   = np.array([[pc11, pc12],[pc21, pc22]], float) @ np.diag([cd1, cd2])
        return CD * (np.pi / 180.0)  # deg/pix → rad/pix

    # Pull beam params (arcsec + PA)
    bmaj_r = float(raw_hdr['BMAJ'])  * 3600.0
    bmin_r = float(raw_hdr['BMIN'])  * 3600.0
    pa_r   = float(raw_hdr.get('BPA', 0.0))
    bmaj_t = float(targ_hdr['BMAJ']) * 3600.0
    bmin_t = float(targ_hdr['BMIN']) * 3600.0
    pa_t   = float(targ_hdr.get('BPA', pa_r))  # fall back to RAW PA if missing

    # World-space covariances and kernel covariance
    C_raw = _beam_cov_radians(bmaj_r, bmin_r, pa_r)
    C_tgt = _beam_cov_radians(bmaj_t, bmin_t, pa_t) * (fudge_scale**2)
    C_ker = C_tgt - C_raw

    # Numerical guard
    w, V = np.linalg.eigh(C_ker)
    w = np.clip(w, 0.0, None)
    C_ker = (V * w) @ V.T

    # Map kernel covariance into pixel coordinates
    J    = _cd_matrix_rad(raw_hdr)       # d(world)/d(pixel)
    Jinv = np.linalg.inv(J)
    Cpix = Jinv @ C_ker @ Jinv.T         # pixel covariance

    # Eigen-decompose in pixel coords
    w_pix, V_pix = np.linalg.eigh(Cpix)
    w_pix = np.clip(w_pix, 0.0, None)
    s_major = float(np.sqrt(w_pix[1]))
    s_minor = float(np.sqrt(w_pix[0]))
    theta   = float(np.arctan2(V_pix[1, 1], V_pix[0, 1]))  # angle of major axis

    eps = 1e-9  # avoid zero-width kernels when beams are almost identical
    s_major = max(s_major, eps)
    s_minor = max(s_minor, eps)

    return Gaussian2DKernel(x_stddev=s_major, y_stddev=s_minor, theta=theta)


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
    """Gaussian beam solid angle in steradians from FITS header (BMAJ/BMIN in deg)."""
    bmaj_deg = float(hdr['BMAJ'])
    bmin_deg = float(hdr['BMIN'])
    bmaj_rad = abs(bmaj_deg) * np.pi / 180.0
    bmin_rad = abs(bmin_deg) * np.pi / 180.0
    return (np.pi / (4.0 * np.log(2.0))) * bmaj_rad * bmin_rad

def convolve_to_target(raw_img, raw_hdr, target_hdr, pixscale_arcsec=None, fudge_scale=1.0):
    """
    Convolve a RAW Jy/beam_native image to the TARGET beam and convert units
    to Jy/beam_target by multiplying with Ω_target / Ω_raw.
    """
    ker = kernel_from_beams(raw_hdr, target_hdr, fudge_scale=fudge_scale)
    out = convolve_fft(raw_img, ker,
                       boundary='fill', fill_value=np.nan,
                       nan_treatment='interpolate',
                       normalize_kernel=True)
    try:
        out *= (_beam_solid_angle_sr(target_hdr) / _beam_solid_angle_sr(raw_hdr))
    except Exception:
        pass
    return out

def convolve_to_target_native(raw_arr, raw_hdr, target_hdr, fudge_scale=1.0):
    """
    Same as convolve_to_target(); provided for convenience when the image is
    already on the RAW grid. Units in = Jy/beam_native; units out = Jy/beam_target.
    """
    ker = kernel_from_beams(raw_hdr, target_hdr, fudge_scale=fudge_scale)
    out = convolve_fft(raw_arr, ker,
                       boundary='fill', fill_value=np.nan,
                       nan_treatment='interpolate',
                       normalize_kernel=True)
    try:
        out *= (_beam_solid_angle_sr(target_hdr) / _beam_solid_angle_sr(raw_hdr))
    except Exception:
        pass
    return out

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


def _cdelt_deg(hdr, axis):
    """Robust pixel step [deg] for axis=1 (x≈RA) or axis=2 (y≈Dec)."""
    key = f'CDELT{axis}'
    if key in hdr:
        return float(hdr[key])
    if 'CD1_1' in hdr:  # fall back to CD
        if axis == 1:
            return float(np.hypot(hdr['CD1_1'], hdr.get('CD1_2', 0.0)))
        else:
            return float(np.hypot(hdr.get('CD2_1', 0.0), hdr['CD2_2']))
    pc11 = hdr.get('PC1_1', 1.0); pc12 = hdr.get('PC1_2', 0.0)
    pc21 = hdr.get('PC2_1', 0.0); pc22 = hdr.get('PC2_2', 1.0)
    cd1  = hdr.get('CDELT1', 1.0); cd2  = hdr.get('CDELT2', 1.0)
    M = np.array([[pc11, pc12],[pc21, pc22]], float) @ np.diag([cd1, cd2])
    return float(np.hypot(M[0, axis-1], M[1, axis-1]))

def image_to_vis(img, img_hdr, beam_hdr=None, divide_by_beam=True,
                 window='tukey', alpha=0.35, pad_factor=2, roi=0.92,
                 subtract_mean=True):
    """
    Convert a 2D sky image to its discrete Fourier transform (visibilities).

    • Optional conversion to Jy/sr (divide by Ω_beam) before the FFT.
    • Robust DC removal: subtract median inside a central elliptical ROI.
    • Strong Tukey apodization to tame edge/aliasing.
    • Optional zero-padding (×2 by default) to reduce wrap-around.
    """
    B = np.array(img, dtype=float)                  # keep NaNs for stats
    A = np.where(np.isfinite(B), B, 0.0)            # NaNs → 0 for FFT path

    # Jy/beam → Jy/sr if requested
    if divide_by_beam and (beam_hdr is not None):
        try:
            A = A / _beam_solid_angle_sr(beam_hdr)
        except Exception:
            pass

    ny, nx = A.shape

    # DC removal using central ellipse
    if subtract_mean and (0.0 < roi <= 1.0):
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
        ry, rx = (roi * ny) / 2.0, (roi * nx) / 2.0
        yy, xx = np.ogrid[:ny, :nx]
        mask_roi = (((yy - cy) / ry)**2 + ((xx - cx) / rx)**2) <= 1.0
        if np.any(mask_roi):
            dc = np.nanmedian(np.where(mask_roi, B, np.nan))
            if np.isfinite(dc):
                A = A - dc

    # Apodize
    if window == 'tukey':
        wy = signal.windows.tukey(ny, alpha=alpha)
        wx = signal.windows.tukey(nx, alpha=alpha)
        A *= np.outer(wy, wx)
    elif window == 'hann':
        wy = signal.windows.hann(ny)
        wx = signal.windows.hann(nx)
        A *= np.outer(wy, wx)

    # Zero-padding
    if pad_factor and pad_factor > 1:
        ny2, nx2 = int(ny * pad_factor), int(nx * pad_factor)
        py = (ny2 - ny) // 2; px = (nx2 - nx) // 2
        A = np.pad(A, ((py, ny2 - ny - py), (px, nx2 - nx - px)),
                   mode='constant', constant_values=0.0)
        ny, nx = A.shape

    # Pixel size [rad]
    dx = abs(_cdelt_deg(img_hdr, 1)) * np.pi / 180.0
    dy = abs(_cdelt_deg(img_hdr, 2)) * np.pi / 180.0

    # Continuous-norm FFT ⇒ |F| in Jy
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A))) * (dx * dy)

    # Frequency axes (cycles/radian = wavelengths)
    u = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    v = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    U, V = np.meshgrid(u, v)
    return U, V, F, np.abs(F)


def _radial_bin(u, v, values, nbins=36, rmin=None, rmax=None, logspace=True, stat='median'):
    """Radial bin a 2D field in uv. values can be real/abs/complex; returns r_centers, stat(values)."""
    R = np.sqrt(u*u + v*v)
    m = np.isfinite(values)
    if rmin is None: rmin = np.nanpercentile(R[m], 1.0)
    if rmax is None: rmax = np.nanmax(R[m])
    edges = (np.geomspace(rmin, rmax, nbins+1) if logspace else np.linspace(rmin, rmax, nbins+1))
    idx = np.digitize(R.ravel(), edges) - 1
    out = []
    for i in range(nbins):
        sel = (idx == i) & m.ravel()
        if not np.any(sel):
            out.append(np.nan); continue
        vals = values.ravel()[sel]
        if stat == 'median':
            out.append(np.nanmedian(vals))
        elif stat == 'mean':
            out.append(np.nanmean(vals))
        elif stat == 'abs-mean':
            out.append(np.nanmean(np.abs(vals)))
        else:
            out.append(np.nanmedian(vals))
    rc = 0.5*(edges[:-1] + edges[1:])
    return rc, np.asarray(out)

def vis_compare_quicklook(name, T, RT, hdr_img, hdr_beam, tag, outdir='.', nbins=36):
    """
    Compact figure comparing visibilities of TARGET image (T) and RAW→TARGET (RT).
    Adds:
      • annulus-wise amplitude mask for Δϕ map,
      • optional coherence mask on the ratio curve to suppress noisy spikes.
    """
    # FFTs
    U,V,FT,AT = image_to_vis(T,  hdr_img, beam_hdr=hdr_beam)
    _,_,FR,AR = image_to_vis(RT, hdr_img, beam_hdr=hdr_beam)

    # Differences
    dA   = np.abs(FT) - np.abs(FR)
    dphi = np.angle(FT * np.conj(FR))

    # Radial profiles
    r, aT = _radial_bin(U,V,AT, nbins=nbins, stat='median')
    _, aR = _radial_bin(U,V,AR, nbins=nbins, stat='median')
    ratio = aT / (aR + 1e-12)

    # Coherence and cross-phase per annulus
    Rgrid = np.sqrt(U*U + V*V)
    edges = np.geomspace(np.nanpercentile(Rgrid,1.0), np.nanmax(Rgrid), nbins+1)

    num_r, den1_r, den2_r, ph_r = [], [], [], []
    for i in range(nbins):
        m = (Rgrid>=edges[i]) & (Rgrid<edges[i+1])
        if not np.any(m):
            num_r.append(np.nan); den1_r.append(np.nan); den2_r.append(np.nan); ph_r.append(np.nan); continue
        num  = np.nanmean(FT[m] * np.conj(FR[m]))
        den1 = np.nanmean(np.abs(FT[m])**2)
        den2 = np.nanmean(np.abs(FR[m])**2)
        num_r.append(np.abs(num)); den1_r.append(den1); den2_r.append(den2); ph_r.append(np.angle(num))
    num_r = np.asarray(num_r); den1_r = np.asarray(den1_r); den2_r = np.asarray(den2_r)
    coh   = num_r / np.sqrt(den1_r * den2_r)

    # Δϕ map: keep only uv pixels above a local amplitude threshold
    amp   = 0.5*(AT + AR)
    keep  = np.zeros_like(amp, dtype=bool)
    for i in range(nbins):
        m = (Rgrid>=edges[i]) & (Rgrid<edges[i+1])
        if not np.any(m): 
            continue
        thr = np.nanpercentile(amp[m], 35.0)
        keep |= (m & (amp > thr))
    dphi_deg_map = np.where(keep, np.rad2deg(dphi), np.nan)

    # ---- figure ----
    fig = plt.figure(figsize=(12.2, 6.6))
    gs  = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.55],
                           height_ratios=[1.0, 0.42], wspace=0.26, hspace=0.28)
    gs_maps = gs[0, 0:2].subgridspec(2, 2, wspace=0.08, hspace=0.12)

    # |F_T| (log10)
    ax00 = fig.add_subplot(gs_maps[0,0])
    ax00.imshow(np.log10(AT + 1e-12), origin='lower', aspect='equal')
    ax00.set_title('|F_T| (log10)'); ax00.set_axis_off()

    # |F_RT| (log10)
    ax01 = fig.add_subplot(gs_maps[0,1])
    ax01.imshow(np.log10(AR + 1e-12), origin='lower', aspect='equal')
    ax01.set_title('|F_RT| (log10)'); ax01.set_axis_off()

    # Δ|F|
    ax10 = fig.add_subplot(gs_maps[1,0])
    Rabs = np.nanpercentile(np.abs(dA[np.isfinite(dA)]), 99.5)
    Rabs = Rabs if np.isfinite(Rabs) and (Rabs > 0) else 1.0
    norm_dA = TwoSlopeNorm(vmin=-Rabs, vcenter=0.0, vmax=+Rabs)
    im10 = ax10.imshow(dA, origin='lower', cmap='RdBu_r', norm=norm_dA, aspect='equal')
    ax10.set_title('Δ|F|'); ax10.set_axis_off()
    div10 = make_axes_locatable(ax10); cax10 = div10.append_axes("right", size="4.5%", pad=0.04)
    fig.colorbar(im10, cax=cax10, label='Δ|F| (Jy)')

    # Δϕ (deg)
    ax11 = fig.add_subplot(gs_maps[1,1])
    im11 = ax11.imshow(dphi_deg_map, origin='lower', cmap='twilight', vmin=-180, vmax=180, aspect='equal')
    ax11.set_title('Δϕ (deg)'); ax11.set_axis_off()
    div11 = make_axes_locatable(ax11); cax11 = div11.append_axes("right", size="4.5%", pad=0.04)
    fig.colorbar(im11, cax=cax11, label='Δϕ (deg)')

    # Right column: radial |F| + ratio/coherence
    axA = fig.add_subplot(gs[:, 2])
    kλ = r/1e3
    axA.plot(kλ, aT, lw=1.5, label='|F_T| (median)')
    axA.plot(kλ, aR, lw=1.5, ls='--', label='|F_RT| (median)')
    axA.set_xscale('log'); axA.set_yscale('log')
    axA.set_xlabel('baseline (kλ)'); axA.set_ylabel('median |F| (Jy)')
    axA.grid(True, which='both', ls=':', alpha=0.4)

    # Suppress meaningless ratio spikes at very low coherence
    ratio_plot = np.where(coh > 0.2, ratio, np.nan)

    axA2 = axA.twinx()
    axA2.plot(kλ, ratio_plot, lw=1.0, label='|F_T| / |F_RT|')
    axA2.plot(kλ, coh,        lw=1.0, label='coherence')
    axA2.set_ylabel('ratio / coherence')
    ymax = np.nanmax([np.nanmax(coh), np.nanmax(ratio_plot), 1.0])
    axA2.set_ylim(0, max(1.05, ymax))

    lines, labels = axA.get_legend_handles_labels()
    lines2, labels2 = axA2.get_legend_handles_labels()
    axA.legend(lines + lines2, labels + labels2, loc='lower left', fontsize=9)

    # Bottom: phase stats vs baseline (amplitude-weighted circular stats)
    axP = fig.add_subplot(gs[1, 0:2])
    phi_bar_deg, sigma_circ_deg, rphi = [], [], []
    for i in range(nbins):
        m = (Rgrid >= edges[i]) & (Rgrid < edges[i+1]) & keep & np.isfinite(dphi)
        if not np.any(m):
            phi_bar_deg.append(np.nan); sigma_circ_deg.append(np.nan); rphi.append(np.nan); continue
        w = AT[m] * AR[m]
        z = np.exp(1j * dphi[m])
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        if np.sum(w) <= 0: w = np.ones_like(w)
        C = np.sum(w * z) / np.sum(w)
        Rlen = np.abs(C)
        phi_bar_deg.append(np.rad2deg(np.angle(C)))
        sigma_circ_deg.append(np.rad2deg(np.sqrt(np.maximum(0.0, -2.0*np.log(Rlen)))))
        rphi.append(0.5*(edges[i] + edges[i+1]))
    rphi = np.asarray(rphi); phi_bar_deg = np.asarray(phi_bar_deg); sigma_circ_deg = np.asarray(sigma_circ_deg)
    axP.plot(rphi/1e3, np.abs(phi_bar_deg), lw=1.2, label='|mean Δϕ|')
    axP.plot(rphi/1e3, sigma_circ_deg, lw=1.2, ls='--', label='circular σ(Δϕ)')
    axP.set_xscale('log'); axP.set_xlabel('baseline (kλ)'); axP.set_ylabel('phase (deg)')
    axP.grid(True, which='both', ls=':', alpha=0.4); axP.legend(loc='upper left', fontsize=9)

    fig.suptitle(f'{name}  –  {tag}', y=0.98, fontsize=12)
    save_dir = os.path.join(outdir, 'uvmaps'); os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f'uvcmp_{name}_{tag}.pdf')
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f'Wrote {out}')


def plot_galaxy_grid(images, filenames, labels,
                     lo=60, hi=95, RES_PCT=99,
                     RES_CMAP     = "RdBu_r",
                     SKIP_CLIP_NORM=False,
                     SCALE_SEPARATE=False,
                     DO_VIS=False):
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
        
        
        # -------- Controls and checks --------
        if DO_VIS:
            # use RAW header (grid) + matching taper header (beam)
            vis_compare_quicklook(name, t25_cut,  rt25_cut,  raw_hdr, t25_hdr,  tag='T25',  outdir='.')
            vis_compare_quicklook(name, t50_cut,  rt50_cut,  raw_hdr, t50_hdr,  tag='T50',  outdir='.')
            vis_compare_quicklook(name, t100_cut, rt100_cut, raw_hdr, t100_hdr, tag='T100', outdir='.')

        
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
    plot_galaxy_grid(eval_imgs, eval_fns, eval_labels, lo=60, hi=95, SKIP_CLIP_NORM=True, SCALE_SEPARATE=True, DO_VIS=True)