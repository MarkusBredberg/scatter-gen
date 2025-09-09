#!/usr/bin/env python3

from utils.data_loader2 import load_galaxies
from utils.data_loader2 import apply_formatting
from utils.calc_tools import check_tensor
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter
import torch
from astropy.wcs import WCS
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Ellipse
from astropy.cosmology import Planck18 as COSMO
import astropy.units as u
from astropy.io import fits
from astropy.convolution import convolve_fft, Gaussian2DKernel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools import lru_cache
import os, glob
try:
    from reproject import reproject_interp
    HAVE_REPROJECT = True
except Exception:
    HAVE_REPROJECT = False
print(f"HAVE_REPROJECT={HAVE_REPROJECT}")
    
print("Loading scatter_galaxies/plot_gauss_blur_grid_editing.py...")

APPLY_UV_TAPER = os.getenv("RT_USE_UV_TAPER","0") == "1"  # default: off
UV_TAPER_FRAC  = float(os.getenv("RT_UV_TAPER_FRAC","0.0"))  # e.g. 0.2 for mild
# Single global broadening factor (calibrate once offline, then freeze)
FUDGE_GLOBAL   = float(os.getenv("RT_FUDGE_SCALE","1.00"))
REQUIRE_REDSHIFT_ONLY = os.getenv("RT_REQUIRE_Z", "1") == "1"
ARCSEC = np.deg2rad(1/3600)
crop_size = (1, 512, 512)  # (T, C, H, W) for 3 channels, 512x512 images

# --- redshift metadata from CSV ---
CLUSTER_METADATA_CSV = "/users/mbredber/scratch/data/PSZ2/cluster_metadata.csv"

def _load_cluster_meta(csv_path):
    import csv, math, os
    d = {}
    if not os.path.exists(csv_path):
        print(f"[meta] CSV not found: {csv_path}")
        return d
    with open(csv_path, newline="") as f:
        R = csv.DictReader(f)
        for row in R:
            slug = (row.get("slug") or "").strip()
            ztxt = (row.get("z") or "").strip()
            try:
                z = float(ztxt)
            except Exception:
                z = float("nan")
            if slug and (z == z) and 0.0 < z < 5.0:   # z==z filters NaN
                d[slug] = z
    print(f"[meta] loaded {len(d)} redshifts from CSV")
    return d

CLUSTER_META = _load_cluster_meta(CLUSTER_METADATA_CSV)
def _z_from_meta(name):
    """Lookup z by PSZ2 slug (we already pass base names like PSZ2G192.18+56.12)."""
    return CLUSTER_META.get(name)

@lru_cache(maxsize=None)
def _z_from_meta_any(name: str):
    """
    CSV lookup that tolerates short slugs like 'PSZ2G192.18+56' by also trying
    the resolved data directory name (e.g. 'PSZ2G192.18+56.12').
    """
    # exact
    z = CLUSTER_META.get(name)
    if z is not None: 
        return float(z)

    # also try the resolved on-disk folder name, if it exists
    bd = _find_base_dir(name)
    if bd:
        base = os.path.basename(bd)
        z = CLUSTER_META.get(base)
        if z is not None:
            return float(z)

    # tolerate spacing/underscore variants that show up in some CSVs
    for alt in (name.replace("PSZ2G", "PSZ2 G"),
                name.replace("PSZ2G", "PSZ2_G")):
        z = CLUSTER_META.get(alt)
        if z is not None:
            return float(z)

    # last resort: unambiguous prefix match (only if it’s unique)
    hits = [k for k in CLUSTER_META.keys() if k.startswith(name)]
    if len(hits) == 1:
        return float(CLUSTER_META[hits[0]])

    return None


def radial_tukey(ny, nx, alpha=0.35):
    y = np.linspace(-1, 1, ny); x = np.linspace(-1, 1, nx)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X*X + Y*Y)
    w = np.ones_like(r)
    r0 = 1.0 - alpha
    m  = (r > r0) & (r < 1.0)
    w[m]  = 0.5 * (1.0 + np.cos(np.pi*(r[m]-r0)/max(alpha,1e-9)))
    w[r>=1.0] = 0.0
    return w

def radial_hann(ny, nx):
    y = np.linspace(-1, 1, ny); x = np.linspace(-1, 1, nx)
    X, Y = np.meshgrid(x, y); r = np.sqrt(X*X + Y*Y)
    w = np.zeros_like(r); m = (r <= 1.0)
    w[m] = 0.5 * (1.0 + np.cos(np.pi * r[m]))
    return w


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


def kpc_to_arcsec(z, L_kpc):
    """Physical size → angle (arcsec) using angular-diameter distance."""
    return ((L_kpc * u.kpc) / COSMO.angular_diameter_distance(float(z))).to(u.arcsec).value

def synth_taper_header_from_kpc(raw_hdr, z, L_kpc,
                                mode="keep_ratio"):  # or "circular"
    """
    Build a *synthetic* target header with only beam keywords filled in,
    derived from redshift and the desired physical FWHM.

    mode="keep_ratio": keep RAW beam PA and axis ratio, scale to the requested
                       geometric-mean FWHM (best match to a uv-taper product).
    mode="circular":   force a circular target beam of FWHM = X kpc.
    """
    # desired FWHM on sky
    phi_as = float(kpc_to_arcsec(z, L_kpc))

    # RAW beam (arcsec)
    bmaj_r = abs(float(raw_hdr['BMAJ'])) * 3600.0
    bmin_r = abs(float(raw_hdr['BMIN'])) * 3600.0
    pa_r   = float(raw_hdr.get('BPA', 0.0))

    # never try to sharpen: target geom. mean ≥ RAW geom. mean
    phi_as = max(phi_as, np.sqrt(bmaj_r * bmin_r))

    if mode == "circular":
        bmaj_t = bmin_t = phi_as
        pa_t   = 0.0
    else:  # keep_ratio
        r = bmaj_r / bmin_r  # RAW axis ratio
        bmin_t = phi_as / np.sqrt(r)
        bmaj_t = phi_as * np.sqrt(r)
        pa_t   = pa_r

    # synth header: only beam terms matter for our kernel & Jy/beam conversion
    thdr = fits.Header()
    thdr['BMAJ'] = bmaj_t / 3600.0   # deg
    thdr['BMIN'] = bmin_t / 3600.0   # deg
    thdr['BPA']  = pa_t
    # copy the RAW WCS so any "reproject_like(..., raw_hdr, thdr)" is a no-op
    for k in ('CTYPE1','CTYPE2','CRVAL1','CRVAL2','CRPIX1','CRPIX2',
              'CDELT1','CDELT2','CD1_1','CD1_2','CD2_1','CD2_2',
              'PC1_1','PC1_2','PC2_1','PC2_2','NAXIS1','NAXIS2'):
        if k in raw_hdr: thdr[k] = raw_hdr[k]
    return thdr

def _sigma_from_fwhm_arcsec(theta_fwhm_arcsec: float) -> float:
    """Convert angular FWHM [arcsec] → Gaussian sigma [radians]."""
    return (theta_fwhm_arcsec * ARCSEC) / (2.0 * np.sqrt(2.0 * np.log(2.0)))

def _make_uv_gaussian_weight(nx, ny, dx, dy, theta_fwhm_arcsec):
    """
    Return W(u,v) = exp[-2*pi^2*sigma_theta^2 * (u^2+v^2)]
    where u,v are in cycles/radian (= wavelengths).
    """
    sigma_th = _sigma_from_fwhm_arcsec(theta_fwhm_arcsec)
    u = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    v = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    U, V = np.meshgrid(u, v)
    return np.exp(-2.0 * (np.pi**2) * (sigma_th**2) * (U*U + V*V))

def apply_uv_gaussian_taper(img, hdr_img, theta_fwhm_arcsec, pad_factor=1):
    """
    Multiply the image's Fourier plane by a Gaussian uv-taper matching the
    requested angular FWHM (from the kpc target at this z), then inverse FFT
    back to image space.  Units are preserved (Jy/beam_*whatever you passed in*).

    We intentionally *do not* do any apodization here to avoid amplitude
    biases; the taper is already smooth.  Optional zero-padding reduces
    wrap-around (pad_factor=2 is safer, 1 is fastest).
    """
    A = np.asarray(img, float)
    A = np.where(np.isfinite(A), A, 0.0)

    ny0, nx0 = A.shape
    if pad_factor > 1:
        ny, nx = int(ny0*pad_factor), int(nx0*pad_factor)
        py = (ny - ny0)//2; px = (nx - nx0)//2
        A_pad = np.pad(A, ((py, ny - ny0 - py), (px, nx - nx0 - px)),
                       mode='constant', constant_values=0.0)
        crop = (slice(py, py+ny0), slice(px, px+nx0))
    else:
        ny, nx = ny0, nx0
        A_pad = A
        crop = (slice(0, ny0), slice(0, nx0))

    # pixel sizes [radian]
    dx = abs(_cdelt_deg(hdr_img, 1)) * np.pi/180.0
    dy = abs(_cdelt_deg(hdr_img, 2)) * np.pi/180.0

    # forward FFT with continuous normalisation
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A_pad))) * (dx * dy)

    # uv Gaussian weight
    W = _make_uv_gaussian_weight(nx, ny, dx, dy, theta_fwhm_arcsec)

    # apply taper and invert
    Fw = F * W
    A_tap = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Fw))).real / (dx * dy)

    return A_tap[crop]


def get_z(name, hdr_primary):
    """Return a usable redshift for this source (CSV → header → siblings)."""
    import re, glob, os
    from astropy.io import fits
    
    z_meta = _z_from_meta_any(name)
    if z_meta is not None:
        print(f"[z] {name}: using z={z_meta:.4f} from CSV")
        return z_meta

    def _coerce_float(val):
        try:
            z = float(val)
            return z if np.isfinite(z) else None
        except Exception:
            pass
        if isinstance(val, (bytes, bytearray)):
            try: val = val.decode('utf-8', 'ignore')
            except Exception: return None
        if isinstance(val, str):
            m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', val)
            if m:
                try: return float(m.group(0))
                except Exception: return None
        return None

    CAND_KEYS = (
        'REDSHIFT','Z','Z_CL','ZCL','ZSPEC','Z_SPEC','ZPHOT','Z_PHOT',
        'Z_BEST','ZBEST','REDSHIFT_CL','REDSHIFT_SPEC','REDSHIFT_PHOT',
        'SIM_Z','Z_MEAN','Z_AVG','Z_LAMBDA','Z_LIT','ZPLANCK','Z_PSZ2'
    )
    BAD_EXACT = {'BZERO','BSCALE','TIMEZERO','MJDZERO','TZERO','ZERO','ZEROPOINT','DZ','DZ_AVG','Z_ERR','ZERR'}

    def _parse_from_keys(hdr):
        for k in CAND_KEYS:
            if k in hdr:
                z = _coerce_float(hdr[k])
                if z is not None and 0.0 < z < 5.0:
                    return z
        return None

    def _parse_by_sweep(hdr):
        for k, v in hdr.items():
            ku = str(k).upper().replace('HIERARCH ', '')
            if ku in BAD_EXACT: 
                continue
            if ('REDSHIFT' in ku) or re.search(r'(^|_)Z($|_|[A-Z0-9])', ku):
                if 'ZERO' in ku or ku.startswith('DZ'):
                    continue
                z = _coerce_float(v)
                if z is not None and 0.0 < z < 5.0:
                    return z
        return None

    # 0) CSV first
    name_base = _name_base_from_fn(name) if ("/" in str(name) or name.endswith(".fits")) else name
    z_meta = _z_from_meta(name_base)
    if z_meta is not None:
        print(f"[z] {name_base}: using z={z_meta:.4f} from CSV")
        return float(z_meta)

    # 1) header we were handed
    z = _parse_from_keys(hdr_primary) or _parse_by_sweep(hdr_primary)
    if z is not None:
        return z

    # 2) sibling headers on disk (prefer CHANDRA if present)
    base_dir = _find_base_dir(name_base)
    if base_dir:
        pats = (sorted(glob.glob(f"{base_dir}/*CHANDRA*.fits")) +
                sorted(glob.glob(f"{base_dir}/*.fits")))
        for p in pats:
            try:
                hdr = fits.getheader(p)
            except Exception:
                continue
            z = _parse_from_keys(hdr) or _parse_by_sweep(hdr)
            if z is not None:
                print(f"[z] {name_base}: using z={z:.4f} from {os.path.basename(p)}")
                return z

    raise KeyError(f"No redshift for {name_base}; not in CSV and no valid header keys.")


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

def _find_raw_path(base_dir, base_name):
    pats = sorted(glob.glob(f"{base_dir}/{base_name}*.fits"))
    if not pats:
        return None
    # Prefer files that don't look like derivatives
    BAD_SUBSTR = ("T25KPC","T50KPC","T100KPC","RMS","NOISE","MASK","BEAM","PSF","PB","WEIGHT","WT","UV")
    good = [p for p in pats if not any(s.lower() in p.lower() for s in BAD_SUBSTR)]
    return (good[0] if good else pats[0])

def _find_base_dir(name):
    variants = [
        name,
        name.replace("PSZ2G", "PSZ2 G"),
        name.replace("PSZ2G", "PSZ2_G"),
    ]
    for v in variants:
        d = _first(f"{ROOT}/fits/{v}*")
        if d: return d
    return None

def _fits_path_triplet(base_dir, real_base):
    raw_path  = f"{base_dir}/{real_base}.fits"
    t25_path  = _first(f"{base_dir}/{real_base}T25kpc*.fits")
    t50_path  = _first(f"{base_dir}/{real_base}T50kpc*.fits")
    t100_path = _first(f"{base_dir}/{real_base}T100kpc*.fits")
    return raw_path, t25_path, t50_path, t100_path

@lru_cache(maxsize=None)
def _has_redshift_by_name(name: str) -> bool:
    # CSV wins first (normalized)
    if _z_from_meta_any(name) is not None:
        return True

    base_dir = _find_base_dir(name)
    if base_dir is None:
        return False
    base = os.path.basename(base_dir)

    cand = []
    cand.extend(sorted(glob.glob(f"{base_dir}/*CHANDRA*.fits")))
    raw_guess = _find_raw_path(base_dir, base)
    if raw_guess: cand.append(raw_guess)
    cand.extend(sorted(glob.glob(f"{base_dir}/{base}T25kpc*.fits")))
    cand.extend(sorted(glob.glob(f"{base_dir}/{base}T50kpc*.fits")))
    cand.extend(sorted(glob.glob(f"{base_dir}/{base}T100kpc*.fits")))
    if not cand:
        cand = sorted(glob.glob(f"{base_dir}/*.fits"))

    for p in cand:
        try:
            hdr = fits.getheader(p)
            z = get_z(base, hdr)   # <-- use the resolved basename here
            if 0.0 < float(z) < 5.0:
                return True
        except Exception:
            continue
    return False
    
    
def _eq_fwhm_as(h):  # geometric-mean FWHM in arcsec
    return np.sqrt((abs(float(h['BMAJ']))*3600.0) * (abs(float(h['BMIN']))*3600.0))

def synth_taper_header_from_ref(raw_hdr, ref_hdr, kpc_target, kpc_ref=50.0, mode="keep_ratio"):
    """Make a target header for kpc_target by scaling a known 'ref' taper header (e.g., T50)."""
    scale = float(kpc_target) / float(kpc_ref)

    # Read reference beam (arcsec)
    bmaj_ref_as = abs(float(ref_hdr['BMAJ'])) * 3600.0
    bmin_ref_as = abs(float(ref_hdr['BMIN'])) * 3600.0
    pa_ref      = float(ref_hdr.get('BPA', raw_hdr.get('BPA', 0.0)))

    # Target geom-mean in arcsec, scaled from the ref beam
    phi_ref_as = np.sqrt(bmaj_ref_as * bmin_ref_as)
    phi_tgt_as = scale * phi_ref_as

    if mode == "circular":
        bmaj_t = bmin_t = phi_tgt_as
        pa_t   = 0.0
    else:  # keep_ratio of the REF beam
        r = bmaj_ref_as / bmin_ref_as
        bmin_t = phi_tgt_as / np.sqrt(r)
        bmaj_t = phi_tgt_as * np.sqrt(r)
        pa_t   = pa_ref

    # Never sharpen beyond RAW geom-mean
    phi_raw_as = np.sqrt((abs(float(raw_hdr['BMAJ']))*3600.0) * (abs(float(raw_hdr['BMIN']))*3600.0))
    if np.sqrt(bmaj_t * bmin_t) < phi_raw_as:
        fac = phi_raw_as / np.sqrt(bmaj_t * bmin_t)
        bmaj_t *= fac; bmin_t *= fac
        print(f"[warn] Requested {kpc_target}kpc < RAW resolution → clamped to RAW beam.")

    # Build header on RAW WCS/grid
    thdr = fits.Header()
    thdr['BMAJ'] = bmaj_t / 3600.0
    thdr['BMIN'] = bmin_t / 3600.0
    thdr['BPA']  = pa_t
    for k in ('CTYPE1','CTYPE2','CRVAL1','CRVAL2','CRPIX1','CRPIX2',
              'CDELT1','CDELT2','CD1_1','CD1_2','CD2_1','CD2_2',
              'PC1_1','PC1_2','PC2_1','PC2_2','NAXIS1','NAXIS2'):
        if k in raw_hdr: thdr[k] = raw_hdr[k]
    return thdr

def make_rt_from_ref(raw_arr, raw_hdr, ref_hdr, kpc_target, kpc_ref=50.0, do_uv_taper=True):
    """RAW → target beam defined by scaling the REF beam (e.g., 40/50)."""
    tgt_hdr = synth_taper_header_from_ref(raw_hdr, ref_hdr, kpc_target, kpc_ref, mode="keep_ratio")
    rt = convolve_to_target_native(raw_arr, raw_hdr, tgt_hdr, fudge_scale=1.0)

    if do_uv_taper:
        # Use ref geom-mean FWHM scaled to target as a uv-taper proxy (no z needed)
        theta_ref_as = _eq_fwhm_as(ref_hdr)
        theta_tgt_as = (kpc_target/float(kpc_ref)) * theta_ref_as
        rt = apply_uv_gaussian_taper(rt, raw_hdr, theta_tgt_as, pad_factor=2)
    return rt, tgt_hdr

def _name_base_from_fn(fn):
    # robust base name; strips any trailing "T25kpc" etc. if present
    stem = Path(str(fn)).stem
    # split only once to avoid clobbering names that legitimately contain 'T'
    return stem.split('T', 1)[0]

def _has_redshift_for_index(idx, filenames) -> bool:
    return _has_redshift_by_name(_name_base_from_fn(filenames[idx]))

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
    base_dir = _find_base_dir(name)
    if base_dir is None:
        raise FileNotFoundError(f"Could not locate directory for {name} under {ROOT}/fits")
    base = os.path.basename(base_dir)

    raw_path = _find_raw_path(base_dir, base)
    if raw_path is None:
        raise FileNotFoundError(f"RAW FITS not found under {base_dir} for base '{base}'")

    raw_hdr = fits.getheader(raw_path)
    raw_arr = np.squeeze(fits.getdata(raw_path)).astype(float)
    # Require a usable redshift up-front (raises KeyError if missing)
    z_required = get_z(name, raw_hdr)
    pix_native = _pixscale_arcsec(raw_hdr)

    real_base = os.path.splitext(os.path.basename(raw_path))[0]
    raw_path, t25_path, t50_path, t100_path = _fits_path_triplet(base_dir, real_base)

    # try to read TXkpc; if missing, synthesize from redshift
    def _hdr_or_synth(tpath, Xkpc, ref_hdr=None, kpc_ref=None):
        if tpath and os.path.exists(tpath):
            return fits.getheader(tpath)
        try:
            z_local = get_z(name, raw_hdr)  # will raise if not found
            return synth_taper_header_from_kpc(raw_hdr, z_local, Xkpc, mode="keep_ratio")
        except Exception as e:
            if REQUIRE_REDSHIFT_ONLY:
                raise
            if ref_hdr is not None and kpc_ref is not None:
                print(f"[z-miss] {name}: {e}. Falling back to REF {int(kpc_ref)}kpc header scaling.")
                return synth_taper_header_from_ref(raw_hdr, ref_hdr, Xkpc, kpc_ref, mode="keep_ratio")
            print(f"[z-miss] {name}: {e}. Proceeding with RAW→target-beam only on RAW grid.")
            # copy RAW beam so code does not crash (no broadening)
            th = fits.Header()
            for k in ('BMAJ','BMIN','BPA','CTYPE1','CTYPE2','CRVAL1','CRVAL2','CRPIX1','CRPIX2',
                    'CDELT1','CDELT2','CD1_1','CD1_2','CD2_1','CD2_2',
                    'PC1_1','PC1_2','PC2_1','PC2_2','NAXIS1','NAXIS2'):
                if k in raw_hdr: th[k] = raw_hdr[k]
            return th

    # Build in an order that allows ref-scaling fallbacks (prefer T50 as ref)
    t50_hdr  = _hdr_or_synth(t50_path,  50)
    t25_hdr  = _hdr_or_synth(t25_path,  25, ref_hdr=t50_hdr,  kpc_ref=50.0)
    t100_hdr = _hdr_or_synth(t100_path, 100, ref_hdr=t50_hdr, kpc_ref=50.0)

    # data arrays only if files exist
    t25_arr  = np.squeeze(fits.getdata(t25_path)).astype(float)   if t25_path  else None
    t50_arr  = np.squeeze(fits.getdata(t50_path)).astype(float)   if t50_path  else None
    t100_arr = np.squeeze(fits.getdata(t100_path)).astype(float)  if t100_path else None


    ch, Hc_raw, Wc_raw = crop_ch, crop_size[-2], crop_size[-1]
    outH, outW = out_hw

    # 1) Keep the *tapered* images on their own grids
    t25_on_t  = t25_arr
    t50_on_t  = t50_arr
    t100_on_t = t100_arr

    # 2) Convolve RAW on its native grid to the target beam
    # Anti-alias only when we actually shrink the image
    ds = int(round(crop_size[-1] / out_hw[1]))  # e.g. 512→128 ⇒ ds≈4
    if ds > 1:
        raw_arr_prefiltered = gaussian_filter(raw_arr, sigma=0.5*ds, mode='nearest')
    else:
        raw_arr_prefiltered = raw_arr
    
    # Compute angular sizes of the targets at this z
    try:
        z = get_z(name, raw_hdr)  # raises if not found
        theta25_as  = kpc_to_arcsec(z,  25.0)
        theta50_as  = kpc_to_arcsec(z,  50.0)
        theta100_as = kpc_to_arcsec(z, 100.0)
    except Exception as e:
        if REQUIRE_REDSHIFT_ONLY:
            raise
        print(f"[z-miss] {name}: {e}. Proceeding without uv-taper; convolving RAW→target beam only.")


    # 2a) convolve RAW → target restoring beam (always)
    r2_25_native  = convolve_to_target_native(raw_arr_prefiltered, raw_hdr, t25_hdr,  fudge_scale=FUDGE_GLOBAL)
    r2_50_native  = convolve_to_target_native(raw_arr_prefiltered, raw_hdr, t50_hdr,  fudge_scale=FUDGE_GLOBAL)
    r2_100_native = convolve_to_target_native(raw_arr_prefiltered, raw_hdr, t100_hdr, fudge_scale=FUDGE_GLOBAL)

    # 2b) uv-taper (disable by default to avoid double-broadening)
    if APPLY_UV_TAPER and UV_TAPER_FRAC > 0:
        r2_25_native  = apply_uv_gaussian_taper(r2_25_native,  raw_hdr, UV_TAPER_FRAC*theta25_as,  pad_factor=2)
        r2_50_native  = apply_uv_gaussian_taper(r2_50_native,  raw_hdr, UV_TAPER_FRAC*theta50_as,  pad_factor=2)
        r2_100_native = apply_uv_gaussian_taper(r2_100_native, raw_hdr, UV_TAPER_FRAC*theta100_as, pad_factor=2)

    # 3) ➜ Reproject **convolved RAW** onto the *tapered* grids
    rt25_on_t  = reproject_like(r2_25_native,  raw_hdr, t25_hdr)  if t25_hdr  is not None else None
    rt50_on_t  = reproject_like(r2_50_native,  raw_hdr, t50_hdr)  if t50_hdr  is not None else None
    rt100_on_t = reproject_like(r2_100_native, raw_hdr, t100_hdr) if t100_hdr is not None else None

    # 4) Downsample everything from its own tapered grid → display size
    def _fmt(arr, ver_hdr):
        if ver_hdr is None or arr is None:
            return None
        scale = abs(raw_hdr['CDELT1'] / ver_hdr['CDELT1'])  # keeps same sky size as RAW crop
        Hc = int(round(Hc_raw * scale)); Wc = int(round(Wc_raw * scale))
        ten = torch.from_numpy(arr).unsqueeze(0).float()
        ten = apply_formatting(ten, (ch, Hc, Wc), (ch, outH, outW)).squeeze(0).numpy()
        return ten

    t25_cut   = _fmt(t25_on_t,  t25_hdr)
    t50_cut   = _fmt(t50_on_t,  t50_hdr)
    t100_cut  = _fmt(t100_on_t, t100_hdr)
    rt25_cut  = _fmt(rt25_on_t, t25_hdr)
    rt50_cut  = _fmt(rt50_on_t, t50_hdr)
    rt100_cut = _fmt(rt100_on_t, t100_hdr)
    
    if t25_cut  is not None: check_tensor(f"t25_cut {name}",  torch.tensor(t25_cut))
    if rt25_cut is not None: check_tensor(f"rt25kpc {name}",   torch.tensor(rt25_cut))
    if t50_cut  is not None: check_tensor(f"t50_cut {name}",  torch.tensor(t50_cut))
    if rt50_cut is not None: check_tensor(f"rt50kpc {name}",   torch.tensor(rt50_cut))
    if t100_cut is not None: check_tensor(f"t100_cut {name}", torch.tensor(t100_cut))
    if rt100_cut is not None: check_tensor(f"rt100kpc {name}", torch.tensor(rt100_cut))

    raw_cut = _fmt(raw_arr, raw_hdr)

    ds_factor = (int(round(Hc_raw)) / outH)
    pix_eff_arcsec = pix_native * ds_factor
    hdr_fft = make_fft_header(raw_hdr, pix_eff_arcsec, (outH, outW))

    return (raw_cut, t25_cut, t50_cut, t100_cut,
            rt25_cut, rt50_cut, rt100_cut,
            raw_hdr, t25_hdr, t50_hdr, t100_hdr, pix_eff_arcsec, hdr_fft)

    
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
    
    # inside kernel_from_beams(), after s_major/s_minor/theta are computed
    nker = int(np.ceil(8.0 * max(s_major, s_minor))) | 1  # make it odd
    ker  = Gaussian2DKernel(x_stddev=s_major, y_stddev=s_minor, theta=theta,
                            x_size=nker, y_size=nker)
    #ker = Gaussian2DKernel(x_stddev=s_major, y_stddev=s_minor, theta=theta)
    return ker

def make_rt_from_raw(raw_arr, raw_hdr, z, L_kpc, mode="keep_ratio"):
    tgt_hdr = synth_taper_header_from_kpc(raw_hdr, z, L_kpc, mode=mode)
    rt = convolve_to_target_native(raw_arr, raw_hdr, tgt_hdr, fudge_scale=1.0)
    return rt, tgt_hdr

def make_fft_header(raw_hdr, pix_eff_arcsec, out_hw):
    """Header describing the downsampled cutout grid for correct uv axes."""
    h = fits.Header()
    # carry the celestial frame/orientation
    for k in ('CTYPE1','CTYPE2','CRVAL1','CRVAL2','PC1_1','PC1_2','PC2_1','PC2_2'):
        if k in raw_hdr: h[k] = raw_hdr[k]
    # choose CDELT form unless your RAW uses CD
    if 'CD1_1' in raw_hdr or 'CD2_2' in raw_hdr:
        # scale the CD matrix to the effective pixel size
        s1 = np.sign(raw_hdr.get('CD1_1', raw_hdr.get('CDELT1', -1.0)))
        s2 = np.sign(raw_hdr.get('CD2_2', raw_hdr.get('CDELT2',  1.0)))
        h['CD1_1'] = s1 * (pix_eff_arcsec/3600.0); h['CD1_2'] = 0.0
        h['CD2_1'] = 0.0;                           h['CD2_2'] = s2 * (pix_eff_arcsec/3600.0)
    else:
        s1 = np.sign(raw_hdr.get('CDELT1', -1.0))
        s2 = np.sign(raw_hdr.get('CDELT2',  1.0))
        h['CDELT1'] = s1 * (pix_eff_arcsec/3600.0)
        h['CDELT2'] = s2 * (pix_eff_arcsec/3600.0)
    outH, outW = out_hw
    h['NAXIS1'] = int(outW); h['NAXIS2'] = int(outH)
    return h

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
    out = convolve_fft(raw_arr, ker, boundary='fill', fill_value=np.nan,
                   nan_treatment='interpolate', normalize_kernel=True,
                   psf_pad=True, fft_pad=True, allow_huge=True)    
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

def auto_fudge_scale(raw_img, raw_hdr, targ_hdr, T_img,
                     s_grid=np.linspace(1.00, 1.20, 11), nbins=36):
    # Work entirely on the tapered grid
    best_s, best_cost = 1.0, np.inf
    U,V,FT,AT = image_to_vis(T_img, targ_hdr, beam_hdr=targ_hdr)

    for s in s_grid:
        RT_native = convolve_to_target_native(raw_img, raw_hdr, targ_hdr, fudge_scale=s)
        RT_on_t   = reproject_like(RT_native, raw_hdr, targ_hdr)
        _,_,FR,AR = image_to_vis(RT_on_t, targ_hdr, beam_hdr=targ_hdr)

        # radial medians + coherence (same as your vis_compare_quicklook)
        r, aT = _radial_bin(U,V,AT, nbins=nbins, stat='median')
        _, aR = _radial_bin(U,V,AR, nbins=nbins, stat='median')

        # crude coherence by annulus
        Rgrid = np.sqrt(U*U + V*V)
        edges = np.geomspace(np.nanpercentile(Rgrid, 1.0), np.nanmax(Rgrid), nbins+1)
        coh = []
        for i in range(nbins):
            m = (Rgrid>=edges[i]) & (Rgrid<edges[i+1])
            if not np.any(m): coh.append(np.nan); continue
            num  = np.nanmean(FT[m]*np.conj(FR[m]))
            den1 = np.nanmean(np.abs(FT[m])**2)
            den2 = np.nanmean(np.abs(FR[m])**2)
            coh.append(np.abs(num)/np.sqrt(den1*den2))
        coh = np.asarray(coh)

        # fit only where the data are trustworthy
        m = (coh > 0.6) & np.isfinite(aT) & np.isfinite(aR) & (r > r[1]) & (r < 0.8*np.nanmax(r))
        if not np.any(m): 
            continue
        # Cost: how flat and close to 1 the ratio is (log space = equal weight)
        ratio = aT[m] / (aR[m] + 1e-12)
        cost = np.nanmedian(np.abs(np.log(ratio)))
        if cost < best_cost:
            best_cost, best_s = cost, float(s)
    return best_s

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
        A *= radial_tukey(ny, nx, alpha=alpha)
    elif window == 'hann':
        A *= radial_hann(ny, nx)
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

    # Δ|F|  <-- MISSING AXIS WAS HERE
    ax10 = fig.add_subplot(gs_maps[1,0])   # <— add this line
    dA_abs = np.abs(FT) - np.abs(FR)
    dA_rel = dA_abs / (0.5*(np.abs(FT)+np.abs(FR)) + 1e-12)
    dA_plot = np.where(keep, dA_rel, np.nan)
    finite = np.isfinite(dA_plot)
    R = (np.nanpercentile(np.abs(dA_plot[finite]), 99.5) if np.any(finite) else 1.0)
    norm_dA = TwoSlopeNorm(vmin=-R, vcenter=0.0, vmax=+R)
    im10 = ax10.imshow(dA_plot, origin='lower', cmap='RdBu_r', norm=norm_dA, aspect='equal')
    ax10.set_title('Δ|F| (relative)'); ax10.set_axis_off()
    div10 = make_axes_locatable(ax10); cax10 = div10.append_axes("right", size="4.5%", pad=0.04)
    fig.colorbar(im10, cax=cax10, label='Δ|F| / ⟨|F|⟩')

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
    good = (coh > 0.6)
    ratio_plot = np.where(good, ratio, np.nan)

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
    
hdr = fits.getheader("/users/mbredber/scratch/data/PSZ2/fits/PSZ2G192.18+56.12/PSZ2G192.18+56.12CHANDRA.fits")
print("SIM_Z raw:", repr(hdr.get("SIM_Z")))

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
    # --- keep arrays aligned ---
    n_imgs = images.shape[0]
    n = min(n_imgs, len(filenames), len(labels))
    if (n_imgs != n) or (len(filenames) != n) or (len(labels) != n):
        print(f"[warn] aligning arrays: images={n_imgs}, filenames={len(filenames)}, labels={len(labels)} → n={n}")
    images    = images[:n]
    filenames = filenames[:n]
    labels    = labels[:n]

    # gather all candidates per class
    de_all  = [i for i, y in enumerate(labels) if int(y) == 50]
    nde_all = [i for i, y in enumerate(labels) if int(y) == 51]

    def _take_with_z(idxs, need=3):
        have, removed = [], []
        for i in idxs:
            if _has_redshift_for_index(i, filenames):
                have.append(i)
                if len(have) == need:
                    break
            else:
                removed.append(_name_base_from_fn(filenames[i]))
        return have, removed

    if REQUIRE_REDSHIFT_ONLY:
        de_idx,  de_rm  = _take_with_z(de_all,  need=3)
        nde_idx, nde_rm = _take_with_z(nde_all, need=3)
        skipped = de_rm + nde_rm
        if skipped:
            preview = ", ".join(skipped[:12]) + (" ..." if len(skipped) > 12 else "")
            print(f"[skip-z] removed {len(skipped)} sources with no redshift: {preview}")
    else:
        de_idx  = de_all[:3]
        nde_idx = nde_all[:3]

        
    if len(de_idx) == 0 and len(nde_idx) == 0:
        sample = [ _name_base_from_fn(f) for f in filenames[:6] ]
        for nm in sample:
            bd = _find_base_dir(nm)
            print(f"[z-scan] {nm}: base_dir={bd}")
            if bd and os.path.isdir(bd):
                for p in sorted(glob.glob(f"{bd}/*.fits"))[:6]:
                    try:
                        hdr = fits.getheader(p)
                        keys = [k for k in hdr.keys() if any(x in k.upper() for x in ("Z","RED"))]
                        print("   ", os.path.basename(p), "keys:", keys)
                    except Exception as e:
                        print("   ", os.path.basename(p), "err:", e)
        raise RuntimeError("No sources with redshift found by header check. Verify filenames/dirs and header keys.")

    # If nothing survives, fall back: take the first few per class and (later) skip uv-taper
    if len(de_idx) == 0 and len(nde_idx) == 0:
        raise RuntimeError("No sources with redshift found by header check. Verify filenames/dirs and header keys.")

    order = de_idx + nde_idx
    if len(order) == 0:
        print("[error] Nothing to plot (no candidates at all).")
        return
    
    # Make sure indices are valid for the (now aligned) arrays
    order = [i for i in order if 0 <= i < len(images)]
    if not order:
        print("[error] selection produced no valid indices after bounds-check.")
        return


    images    = images[order]
    filenames = [filenames[i] for i in order]

    # dynamic grid
    top    = min(3, len(de_idx))
    bot    = min(3, len(nde_idx))
    spacer = 1 if (top > 0 and bot > 0) else 0

    n_cols = 9
    n_rows = top + spacer + bot
    cell   = 1.6
    height_ratios = [1]*top + ([0.12] if spacer else []) + [1]*bot

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols*cell, n_rows*cell*0.98),
        gridspec_kw=dict(
            left=0.04, right=0.995, top=0.92, bottom=0.05,
            wspace=0.04, hspace=0.04,
            height_ratios=height_ratios),
        constrained_layout=False
    )
    # ensure axes is 2D even when n_rows==1
    import numpy as _np
    if n_rows == 1:
        axes = _np.array([axes])

    col_titles = [
        "T25 kpc", "RAW → 25 kpc", "res 25 kpc",
        "T50 kpc", "RAW → 50 kpc", "res 50 kpc",
        "T100 kpc", "RAW → 100 kpc", "res 100 kpc",
    ]
    for ax, t in zip(axes[0], col_titles):
        ax.set_title(t, fontsize=12, pad=6)

    # row positions for the chosen sources (skip the spacer index if present)
    row_map = list(range(top)) + list(range(top + spacer, top + spacer + bot))
    outH, outW = images.shape[-2], images.shape[-1]

    # ---------- PASS 1: load, convolve, and paired debug ----------
    rows = []
    for i_src, grid_row in enumerate(row_map):
        name = _name_base_from_fn(filenames[i_src])
        try:
            (raw_cut, t25_cut, t50_cut, t100_cut,
            rt25_cut, rt50_cut, rt100_cut,
            raw_hdr, t25_hdr, t50_hdr, t100_hdr,
            pix_eff, hdr_fft) = _load_fits_arrays_scaled(name, crop_ch=1, out_hw=(outH, outW))
        except Exception as e:
            print(f"[skip-z] {name}: {e}")
            continue

        # Debug prints: T next to RT
        if t25_cut  is not None: check_tensor(f"t25_cut {name}",  torch.tensor(t25_cut))
        if rt25_cut is not None: check_tensor(f"rt25kpc {name}",  torch.tensor(rt25_cut))
        if t50_cut  is not None: check_tensor(f"t50_cut {name}",  torch.tensor(t50_cut))
        if rt50_cut is not None: check_tensor(f"rt50kpc {name}",  torch.tensor(rt50_cut))
        if t100_cut is not None: check_tensor(f"t100_cut {name}", torch.tensor(t100_cut))
        if rt100_cut is not None:  check_tensor(f"rt100kpc {name}", torch.tensor(rt100_cut))    
        
        rows.append(dict(
            name=name, grid_row=grid_row,
            t25=t25_cut, t50=t50_cut, t100=t100_cut,
            rt25=rt25_cut, rt50=rt50_cut, rt100=rt100_cut,
            hdr25=t25_hdr, hdr50=t50_hdr, hdr100=t100_hdr,
            hdr_fft=hdr_fft, pix_eff=pix_eff
        ))
        
        # -------- Controls and checks --------
        if DO_VIS:
            if (t25_cut is not None) and (rt25_cut is not None):
                vis_compare_quicklook(name, t25_cut,  rt25_cut,  hdr_fft, t25_hdr,  tag='T25')
            if (t50_cut is not None) and (rt50_cut is not None):
                vis_compare_quicklook(name, t50_cut,  rt50_cut,  hdr_fft, t50_hdr,  tag='T50')
            if (t100_cut is not None) and (rt100_cut is not None):
                vis_compare_quicklook(name, t100_cut, rt100_cut, hdr_fft, t100_hdr, tag='T100')



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
    try:
        if (raw_hdr is not None) and (t50_hdr is not None):
            print("RAW:", describe(raw_hdr), "  T50:", describe(t50_hdr))
    except NameError:
        pass


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

        # spacer strip + group labels (dynamic)
        if spacer:
            s = top
            for j in range(n_cols):
                axes[s, j].axis('off')
                axes[s, j].set_facecolor('white')
                if s-1 >= 0:
                    axes[s-1, j].spines['bottom'].set_color('white')
                    axes[s-1, j].spines['bottom'].set_linewidth(3)
                if s+1 < n_rows:
                    axes[s+1, j].spines['top'].set_color('white')
                    axes[s+1, j].spines['top'].set_linewidth(3)

        # place "DE" and "NDE" at the center row of each group
        if top > 0:
            r_mid_de = (top-1)//2
            axes[r_mid_de, 0].text(-0.20, 0.5, "DE",
                                transform=axes[r_mid_de,0].transAxes,
                                va='center', ha='right', fontsize=13, fontweight='bold')
        if bot > 0:
            r0_bot   = top + spacer
            r_mid_nd = r0_bot + (bot-1)//2
            axes[r_mid_nd, 0].text(-0.20, 0.5, "NDE",
                                transform=axes[r_mid_nd,0].transAxes,
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

    # spacer strip + group labels (dynamic)
    if spacer:
        s = top
        for j in range(n_cols):
            axes[s, j].axis('off')
            axes[s, j].set_facecolor('white')
            if s-1 >= 0:
                axes[s-1, j].spines['bottom'].set_color('white')
                axes[s-1, j].spines['bottom'].set_linewidth(3)
            if s+1 < n_rows:
                axes[s+1, j].spines['top'].set_color('white')
                axes[s+1, j].spines['top'].set_linewidth(3)

    if top > 0:
        r_mid_de = (top-1)//2
        axes[r_mid_de, 0].text(-0.20, 0.5, "DE",
                               transform=axes[r_mid_de,0].transAxes,
                               va='center', ha='right', fontsize=13, fontweight='bold')
    if bot > 0:
        r0_bot   = top + spacer
        r_mid_nd = r0_bot + (bot-1)//2
        axes[r_mid_nd, 0].text(-0.20, 0.5, "NDE",
                               transform=axes[r_mid_nd,0].transAxes,
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
    plot_galaxy_grid(eval_imgs, eval_fns, eval_labels, lo=60, hi=100, SKIP_CLIP_NORM=False, SCALE_SEPARATE=True, DO_VIS=True)