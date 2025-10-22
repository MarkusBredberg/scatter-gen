from utils.data_loader import load_galaxies
from utils.data_loader import apply_formatting
from utils.calc_tools import check_tensor
import numpy as np
import torch
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as _imgshift
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.wcs import WCS
from astropy.cosmology import Planck18 as COSMO
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.convolution import convolve_fft, Gaussian2DKernel
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os, glob, re

def _fft_forward(A, dx, dy):
    """Continuous-norm forward FFT used elsewhere in this file: F = FFT(A) * (dx*dy)."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A))) * (dx * dy)

def _ifft_inverse(F, dx, dy):
    """Inverse for the above convention: a = IFFT(F) / (dx*dy)."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F))).real / (dx * dy)

def _make_W_from_sigma_theta(U, V, sigma_theta_rad, anisotropic=None):
    """
    Analytic W(u,v). If 'anisotropic' is a 2x2 covariance matrix (rad^2),
    use exp[-2π^2 k^T C k]. Otherwise assume isotropic sigma_theta_rad.
    """
    if anisotropic is not None:
        # U,V are in cycles/radian. Build quadratic form per pixel.
        # k = [U, V]; exponent = -2π^2 * (k^T C k)
        a, b, c = float(anisotropic[0,0]), float(anisotropic[0,1]), float(anisotropic[1,1])
        # k^T C k = a U^2 + 2b U V + c V^2
        Q = a*(U*U) + 2.0*b*(U*V) + c*(V*V)
        return np.exp(-2.0 * (np.pi**2) * Q)
    else:
        R2 = (U*U + V*V)
        return np.exp(-2.0 * (np.pi**2) * (sigma_theta_rad**2) * R2)
    

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
    pix_native = _pixscale_arcsec(raw_hdr)

    real_base = os.path.splitext(os.path.basename(raw_path))[0]
    raw_path, t25_path, t50_path, t100_path = _fits_path_triplet(base_dir, real_base)

    # try to read TXkpc; if missing, synthesize from redshift
    def _hdr_or_synth(tpath, Xkpc, ref_hdr=None, kpc_ref=None):
        """
        Prefer on-disk header; otherwise build synthetic beam.
        Try z→kpc; if z is missing but a reference taper header is available
        (e.g. T50), fall back to scaling that header to the requested kpc.
        """
        if tpath and os.path.exists(tpath):
            return fits.getheader(tpath)
        try:
            z_local = get_z(name, raw_hdr)
            return synth_taper_header_from_kpc(raw_hdr, z_local, Xkpc, mode="keep_ratio")
        except Exception as e:
            if ref_hdr is not None and kpc_ref is not None:
                print(f"[z-miss] {name}: {e}. Falling back to REF {int(kpc_ref)}kpc header scaling.")
                return synth_taper_header_from_ref(raw_hdr, ref_hdr, Xkpc, kpc_ref, mode="keep_ratio")
            print(f"[z-miss] {name}: {e}. Proceeding with RAW→target-beam only on RAW grid.")
            # Final fallback: copy RAW beam so code does not crash (no broadening).
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
    
    # Try redshift ➜ kpc→arcsec. If missing, skip uv-taper (still make rtX via image-domain PSF match).
    theta25_as = theta50_as = theta100_as = None
    try:
        z = get_z(name, raw_hdr)
        theta25_as  = kpc_to_arcsec(z,  25.0)
        theta50_as  = kpc_to_arcsec(z,  50.0)
        theta100_as = kpc_to_arcsec(z, 100.0)
    except Exception as e:
        print(f"[z-miss] {name}: {e}. Proceeding without uv-taper; convolving RAW→target beam only.")
        
    # --- per-taper fudge to fix the long-baseline mismatch (esp. T25) ---
    s25 = s50 = s100 = FUDGE_GLOBAL
    if AUTO_FUDGE:
        if t25_arr is not None:
            s25 = auto_fudge_scale(raw_arr_prefiltered, raw_hdr, t25_hdr, t25_arr,
                                s_grid=np.linspace(1.00, 1.35, 15), nbins=48)
            print(f"[fudge] {name} T25: using s={s25:.3f}")
        if t50_arr is not None:
            s50 = auto_fudge_scale(raw_arr_prefiltered, raw_hdr, t50_hdr, t50_arr,
                                s_grid=np.linspace(1.00, 1.20, 11), nbins=48)
            print(f"[fudge] {name} T50: using s={s50:.3f}")
        if t100_arr is not None:
            s100 = auto_fudge_scale(raw_arr_prefiltered, raw_hdr, t100_hdr, t100_arr,
                                    s_grid=np.linspace(1.00, 1.15, 8), nbins=48)
            print(f"[fudge] {name} T100: using s={s100:.3f}")


    # 2a) convolve RAW → target restoring beam (always)
    r2_25_native  = convolve_to_target(raw_arr_prefiltered, raw_hdr, t25_hdr,  fudge_scale=s25)
    r2_50_native  = convolve_to_target(raw_arr_prefiltered, raw_hdr, t50_hdr,  fudge_scale=s50)
    r2_100_native = convolve_to_target(raw_arr_prefiltered, raw_hdr, t100_hdr, fudge_scale=s100)

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
        # use robust pixel size along x
        s_raw = abs(_cdelt_deg(raw_hdr, 1))
        s_ver = abs(_cdelt_deg(ver_hdr, 1))
        scale = s_raw / s_ver
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
    


def reproject_like(arr, src_hdr, dst_hdr):
    """
    Reproject a 2-D image from src_hdr WCS to dst_hdr WCS.

    If the 'reproject' package is available and both headers contain a valid
    2-D celestial WCS, use bilinear interpolation. Otherwise fall back to a
    center-alignment translation (keeps shape, best-effort alignment).
    """
    if arr is None or src_hdr is None or dst_hdr is None:
        return None

    # FAST PATH: identical grid → return as-is
    if _same_pixel_grid(src_hdr, dst_hdr):
        return np.asarray(arr, float)

    # Otherwise do the real thing (but with a cached WCS)
    try:
        w_src = _wcs_celestial_cached(src_hdr)
        w_dst = _wcs_celestial_cached(dst_hdr)
    except Exception:
        w_src = w_dst = None

    if HAVE_REPROJECT and (w_src is not None) and (w_dst is not None):
        ny_out = int(dst_hdr['NAXIS2']); nx_out = int(dst_hdr['NAXIS1'])
        reproj, _ = reproject_interp((arr, w_src), w_dst,
                                     shape_out=(ny_out, nx_out),
                                     order='bilinear')
        return reproj.astype(float)

    # Fallback: simple center alignment with subpixel shift
    if (w_src is None) or (w_dst is None):
        return arr.astype(float)

    ny, nx = arr.shape
    (ra, dec) = w_src.wcs_pix2world([[nx/2.0, ny/2.0]], 0)[0]
    (x_dst, y_dst) = w_dst.wcs_world2pix([[ra, dec]], 0)[0]
    dx = (float(dst_hdr['NAXIS1'])/2.0) - x_dst
    dy = (float(dst_hdr['NAXIS2'])/2.0) - y_dst
    return _imgshift(arr, shift=(dy, dx), order=1, mode="nearest").astype(float)


def kpc_to_arcsec(z, L_kpc):
    """Physical size → angle (arcsec) using θ ≈ L / D_A."""
    D_A = COSMO.angular_diameter_distance(float(z)).to(u.kpc)
    theta = (L_kpc * u.kpc / D_A) * u.rad  # mark the dimensionless ratio as radians
    return theta.to(u.arcsec).value

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

def plot_one_source_full(
    name,
    target_kpc=50,                   # 25, 50, or 100
    out_hw=(128, 128),
    fudge_scale_demo=1.0,
    save_dir="uvmaps",
    blur_fwhm_px=None,               # legacy, ignored (kernel comes from beams)
    use_anisotropic_beam_demo=False, # legacy, ignored (anisotropy handled via beams)
    **kwargs                          # absorb any other legacy kwargs safely
):
    """
    Make a single 2x4 figure for `name` showing both routes:

      TOP (image plane; shared colorbar for first 3):
        [ I ,  I ⊗ G  (image-space RT) ,  𝔉⁻¹{ 𝔉(I) · W } (uv-space RT) ,  G ]

      BOTTOM (uv plane; shared colorbar for first 3, log10 amplitude):
        [ |𝔉(I)| ,  |𝔉(I ⊗ G)| ,  |𝔉(I) · W| ,  W(u,v) ]

    Notes
    -----
    • G is the Gaussian kernel that turns the RAW restoring beam into the TARGET beam
      (from `kernel_from_beams_cached` on the RAW grid), normalized to unit integral.
    • W(u,v) is the continuous-norm FFT of G, rescaled so DC=1. This ensures that
      𝔉(I ⊗ G) = 𝔉(I) · W holds numerically on the same grid.
    • The uv-route image 𝔉⁻¹{𝔉(I)·W} is multiplied by Ω_tgt/Ω_raw to match the
      Jy/beam_target units returned by `convolve_to_target`.
    """


    os.makedirs(save_dir, exist_ok=True)

    # ---------- helpers (continuous-normalized FFT/IFFT) ----------
    def _fft_forward(A, dx, dy):
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.asarray(A, float)))) * (dx * dy)

    def _ifft_inverse(F, dx, dy):
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(np.asarray(F, complex)))).real / (dx * dy)

    # ---------- load arrays on a common, FFT-friendly grid ----------
    (raw_cut, t25_cut, t50_cut, t100_cut,
     rt25_cut, rt50_cut, rt100_cut,
     raw_hdr, t25_hdr, t50_hdr, t100_hdr,
     pix_eff_arcsec, hdr_fft) = _load_fits_arrays_scaled(name, crop_ch=1, out_hw=out_hw)

    # choose target
    if int(target_kpc) == 25:
        T, RT, H_targ = t25_cut, rt25_cut, t25_hdr
        t_tag, rt_tag = "T25", "RT25"
    elif int(target_kpc) == 50:
        T, RT, H_targ = t50_cut, rt50_cut, t50_hdr
        t_tag, rt_tag = "T50", "RT50"
    else:
        T, RT, H_targ = t100_cut, rt100_cut, t100_hdr
        t_tag, rt_tag = "T100", "RT100"


    I = raw_cut
    if I is None or RT is None or H_targ is None:
        raise RuntimeError(f"Missing planes for {name}: I={I is not None}, RT={RT is not None}, H_targ={H_targ is not None}")

    H, W = I.shape
    dx = abs(_cdelt_deg(hdr_fft, 1)) * np.pi/180.0
    dy = abs(_cdelt_deg(hdr_fft, 2)) * np.pi/180.0

    # ---------- image-space kernel G (RAW → TARGET) ----------
    ker = kernel_from_beams_cached(raw_hdr, H_targ, fudge_scale=fudge_scale_demo)

    # small, native kernel for DISPLAY
    G_small = np.asarray(ker.array, float)
    G_small /= (np.nansum(G_small) + 1e-12)  # unit-integral

    RT_img = convolve_to_target(I, raw_hdr, H_targ, fudge_scale=fudge_scale_demo)
    
    # ---------- uv-space weight W from analytic Gaussian (DC=1) ----------
    # uv grid in cycles/radian
    u = np.fft.fftshift(np.fft.fftfreq(W, d=dx))
    v = np.fft.fftshift(np.fft.fftfreq(H, d=dy))
    U, V = np.meshgrid(u, v)

    # world-space covariance of RAW→TARGET kernel (rad^2)
    Cw = _kernel_world_covariance_from_headers(raw_hdr, H_targ,
                                            fudge_scale=fudge_scale_demo)
    a, b, c = float(Cw[0,0]), float(Cw[0,1]), float(Cw[1,1])

    # W(u,v) = exp(-2π² · [a U² + 2b UV + c V²]); DC=1 by construction
    Wuv = np.exp(-2.0 * (np.pi**2) * (a*(U**2) + 2.0*b*(U*V) + c*(V**2)))


    # uv-route: multiply 𝔉(I) by W, inverse FFT, then convert units to Jy/beam_target
    F_I_raw = _fft_forward(I, dx, dy)
    F_uv    = F_I_raw * Wuv
    I_uv    = _ifft_inverse(F_uv, dx, dy)
    # match convolve_to_target() output units:
    unit_scale = (_beam_solid_angle_sr(H_targ) / _beam_solid_angle_sr(raw_hdr))
    RT_uv = I_uv * unit_scale

    # For the uv-row we also need |𝔉(I ⊗ G)|
    F_img = _fft_forward(RT_img, dx, dy)            # this already includes unit_scale
    F_uv_scaled = F_uv * unit_scale                 # matches units of F_img
    
    T_true = T  # the real uv-tapered map loaded from disk and resampled
    F_true = _fft_forward(T_true, dx, dy) if T_true is not None else None


    # ---------- shared scales ----------
    # top-row shared (images)
    top_stack = np.concatenate([
        I[np.isfinite(I)].ravel(),
        RT_img[np.isfinite(RT_img)].ravel(),
        RT_uv[np.isfinite(RT_uv)].ravel(),
        (T_true[np.isfinite(T_true)].ravel() if T_true is not None else np.array([]))
    ])

    vmin_top = float(np.nanpercentile(top_stack, 1.0))
    vmax_top = float(np.nanpercentile(top_stack, 99.5))
    norm_top = mcolors.Normalize(vmin=vmin_top, vmax=vmax_top)

    # bottom-row shared (log |F| )
    A_raw  = np.log10(np.abs(F_I_raw)         + 1e-12)
    A_img  = np.log10(np.abs(F_img)           + 1e-12)
    A_uv   = np.log10(np.abs(F_uv_scaled)     + 1e-12)
    A_true = np.log10(np.abs(F_true) + 1e-12) if F_true is not None else None

    bot_vals = np.concatenate([
        A_raw[np.isfinite(A_raw)].ravel(),
        A_img[np.isfinite(A_img)].ravel(),
        A_uv [np.isfinite(A_uv )].ravel(),
        (A_true[np.isfinite(A_true)].ravel() if A_true is not None else np.array([]))
    ])

    vmin_bot = float(np.nanpercentile(bot_vals, 1.0))
    vmax_bot = float(np.nanpercentile(bot_vals, 99.5))
    norm_bot = mcolors.Normalize(vmin=vmin_bot, vmax=vmax_bot)

    # W(u,v) display (log)
    W_show = np.log10(np.abs(Wuv) + 1e-12)
    vmin_W = float(np.nanpercentile(W_show, 2.0))
    vmax_W = float(np.nanpercentile(W_show, 98.0))
    norm_W = mcolors.Normalize(vmin=vmin_W, vmax=vmax_W)

    # ---------- figure ----------
    fig = plt.figure(figsize=(16.0, 7.8))
    gs  = fig.add_gridspec(2, 5, wspace=0.10, hspace=0.16)

    # ---- TOP ROW (image plane) ----
    axI   = fig.add_subplot(gs[0,0]); im0 = axI.imshow(I,       origin='lower', cmap='viridis', norm=norm_top); axI.set_axis_off(); axI.set_title('RAW  $I(\\ell,m)$')
    axRTi = fig.add_subplot(gs[0,1]); im1 = axRTi.imshow(RT_img,origin='lower', cmap='viridis', norm=norm_top); axRTi.set_axis_off(); axRTi.set_title(f'{rt_tag}: $I\\,*\\,G$  (image space)')
    axRTu = fig.add_subplot(gs[0,2]); im2 = axRTu.imshow(RT_uv, origin='lower', cmap='viridis', norm=norm_top); axRTu.set_axis_off(); axRTu.set_title(f'{rt_tag}: $\\mathcal{{F}}^{{-1}}[\\mathcal{{F}}(I)\\,W]$  (uv space)')
    axT   = fig.add_subplot(gs[0,3]); im3 = axT.imshow(T_true,  origin='lower', cmap='viridis', norm=norm_top); axT.set_axis_off();  axT.set_title(f'{t_tag}  (from FITS)')
    axG   = fig.add_subplot(gs[0,4]); axG.imshow(G_small,       origin='lower', cmap='magma');                  axG.set_axis_off();  axG.set_title('Gaussian kernel  $G$')

    # one shared colorbar for the four image maps (exclude G)
    fig.colorbar(ScalarMappable(norm=norm_top, cmap='viridis'),
                ax=[axI, axRTi, axRTu, axT], fraction=0.046, pad=0.02,
                label='Jy/beam')

    fig.text(0.015, 0.92, 'Image plane: RAW, two RT constructions, and the true tapered map',
            fontsize=11, weight='bold')

    # ---- BOTTOM ROW (uv plane) ----
    axF0 = fig.add_subplot(gs[1,0]); imF0 = axF0.imshow(A_raw,  origin='lower', cmap='viridis', norm=norm_bot); axF0.set_axis_off(); axF0.set_title(r'RAW  $|\mathcal{F}(I)|$  (log)')
    axF1 = fig.add_subplot(gs[1,1]); imF1 = axF1.imshow(A_img,  origin='lower', cmap='viridis', norm=norm_bot); axF1.set_axis_off(); axF1.set_title(rf'{rt_tag}: $|\mathcal{{F}}(I*G)|$  (log)')
    axF2 = fig.add_subplot(gs[1,2]); imF2 = axF2.imshow(A_uv,   origin='lower', cmap='viridis', norm=norm_bot); axF2.set_axis_off(); axF2.set_title(rf'{rt_tag}: $|\mathcal{{F}}(I)\,W|$  (log)')
    axF3 = fig.add_subplot(gs[1,3]); imF3 = axF3.imshow(A_true, origin='lower', cmap='viridis', norm=norm_bot); axF3.set_axis_off(); axF3.set_title(rf'{t_tag}: $|\mathcal{{F}}(T)|$  (log)')
    axWv = fig.add_subplot(gs[1,4]); axWv.imshow(np.log10(np.abs(Wuv)+1e-12), origin='lower', cmap='viridis', norm=norm_W); axWv.set_axis_off(); axWv.set_title('$W(u,v)$  (log)')

    # one shared colorbar for the four |F| maps (exclude W)
    fig.colorbar(ScalarMappable(norm=norm_bot, cmap='viridis'),
                ax=[axF0, axF1, axF2, axF3], fraction=0.046, pad=0.02,
                label=r'$\log_{10}$ amplitude (Jy)')

    fig.text(0.015, 0.50, 'UV plane: amplitudes for RAW, both RT routes, and the true tapered map',
            fontsize=11, weight='bold')

    # overall title (use RTxx)
    fig.suptitle(f"{name}  —  {rt_tag}  —  RAW, RT, visibilities, and convolution theorem (two paths)",
                y=0.995, fontsize=12)


    out = os.path.join(save_dir, f"one_source_{name}_{targ_tag}_two_paths.pdf")
    fig.savefig(out, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"[one-source] wrote {out}")

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
    
    for probe in ["PSZ2G120.08-44", "PSZ2G150.56+58", "PSZ2G031.93+78"]:
        z = _z_from_meta(probe)
        print(f"[meta-test] {probe} → z={z}")
        
    # --- Example: make the one-source figure for the first eval example ---
    example_name = _name_base_from_fn(eval_fns[0])
    print("[one-source] building figure for:", example_name)
    plot_one_source_full(
        name=example_name,
        target_kpc=50,             # choose 25 / 50 / 100
        out_hw=(128, 128),         # should match the loader’s downsample size
        blur_fwhm_px=5.0,          # size of the demo Gaussian in image pixels
        use_anisotropic_beam_demo=False  # set True to base W(u,v) on the beam difference
    )