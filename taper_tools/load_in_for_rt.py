#!/usr/bin/env python3
"""
Used to build RT functionality in the data_loader.py module.
Batch RT maker (image-space): make cropped (FOV) RT = I*G only where T50 exists.
Also save PNG samples, including single-row plots for 5 sources to verify crops.
"""

# ------------------------------ CONFIG SECTION ------------------------------
CONFIG = {
    "root": "/home/sysadmin/Scripts/scatter_galaxies/data/PSZ2/fits",
    "recursive": False,
    "overwrite": False,
    "fov_arcmin": 50.0,
    "max_sources": 5,     # ← five sources by default
}
# ---------------------------------------------------------------------------

import os, sys, glob, argparse, warnings, numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve_fft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------- FITS / WCS helpers ----------------------------
def _cd_matrix_rad(h):
    if 'CD1_1' in h:
        M = np.array([[h['CD1_1'], h.get('CD1_2', 0.0)],
                      [h.get('CD2_1', 0.0), h['CD2_2']]], float)
    else:
        pc11=h.get('PC1_1',1.0); pc12=h.get('PC1_2',0.0)
        pc21=h.get('PC2_1',0.0); pc22=h.get('PC2_2',1.0)
        cd1 =h.get('CDELT1', 1.0); cd2 =h.get('CDELT2', 1.0)
        M = np.array([[pc11, pc12],[pc21, pc22]], float) @ np.diag([cd1, cd2])
    return M * (np.pi/180.0)

def fwhm_major_as(h):
    return max(float(h["BMAJ"]), float(h["BMIN"])) * 3600.0  # arcsec

def arcsec_per_pix(h):
    J = _cd_matrix_rad(h)
    dx = np.hypot(J[0,0], J[1,0]); dy = np.hypot(J[0,1], J[1,1])
    ARCSEC_PER_RAD = 206264.80624709636
    return dx*ARCSEC_PER_RAD, dy*ARCSEC_PER_RAD

def crop_to_side_arcsec_on_raw(I, Hraw, side_arcsec, *arrs):
    """Square crop on RAW grid with side length in arcsec, centered on RAW center."""
    asx, asy = arcsec_per_pix(Hraw)
    nx = int(round(side_arcsec / asx))
    ny = int(round(side_arcsec / asy))
    m  = max(1, min(nx, ny))                   # force square, ≥1 px
    nx = min(m, I.shape[1]); ny = min(m, I.shape[0])
    cy, cx = (I.shape[0]-1)/2.0, (I.shape[1]-1)/2.0
    out = [center_crop_at(a, ny, nx, cy, cx) for a in (I,) + arrs]
    return out, (ny, nx)


def _arcsec_per_pix(h):
    J = _cd_matrix_rad(h)
    dx = np.hypot(J[0,0], J[1,0])
    dy = np.hypot(J[0,1], J[1,1])
    ARCSEC_PER_RAD = 206264.80624709636
    return dx*ARCSEC_PER_RAD, dy*ARCSEC_PER_RAD

def _fwhm_as_to_sigma_rad(fwhm_as):
    return (float(fwhm_as)/(2.0*np.sqrt(2.0*np.log(2.0)))) * (np.pi/(180.0*3600.0))

def beam_cov_world(h):
    bmaj_as = float(h['BMAJ'])*3600.0
    bmin_as = float(h['BMIN'])*3600.0
    pa_deg  = float(h.get('BPA', 0.0))
    sx, sy  = _fwhm_as_to_sigma_rad(bmaj_as), _fwhm_as_to_sigma_rad(bmin_as)
    th      = np.deg2rad(pa_deg)
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]], float)
    S = np.diag([sx*sx, sy*sy])
    return R @ S @ R.T

def beam_solid_angle_sr(h):
    bmaj = abs(float(h['BMAJ'])) * np.pi/180.0
    bmin = abs(float(h['BMIN'])) * np.pi/180.0
    return (np.pi/(4.0*np.log(2.0))) * bmaj * bmin

def kernel_from_beams(raw_hdr, tgt_hdr):
    C_raw = beam_cov_world(raw_hdr)
    C_tgt = beam_cov_world(tgt_hdr)
    C_ker = C_tgt - C_raw
    w, V = np.linalg.eigh(C_ker)
    w = np.clip(w, 0.0, None)
    C_ker = (V * w) @ V.T
    J    = _cd_matrix_rad(raw_hdr)
    Jinv = np.linalg.inv(J)
    Cpix = Jinv @ C_ker @ Jinv.T
    wp, Vp = np.linalg.eigh(Cpix)
    wp = np.clip(wp, 1e-18, None)
    s_minor = float(np.sqrt(wp[0]))
    s_major = float(np.sqrt(wp[1]))
    theta   = float(np.arctan2(Vp[1,1], Vp[0,1]))
    nker    = int(np.ceil(8.0*max(s_major, s_minor))) | 1
    return Gaussian2DKernel(x_stddev=s_major, y_stddev=s_minor, theta=theta,
                            x_size=nker, y_size=nker)

def read_fits(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = np.squeeze(fits.getdata(path)).astype(float)
        hdr  = fits.getheader(path)
    return data, hdr

def reproject_like(arr, src_hdr, dst_hdr):
    try:
        from reproject import reproject_interp
        from astropy.wcs import WCS
        w_src = WCS(src_hdr).celestial
        w_dst = WCS(dst_hdr).celestial
        ny_out = int(dst_hdr['NAXIS2']); nx_out = int(dst_hdr['NAXIS1'])
        out, _ = reproject_interp((arr, w_src), w_dst, shape_out=(ny_out, nx_out), order='bilinear')
        return out.astype(float)
    except Exception:
        from scipy.ndimage import zoom as _zoom
        ny_out = int(dst_hdr['NAXIS2']); nx_out = int(dst_hdr['NAXIS1'])
        ny_in, nx_in = arr.shape
        zy = ny_out / max(ny_in, 1); zx = nx_out / max(nx_in, 1)
        y = _zoom(arr, zoom=(zy, zx), order=1)
        y = y[:ny_out, :nx_out]
        if y.shape != (ny_out, nx_out):
            pad_y = ny_out - y.shape[0]; pad_x = nx_out - y.shape[1]
            y = np.pad(y, ((0, max(0, pad_y)), (0, max(0, pad_x))), mode='edge')[:ny_out, :nx_out]
        return y.astype(float)

def center_crop_at(arr, ny_target, nx_target, cy, cx):
    ny, nx = arr.shape
    y0 = int(round(cy - ny_target/2)); x0 = int(round(cx - nx_target/2))
    y0 = max(0, min(y0, ny - ny_target)); x0 = max(0, min(x0, nx - nx_target))
    return arr[y0:y0+ny_target, x0:x0+nx_target]

def crop_to_fov_on_raw(I, Hraw, fov_arcmin, *arrs):
    asx, asy = _arcsec_per_pix(Hraw)
    fov_as   = float(fov_arcmin) * 60.0
    nx_crop  = int(round(fov_as / asx))
    ny_crop  = int(round(fov_as / asy))
    m        = min(nx_crop, ny_crop)
    nx_crop  = min(m, I.shape[1])
    ny_crop  = min(m, I.shape[0])
    cy, cx   = (I.shape[0] - 1)/2.0, (I.shape[1] - 1)/2.0
    out = [center_crop_at(a, ny_crop, nx_crop, cy, cx) for a in (I,) + arrs]
    return out, (ny_crop, nx_crop)

def write_fits(path_out, data, hdr_base, hdr_beam=None, overwrite=False, note=None):
    h = hdr_base.copy()
    if hdr_beam is not None:
        for k in ("BMAJ","BMIN","BPA"):
            if k in hdr_beam:
                h[k] = hdr_beam[k]
    if note:
        try: h.add_history(note)
        except Exception: pass
    fits.writeto(path_out, data.astype(np.float32), h, overwrite=overwrite)

# --------------------------------- Discovery --------------------------------
def discover(root, recursive=False):
    print(f"[scan] walking: {root}  (recursive={recursive})")
    if not os.path.isdir(root):
        print(f"[scan] ERROR: root is not a directory: {root}")
        return []
    if recursive:
        dirs = []
        for d, _, files in os.walk(root):
            if any(f.lower().endswith(".fits") for f in files):
                dirs.append(d)
        print(f"[scan] recursive candidate dirs (contain .fits): {len(dirs)}")
    else:
        dirs = [p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)]
        print(f"[scan] depth-1 candidate dirs: {len(dirs)}")
    if dirs:
        first = "\n       ".join(sorted(dirs)[:5])
        print(f"[scan] first few dirs:\n       {first}" + ("" if len(dirs)<=5 else "\n       ..."))
    items = []
    matched = with_t50 = without_t50 = 0
    for d in sorted(dirs):
        base = Path(d).name
        raw = t50 = None
        for ext in (".fits", ".FITS", ".Fits"):
            cand = os.path.join(d, f"{base}{ext}")
            if os.path.exists(cand):
                raw = cand; break
        if not raw:
            continue
        for ext in (".fits", ".FITS", ".Fits"):
            cand = os.path.join(d, f"{base}T50kpc{ext}")
            if os.path.exists(cand):
                t50 = cand; break
        matched += 1
        with_t50 += int(t50 is not None)
        without_t50 += int(t50 is None)
        items.append({"dir": d, "stem": base, "raw": raw, "t50": t50})
    print(f"[scan] RAW found: {matched}  |  with T50: {with_t50}  |  without T50: {without_t50}")
    return items

# --------------------------------- PNG helpers -------------------------------
def _stretch01(a):
    a = np.asarray(a, float)
    lo, hi = np.nanpercentile(a, 1.0), np.nanpercentile(a, 99.5)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(a), np.nanmax(a)
        if hi <= lo: hi = lo + 1.0
    return np.clip((a - lo)/(hi - lo + 1e-12), 0, 1)

def save_sample_png(path_png, img, title):
    plt.figure(figsize=(4.2, 4.0))
    plt.imshow(_stretch01(img), origin='lower', cmap='viridis', interpolation='nearest')
    plt.axis('off'); plt.title(title, fontsize=10)
    plt.tight_layout(pad=0.1)
    plt.savefig(path_png, dpi=180, bbox_inches='tight')
    plt.close()

def save_grid_3xN_from_triplets(path_png, samples, ncols=5):
    cols = min(ncols, len(samples))
    if cols < 1:
        return
    fig, axs = plt.subplots(3, cols, figsize=(3.6*cols, 3.7*3), squeeze=False)
    row_names = ["RAW (crop)", "RT = I*G (crop)", "T50 on RAW (crop)"]
    for j in range(cols):
        tag, Ieq, IGeq, Teq = samples[j]
        for i, arr in enumerate([Ieq, IGeq, Teq]):
            ax = axs[i, j]
            ax.imshow(_stretch01(arr), origin='lower', cmap='viridis', interpolation='nearest')
            ax.set_axis_off()
            if i == 0:
                ax.set_title(tag, fontsize=10)
    for i in range(3):
        axs[i,0].text(-0.02, 0.5, row_names[i], transform=axs[i,0].transAxes,
                      va='center', ha='right', fontsize=11)
    plt.tight_layout(pad=0.3)
    plt.savefig(path_png, dpi=180, bbox_inches='tight')
    plt.close(fig)

# ----------------------------------- Main ------------------------------------
def _parse_max(s):
    if s is None: return None
    s = str(s).strip()
    if s.lower() == "none": return None
    return int(s)

def build_args_from_config():
    ap = argparse.ArgumentParser(description="Create cropped RT images for all RAW/T50 pairs; count all RAWs.")
    ap.add_argument("root", nargs="?", default=CONFIG["root"], help="Root directory")
    ap.add_argument("--recursive", action="store_true", default=CONFIG["recursive"], help="Recurse into subdirectories")
    ap.add_argument("--overwrite", action="store_true", default=CONFIG["overwrite"], help="Overwrite existing outputs")
    ap.add_argument("--fov-arcmin", type=float, default=CONFIG["fov_arcmin"], help="Square FOV in arcmin")
    ap.add_argument("--max-sources", "--n-galaxies", dest="max_sources", default=CONFIG["max_sources"],
                    help="Max number of sources to process (int) or 'None' for all")
    return ap

def main():
    parser = build_args_from_config()
    args = parser.parse_args()
    args.max_sources = _parse_max(args.max_sources)

    print(f"[config] root={args.root}")
    print(f"[config] recursive={args.recursive}  overwrite={args.overwrite}")
    print(f"[config] fov_arcmin={args.fov_arcmin}  max_sources={args.max_sources}")

    if not os.path.isdir(args.root):
        raise SystemExit(f"[error] root does not exist or is not a directory: {args.root}")

    root = os.path.abspath(args.root)
    items = discover(root, recursive=args.recursive)

    if args.max_sources is not None:
        items = items[:max(0, int(args.max_sources))]

    total_raw = len(items)
    with_t50  = sum(1 for it in items if it["t50"] is not None)
    print(f"[run] will process: {total_raw} sources (with T50 among them: {with_t50})")
    
    # ----- pre-scan: T50 beam counts across the native T50 frames -----
    t50_stats = []  # (tag, n_beams, fwhm_as, fov_as_min)
    for it in items:
        if it["t50"] is None:
            continue
        tag = Path(it["dir"]).name
        _, Ht = read_fits(it["t50"])
        fwhm_as = fwhm_major_as(Ht)
        asx_t, asy_t = arcsec_per_pix(Ht)
        fovx_as = int(Ht["NAXIS1"]) * asx_t
        fovy_as = int(Ht["NAXIS2"]) * asy_t
        fov_as_min = min(fovx_as, fovy_as)
        n_beams = fov_as_min / max(fwhm_as, 1e-9)
        t50_stats.append((tag, n_beams, fwhm_as, fov_as_min))

    if not t50_stats:
        print("[info] No T50kpc images found; skipping equal-beams cropping.")
        n_beams_min = None
    else:
        # print per-source and choose the smallest
        print("\n[T50 beam counts across native frames]")
        for tag, nb, fwhm_as, fov_as in t50_stats:
            print(f"  {tag:>20s} : n_beams≈{nb:.2f}  (FWHM={fwhm_as:.2f}\"  FOV_min={fov_as/60:.2f} arcmin)")
        n_beams_min = min(nb for (_, nb, _, _) in t50_stats)
        print(f"\n[select] Using n_beams = {n_beams_min:.2f} (smallest across T50 set)\n")


    made_rt = made_raw = made_t50 = skipped_rt = failures = 0

    #sample_raw, sample_rt, sample_t50 = [], [], []
    samples = []

    for it in items:
        d, stem, raw_path, t50_path = it["dir"], it["stem"], it["raw"], it["t50"]
        tag = Path(d).name

        try:
            I, Hraw = read_fits(raw_path)

            raw_crop_out = os.path.join(d, f"{stem}_raw_crop{int(args.fov_arcmin):02d}.fits")
            (I_crop,), _ = crop_to_fov_on_raw(I, Hraw, args.fov_arcmin)
            if args.overwrite or (not os.path.exists(raw_crop_out)):
                write_fits(raw_crop_out, I_crop, Hraw, hdr_beam=None, overwrite=True,
                           note=f"{args.fov_arcmin}-arcmin square crop centered on RAW.")
                made_raw += 1

            if t50_path is None:
                skipped_rt += 1
                continue

            T_native, Htgt = read_fits(t50_path)
            T_on_raw = reproject_like(T_native, Htgt, Hraw)
            (I_crop, T_crop), _ = crop_to_fov_on_raw(I, Hraw, args.fov_arcmin, T_on_raw)

            t50_out = os.path.join(d, f"{stem}_t50_onraw_crop{int(args.fov_arcmin):02d}.fits")
            if args.overwrite or (not os.path.exists(t50_out)):
                write_fits(t50_out, T_crop, Hraw, hdr_beam=Htgt, overwrite=True,
                           note=f"T50 reprojected to RAW grid; {args.fov_arcmin}-arcmin crop; beam=TARGET.")
                made_t50 += 1

            ker = kernel_from_beams(Hraw, Htgt)
            IG_full = convolve_fft(
                I, ker, boundary='fill', fill_value=np.nan,
                nan_treatment='interpolate', normalize_kernel=True,
                psf_pad=True, fft_pad=True, allow_huge=True
            )
            scale = beam_solid_angle_sr(Htgt) / beam_solid_angle_sr(Hraw)
            IG_full *= scale
            (_, IG_crop), _ = crop_to_fov_on_raw(I, Hraw, args.fov_arcmin, IG_full)

            rt_out = os.path.join(d, f"{stem}_rt_crop{int(args.fov_arcmin):02d}.fits")
            if args.overwrite or (not os.path.exists(rt_out)):
                write_fits(rt_out, IG_crop, Hraw, hdr_beam=Htgt, overwrite=True,
                           note=f"RT (I*G) in Jy/beam_tgt; {args.fov_arcmin}-arcmin crop; beam=TARGET.")
                made_rt += 1
                
            # ---- equal-beams side (arcsec) for THIS source, measured using its T50 beam ----
            if n_beams_min is not None:
                fwhm_t50_as = fwhm_major_as(Htgt)          # THIS source's T50 FWHM (arcsec)
                side_as_eq  = n_beams_min * fwhm_t50_as    # all sources will have the same # of beams

                # Crop RAW, T50-on-RAW, and RT to that side on the RAW grid (same center)
                (I_eq_raw, T_eq, IG_eq), (nyc, nxc) = crop_to_side_arcsec_on_raw(
                    I, Hraw, side_as_eq, T_on_raw, IG_full
                )

                # (optional) write equal-beams FITS too
                write_fits(os.path.join(d, f"{stem}_raw_eqbeams.fits"), I_eq_raw, Hraw, hdr_beam=None,
                        overwrite=True, note=f"equal-beams crop side={side_as_eq:.2f}\"")
                write_fits(os.path.join(d, f"{stem}_t50_onraw_eqbeams.fits"), T_eq, Hraw, hdr_beam=Htgt,
                        overwrite=True, note=f"T50→RAW; equal-beams crop side={side_as_eq:.2f}\"")
                write_fits(os.path.join(d, f"{stem}_rt_eqbeams.fits"), IG_eq, Hraw, hdr_beam=Htgt,
                        overwrite=True, note=f"RT (I*G); equal-beams crop side={side_as_eq:.2f}\"")

                # Collect ONE aligned triplet per source (keeps column order identical)
                if len(samples) < 5:
                    samples.append((tag, I_eq_raw, IG_eq, T_eq))
            else:
                # Fallback (no T50s found): don’t collect samples, to avoid misalignment
                pass

        except Exception as e:
            print(f"[ERR] {tag}: {e!r}")
            failures += 1

    print(f"\nSummary:")
    print(f"  RAW counted            : {total_raw}")
    print(f"  T50 available          : {with_t50}")
    print(f"  RT written             : {made_rt}")
    print(f"  RAW-crop written       : {made_raw}")
    print(f"  T50-on-RAW-crop written: {made_t50}")
    print(f"  Missing T50 (no RT)    : {skipped_rt}")
    print(f"  Failures               : {failures}")

    # ---- Make row plots for 5 different sources when available ----
    # Make a 3×5 montage from aligned triplets
    if len(samples) >= 5 and (CONFIG["max_sources"] is None or args.max_sources is None or args.max_sources >= 5):
        save_grid_3xN_from_triplets("/home/sysadmin/Scripts/samples_grid_3x5_eqbeams.png", samples, ncols=5)
        print("[png] wrote samples_grid_3x5_eqbeams.png")
    elif len(samples) >= 3:
        save_grid_3xN_from_triplets("/home/sysadmin/Scripts/samples_grid_3xN_eqbeams.png", samples, ncols=min(5, len(samples)))
        print("[png] wrote samples_grid_3xN_eqbeams.png")

if __name__ == "__main__":
    sys.exit(main())
