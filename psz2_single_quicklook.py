# psz2_single_quicklook.py  (fixed: search all PSZ2 classes)
from utils.data_loader import load_galaxies, get_classes, root_path, apply_formatting
import os, math, re, glob
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from astropy.io import fits

# --------------------- CONFIG ---------------------
SOURCE = "PSZ2G136.64-25.03"      # z ≈ 0.016
# leave this as a *hint*; we’ll fall back to all PSZ2 classes automatically
GALAXY_CLASSES = [50, 51]
CROP_HW = (512, 512)
OUT_HW  = (128, 128)
PCT_LO, PCT_HI = 60, 99
DO_STRETCH = True
USE_ASINH = True

percentile_lo, percentile_hi = PCT_LO, PCT_HI
crop_size       = (1, CROP_HW[0], CROP_HW[1])
downsample_size = (1, OUT_HW[0],  OUT_HW[1])

# --------------------- HELPERS ---------------------


# --- quick filename probe for PSZ2/classified ---

SOURCE = "PSZ2G136.64-25.03"
PREFIX = "PSZ2G136.64-25"   # looser match
VERSIONS = ["RAW", "T50kpc", "T50kpcSUB"]
CLASS_TAGS = [50, 51]       # you said it should be in 50 or 51

# --- NEW: format a FITS file exactly like the loader does (crop/downsample + stretch + asinh) ---
def _format_fits_like_loader(fits_path):
    arr = np.squeeze(fits.getdata(fits_path)).astype(float)
    arr = np.nan_to_num(arr, copy=False)
    t = torch.from_numpy(arr).float().unsqueeze(0)  # [1,H,W]
    img = apply_formatting(
        t,
        crop_size=(1, CROP_HW[0], CROP_HW[1]),
        downsample_size=(1, OUT_HW[0], OUT_HW[1])
    ).squeeze(0)  # [1,h,w]
    if DO_STRETCH:
        # percentile stretch to [0,1], then optional arcsinh like in your pipeline
        flat = img.reshape(-1)
        plo = torch.quantile(flat, PCT_LO/100.0)
        phi = torch.quantile(flat, PCT_HI/100.0)
        img = ((img - plo) / (phi - plo + 1e-6)).clamp(0, 1)
    if USE_ASINH:
        img = torch.asinh(10.0 * img) / math.asinh(10.0)
    return img  # [1,h,w]

def _resolve_version_dir(version):
    base = os.path.join(root_path, "PSZ2", "classified")
    candidates = [version, version.capitalize(), version.lower(), version.upper()]
    for v in candidates:
        d = os.path.join(base, v)
        if os.path.isdir(d):
            return d, v
    return os.path.join(base, version), version

# --- NEW: direct file-system fetch (exact match preferred; prefix tolerated) ---
def _load_direct_from_disk(version, basename, classes_hint=(50,51)):
    cm = {c["tag"]: c["description"] for c in get_classes()}
    base_dir, chosen = _resolve_version_dir(version)
    class_tags = [t for t in cm if 50 <= t <= 51]
    # try exact → then prefix
    candidates = []
    for tag in class_tags:
        cls_name = cm[tag]
        folder = os.path.join(base_dir, cls_name)
        if not os.path.isdir(folder):
            continue
        exact = os.path.join(folder, f"{basename}.fits")
        if os.path.isfile(exact):
            candidates.append(exact)
        # also allow prefix (e.g., if user typed shorter slug)
        candidates += sorted(glob.glob(os.path.join(folder, f"{basename}*.fits")))
    # de-dup and prefer exact match
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return None, None
    # Prefer the plain name without T-suffix if both exist
    preferred = None
    for p in candidates:
        stem = Path(p).stem
        if stem == basename:
            preferred = p; break
    fits_path = preferred or candidates[0]
    img = _format_fits_like_loader(fits_path)
    print(f"[fallback] Loaded directly from disk: {fits_path}")
    return img, fits_path


def _class_map():
    return {c["tag"]: c["description"] for c in get_classes()}

def peek_classified(version="RAW", classes=(50,51), limit=30):
    base = os.path.join(root_path, "PSZ2", "classified", version)
    cm = _class_map()
    print(f"\n=== Version: {version} ===")
    for tag in classes:
        cls_name = cm.get(tag, str(tag))
        folder = os.path.join(base, cls_name)
        if not os.path.isdir(folder):
            print(f"  [missing] {folder}")
            continue
        fns = [os.path.splitext(fn)[0] for fn in os.listdir(folder)
               if fn.lower().endswith(".fits")]
        fns = sorted(fns)
        print(f"  Class {tag} ({cls_name}): {len(fns)} files")
        for s in fns[:limit]:
            print("    ", s)
        if len(fns) > limit:
            print(f"    … (+{len(fns)-limit} more)")
    print()

def find_candidates(prefix, versions=VERSIONS, classes=(50,51)):
    cm = _class_map()
    hits = []
    for v in versions:
        base = os.path.join(root_path, "PSZ2", "classified", v)
        for tag in classes:
            cls_name = cm.get(tag, str(tag))
            folder = os.path.join(base, cls_name)
            if not os.path.isdir(folder): 
                continue
            pattern = os.path.join(folder, f"{prefix}*.fits")
            for p in sorted(glob.glob(pattern)):
                hits.append((v, tag, cls_name, os.path.splitext(os.path.basename(p))[0], p))
    return hits

if __name__ == "__main__":
    # 1) Print a handful of filenames so we see the exact naming convention
    for v in VERSIONS:
        peek_classified(version=v, classes=CLASS_TAGS, limit=25)

    # 2) Try exact and prefix searches
    for query in [SOURCE, PREFIX]:
        print(f"\n=== Searching for '{query}' in {CLASS_TAGS} and versions {VERSIONS} ===")
        found = find_candidates(query, versions=VERSIONS, classes=CLASS_TAGS)
        if not found:
            print("  (no matches)")
        else:
            for v, tag, cls_name, base, full in found:
                print(f"  hit: version={v} class={tag}({cls_name}) name={base}  →  {full}")


def _as_2d_numpy(timg):
    x = timg.detach().cpu()
    if x.ndim == 4: x = x[0]
    if x.ndim == 3: x = x[0] if x.shape[0] > 1 else x.squeeze(0)
    return x.numpy()

def _find_by_basename(imgs, fns, basename):
    to_base = lambda s: Path(str(s)).stem.split('T', 1)[0]
    for i, fn in enumerate(fns or []):
        if to_base(fn) == basename:
            return imgs[i], i
    return None, None

def _load_one(version):
    def _try(classes):
        out = load_galaxies(
            galaxy_classes=classes,
            versions=[version],
            fold=5,
            sample_size=10**9,
            crop_size=CROP_HW,
            downsample_size=OUT_HW,
            STRETCH=True,
            percentile_lo=PCT_LO, percentile_hi=PCT_HI,
            NORMALISE=True,
            USE_GLOBAL_NORMALISATION=False,
            GLOBAL_NORM_MODE="percentile",
            PRINTFILENAMES=True,
            REMOVEOUTLIERS=False,
            AUGMENT=False,
            train=False
        )
        tr_imgs, tr_lbls, ev_imgs, ev_lbls, tr_fns, ev_fns = out
        img, idx = _find_by_basename(ev_imgs, ev_fns, SOURCE)
        if img is not None:
            return img, ev_fns[idx]
        img, idx = _find_by_basename(tr_imgs, tr_fns, SOURCE)
        if img is not None:
            return img, tr_fns[idx]
        return None, None

    # 1) try the hinted classes
    img, fn = _try(GALAXY_CLASSES)
    if img is None:
        # 2) try all PSZ2 classes
        all_psz2 = [c["tag"] for c in get_classes() if 50 <= c["tag"] <= 59]
        img, fn = _try(all_psz2)

    if img is not None:
        return img, fn

    # 3) FINAL FALLBACK: read the FITS from disk and format like the loader
    img, path = _load_direct_from_disk(version, SOURCE, classes_hint=GALAXY_CLASSES)
    if img is not None:
        return img, path
    
    print("Looked for ", SOURCE, "at the locations below:")
    for v in VERSIONS:
        base = os.path.join(root_path, "PSZ2", "classified", v)
        for tag in GALAXY_CLASSES:
            cls_name = _class_map().get(tag, str(tag))
            folder = os.path.join(base, cls_name)
            print("  ", os.path.join(folder, f"{SOURCE}.fits"))
            print("  ", os.path.join(folder, f"{PREFIX}*.fits"))
    raise RuntimeError(f"Could not find {SOURCE} in version={version} anywhere under PSZ2.")


def apply_taper_to_tensor(
    imgs, mode, filenames,
    crop_size=(1,512,512), downsample_size=(1,128,128),
    percentile_lo=60, percentile_hi=95,
    do_stretch=True, use_asinh=True,
    require_fixed_header=False,
    ref_sigma_map=None, bg_inner=64,
    debug_dir=None
):
    """
    Build runtime-tapered planes (rtXX) by:
      RAW → (PSF-match to target restoring beam on RAW grid) → (optional uv-taper)
      → reproject onto target grid → crop/resize → percentile stretch → asinh.
    Returns (stack[B,1,H,W], keep_mask[B], kept_fns[list], skipped[list]).
    """

    mode = str(mode).lower()
    m = re.fullmatch(r'rt(\d+)', mode)
    want_kpc = int(m.group(1)) if m else None
    if want_kpc is None:
        # nothing special requested: just ensure [B,1,H,W]
        keep_mask = torch.ones(len(filenames), dtype=torch.bool)
        return (imgs if imgs.dim()==4 else imgs.unsqueeze(1)), keep_mask, list(map(str, filenames)), []

    device = imgs.device if torch.is_tensor(imgs) else torch.device('cpu')
    dtype  = imgs.dtype  if torch.is_tensor(imgs) else torch.float32
    Hout, Wout = downsample_size[-2], downsample_size[-1]

    out, kept_fns, kept_flags, skipped = [], [], [], []

    for base in map(str, filenames):
        try:
            raw_hdr, t25_hdr, t50_hdr, t100_hdr, pix_native_as, raw_path = _headers_for_name(base)
        except Exception as e:
            skipped.append(base); kept_flags.append(False); continue

        # Choose or synthesize the target header
        targ_hdr = None
        if   want_kpc == 25:   targ_hdr = t25_hdr
        elif want_kpc == 50:   targ_hdr = t50_hdr
        elif want_kpc == 100:  targ_hdr = t100_hdr
        else:
            # interpolate between existing fixed tapers if possible
            def _interp_hdr(k_lo, h_lo, k_hi, h_hi, k_want):
                if h_lo is None and h_hi is None:
                    return None
                if h_lo is None or h_hi is None:
                    return (h_lo or h_hi).copy()
                w = (k_want - k_lo) / float(k_hi - k_lo)
                outH = h_lo.copy()
                for key in ("BMAJ", "BMIN"):
                    v_lo = float(h_lo[key]); v_hi = float(h_hi[key])
                    outH[key] = v_lo*(1.0-w) + v_hi*w
                bpa_lo = float(h_lo.get("BPA", h_hi.get("BPA", 0.0)))
                bpa_hi = float(h_hi.get("BPA", h_lo.get("BPA", 0.0)))
                outH["BPA"] = bpa_lo if (k_want - k_lo) <= (k_hi - k_want) else bpa_hi
                # keep the RAW WCS so reproject_like(raw→targ) can be a no-op when grids match
                for k in ('CTYPE1','CTYPE2','CRVAL1','CRVAL2','CRPIX1','CRPIX2',
                          'CDELT1','CDELT2','CD1_1','CD1_2','CD2_1','CD2_2',
                          'PC1_1','PC1_2','PC2_1','PC2_2','NAXIS1','NAXIS2'):
                    if k in raw_hdr: outH[k] = raw_hdr[k]
                return outH
            if want_kpc < 50:
                targ_hdr = _interp_hdr(25, t25_hdr, 50, t50_hdr, want_kpc)
            elif want_kpc < 100:
                targ_hdr = _interp_hdr(50, t50_hdr, 100, t100_hdr, want_kpc)
            else:
                targ_hdr = _interp_hdr(50, t50_hdr, 100, t100_hdr, want_kpc)

        # try final synthesis if needed (from redshift or scaled ref header)
        if targ_hdr is None:
            try:
                z_here  = get_z(base, raw_hdr)
                targ_hdr = synth_taper_header_from_kpc(raw_hdr, z_here, want_kpc, mode="keep_ratio")
            except Exception:
                # scale from T50 if available
                if t50_hdr is not None:
                    try:
                        targ_hdr = synth_taper_header_from_ref(raw_hdr, t50_hdr, want_kpc, kpc_ref=50.0, mode="keep_ratio")
                    except Exception:
                        targ_hdr = None

        # Enforce parity (optional)
        if require_fixed_header and want_kpc in (25, 50, 100) and targ_hdr is None:
            skipped.append(base); kept_flags.append(False); continue
        if (t25_hdr is None) and (t50_hdr is None) and (t100_hdr is None) and (targ_hdr is None):
            skipped.append(base); kept_flags.append(False); continue

        # 1) Load RAW map (native grid)
        try:
            raw_native = np.squeeze(fits.getdata(raw_path)).astype(float)
        except Exception:
            skipped.append(base); kept_flags.append(False); continue
        raw_native = np.nan_to_num(raw_native, copy=False)

        # 2) Anti-alias (only if we’ll shrink to Hout×Wout)
        try:
            ds = int(round(crop_size[-2] / float(Hout)))
            if ds > 1:
                raw_pref = gaussian_filter(raw_native, sigma=0.5*ds, mode='nearest')
            else:
                raw_pref = raw_native
        except Exception:
            raw_pref = raw_native

        # 3) PSF-match RAW → TARGET beam on RAW grid, with optional global fudge
        targ_hdr_eff = targ_hdr
        fudge = float(FUDGE_GLOBAL)
        if os.getenv("RT_AUTO_FUDGE", "1") == "1" and (targ_hdr_eff is not None):
            try:
                tpath = _first(f"{os.path.dirname(raw_path)}/{_name_base_from_fn(base)}T{want_kpc}kpc*.fits")
                if tpath:
                    T_img = np.squeeze(fits.getdata(tpath)).astype(float)
                    fudge = auto_fudge_scale(
                        raw_pref, raw_hdr, targ_hdr_eff, T_img,
                        s_grid=np.linspace(1.00, 1.20, 11), nbins=48
                    )
            except Exception:
                pass
        matched_native = convolve_to_target(raw_pref, raw_hdr, targ_hdr_eff or raw_hdr, fudge_scale=fudge)

        # 4) Optional uv-taper (disabled by default)
        if APPLY_UV_TAPER and (UV_TAPER_FRAC > 0):
            try:
                z_here = get_z(base, raw_hdr)
                theta_as = kpc_to_arcsec(z_here, float(want_kpc))  # arcsec
                matched_native = apply_uv_gaussian_taper(matched_native, raw_hdr, theta_as * UV_TAPER_FRAC, pad_factor=2)
            except Exception:
                pass

        # 5) Reproject RAW→TARGET (image we just convolved) onto TARGET grid
        matched_on_t = reproject_like(matched_native, raw_hdr, targ_hdr_eff or raw_hdr)

        # 6) Crop+resize from TARGET grid → (Hout,Wout) using the same pipeline tool
        t = torch.from_numpy(np.nan_to_num(matched_on_t, copy=False)).float().unsqueeze(0)  # [1,H,W]
        crop_eff = _effective_crop_on_raw(raw_hdr, targ_hdr_eff, crop_size)
        formatted = apply_formatting(t, crop_size=crop_eff,
                                     downsample_size=(1, Hout, Wout)).squeeze(0)   # [1,Hout,Wout]

        # 7) Percentile stretch + optional asinh
        if do_stretch:
            stretched = _per_image_percentile_stretch(formatted.squeeze(0), percentile_lo, percentile_hi).unsqueeze(0)
        else:
            stretched = formatted
        if use_asinh:
            stretched = torch.asinh(10.0 * stretched) / math.asinh(10.0)

        # 8) Optional noise match to a reference sigma map
        if (ref_sigma_map is not None) and (base in ref_sigma_map):
            mask = _background_ring_mask(Hout, Wout, inner=bg_inner)
            sig_fake = _robust_sigma(stretched.squeeze(0)[mask])
            sig_want = torch.tensor(ref_sigma_map[base], dtype=stretched.dtype)
            add = torch.clamp(sig_want - sig_fake, min=0.0)
            if add > 1e-8:
                noise = torch.randn_like(stretched)
                s = stretched.clone()
                s[:, 0][mask] = (s[:, 0][mask] + noise[:, 0][mask]*add).clamp(0, 1)
                stretched = s

        out.append(stretched)
        kept_fns.append(base)
        kept_flags.append(True)

        # Optional debug figure per source
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            fig, ax = plt.subplots(1,2, figsize=(6,3))
            ax[0].imshow(formatted.squeeze(0).cpu().numpy(), cmap='viridis', origin='lower'); ax[0].axis('off'); ax[0].set_title('PSF-matched')
            ax[1].imshow(stretched.squeeze(0).cpu().numpy(), cmap='viridis', origin='lower'); ax[1].axis('off'); ax[1].set_title('final')
            fig.suptitle(base); fig.tight_layout()
            tag = f"rt{want_kpc}"
            fig.savefig(os.path.join(debug_dir, f"{base}_{tag}{_versions_to_load}.png"), dpi=140)
            plt.close(fig)

    keep_mask = torch.tensor(kept_flags, dtype=torch.bool)
    out = torch.stack(out, dim=0).to(device=device, dtype=dtype) if out else torch.empty((0,1,Hout,Wout), device=device, dtype=dtype)
    return out, keep_mask, kept_fns, skipped


def _synthesize_rt50_from_raw(raw_img, raw_fn):
    imgs = raw_img.unsqueeze(0) if raw_img.ndim == 3 else raw_img  # [1,1,H,W]
    if imgs.ndim == 2:
        imgs = imgs.unsqueeze(0).unsqueeze(0)
    rt, keep_mask, kept, skipped = apply_taper_to_tensor(
        imgs, mode="rt50",
        filenames=[Path(str(raw_fn)).stem.split('T',1)[0]],
        crop_size=crop_size,
        downsample_size=downsample_size,
        percentile_lo=PCT_LO, percentile_hi=PCT_HI,
        do_stretch=DO_STRETCH, use_asinh=USE_ASINH,
        require_fixed_header=False
    )
    if rt.shape[0] == 0:
        raise RuntimeError(f"RT50 synthesis failed for {SOURCE}. Skipped={skipped}")
    return rt[0]  # [1,H,W]

# ---- FoV helpers (need the header utilities from your environment) ----
def _fov_arcmin_and_pixfinal_for_raw_like_t50(base_name):
    raw_hdr, t25_hdr, t50_hdr, t100_hdr, pix_native, _ = _headers_for_name(base_name)
    if t50_hdr is None:
        z = get_z(base_name, raw_hdr)
        t50_hdr = synth_taper_header_from_kpc(raw_hdr, z, 50.0, mode="keep_ratio")
    ref_pixdeg = _cdelt_deg(t50_hdr, 1)           # deg/px on T50 grid
    fov_deg_x  = CROP_HW[1] * ref_pixdeg
    pix_final_as = (fov_deg_x / OUT_HW[1]) * 3600.0
    return fov_deg_x*60.0, pix_final_as

def _fov_arcmin_and_pixfinal_for_T50(base_name):
    raw_hdr, t25_hdr, t50_hdr, t100_hdr, pix_native, _ = _headers_for_name(base_name)
    if t50_hdr is None:
        z = get_z(base_name, raw_hdr)
        t50_hdr = synth_taper_header_from_kpc(raw_hdr, z, 50.0, mode="keep_ratio")
    ref_pixdeg = _cdelt_deg(t50_hdr, 1)
    fov_deg_x  = CROP_HW[1] * ref_pixdeg
    pix_final_as = (fov_deg_x / OUT_HW[1]) * 3600.0
    return fov_deg_x*60.0, pix_final_as

# --------------------- MAIN ---------------------
if __name__ == "__main__":
    raw_img, raw_fn = _load_one("raw")
    t50_img, t50_fn = _load_one("T50kpc")

    rt50_img = _synthesize_rt50_from_raw(raw_img, raw_fn)

    fov_raw_am,  pixf_raw_as  = _fov_arcmin_and_pixfinal_for_raw_like_t50(SOURCE)
    fov_rt_am,   pixf_rt_as   = _fov_arcmin_and_pixfinal_for_raw_like_t50(SOURCE)
    fov_t50_am,  pixf_t50_as  = _fov_arcmin_and_pixfinal_for_T50(SOURCE)

    print(f"\nFoV_x (arcmin) and final pixel scale (arcsec/px) for {SOURCE}:")
    print(f"  RAW    : FoV_x ≈ {fov_raw_am:.2f}′,  pix_final ≈ {pixf_raw_as:.3f}\"/px")
    print(f"  RT50   : FoV_x ≈ {fov_rt_am:.2f}′,   pix_final ≈ {pixf_rt_as:.3f}\"/px")
    print(f"  T50kpc : FoV_x ≈ {fov_t50_am:.2f}′, pix_final ≈ {pixf_t50_as:.3f}\"/px")

    A = _as_2d_numpy(raw_img)
    B = _as_2d_numpy(rt50_img)
    C = _as_2d_numpy(t50_img)

    fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.6))
    for ax, im, title in zip(
        axs,
        (A, B, C),
        (f"RAW\nFoV≈{fov_raw_am:.2f}′, {pixf_raw_as:.2f}\"/px",
         f"RT50\nFoV≈{fov_rt_am:.2f}′, {pixf_rt_as:.2f}\"/px",
         f"T50kpc\nFoV≈{fov_t50_am:.2f}′, {pixf_t50_as:.2f}\"/px")
    ):
        ax.imshow(im, origin='lower', cmap='viridis')
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(SOURCE, fontsize=11)
    fig.tight_layout()
    os.makedirs("./quicklooks", exist_ok=True)
    outpath = f"./quicklooks/{SOURCE}_raw_rt50_t50.png"
    fig.savefig(outpath, dpi=160, bbox_inches='tight')
    print(f"\nWrote figure: {outpath}")
