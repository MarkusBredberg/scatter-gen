import os, numpy as np, torch, matplotlib.pyplot as plt
from utils.data_loader import load_galaxies as _loader_processed # Prefer processed images
from utils.data_loader3 import load_galaxies as _loader_original # Prefer original images

def _to_numpy(img):
    # img can be Tensor [C,H,W] or [H,W]; return 2D np.ndarray
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.asarray(img)
    if arr.ndim == 3:  # (C,H,W) -> take first channel
        arr = arr[0]
    return np.squeeze(arr).astype(np.float32)

def _pearsonr(a, b, eps=1e-12):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    a -= a.mean(); b -= b.mean()
    denom = (np.sqrt((a*a).sum()) * np.sqrt((b*b).sum())) + eps
    return float((a*b).sum() / denom)

def compare_raw_sets(processed_images, processed_names, original_images, original_names, outdir="./raw_compare", n_examples=6, hist_samples=200_000, seed=42):
    """
    processed_images/original_images: list or Tensor of images (B,1,H,W) or (B,H,W)
    processed_names/original_names:   list of basenames matching each image
    """
    os.makedirs(outdir, exist_ok=True)

    # Make dicts keyed by basename
    processed_map = {n: processed_images[i] for i, n in enumerate(processed_names)}
    original_map = {n: original_images[i] for i, n in enumerate(original_names)}
    common = sorted(set(processed_map) & set(original_map))
    assert common, "No overlapping basenames to compare."

    # Summaries
    shapes_processed = set()
    shapes_original = set()
    for n in common:
        shapes_processed.add(tuple(np.asarray(_to_numpy(processed_map[n])).shape))
        shapes_original.add(tuple(np.asarray(_to_numpy(original_map[n])).shape))
    print(f"[RAW] matched={len(common)}; unique processed shapes={shapes_processed}; unique original shapes={shapes_original}")

    # Pixel stats over matched set (means, stds)
    def _stats_one(arr): 
        return float(np.nanmean(arr)), float(np.nanstd(arr))
    means_processed, stds_processed, means_original, stds_original = [], [], [], []
    for n in common:
        a = _to_numpy(processed_map[n]); b = _to_numpy(original_map[n])
        means_processed.append(np.mean(a)); stds_processed.append(np.std(a))
        means_original.append(np.mean(b)); stds_original.append(np.std(b))
    print(f"[RAW] mean(processed)={np.mean(means_processed):.4g}±{np.std(means_processed):.4g}  "
          f"mean(original)={np.mean(means_original):.4g}±{np.std(means_original):.4g}")
    print(f"[RAW] std(processed) ={np.mean(stds_processed):.4g}±{np.std(stds_processed):.4g}   "
          f"std(original) ={np.mean(stds_original):.4g}±{np.std(stds_original):.4g}")

    # Global histogram overlay on a pixel subset
    rng = np.random.default_rng(seed)
    pool = rng.choice(len(common), size=min(len(common), 50), replace=False)
    samp_processed, samp_original = [], []
    for idx in pool:
        n = common[idx]
        samp_processed.append(_to_numpy(processed_map[n]).ravel())
        samp_original.append(_to_numpy(original_map[n]).ravel())
    processed_flat = np.concatenate(samp_processed); original_flat = np.concatenate(samp_original)
    if processed_flat.size > hist_samples:
        pick = rng.choice(processed_flat.size, size=hist_samples, replace=False)
        processed_flat = processed_flat[pick]; original_flat = original_flat[pick]
    vmin = float(min(processed_flat.min(), original_flat.min()))
    vmax = float(max(processed_flat.max(), original_flat.max()))
    edges = np.linspace(vmin, vmax, 60)
    plt.figure(figsize=(6,3.2))
    plt.hist(processed_flat, bins=edges, alpha=0.6, label="processed RAW", density=True)
    plt.hist(original_flat, bins=edges, alpha=0.6, label="original RAW", density=True)
    plt.xlabel("pixel"); plt.ylabel("density"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "hist_overlay.png"), dpi=150); plt.close()

    # Example panels: processed vs original vs difference with metrics
    ex = rng.choice(len(common), size=min(n_examples, len(common)), replace=False)
    for k, idx in enumerate(ex):
        name = common[idx]
        A = _to_numpy(processed_map[name]); B = _to_numpy(original_map[name])
        assert A.shape == B.shape, f"Shape mismatch for {name}: {A.shape} vs {B.shape}"
        D = B - A
        mse = float(np.mean((B - A) ** 2))
        mae = float(np.mean(np.abs(B - A)))
        corr = _pearsonr(A, B)

        fig, axs = plt.subplots(1, 3, figsize=(10,3.2))
        im0 = axs[0].imshow(A, origin="lower"); axs[0].set_title("processed RAW"); axs[0].axis("off"); fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        im1 = axs[1].imshow(B, origin="lower"); axs[1].set_title("original RAW"); axs[1].axis("off"); fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
        im2 = axs[2].imshow(D, origin="lower"); axs[2].set_title("original - processed"); axs[2].axis("off"); fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
        fig.suptitle(f"{name}   MSE={mse:.3e}  MAE={mae:.3e}  r={corr:.4f}")
        plt.tight_layout(rect=[0,0.02,1,0.95])
        fig.savefig(os.path.join(outdir, f"panel_{k}_{name}.png"), dpi=150)
        print("Saved figure at:", os.path.join(outdir, f"panel_{k}_{name}.png"))
        plt.close(fig)

    # Quick sanity: fraction of identical images (bitwise)
    identical = 0
    for n in common:
        A = _to_numpy(processed_map[n]); B = _to_numpy(original_map[n])
        if A.shape == B.shape and np.array_equal(A, B):
            identical += 1
    frac_ident = identical / len(common)
    print(f"[RAW] exactly-identical pairs: {identical}/{len(common)}  ({100*frac_ident:.1f}%)")

    print(f"[RAW] wrote: {outdir}/hist_overlay.png and {len(ex)} example panels.")


def _radial_psd(img2d):
    x = img2d.astype(np.float64)
    x = x - np.nanmean(x)
    # simple cosine apodization to reduce edge power
    h, w = x.shape
    wy = 0.5 * (1 - np.cos(2*np.pi*np.arange(h)/(h-1)))
    wx = 0.5 * (1 - np.cos(2*np.pi*np.arange(w)/(w-1)))
    window = np.outer(wy, wx)
    xw = x * window

    F  = np.fft.rfft2(xw)
    Pk = (F * np.conj(F)).real  # power

    ky = np.fft.fftfreq(h) * h
    kx = np.fft.rfftfreq(w) * w
    ky = ky[:, None]
    k  = np.sqrt(ky**2 + kx[None, :]**2)

    kbin = np.floor(k).astype(int)
    kmax = kbin.max()
    num  = np.bincount(kbin.ravel(), weights=Pk.ravel(), minlength=kmax+1)
    den  = np.bincount(kbin.ravel(), minlength=kmax+1)
    psd  = num / np.maximum(den, 1)
    kk   = np.arange(psd.size)
    return kk, psd

def _to_np_2d(img):
    arr = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
    if arr.ndim == 3: arr = arr[0]
    return np.squeeze(arr).astype(np.float32)

galaxy_classes = [50, 51]  # example classes
versions = ["RAW"]
percentile_lo, percentile_hi = 30, 99
STRETCH = True  # Arcsinh stretch
USE_GLOBAL_NORMALISATION = False           # single on/off switch . False - image-by-image normalisation 
GLOBAL_NORM_MODE = "percentile"           # "percentile" or "flux"
AUGMENT = True  # Apply classical data augmentation (flips, rotations)
NORMALISEIMGS = True  # Globally normalise images to [0, 1]
NORMALISEIMGSTOPM = False  # Globally normalise images to [-1, 1]
PRINTFILENAMES = True

#n_beams_min, _T_header_cache = None, {}
processed_train_images, processed_train_labels, _, _, processed_train_names, _ = _loader_processed(
    galaxy_classes=galaxy_classes,
    versions=versions,
    STRETCH=STRETCH,
    percentile_lo=percentile_lo,
    percentile_hi=percentile_hi,
    AUGMENT=AUGMENT,                  # (late aug happens after rt*)
    NORMALISE=NORMALISEIMGS,
    NORMALISETOPM=NORMALISEIMGSTOPM,
    USE_GLOBAL_NORMALISATION=USE_GLOBAL_NORMALISATION,
    GLOBAL_NORM_MODE=GLOBAL_NORM_MODE,
    PRINTFILENAMES=True,                   # ensure filenames for TEST
    train=False
)

#EQUAL_TAPER = _pick_equal_taper_from(versions)  # T50kpc by default
#n_beams_min, _T_header_cache = _scan_min_beams(path, target_classes, taper=EQUAL_TAPER)
original_train_images, original_train_labels, _, _, original_train_names, _ = _loader_original(
    galaxy_classes=galaxy_classes,
    versions=versions,
    STRETCH=STRETCH,
    percentile_lo=percentile_lo,
    percentile_hi=percentile_hi,
    AUGMENT=AUGMENT,                  # (late aug happens after rt*)
    NORMALISE=NORMALISEIMGS,
    NORMALISETOPM=NORMALISEIMGSTOPM,
    USE_GLOBAL_NORMALISATION=USE_GLOBAL_NORMALISATION,
    GLOBAL_NORM_MODE=GLOBAL_NORM_MODE,
    PRINTFILENAMES=True,                   # ensure filenames for TEST
    train=False
)

# 1) run the existing visual diffs
compare_raw_sets(
    processed_images=processed_train_images, processed_names=processed_train_names,
    original_images=original_train_images, original_names=original_train_names,
    outdir="./raw_compare", n_examples=6
)

# 2) original: PSD comparison on matched subset
os.makedirs("./raw_compare", exist_ok=True)

processed_map = {n: processed_train_images[i] for i, n in enumerate(processed_train_names)}
original_map = {n: original_train_images[i] for i, n in enumerate(original_train_names)}
common  = sorted(set(processed_map) & set(original_map))

# take up to 64 random matches
rng = np.random.default_rng(42)
sel = rng.choice(len(common), size=min(64, len(common)), replace=False)

# accumulate average PSDs
psd_processed_sum = psd_original_sum = None
for idx in sel:
    name = common[idx]
    A = _to_np_2d(processed_map[name]); B = _to_np_2d(original_map[name])
    assert A.shape == B.shape, f"shape mismatch for {name}: {A.shape} vs {B.shape}"
    kA, pA = _radial_psd(A)
    kB, pB = _radial_psd(B)
    k  = kA if psd_processed_sum is None else k
    psd_processed_sum = pA if psd_processed_sum is None else (psd_processed_sum + pA)
    psd_original_sum = pB if psd_original_sum is None else (psd_original_sum + pB)

psd_processed = psd_processed_sum / len(sel)
psd_original = psd_original_sum / len(sel)
ratio   = psd_original / np.maximum(psd_processed, 1e-12)

# plots: mean PSDs and their ratio
plt.figure(figsize=(6.5,3.2))
plt.loglog(k[1:], psd_processed[1:], label="processed RAW")
plt.loglog(k[1:], psd_original[1:], label="original RAW")
plt.xlabel("spatial frequency k"); plt.ylabel("power"); plt.legend(); plt.tight_layout()
plt.savefig("./raw_compare/psd_processed_original.png", dpi=150); plt.close()

plt.figure(figsize=(6.5,3.0))
plt.semilogx(k[1:], ratio[1:])
plt.axhline(1, linestyle=":")
plt.xlabel("spatial frequency k"); plt.ylabel("original / processed power")
plt.tight_layout(); plt.savefig("./raw_compare/psd_ratio.png", dpi=150); plt.close()
