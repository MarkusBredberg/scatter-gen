import os, numpy as np, torch, matplotlib.pyplot as plt
from utils.data_loader import load_galaxies as _loader_bad
from utils.data_loader3 import load_galaxies as _loader_good

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

def compare_raw_sets(bad_images, bad_names, good_images, good_names, outdir="./raw_compare", n_examples=6, hist_samples=200_000, seed=42):
    """
    bad_images/good_images: list or Tensor of images (B,1,H,W) or (B,H,W)
    bad_names/good_names:   list of basenames matching each image
    """
    os.makedirs(outdir, exist_ok=True)

    # Make dicts keyed by basename
    bad_map = {n: bad_images[i] for i, n in enumerate(bad_names)}
    good_map = {n: good_images[i] for i, n in enumerate(good_names)}
    common = sorted(set(bad_map) & set(good_map))
    assert common, "No overlapping basenames to compare."

    # Summaries
    shapes_bad = set()
    shapes_good = set()
    for n in common:
        shapes_bad.add(tuple(np.asarray(_to_numpy(bad_map[n])).shape))
        shapes_good.add(tuple(np.asarray(_to_numpy(good_map[n])).shape))
    print(f"[RAW] matched={len(common)}; unique bad shapes={shapes_bad}; unique good shapes={shapes_good}")

    # Pixel stats over matched set (means, stds)
    def _stats_one(arr): 
        return float(np.nanmean(arr)), float(np.nanstd(arr))
    means_bad, stds_bad, means_good, stds_good = [], [], [], []
    for n in common:
        a = _to_numpy(bad_map[n]); b = _to_numpy(good_map[n])
        means_bad.append(np.mean(a)); stds_bad.append(np.std(a))
        means_good.append(np.mean(b)); stds_good.append(np.std(b))
    print(f"[RAW] mean(bad)={np.mean(means_bad):.4g}±{np.std(means_bad):.4g}  "
          f"mean(good)={np.mean(means_good):.4g}±{np.std(means_good):.4g}")
    print(f"[RAW] std(bad) ={np.mean(stds_bad):.4g}±{np.std(stds_bad):.4g}   "
          f"std(good) ={np.mean(stds_good):.4g}±{np.std(stds_good):.4g}")

    # Global histogram overlay on a pixel subset
    rng = np.random.default_rng(seed)
    pool = rng.choice(len(common), size=min(len(common), 50), replace=False)
    samp_bad, samp_good = [], []
    for idx in pool:
        n = common[idx]
        samp_bad.append(_to_numpy(bad_map[n]).ravel())
        samp_good.append(_to_numpy(good_map[n]).ravel())
    bad_flat = np.concatenate(samp_bad); good_flat = np.concatenate(samp_good)
    if bad_flat.size > hist_samples:
        pick = rng.choice(bad_flat.size, size=hist_samples, replace=False)
        bad_flat = bad_flat[pick]; good_flat = good_flat[pick]
    vmin = float(min(bad_flat.min(), good_flat.min()))
    vmax = float(max(bad_flat.max(), good_flat.max()))
    edges = np.linspace(vmin, vmax, 60)
    plt.figure(figsize=(6,3.2))
    plt.hist(bad_flat, bins=edges, alpha=0.6, label="bad RAW", density=True)
    plt.hist(good_flat, bins=edges, alpha=0.6, label="good RAW", density=True)
    plt.xlabel("pixel"); plt.ylabel("density"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "hist_overlay.png"), dpi=150); plt.close()

    # Example panels: bad vs good vs difference with metrics
    ex = rng.choice(len(common), size=min(n_examples, len(common)), replace=False)
    for k, idx in enumerate(ex):
        name = common[idx]
        A = _to_numpy(bad_map[name]); B = _to_numpy(good_map[name])
        assert A.shape == B.shape, f"Shape mismatch for {name}: {A.shape} vs {B.shape}"
        D = B - A
        mse = float(np.mean((B - A) ** 2))
        mae = float(np.mean(np.abs(B - A)))
        corr = _pearsonr(A, B)

        fig, axs = plt.subplots(1, 3, figsize=(10,3.2))
        im0 = axs[0].imshow(A, origin="lower"); axs[0].set_title("bad RAW"); axs[0].axis("off"); fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        im1 = axs[1].imshow(B, origin="lower"); axs[1].set_title("good RAW"); axs[1].axis("off"); fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
        im2 = axs[2].imshow(D, origin="lower"); axs[2].set_title("good - bad"); axs[2].axis("off"); fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
        fig.suptitle(f"{name}   MSE={mse:.3e}  MAE={mae:.3e}  r={corr:.4f}")
        plt.tight_layout(rect=[0,0.02,1,0.95])
        fig.savefig(os.path.join(outdir, f"panel_{k}_{name}.png"), dpi=150)
        print("Saved figure at:", os.path.join(outdir, f"panel_{k}_{name}.png"))
        plt.close(fig)

    # Quick sanity: fraction of identical images (bitwise)
    identical = 0
    for n in common:
        A = _to_numpy(bad_map[n]); B = _to_numpy(good_map[n])
        if A.shape == B.shape and np.array_equal(A, B):
            identical += 1
    frac_ident = identical / len(common)
    print(f"[RAW] exactly-identical pairs: {identical}/{len(common)}  ({100*frac_ident:.1f}%)")

    print(f"[RAW] wrote: {outdir}/hist_overlay.png and {len(ex)} example panels.")


def _radial_psd(img2d):
    import numpy as np
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
    import numpy as np, torch
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
bad_train_images, bad_train_labels, _, _, bad_train_names, _ = _loader_bad(
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
good_train_images, good_train_labels, _, _, good_train_names, _ = _loader_good(
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
    bad_images=bad_train_images, bad_names=bad_train_names,
    good_images=good_train_images, good_names=good_train_names,
    outdir="./raw_compare", n_examples=6
)

# 2) good: PSD comparison on matched subset
import os, numpy as np, matplotlib.pyplot as plt
os.makedirs("./raw_compare", exist_ok=True)

bad_map = {n: bad_train_images[i] for i, n in enumerate(bad_train_names)}
good_map = {n: good_train_images[i] for i, n in enumerate(good_train_names)}
common  = sorted(set(bad_map) & set(good_map))

# take up to 64 random matches
rng = np.random.default_rng(42)
sel = rng.choice(len(common), size=min(64, len(common)), replace=False)

# accumulate average PSDs
psd_bad_sum = psd_good_sum = None
for idx in sel:
    name = common[idx]
    A = _to_np_2d(bad_map[name]); B = _to_np_2d(good_map[name])
    assert A.shape == B.shape, f"shape mismatch for {name}: {A.shape} vs {B.shape}"
    kA, pA = _radial_psd(A)
    kB, pB = _radial_psd(B)
    k  = kA if psd_bad_sum is None else k
    psd_bad_sum = pA if psd_bad_sum is None else (psd_bad_sum + pA)
    psd_good_sum = pB if psd_good_sum is None else (psd_good_sum + pB)

psd_bad = psd_bad_sum / len(sel)
psd_good = psd_good_sum / len(sel)
ratio   = psd_good / np.maximum(psd_bad, 1e-12)

# plots: mean PSDs and their ratio
plt.figure(figsize=(6.5,3.2))
plt.loglog(k[1:], psd_bad[1:], label="bad RAW")
plt.loglog(k[1:], psd_good[1:], label="good RAW")
plt.xlabel("spatial frequency k"); plt.ylabel("power"); plt.legend(); plt.tight_layout()
plt.savefig("./raw_compare/psd_bad_good.png", dpi=150); plt.close()

plt.figure(figsize=(6.5,3.0))
plt.semilogx(k[1:], ratio[1:])
plt.axhline(1, linestyle=":")
plt.xlabel("spatial frequency k"); plt.ylabel("good / bad power")
plt.tight_layout(); plt.savefig("./raw_compare/psd_ratio.png", dpi=150); plt.close()
