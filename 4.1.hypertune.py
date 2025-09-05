import numpy as np
import torch, math, time, random
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset
from torchsummary import summary
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from utils.data_loader import load_galaxies, get_classes,  get_synthetic, augment_images, apply_formatting
from utils.classifiers import RustigeClassifier, TinyCNN, MLPClassifier, SCNN, CNNSqueezeNet, ScatterResNet, DANNClassifier, BinaryClassifier, ScatterSqueezeNet, ScatterSqueezeNet2, DualCNNSqueezeNet
from utils.training_tools import EarlyStopping, reset_weights
from utils.calc_tools import cluster_metrics, normalise_images, check_tensor, fold_T_axis, compute_scattering_coeffs, custom_collate
from utils.plotting import plot_histograms, plot_images_by_class, plot_image_grid, plot_background_histogram
from torchvision.utils import make_grid, save_image
from kymatio.torch import Scattering2D
from scipy.ndimage import gaussian_filter
from collections import defaultdict, Counter
from astropy.io import fits
from astropy.convolution import convolve_fft, Gaussian2DKernel
from functools import lru_cache
import pandas as pd
import pickle
from tqdm import tqdm
from torch.optim import AdamW
import itertools
from itertools import product
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib 
import matplotlib.pyplot as plt
import sys, os, glob, re

SEED = 42  # Set a seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # make cuDNN deterministic (may slow a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)


os.makedirs('./classifier/trained_models', exist_ok=True)
os.makedirs('./classifier/trained_models_filtered', exist_ok=True)
matplotlib.use('Agg')
tqdm.pandas(disable=True)

print("Running ht1 with seed", SEED)


#############################################
################ CONFIGURATION ################
###############################################

galaxy_classes    = [50, 51]
max_num_galaxies  = 1000000
dataset_portions  = [1]
J, L, order       = 2, 12, 2
gen_model_names   = ['DDPM'] #['ST', 'DDPM', 'wGAN', 'GAN', 'Dual', 'CNN', 'STMLP', 'lavgSTMLP', 'ldiffSTMLP'] # Specify the generative model_name
num_epochs_cuda = 200
num_epochs_cpu = 100
folds = [5]  # e.g., [1, 2, 3, 4, 5] for five-fold cross-validation
max_num_galaxies = 1000000  # Upper limit for the all-classes combined training data before classical augmentation
lambda_values = [0]  # Ratio between generated images and original images per class. 8 is reserfved for TRAINONGENERATED
num_experiments = 3

# pick exactly one classifier
classifier        = ["TinyCNN",       # Very Simple CNN
                     "Rustige",       # from Rustige et al. 2023
                     "SCNN",          # simple CNN variant
                     "CNNSqueezeNet", # with SE blocks
                     "DualCNNSqueezeNet",
                     "CloudNet",      # from SorourMo/Cloud-Net
                     "DANN",          # domain‐adversarial NN
                     "ScatterNet",
                     "ScatterSqueezeNet",
                     "ScatterSqueezeNet2",
                     "Binary",
                     "ScatterResNet"][-4]

# Define every value you want to try
param_grid = {
    'lr':            [1e-4, 1e-3],
    'reg':           [1e-4, 1e-3],
    'label_smoothing':[0.1],
    'J':             [2],
    'L':             [12],
    'order':         [2],
    'percentile_lo': [1, 30, 60],   
    'percentile_hi': [80, 90, 99], 
    'crop_size':     [(512,512)],
    'downsample_size':[(128,128)],
    'versions':       ['T100kpcSUB']  # 'raw', 'T50kpc', ad hoc tapering: e.g. 'rt50'  strings in list → product() iterates them individually
} #'versions': [('raw', 'rt50')]  # tuple signals “stack these”

STRETCH = True  # Arcsinh stretch
USE_GLOBAL_NORMALISATION = False           # single on/off switch . False - image-by-image normalisation 
GLOBAL_NORM_MODE = "percentile"           # "percentile" or "flux"
ES, patience = True, 10  # Use early stopping
SCHEDULER = False  # Use a learning rate scheduler
SHOWIMGS = False  # Show some generated images for each class (Tool for control)

NORMALISEIMGS = True  # Globally normalise images to [0, 1]
NORMALISEIMGSTOPM = False  # Globally normalise images to [-1, 1] 
NORMALISESCS = False  # Normalise scattering coefficients to [0, 1]
NORMALISESCSTOPM = False  # Normalise scattering coefficients to [-1, 1]

BALANCE = True if galaxy_classes == [52, 53] else False  # Balance the dataset by undersampling the majority class
USE_CLASS_WEIGHTS = True  # Set to False to disable class weights
TRAINONGENERATED = False  # Use generated data as testdata
FILTERED = True  # Remove in training, validation and test data for the classifier
FILTERGEN = False  # Remove generated images that are too similar to other generated images

HISTOGRAMMATCHING = False  # Histogram matching for generated images
USE_MEMMAP = False  # Use memmap for scattering coefficients
SIGMACLIPPONDDPM = False  # Apply sigma clipping to DDPM data


# -------------------------- GAN CONFIGURATION --------------------------
gan_epoch = 5000           # e.g., epoch number to load from
gan_gen_loss = 'MSE'       # e.g., generator loss value (as used in filename)
gan_disc_loss = 'BCE'      # e.g., discriminator loss value (as used in filename)
gan_latent_dim = 128
gan_sample_size = 100
lr_gen = 1e-3
lr_disc = 1e-4
gan_adam_beta = (0.5, 0.999)
gan_weight_decay = 0
gan_label_smoothing = 0.7
gan_lambda_div = 0
gan_type = ['Simple', 'Advanced'][0]
gan_data_version = 'clean' if FILTERED else 'full'  # 'full' or 'clean'


# ---------------------------- VAE CONFIGURATION -----------------------------
VAE_train_size = 1101128
forbidden_classes = 12  # Generated bent sources look awful



########################################################################
##################### AUTOMATIC CONFIGURATION ##########################
########################################################################

# get the full list of available classes
classes = get_classes()
base_cls = min(galaxy_classes)

if torch.cuda.is_available():
    DEVICE = "cuda"
    num_epochs = num_epochs_cuda
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"   # Apple silicon fallback if relevant
    num_epochs = num_epochs_cpu
else:
    DEVICE = "cpu"
    num_epochs = num_epochs_cpu
print(f"{DEVICE.upper()} is available. Setting epochs to {num_epochs}.")


ARCSEC = np.deg2rad(1/3600.0)
PSZ2_ROOT = "/users/mbredber/scratch/data/PSZ2"  # FITS root used below

if TRAINONGENERATED:
    lambda_values = [8]  # To identify and distinguish TRAINONGENERATED from other runs
    print("Using generated data for testing.")

########################################################################
##################### HELPER FUNCTIONS #################################
########################################################################


def _append_rt_versions(imgs, fns, gen_versions, labels=None):
    """
    Append runtime-tapered planes to the loaded images and drop samples
    that are missing any requested taper.

    Inputs
    ------
    imgs: torch.Tensor [B, 1, H, W]
    fns:  list[str] length B
    gen_versions: e.g. ['rt50'] or ['rt50','rt100']
    labels: optional torch.Tensor [B] (kept in sync if provided)

    Returns
    -------
    imgs_out:   [B_kept, 1+len(gen_versions), H, W]
    labels_out: [B_kept] or None if labels=None
    fns_out:    list[str] of length B_kept
    info:       dict with removal counts
    """
    if isinstance(gen_versions, (str,)):
        gen_versions = [gen_versions]
    gen_versions = [str(gv).lower() for gv in gen_versions]
    assert imgs.dim() == 4 and imgs.size(1) == 1, "Expecting [B,1,H,W] images before appending"

    B, _, H, W = imgs.shape
    fns_str = list(map(str, fns))

    # Call the taper once per version; record which filenames survived and the plane per file.
    kept_sets = {}
    planes_by_fn = {}
    removed_by_version = {}

    for gv in gen_versions:
        tapered, keep_mask, kept_fns, skipped = apply_taper_to_tensor(
            imgs, gv, filenames=fns_str,
            crop_size=crop_size,
            downsample_size=downsample_size,
            percentile_lo=percentile_lo, percentile_hi=percentile_hi,
            do_stretch=STRETCH
        )
        kept_sets[gv] = set(kept_fns)
        removed_by_version[gv] = len(skipped)

        # Map filename -> tapered plane tensor [1,H,W]
        planes_by_fn[gv] = {fn: tapered[i] for i, fn in enumerate(kept_fns)}

    # Keep only samples that have *all* requested versions
    if gen_versions:
        keep_fns = [fn for fn in fns_str if all(fn in kept_sets[gv] for gv in gen_versions)]
    else:
        keep_fns = fns_str[:]  # nothing to filter

    # Indices in the original order
    keep_idx = [i for i, fn in enumerate(fns_str) if fn in keep_fns]
    removed_total = B - len(keep_idx)

    # Filter base images/labels/fns
    imgs_kept = imgs[keep_idx]
    labels_kept = labels[keep_idx] if labels is not None else None
    fns_kept = [fns[i] for i in keep_idx]

    # Gather tapered planes in the same order as fns_kept
    extra_planes = []
    for gv in gen_versions:
        if len(keep_fns) == 0:
            # preserve shape/device/dtype even if empty
            plane_stack = torch.empty((0, 1, H, W), device=imgs.device, dtype=imgs.dtype)
        else:
            # stack [B_kept, 1, H, W] in filename order
            plane_stack = torch.stack([planes_by_fn[gv][str(fn)] for fn in fns_kept], dim=0)
            plane_stack = plane_stack.to(device=imgs_kept.device, dtype=imgs_kept.dtype)
        extra_planes.append(plane_stack)

    # Concatenate base + all tapered planes along channel dim
    planes = [imgs_kept] + extra_planes
    imgs_out = torch.cat(planes, dim=1) if planes else imgs_kept

    info = {
        "initial": B,
        "kept": len(keep_idx),
        "removed_total": removed_total,
        "removed_by_version": removed_by_version
    }

    # Human-readable log
    if gen_versions:
        per_ver = ", ".join(f"{gv}:{removed_by_version.get(gv,0)}" for gv in gen_versions)
        print(f"[rt-append] Removed {removed_total} / {B} due to missing tapers "
              f"(per-version: {per_ver}). Kept {len(keep_idx)}.")
    else:
        print(f"[rt-append] No taper versions requested. Kept all {B} samples.")

    return imgs_out, labels_kept, fns_kept, info

def _first(pattern: str):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None

# After you get tapered images back:
def bg_sigma(t):
    m = _background_ring_mask(t.shape[-2], t.shape[-1], inner=64)
    return _robust_sigma(t.squeeze(1)[..., m])  # [B] robust σ


def _pixscale_arcsec(hdr):
    if 'CDELT1' in hdr:  # deg/pix
        return abs(hdr['CDELT1']) * 3600.0
    cd11 = hdr.get('CD1_1'); cd12 = hdr.get('CD1_2', 0.0)
    if cd11 is not None:
        return float(np.hypot(cd11, cd12)) * 3600.0
    raise KeyError("No CDELT* or CD* keywords in FITS header")

def fwhm_to_sigma_rad(fwhm_arcsec):
    return (fwhm_arcsec*ARCSEC) / (2*np.sqrt(2*np.log(2)))

def _cov_from_beam(bmaj_as, bmin_as, pa_deg):
    sx = fwhm_to_sigma_rad(bmaj_as)   # radians
    sy = fwhm_to_sigma_rad(bmin_as)
    th = np.deg2rad(pa_deg)
    R = np.array([[np.cos(th), -np.sin(th)],
                  [np.sin(th),  np.cos(th)]], dtype=float)
    S = np.diag([sx*sx, sy*sy])
    return R @ S @ R.T

def _kernel_from_headers(raw_hdr, targ_hdr, _pixscale_arcsec_unused=None):
    """
    Build the Gaussian2DKernel that converts the RAW beam into the TARGET beam,
    using the full WCS CD/PC matrix (handles rotation & anisotropy). Returns
    None if no additional blur is needed.
    """
    if targ_hdr is None:
        return None

    def _beam_cov_matrix(bmaj_as, bmin_as, pa_deg):
        ARCSEC = np.deg2rad(1/3600.0)
        sx = (bmaj_as*ARCSEC) / (2*np.sqrt(2*np.log(2)))
        sy = (bmin_as*ARCSEC) / (2*np.sqrt(2*np.log(2)))
        th = np.deg2rad(pa_deg)
        R = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th),  np.cos(th)]], dtype=float)
        S = np.diag([sx*sx, sy*sy])
        return R @ S @ R.T

    def _cd_matrix_rad(hdr):
        # Return 2×2 CD in radians per pixel (supports CD or PC+CDELT)
        if 'CD1_1' in hdr:
            CD = np.array([[hdr['CD1_1'], hdr.get('CD1_2', 0.0)],
                           [hdr.get('CD2_1', 0.0), hdr['CD2_2']]], dtype=float)
        else:
            pc11 = hdr.get('PC1_1', 1.0); pc12 = hdr.get('PC1_2', 0.0)
            pc21 = hdr.get('PC2_1', 0.0); pc22 = hdr.get('PC2_2', 1.0)
            cdelt1 = hdr.get('CDELT1', 1.0); cdelt2 = hdr.get('CDELT2', 1.0)
            CD = np.array([[pc11, pc12],[pc21, pc22]], dtype=float) @ np.diag([cdelt1, cdelt2])
        return CD * (np.pi/180.0)

    # Beams in arcsec (+ PA in deg)
    bmaj_r = float(raw_hdr['BMAJ'])*3600.0
    bmin_r = float(raw_hdr['BMIN'])*3600.0
    pa_r   = float(raw_hdr.get('BPA', 0.0))

    bmaj_t = float(targ_hdr['BMAJ'])*3600.0
    bmin_t = float(targ_hdr['BMIN'])*3600.0
    pa_t   = float(targ_hdr.get('BPA', pa_r))

    # Covariances in radians^2 (world coords)
    C_raw = _beam_cov_matrix(bmaj_r, bmin_r, pa_r)
    C_tgt = _beam_cov_matrix(bmaj_t, bmin_t, pa_t)

    # Desired kernel covariance: C_ker = C_tgt - C_raw
    C_ker = C_tgt - C_raw
    w, V = np.linalg.eigh(C_ker)
    w = np.clip(w, 0.0, None)        # numerical guard
    C_ker = (V * w) @ V.T
    if not np.any(w > 0):
        return None  # nothing to blur

    # Transform to pixel coords on the RAW grid: C_pix = J^{-1} C_ker J^{-T}
    J = _cd_matrix_rad(raw_hdr)
    Jinv = np.linalg.inv(J)
    C_pix = Jinv @ C_ker @ Jinv.T

    # Eigen-decompose in pixels
    w_pix, V_pix = np.linalg.eigh(C_pix)
    w_pix = np.clip(w_pix, 0.0, None)
    if not np.any(w_pix > 0):
        return None
    sx_pix = float(np.sqrt(w_pix[1]))  # major
    sy_pix = float(np.sqrt(w_pix[0]))  # minor
    theta  = float(np.arctan2(V_pix[1,1], V_pix[0,1]))

    if sx_pix == 0.0 and sy_pix == 0.0:
        return None

    return Gaussian2DKernel(x_stddev=sx_pix, y_stddev=sy_pix, theta=theta)

def _beam_solid_angle_sr(hdr):
    """Gaussian beam solid angle in steradians; BMAJ/BMIN in degrees."""
    bmaj = float(hdr['BMAJ']) * (np.pi/180.0)
    bmin = float(hdr['BMIN']) * (np.pi/180.0)
    return (np.pi / (4.0*np.log(2.0))) * bmaj * bmin


@lru_cache(maxsize=None)
def _headers_for_name(base_name: str):
    """
    Return (raw_hdr, t25_hdr, t50_hdr, t100_hdr, pix_native_arcsec, raw_fits_path)
    """
    base_dir = _first(f"{PSZ2_ROOT}/fits/{base_name}*") or f"{PSZ2_ROOT}/fits/{base_name}"
    raw_path = _first(f"{base_dir}/{os.path.basename(base_dir)}.fits") \
            or _first(f"{base_dir}/{os.path.basename(base_dir)}*.fits")
    if raw_path is None:
        raise FileNotFoundError(f"RAW FITS not found under {base_dir}")

    # look beside RAW first
    t25_path  = _first(f"{base_dir}/{base_name}T25kpc*.fits")
    t50_path  = _first(f"{base_dir}/{base_name}T50kpc*.fits")
    t100_path = _first(f"{base_dir}/{base_name}T100kpc*.fits")

    # fallbacks to 'classified' trees
    if t25_path is None:
        t25_path = _first(f"{PSZ2_ROOT}/classified/T25kpc/*/{base_name}.fits") or \
                   _first(f"{PSZ2_ROOT}/classified/T25kpcSUB/*/{base_name}.fits")
    if t50_path is None:
        t50_path = _first(f"{PSZ2_ROOT}/classified/T50kpc/*/{base_name}.fits") or \
                   _first(f"{PSZ2_ROOT}/classified/T50kpcSUB/*/{base_name}.fits")
    if t100_path is None:
        t100_path = _first(f"{PSZ2_ROOT}/classified/T100kpc/*/{base_name}.fits") or \
                    _first(f"{PSZ2_ROOT}/classified/T100kpcSUB/*/{base_name}.fits")

    raw_hdr  = fits.getheader(raw_path)
    t25_hdr  = fits.getheader(t25_path)  if t25_path  else None
    t50_hdr  = fits.getheader(t50_path)  if t50_path  else None
    t100_hdr = fits.getheader(t100_path) if t100_path else None
    pix_native = _pixscale_arcsec(raw_hdr)
    return raw_hdr, t25_hdr, t50_hdr, t100_hdr, pix_native, raw_path


def plot_raw_vs_fake_taper(raw_imgs, tapered_imgs, filenames, taper_mode,
                           save_dir="./classifier", max_n=4):
    n = min(max_n, raw_imgs.size(0), tapered_imgs.size(0), len(filenames))
    fig, axes = plt.subplots(2, n, figsize=(3*n, 6), constrained_layout=True)
    for i in range(n):
        r = raw_imgs[i, 0].detach().cpu().numpy()
        t = tapered_imgs[i, 0].detach().cpu().numpy()
        axes[0, i].imshow(r, cmap="viridis", origin="lower", vmin=0, vmax=1)
        axes[0, i].axis("off"); axes[0, i].set_title(str(filenames[i]), fontsize=8)
        axes[1, i].imshow(t, cmap="viridis", origin="lower", vmin=0, vmax=1)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("RAW", fontsize=11)
    axes[1, 0].set_ylabel("RAW → " + ("50 kpc" if taper_mode=="rt50" else "100 kpc"), fontsize=11)
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"raw_vs_{taper_mode}_eval_{n}x2_{raw_imgs.shape[-2]}x{raw_imgs.shape[-1]}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")

def permute_like(x, perm):
    if x is None: return None
    idx = perm.cpu().tolist()
    if isinstance(x, torch.Tensor): return x[perm]
    if isinstance(x, np.ndarray):   return x[idx]
    if isinstance(x, (list, tuple)): return [x[i] for i in idx]
    return x

base_cls = min(galaxy_classes)
def relabel(y): 
    return (y - base_cls).long()


def _maybe_resize_center(img_chw, out_hw):
    """Center-crop + resize but skips resize if shape already matches (C)."""
    # img_chw: [1,H,W] torch; out_hw: (Hout, Wout)
    C, H0, W0 = img_chw.shape
    Hout, Wout = out_hw
    # center-crop to requested crop_size (same as your apply_formatting logic)
    y0, x0 = H0//2, W0//2
    y1, y2 = y0 - Hout//2, y0 + (Hout - Hout//2)
    x1, x2 = x0 - Wout//2, x0 + (Wout - Wout//2)
    y1, y2 = max(0, y1), min(H0, y2)
    x1, x2 = max(0, x1), min(W0, x2)
    crop = img_chw[:, y1:y2, x1:x2]
    if crop.shape[-2:] == (Hout, Wout):
        return crop
    return torch.nn.functional.interpolate(
        crop.unsqueeze(0), size=(Hout, Wout), mode='bilinear', align_corners=False
    ).squeeze(0)
    

def _background_ring_mask(h, w, inner=64, pad=8):
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    cy, cx = h//2, w//2
    half = inner//2
    # guard band
    mask_c = (yy >= cy-(half+pad)) & (yy <= cy+(half+pad)) & (xx >= cx-(half+pad)) & (xx <= cx+(half+pad))
    return ~mask_c

def _robust_sigma(x2d):
    x = torch.as_tensor(x2d, dtype=torch.float32)
    med = x.median()
    return 1.4826 * (x - med).abs().median()

def build_ref_sigma_map(real_imgs, filenames, inner=64):
    """Measure background σ on the already-loaded images (top row)."""
    ref = {}
    for img, name in zip(real_imgs, filenames):
        im = torch.as_tensor(img, dtype=torch.float32).squeeze(0)  # [H,W]
        mask = _background_ring_mask(*im.shape, inner=inner)
        ref[str(name)] = _robust_sigma(im[mask]).item()
    return ref

def _per_image_percentile_stretch(x2d, lo=60, hi=95):
    t = torch.as_tensor(x2d, dtype=torch.float32)
    pl = torch.quantile(t.reshape(-1), lo/100.0)
    ph = torch.quantile(t.reshape(-1), hi/100.0)
    y = (t - pl) / (ph - pl + 1e-6)
    return y.clamp(0, 1)


def apply_taper_to_tensor(
    imgs, mode, filenames,
    crop_size=(1,512,512), downsample_size=(1,128,128),
    percentile_lo=60, percentile_hi=95,
    do_stretch=True, use_asinh=True,
    ref_sigma_map=None, bg_inner=64,
    debug_dir=None
):
    mode = str(mode).lower()
    m = re.fullmatch(r'rt(\d+)', mode)
    want_kpc = int(m.group(1)) if m else None
    if want_kpc is None:
        keep_mask = torch.ones(len(filenames), dtype=torch.bool)
        return (imgs if imgs.dim()==4 else imgs.unsqueeze(1)), keep_mask, list(map(str, filenames)), []

    device = imgs.device if torch.is_tensor(imgs) else torch.device('cpu')
    dtype  = imgs.dtype  if torch.is_tensor(imgs) else torch.float32
    Hout, Wout = downsample_size[-2], downsample_size[-1]

    out, kept_fns, kept_flags, skipped = [], [], [], []
    for base in map(str, filenames):
        # NOTE: your _headers_for_name returns in this order:
        raw_hdr, t25_hdr, t50_hdr, t100_hdr, pix_native_as, raw_path = _headers_for_name(base)

        # Choose (or synthesize) the target header
        if   want_kpc == 25:   targ_hdr = t25_hdr
        elif want_kpc == 50:   targ_hdr = t50_hdr
        elif want_kpc == 100:  targ_hdr = t100_hdr
        else:
            # Off-grid (e.g. rt60): build a synthetic target by interpolating the beam
            # between the nearest fixed tapers that exist. Falls back to nearest neighbour.
            def _interp_hdr(k_lo, h_lo, k_hi, h_hi, k_want):
                if h_lo is None and h_hi is None:
                    return None
                if h_lo is None or h_hi is None:
                    return (h_lo or h_hi).copy()
                w = (k_want - k_lo) / float(k_hi - k_lo)
                out = h_lo.copy()
                for key in ("BMAJ", "BMIN"):
                    v_lo = float(h_lo[key])
                    v_hi = float(h_hi[key])
                    out[key] = v_lo * (1.0 - w) + v_hi * w
                # Use BPA from the closer endpoint when available
                bpa_lo = float(h_lo.get("BPA", h_hi.get("BPA", 0.0)))
                bpa_hi = float(h_hi.get("BPA", h_lo.get("BPA", 0.0)))
                out["BPA"] = bpa_lo if (k_want - k_lo) <= (k_hi - k_want) else bpa_hi
                return out

            if want_kpc < 50:
                targ_hdr = _interp_hdr(25, t25_hdr, 50, t50_hdr, want_kpc)
            elif want_kpc < 100:
                targ_hdr = _interp_hdr(50, t50_hdr, 100, t100_hdr, want_kpc)
            else:
                # ≥100 kpc: extrapolate from (50,100) if both exist, else nearest
                targ_hdr = _interp_hdr(50, t50_hdr, 100, t100_hdr, want_kpc)


        # —— enforce parity with fixed sets ——
        if want_kpc in (25, 50, 100):
            # exact parity: require that exact fixed header exists
            if targ_hdr is None:
                skipped.append(base); kept_flags.append(False); continue
        else:
            # off-grid: require that the source *has* a redshift (at least one fixed header present)
            if (t25_hdr is None) and (t50_hdr is None) and (t100_hdr is None):
                skipped.append(base); kept_flags.append(False); continue
            if targ_hdr is None:
                # could not synthesize even though some fixed headers exist → skip
                skipped.append(base); kept_flags.append(False); continue

        # 1) Load RAW
        raw_native = np.squeeze(fits.getdata(raw_path)).astype(float)
        raw_native = np.nan_to_num(raw_native, copy=False)

        # 2) PSF kernel; if it degenerates, skip (no raw-through)
        ker = _kernel_from_headers(raw_hdr, targ_hdr, pix_native_as)
        if ker is None:
            # No extra blur needed — do NOT drop; just use raw map.
            matched = raw_native.copy()
        else:
            matched = convolve_fft(
                raw_native, ker, boundary='fill', fill_value=np.nan,
                nan_treatment='interpolate', normalize_kernel=True
            )

        # Convert Jy/beam_native → Jy/beam_target
        try:
            omega_raw = _beam_solid_angle_sr(raw_hdr)
            omega_tgt = _beam_solid_angle_sr(targ_hdr)
            matched = matched * (omega_tgt / omega_raw)
        except Exception:
            pass

        matched = np.nan_to_num(matched, copy=False)

        # 3) center-crop + resize
        t = torch.from_numpy(matched).float().unsqueeze(0)
        formatted = apply_formatting(t, crop_size=(1, crop_size[-2], crop_size[-1]),
                                     downsample_size=(1, Hout, Wout)).squeeze(0)

        # 4) percentile stretch
        stretched = _per_image_percentile_stretch(formatted.squeeze(0), percentile_lo, percentile_hi).unsqueeze(0) if do_stretch else formatted

        # 5) asinh
        if use_asinh:
            stretched = torch.asinh(10.0 * stretched) / math.asinh(10.0)

        # 6) optional noise-match
        if ref_sigma_map is not None and base in ref_sigma_map:
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

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            fig, ax = plt.subplots(1,2, figsize=(6,3))
            ax[0].imshow(formatted.squeeze(0), cmap='viridis', origin='lower'); ax[0].axis('off'); ax[0].set_title('PSF-matched')
            ax[1].imshow(stretched.squeeze(0), cmap='viridis', origin='lower'); ax[1].axis('off'); ax[1].set_title('final')
            fig.suptitle(base); fig.tight_layout()
            fig.savefig(os.path.join(debug_dir, f"{base}_rt{want_kpc}.png"), dpi=140)
            plt.close(fig)

    keep_mask = torch.tensor(kept_flags, dtype=torch.bool)
    out = torch.stack(out, dim=0).to(device=device, dtype=dtype) if out else torch.empty((0,1,Hout,Wout), device=device, dtype=dtype)
    return out, keep_mask, kept_fns, skipped



def replicate_list(x, n):
    return [v for v in x for _ in range(int(n))]

def shuffle_with_filenames(images, labels, filenames=None):
    perm = torch.randperm(images.size(0))
    images, labels = images[perm], labels[perm]
    if filenames is not None:
        filenames = [filenames[i] for i in perm.tolist()]
    return images, labels, filenames

def late_augment(images, labels, filenames=None, *, st_aug=False):
    """
    Apply your normal augmentations AFTER tapering.
    Returns (imgs_aug, labels_aug, filenames_aug).
    The replication factor n_aug is inferred from sizes.
    """
    imgs_aug, labels_aug = augment_images(images, labels, ST_augmentation=st_aug)
    n_aug = imgs_aug.size(0) // max(1, images.size(0))     # infer n_aug (e.g. 8 for 4 rotations × 2 flips)
    if filenames is not None:
        filenames = replicate_list(filenames, n_aug)
    return imgs_aug, labels_aug, filenames




###############################################
########### DATA STORING FUNCTIONS ###############
###############################################


def initialize_metrics(metrics,
                    model_name, subset_size, fold, experiment,
                    lr, reg, lam,
                    crop, down, ver):
    # make a short stable string for each hyperparam
    cs = f"{crop[0]}x{crop[1]}"
    ds = f"{down[0]}x{down[1]}"

    key_base = (
    f"{model_name}"
    f"_ss{subset_size}"
    f"_f{fold}"
    f"_lr{lr}"
    f"_reg{reg}"
    f"_lam{lam}"
    f"_cs{cs}"
    f"_ds{ds}"
    f"_ver{ver}"
    )
    
    for k in [
        f"{key_base}_accuracy",
        f"{key_base}_precision",
        f"{key_base}_recall",
        f"{key_base}_f1_score",
    ]:
        if k not in metrics:
            metrics[k] = []


def update_metrics(metrics,
                model_name, subset_size, fold, experiment,
                lr, reg, accuracy, precision, recall, f1, lam,
                crop, down, ver):
    cs = f"{crop[0]}x{crop[1]}"
    ds = f"{down[0]}x{down[1]}"
    
    key_base = (
    f"{model_name}"
    f"_ss{subset_size}"
    f"_f{fold}"
    f"_lr{lr}"
    f"_reg{reg}"
    f"_lam{lam}"
    f"_cs{cs}"
    f"_ds{ds}"
    f"_ver{ver}"
    )

    metrics[f"{key_base}_accuracy"].append(accuracy)
    metrics[f"{key_base}_precision"].append(precision)
    metrics[f"{key_base}_recall"].append(recall)
    metrics[f"{key_base}_f1_score"].append(f1)
    
def initialize_history(history, model_name, subset_size, fold, experiment, lr, reg, lambda_generate):
    if model_name not in history:
        history[model_name] = {}

    loss_key = f"{model_name}_loss_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
    val_loss_key = f"{model_name}_val_loss_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"

    if loss_key not in history[model_name]:
        history[model_name][loss_key] = []
    if val_loss_key not in history[model_name]:
        history[model_name][val_loss_key] = []

def initialize_labels(all_true_labels, all_pred_labels, model_name, subset_size, fold, experiment, lr, reg, lambda_generate):
    key = f"{model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
    all_true_labels[key] = []
    all_pred_labels[key] = []


###########################################################################
################# MAIN LOOP FOR PLOTTING GAUSSIAN BLUR GRID #################
###########################################################################

for lr, reg, ls, J_val, L_val, order_val, lo_val, hi_val, crop, down, vers in product(
        param_grid['lr'],
        param_grid['reg'],
        param_grid['label_smoothing'],
        param_grid['J'],
        param_grid['L'],
        param_grid['order'],
        param_grid['percentile_lo'],
        param_grid['percentile_hi'],
        param_grid['crop_size'],
        param_grid['downsample_size'],
        param_grid['versions']
    ):
    percentile_lo = lo_val
    percentile_hi = hi_val

    # Assign into your existing variables
    learning_rates       = [lr]
    regularization_params = [reg]
    label_smoothing      = ls
    J, L, order          = J_val, L_val, order_val
    crop_size            = crop
    downsample_size      = down
    versions              = vers
    
    print(f"\n▶ Experiment: g_classes={galaxy_classes}, lr={lr}, reg={reg}, ls={ls}, "
        f"J={J}, L={L}, crop={crop_size}, down={downsample_size}, ver={versions}, "
        f"lo={percentile_lo}, hi={percentile_hi}, classifier={classifier}, "
        f"global_norm={USE_GLOBAL_NORMALISATION}, norm_mode={GLOBAL_NORM_MODE} ◀\n")

    if any (cls in galaxy_classes for cls in [10, 11, 12, 13]):
        crop_size = (128, 128)  # Crop size for the images
        downsample_size = (128, 128)  # Downsample size for the images
        batch_size = 128
    elif galaxy_classes[0] in list(range(40, 49)):
        crop_size = (1600, 1600)  # Crop size for the images
        downsample_size = (128, 128)  # Downsample size for the images
        batch_size = 16
    elif galaxy_classes[0] in list(range(50, 60)):
        crop_size = (1, 512, 512)  # Crop size for the images
        downsample_size = (1, 128, 128)  # Downsample size for the images
        batch_size = 16 

    img_shape = downsample_size

    # parse versions
    def _split_versions(v):
        v = v if isinstance(v, (list, tuple)) else [v]
        v = [str(x).lower() for x in v]
        gen, load = [], []
        for x in v:
            m = re.fullmatch(r'rt(\d+)(?:kpc)?', x)
            if m:
                gen.append(f"rt{m.group(1)}")
            else:
                load.append(x)
        return load, gen

    _load_versions, _gen_versions = _split_versions(versions)

    # NEW: align RTxx exactly to the fixed uv–tapered set Txxkpc by using that
    ALIGN_RT_WITH_FIXED = True
    if ALIGN_RT_WITH_FIXED and _gen_versions:
        def _rt_anchor(g):
            m = re.fullmatch(r'rt(\d+)', g)
            if not m:
                return None
            k = int(m.group(1))
            # nearest fixed anchor among 25/50/100 kpc (midpoints at 37.5 and 75)
            if k < 38:
                anchor = 25
            elif k < 75:
                anchor = 50
            else:
                anchor = 100
            return f"T{anchor}kpc"
        anchors = [a for a in (_rt_anchor(g) for g in _gen_versions) if a]
        _versions_to_load = [anchors[0]] if anchors else (_load_versions if _load_versions else ['raw'])
    else:
        _versions_to_load = _load_versions if _load_versions else ['raw']


    print(f"_versions_to_load={_versions_to_load}, _gen_versions={_gen_versions}")

    REPLACE_WITH_RT = (
        (isinstance(versions, str) and versions.lower().startswith('rt')) or
        (isinstance(versions, (list, tuple)) and len(versions) == 1
        and isinstance(versions[0], str) and versions[0].lower().startswith('rt'))
    )

    # drive augmentation/filenames solely from presence of rt*
    LATE_AUG = bool(_gen_versions) # True if any(v.startswith('rt') for v in _gen_versions)
    PRINTFILENAMES = bool(_gen_versions)
    EXTRAVARS = False  # Use extra features (redshift, mass, size) for the classifier. Will automatically be true if test_meta is not None.

    if set(galaxy_classes) & {18} or set(galaxy_classes) & {19}:
        galaxy_classes = [20, 21, 21, 22, 23, 24, 25, 26, 27, 28, 29]  # Include all digits if 18 or 19 is in target_classes
    else:
        galaxy_classes = galaxy_classes
    num_classes = len(galaxy_classes)

    EXTRAVARS = False  # Use extra features (redshift, mass, size) for the classifier. Will automatically be true if test_meta is not None.
    LATE_AUG = bool(_gen_versions) # True if any(v.startswith('rt') for v in _gen_versions)
    PRINTFILENAMES = bool(_gen_versions)
    
    def _verkey(v):
        if isinstance(v, (list, tuple)):
            return "+".join(map(str, v))
        return str(v)
    ver_key = _verkey(versions)

    ###############################################
    ########## INITIALIZE DICTIONARIES ############
    ###############################################

    metrics = {
        "accuracy": {},
        "precision": {},
        "recall": {},
        "f1_score": {}
    }

    metric_colors = {
        "accuracy": 'blue',
        "precision": 'green',
        "recall": 'red',
        "f1_score": 'orange'
    }

    dataset_sizes   = defaultdict(list)
    metrics         = defaultdict(list)
    all_true_labels = defaultdict(list)
    all_pred_labels = defaultdict(list)
    all_pred_probs  = defaultdict(list)
    training_times = {}  # dict of dicts of lists
    history        = {}  # dict of dicts of lists


    ###############################################
    ########### READ IN TEST DATA #################
    ######## Needs only be done once ##############
    ###############################################

    scattering = Scattering2D(J=J, L=L, shape=img_shape[-2:], max_order=order)      
    for gen_model_name in gen_model_names:

        with torch.no_grad():
            dummy = torch.zeros((1, *img_shape)).cpu()
            scat_dummy = scattering(dummy)
            if scat_dummy.dim()==5:
                # fold T into channels
                scat_dummy = scat_dummy.flatten(1,2)
            # now scat_dummy.shape == [1, C, H, W]
            scatshape = tuple(scat_dummy.shape[1:])   # (C, H, W)
        hidden_dim1 = 256
        hidden_dim2 = 128
        vae_latent_dim = 64
        
        _out  = load_galaxies(galaxy_classes=galaxy_classes,
                    versions=_versions_to_load or ['raw'], 
                    fold=max(folds), #Any fold other than 5 gives me the test data for the five fold cross validation
                    crop_size=crop_size,
                    downsample_size=downsample_size,
                    sample_size=max_num_galaxies, 
                    REMOVEOUTLIERS=FILTERED,
                    BALANCE=BALANCE,           # Reduce the larger classes to the size of the smallest class
                    STRETCH=STRETCH,
                    percentile_lo=percentile_lo,  # Percentile stretch lower bound
                    percentile_hi=percentile_hi,  # Percentile stretch upper bound
                    AUGMENT=not LATE_AUG,
                    NORMALISE=NORMALISEIMGS,
                    NORMALISETOPM=NORMALISEIMGSTOPM,
                    USE_GLOBAL_NORMALISATION=USE_GLOBAL_NORMALISATION,
                    GLOBAL_NORM_MODE=GLOBAL_NORM_MODE,
                    PRINTFILENAMES=PRINTFILENAMES,
                    train=False)

        if len(_out) == 4:
            train_images, train_labels, test_images, test_labels = _out
            train_fns = test_fns = None
            perm_test = torch.randperm(test_images.size(0))
            test_images, test_labels = test_images[perm_test], test_labels[perm_test]
        elif len(_out) == 6:
            train_images, train_labels, test_images, test_labels, train_fns, test_fns = _out
            test_images, test_labels, test_fns = shuffle_with_filenames(test_images, test_labels, test_fns)
        else:
            raise ValueError(f"load_galaxies returned {len(_out)} values, expected 4 or 6")

        train_labels = relabel(train_labels)  # Will be used for the sanity check below
        test_labels  = relabel(test_labels)   # [0,1,...] aligning with galaxy_classes
        unique_labels, counts = torch.unique(test_labels, return_counts=True)

        # --- PSF matching for the test/eval set only (RAW → 50/100) ---
        if _gen_versions:
            if test_fns is None:
                raise RuntimeError("versions includes rt*, but filenames were not returned. Set PRINTFILENAMES=True.")
            test_images, test_labels, test_fns, info_test = _append_rt_versions(test_images, test_fns, _gen_versions, labels=test_labels)
            print(f"[TEST] dropped {info_test['removed_total']} (kept {info_test['kept']}/{info_test['initial']})")
            assert info_test['removed_total'] == 0, (
                f"RT alignment error (TEST): expected 0 drops with {_versions_to_load} "
                f"as anchor, but dropped {info_test['removed_total']} out of {info_test['initial']}."
            )

            if REPLACE_WITH_RT:
                test_images = test_images[:, 1:2]  # keep only the runtime-tapered plane
                print(f"[TEST] after replacing with RT: {test_images.size(0)} images")

            if LATE_AUG:
                test_images, test_labels, test_fns = late_augment(test_images, test_labels, test_fns)
                print(f"[TEST] after late augmentation: {test_images.size(0)} images")


        # ——— Data sanity checks ———
        #for i, cls in enumerate(galaxy_classes):
        #    cls_mask = (train_labels == i)
        #    cls_images = train_images[cls_mask]
        #    check_tensor(f"Train images for class {cls} (idx={i})", cls_images)
        #    cls_mask = (test_labels == i)
        #    cls_images = test_images[cls_mask]
        #    check_tensor(f"Test images for class {cls} (idx={i})", cls_images)

        import hashlib

        def img_hash(img: torch.Tensor) -> str:
            # ensure CPU & contiguous
            arr = img.cpu().contiguous().numpy()
            return hashlib.sha1(arr.tobytes()).hexdigest()

        # after loading train_images, test_images:
        train_hss = {img_hash(img) for img in train_images}
        test_hashes   = {img_hash(img) for img in test_images}

        common = train_hss & test_hashes
        assert not common, f"Overlap detected: {len(common)} images appear in both train and test validation!"
        # ————————————————————————


        # Produce an empty tensor to occupy the not used component of the datasets. 
        mock_tensor = torch.zeros_like(test_images)
        if classifier in ['ScatterNet', 'ScatterResNet', 'ScatterSqueezeNet', 'ScatterSqueezeNet2']:
            test_images = fold_T_axis(test_images)
            test_scat_coeffs = compute_scattering_coeffs(test_images, scattering, batch_size=128, device='cpu')
            if test_scat_coeffs.dim() == 5:
                # [B, C_in, C_scat, H, W] → [B, C_in*C_scat, H, W]
                test_scat_coeffs = test_scat_coeffs.flatten(start_dim=1, end_dim=2)
                            
            if NORMALISESCS or NORMALISESCSTOPM:
                if NORMALISESCSTOPM:
                    test_scat_coeffs = normalise_images(test_scat_coeffs, -1, 1)
                else:
                    test_scat_coeffs = normalise_images(test_scat_coeffs, 0, 1)

            if classifier in ['ScatterNet', 'ScatterResNet']:
                mock_test = torch.zeros_like(test_images)  # images are ignored
                test_dataset = TensorDataset(mock_test, test_scat_coeffs, test_labels)
            elif classifier in ['ScatterSqueezeNet', 'ScatterSqueezeNet2']:
                if EXTRAVARS:
                    print("Not implemented yet for ScatterSqueezeNet with extra variables")
                    exit(1)
                else:
                    test_dataset = TensorDataset(test_images, test_scat_coeffs, test_labels)

        else:
            if test_images.dim() == 5:
                test_images = fold_T_axis(test_images)  # [B,T,1,H,W] -> [B,T,H,W]
            mock_tensor = torch.zeros_like(test_images)
            assert test_images.dim() == 4, f"test_images should be [B,C,H,W], got {tuple(test_images.shape)}"
            test_dataset = TensorDataset(test_images, mock_tensor, test_labels)

                                
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=False)

        ###############################################
        ########### LOOP OVER DATA FOLD ###############
        ###############################################            

        FIRSTTIME = True  # Set to True to print model summaries only once
        param_combinations = list(itertools.product(folds, learning_rates, regularization_params, lambda_values))
        for fold, lr, reg, lambda_generate in param_combinations:
            torch.cuda.empty_cache()
            runname = f"{galaxy_classes}_{gen_model_name}_lr{lr}_reg{reg}_lo{percentile_lo}_hi{percentile_hi}_cs{crop_size[0]}x{crop_size[1]}"

            log_path = f"./classifier/log_{runname}.txt"
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            #file = open(log_path, 'w')
                
            # ——— Data loading for training/validation ———
            if TRAINONGENERATED:
                # only synthetic for training; reserve real for validation
                train_images = torch.empty((0, *img_shape), device=DEVICE)
                train_labels = torch.empty((0,), dtype=torch.long, device=DEVICE)
                _out = load_galaxies(
                    galaxy_classes=galaxy_classes,
                    versions=_versions_to_load or ['raw'],
                    fold=fold,
                    crop_size=crop_size,
                    downsample_size=downsample_size,
                    sample_size=max_num_galaxies,
                    REMOVEOUTLIERS=FILTERED,
                    BALANCE=BALANCE,           # Reduce the larger classes to the size of the smallest class
                    AUGMENT=False,   
                    NORMALISE=NORMALISEIMGS,
                    NORMALISETOPM=NORMALISEIMGSTOPM,
                    USE_GLOBAL_NORMALISATION=USE_GLOBAL_NORMALISATION,
                    GLOBAL_NORM_MODE=GLOBAL_NORM_MODE,
                    train=True)
                
                if len(_out) == 4:
                    _, _, valid_images, valid_labels = _out
                    valid_data = None
                elif len(_out) == 6:
                    _, _, valid_images, valid_labels, _, valid_data = _out

                # Relabel and shuffle validation data
                valid_labels = relabel(valid_labels)  # [0,1,2,...] for classes [50,51,...]
                perm_valid = torch.randperm(valid_images.size(0))
                valid_images, valid_labels = valid_images[perm_valid], valid_labels[perm_valid]
                if PRINTFILENAMES: valid_fns = permute_like(valid_fns, perm_valid)

            else:
                # real train + valid
                _out = load_galaxies(
                    galaxy_classes=galaxy_classes,
                    versions=_versions_to_load or ['raw'],
                    fold=max(folds),
                    crop_size=crop_size,
                    downsample_size=downsample_size,
                    sample_size=max_num_galaxies, 
                    REMOVEOUTLIERS=FILTERED,
                    BALANCE=BALANCE,
                    STRETCH=STRETCH,
                    percentile_lo=percentile_lo,
                    percentile_hi=percentile_hi,
                    AUGMENT=not LATE_AUG,
                    NORMALISE=NORMALISEIMGS,
                    NORMALISETOPM=NORMALISEIMGSTOPM,
                    USE_GLOBAL_NORMALISATION=USE_GLOBAL_NORMALISATION,
                    GLOBAL_NORM_MODE=GLOBAL_NORM_MODE,
                    PRINTFILENAMES=PRINTFILENAMES,
                    train=True)

                if len(_out) == 4:
                    train_images, train_labels, valid_images, valid_labels = _out
                    train_fns = test_fns = None

                    perm_train = torch.randperm(train_images.size(0))
                    train_images, train_labels = train_images[perm_train], train_labels[perm_train]
                    
                    perm_valid = torch.randperm(valid_images.size(0))
                    valid_images, valid_labels = valid_images[perm_valid], valid_labels[perm_valid]

                elif len(_out) == 6:
                    if PRINTFILENAMES:
                        train_images, train_labels, valid_images, valid_labels, train_fns, valid_fns = _out
                    else:
                        print("Not implemented extradata yet")

                    # --- PSF matching for train/valid when requested ---
                    if _gen_versions:
                        if train_fns is None or valid_fns is None:
                            raise RuntimeError("versions includes rt*, but train/valid filenames were not returned.")
                        
                        # TRAIN/VALID sets
                        train_images, train_labels, train_fns, info_tr = _append_rt_versions(train_images, train_fns, _gen_versions, labels=train_labels)
                        valid_images, valid_labels, valid_fns, info_va = _append_rt_versions(valid_images, valid_fns, _gen_versions, labels=valid_labels)
                        print(f"[TRAIN] dropped {info_tr['removed_total']} (kept {info_tr['kept']}/{info_tr['initial']})")
                        print(f"[VALID] dropped {info_va['removed_total']} (kept {info_va['kept']}/{info_va['initial']})")
                        assert info_tr['removed_total'] == 0, (
                            f"RT alignment error (TRAIN): expected 0 drops with {_versions_to_load} "
                            f"as anchor, but dropped {info_tr['removed_total']} out of {info_tr['initial']}."
                        )
                        assert info_va['removed_total'] == 0, (
                            f"RT alignment error (VALID): expected 0 drops with {_versions_to_load} "
                            f"as anchor, but dropped {info_va['removed_total']} out of {info_va['initial']}."
                        )
                        
                        if REPLACE_WITH_RT:
                            train_images = train_images[:, 1:2]
                            valid_images = valid_images[:, 1:2]
                        
                    if LATE_AUG:
                        before = len(train_images)
                        train_images, train_labels, train_fns = late_augment(train_images, train_labels, train_fns)
                        valid_images, valid_labels, valid_fns = late_augment(valid_images, valid_labels, valid_fns)
                        n_aug = len(train_images) // max(1, before)
                        print(f"[AUG] late_augment replicated train by ~x{n_aug} ({before} → {len(train_images)})")
                    
                    if EXTRAVARS:
                        train_data = train_fns
                        valid_data = valid_fns
                dataset_sizes[fold] = [max(2, int(len(train_images) * p)) for p in dataset_portions]
                
                train_labels = relabel(train_labels)  # [0,1,2,...] for classes [50,51,...]
                valid_labels = relabel(valid_labels)  # [0,1,2,...] for classes [50,51,...]
                
            ##########################################################
            ############# READ IN GENERATED DATA #####################
            ##########################################################
                        
            # Generate more training data if requested
            if lambda_generate not in [0]:
                batch_size_generate = 500 
                if train_images.numel() > 0:
                    num_generate = int(lambda_generate / len(galaxy_classes) * len(train_images))
                else:
                    num_generate = int(lambda_generate * 125)

                
                for cls in galaxy_classes:
                    # use your shared helper to do all generation + filtering
                    gen_imgs_list, gen_lbls_list = get_synthetic(
                        num_generate=num_generate,
                        gen_model_name=gen_model_name,
                        cls=cls,
                        galaxy_classes=galaxy_classes,
                        img_shape=img_shape,
                        FILTERGEN=FILTERGEN,
                        CLIPDDPM=True,
                        model_kwargs={
                            'gan_type': gan_type,
                            'gan_latent_dim': gan_latent_dim,
                            'gan_sample_size': gan_sample_size,
                            'lr_gen': lr_gen,
                            'lr_disc': lr_disc,
                            'gan_gen_loss': gan_gen_loss,
                            'gan_disc_loss': gan_disc_loss,
                            'gan_adam_beta': gan_adam_beta,
                            'gan_weight_decay': gan_weight_decay,
                            'gan_label_smoothing': gan_label_smoothing,
                            'gan_lambda_div': gan_lambda_div,
                            'gan_data_version': gan_data_version,
                            'gan_epoch': gan_epoch,
                            'NORMALISEIMGS': NORMALISEIMGS,
                            'NORMALISETOPM': NORMALISEIMGSTOPM,
                            'VAE_train_size': VAE_train_size,
                            'scatshape': scatshape,
                            'hidden_dim1': hidden_dim1,
                            'hidden_dim2': hidden_dim2,
                            'vae_latent_dim': vae_latent_dim
                        },
                        fold=fold,
                        device=DEVICE
                    )
                    # flatten and append
                    if isinstance(gen_imgs_list, torch.Tensor):
                        generated_images = gen_imgs_list
                        print("Generated images shape: ", generated_images.shape)
                    else:
                        generated_images = torch.cat(gen_imgs_list, dim=0)
                        
                    if isinstance(gen_lbls_list, torch.Tensor):
                        generated_labels = gen_lbls_list
                    else:
                        generated_labels = torch.cat(gen_lbls_list, dim=0)

                    if train_images.numel() > 0:     
                        train_images = torch.cat([train_images, generated_images.to(train_images.device)])
                        train_labels = torch.cat([train_labels, generated_labels])
                    else:
                        train_images = generated_images.to(DEVICE)
                        train_labels = generated_labels
                        
                    # store per-class generated images for sanity checking later
                    if 'generated_by_class' not in locals() and SHOWIMGS:
                        generated_by_class = {}
                    generated_by_class[cls] = generated_images.cpu()

                    # Append the filtered images and labels to your training data:
                    if SHOWIMGS and lambda_generate not in [0, 8]:
                        pristine_train_images = train_images
                        pristine_train_labels = train_labels
                        
                # Check the tensor for generated images
                for cls in galaxy_classes:
                    cls_mask = (train_labels == cls)
                    cls_images = train_images[cls_mask]
                    check_tensor(f"Generated images for class {cls} with model {gen_model_name}", cls_images)
                train_labels = relabel(train_labels)  # [0,1,2,...] for classes [50,51,...]
                        
            if TRAINONGENERATED:
                # For each class, select the correct slice of (images, labels), then concatenate all.
                class_imgs, class_lbls = [], []
                offset = min(galaxy_classes)
                for cls in galaxy_classes:
                    idx = cls - offset
                    imgs = train_images[train_labels == idx]
                    lbls = train_labels[train_labels == idx]
                    if  cls == 10:
                        imgs, lbls = imgs[:389], lbls[:389]
                    elif cls == 11:
                        imgs, lbls = imgs[:816], lbls[:816]
                    elif cls == 12:
                        imgs, lbls = imgs[:292], lbls[:292]
                    elif cls == 13:
                        imgs, lbls = imgs[:242], lbls[:242]
                    class_imgs.append(imgs)
                    class_lbls.append(lbls)
                train_images = torch.cat(class_imgs, dim=0)
                train_labels = torch.cat(class_lbls, dim=0)
                        
                # Now apply augmentation to both images and labels
                train_images, train_labels = augment_images(train_images, train_labels)
                USE_CLASS_WEIGHTS = True
                
                # Shuffle the training data
                perm = torch.randperm(train_images.size(0))
                train_images = train_images[perm]
                train_labels = train_labels[perm]
                if EXTRAVARS:
                    print("Cannot use extra features with TRAINONGENERATED, setting EXTRAVARS to False")
                    EXTRAVARS = False
                
            unique_labels, counts = torch.unique(train_labels, return_counts=True)

            if dataset_sizes == {}:
                dataset_sizes[fold] = [int(len(train_images) * p) for p in dataset_portions]

                    
            ##########################################################
            ############ NORMALISE AND PACKAGE THE INPUT #############
            ##########################################################

            if classifier in ['Rustige', 'CNNSqueezeNet', 'SCNN', 'ScatterSqueezeNet', 'ScatterSqueezeNet2', 'Binary']:
                if lambda_generate not in [0, 8]:
                    # 1. concatenate images
                    img_splits = [
                        pristine_train_images.to(DEVICE),
                        generated_images.to(DEVICE),
                        valid_images.to(DEVICE)
                    ]
                    img_lengths = [len(t) for t in img_splits]
                    all_images = torch.cat(img_splits, dim=0)

                    # 2. concatenate labels in the same order
                    lbl_splits = [
                        pristine_train_labels.to(DEVICE),
                        generated_labels.to(DEVICE),
                        valid_labels.to(DEVICE)
                    ]
                    all_labels = torch.cat(lbl_splits, dim=0)

                    # 3. compute split boundaries once
                    boundaries = [0] + list(torch.cumsum(torch.tensor(img_lengths), dim=0).numpy())

                    # 4. slice images back into (pristine, generated, valid)
                    chunked_imgs = [
                        all_images[boundaries[i]:boundaries[i+1]]
                        for i in range(len(img_lengths))
                    ]
                    # 5. slice labels back in exactly the same way
                    chunked_lbls = [
                        all_labels[boundaries[i]:boundaries[i+1]]
                        for i in range(len(img_lengths))
                    ]

                    # 6. reassign train/valid splits
                    pristine_train_images, generated_images, valid_images = chunked_imgs
                    pristine_train_labels, generated_labels, valid_labels = chunked_lbls

                    # 7. rebuild train_images and train_labels without pulling validation back in
                    train_images = torch.cat([pristine_train_images, generated_images], dim=0)
                    train_labels = torch.cat([pristine_train_labels, generated_labels], dim=0)

                else:
                    # same idea if lambda_generate == 0 (i.e. only real train+valid)
                    img_splits = [
                        train_images.to(DEVICE),
                        valid_images.to(DEVICE)
                    ]
                    img_lengths = [len(t) for t in img_splits]
                    all_images = torch.cat(img_splits, dim=0)

                    lbl_splits = [
                        train_labels.to(DEVICE),
                        valid_labels.to(DEVICE)
                    ]
                    all_labels = torch.cat(lbl_splits, dim=0)

                    boundaries = [0] + list(torch.cumsum(torch.tensor(img_lengths), dim=0).numpy())
                    chunked_imgs = [
                        all_images[boundaries[i]:boundaries[i+1]]
                        for i in range(len(img_lengths))
                    ]
                    chunked_lbls = [
                        all_labels[boundaries[i]:boundaries[i+1]]
                        for i in range(len(img_lengths))
                    ]

                    train_images, valid_images = chunked_imgs
                    train_labels, valid_labels = chunked_lbls            
            
            # ── SANITY-CHECK PLOTS ON FIRST FOLD ONLY ──
            if fold == folds[0] and SHOWIMGS and downsample_size == (1, 128, 128):
                
                if len(galaxy_classes) == 2:
                    # Plot histograms for the two classes
                    train_images_cls1 = train_images[train_labels == galaxy_classes[0] - min(galaxy_classes)]
                    train_images_cls2 = train_images[train_labels == galaxy_classes[1] - min(galaxy_classes)]
                    
                    #Make sure the images are not tupples
                    if isinstance(train_images_cls1, tuple):
                        train_images_cls1 = train_images_cls1[0]
                    if isinstance(train_images_cls2, tuple):
                        train_images_cls2 = train_images_cls2[0]
                    
                    plot_histograms(
                        train_images_cls1.cpu(),
                        train_images_cls2.cpu(),
                        title1=f"Class {galaxy_classes[0]}",
                        title2=f"Class {galaxy_classes[1]}",
                        save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_histogram.png"
                    )

                    plot_background_histogram(
                        train_images_cls1.cpu(),        # shape (936, 1, 128, 128)
                        train_images_cls2.cpu(),        # shape (720, 1, 128, 128)
                        img_shape=(1, 128, 128),
                        title="Background histograms",
                        save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_background_hist.png"
                    )

                    for cls in galaxy_classes:
                        orig_imgs = train_images[train_labels == (cls - min(galaxy_classes))][:36]
                        test_imgs = test_images[test_labels == (cls - min(galaxy_classes))][:36]
                                    
                        plot_image_grid(
                            orig_imgs.cpu(),
                            num_images=36,
                            save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_{cls}_train_grid.png"
                        )
                        plot_image_grid(
                            test_imgs.cpu(),
                            num_images=36,
                            save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_{cls}_test_grid.png"
                        )
                        
                        if lambda_generate not in [0, 8]:
                            gen_imgs = generated_by_class[cls][:36]

                            plot_image_grid(
                                gen_imgs,
                                num_images=36,
                                save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_{cls}_generated_grid.png"
                            )
                            plot_histograms(
                                gen_imgs,
                                orig_imgs.cpu(),
                                title1="Generated Images",
                                title2="Train Images",
                                save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_{cls}_histogram.png"
                            )
                            plot_background_histogram(
                                orig_imgs,
                                gen_imgs,
                                img_shape=(1, 128, 128),
                                title="Background histograms",
                                save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_{cls}_background_hist.png")
            
            if USE_CLASS_WEIGHTS:
                unique, counts = np.unique(train_labels.cpu().numpy(), return_counts=True)
                total_count = sum(counts)
                class_weights = {i: total_count / count for i, count in zip(unique, counts)}
                weights = torch.tensor([class_weights.get(i, 1.0) for i in range(num_classes)],
                                    dtype=torch.float, device=DEVICE)
                missing_classes = [cls for cls in unique if cls not in class_weights]
                if missing_classes:
                    print(f"Warning: Missing classes in dataset: {missing_classes}")
                    class_weights.update({int(cls): 1.0 for cls in missing_classes})
                    
                unique_valid, counts_valid = np.unique(valid_labels.cpu().numpy(), return_counts=True)
                unique_test, counts_test = np.unique(test_labels.cpu().numpy(), return_counts=True)
                class_counts_valid = dict(zip(map(int, unique_valid), map(int, counts_valid)))
                class_counts_test = dict(zip(map(int, unique_test), map(int, counts_test)))
            else:
                weights = None

            if fold in [0, 5] and SHOWIMGS:
                imgs = train_images.detach().cpu().numpy()
                lbls = (train_labels + min(galaxy_classes)).detach().cpu().numpy() # This is to match the original class labels
                plot_images_by_class(imgs, labels=lbls, num_images=5, save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_example_train_data.pdf")
            
            # Prepare input data
            mock_tensor = torch.zeros_like(train_images)
            valid_mock_tensor = torch.zeros_like(valid_images)
            if classifier in ['ScatterNet', 'ScatterResNet', 'ScatterSqueezeNet', 'ScatterSqueezeNet2']:
                # Define cache paths (you can adjust these names as needed)
                train_cache_path = f"./.cache/train_scat_{galaxy_classes}_{fold}_{lambda_generate}_{dataset_portions[0]}_{FILTERED}_{TRAINONGENERATED}.npy"
                valid_cache_path = f"./.cache/valid_scat_{galaxy_classes}_{fold}_{lambda_generate}_{dataset_portions[0]}_{FILTERED}_{TRAINONGENERATED}.npy"
                
                if USE_MEMMAP:
                    # Use memmap-based caching (from script 1)
                    train_cache_file, train_full_shape = cache_scattering_memmap(train_images, scattering, train_cache_path, batch_size=128, device="cpu")
                    valid_cache_file, valid_full_shape = cache_scattering_memmap(valid_images, scattering, valid_cache_path, batch_size=128, device="cpu")
                    # Create dataset using the memmap cache
                    train_dataset = CachedScatterDataset(train_images, train_labels, train_cache_file, train_full_shape)
                    valid_dataset = CachedScatterDataset(valid_images, valid_labels, valid_cache_file, valid_full_shape)
                    scatdim = train_full_shape[1:]  # e.g., (C, H, W)
                    
                else:
                    # fold T into C on both real & scattering inputs
                    train_images = fold_T_axis(train_images) # Merges the image version into the channel dimension
                    valid_images = fold_T_axis(valid_images)
                    mock_train = torch.zeros_like(train_images)
                    mock_valid = torch.zeros_like(valid_images)


                    train_cache = f"./.cache/train_scat_{galaxy_classes}_{fold}_{lambda_generate}_{dataset_portions[0]}_{FILTERED}.pt"
                    valid_cache = f"./.cache/valid_scat_{galaxy_classes}_{fold}_{lambda_generate}_{dataset_portions[0]}_{FILTERED}.pt"
                    train_scat_coeffs = compute_scattering_coeffs(train_images, scattering, batch_size=128, device="cpu")
                    valid_scat_coeffs = compute_scattering_coeffs(valid_images, scattering, batch_size=128, device="cpu")

                    if train_scat_coeffs.dim() == 5:
                        # [B, C_in, C_scat, H, W] → [B, C_in*C_scat, H, W]
                        print("Shape of train_scat_coeffs before flattening: ", train_scat_coeffs.shape)
                        train_scat_coeffs = train_scat_coeffs.flatten(start_dim=1, end_dim=2)
                        valid_scat_coeffs = valid_scat_coeffs.flatten(start_dim=1, end_dim=2)
                        print("Shape of train_scat_coeffs after flattening: ", train_scat_coeffs.shape)

                    all_scat = torch.cat([train_scat_coeffs, valid_scat_coeffs], dim=0)
                    if NORMALISESCS or NORMALISESCSTOPM:
                        if NORMALISESCSTOPM:
                            all_scat = normalise_images(all_scat, -1, 1)
                        else:
                            all_scat = normalise_images(all_scat, 0, 1)
                    train_scat_coeffs, valid_scat_coeffs = all_scat[:len(train_scat_coeffs)], all_scat[len(train_scat_coeffs):]

                    scatdim = train_scat_coeffs.shape[1:]   # tuple(C, H, W)

                    if classifier in ['ScatterNet', 'ScatterResNet']:
                        train_dataset = TensorDataset(mock_train, train_scat_coeffs, train_labels)
                        valid_dataset = TensorDataset(mock_valid, valid_scat_coeffs, valid_labels)
                    else: # if classifier in ['ScatterSqueezeNet', 'ScatterSqueezeNet2']:
                        if EXTRAVARS:
                            print(train_images.shape)
                            print(train_scat_coeffs.shape)
                            print(train_data.shape)
                            print(train_labels.shape)
                            train_dataset = TensorDataset(train_images, train_scat_coeffs, train_data, train_labels)
                            valid_dataset = TensorDataset(valid_images, valid_scat_coeffs, valid_data, valid_labels)
                        else:
                            train_dataset = TensorDataset(train_images, train_scat_coeffs, train_labels)
                            valid_dataset = TensorDataset(valid_images, valid_scat_coeffs, valid_labels)
            else:
                if train_images.dim() == 5:
                    train_images = fold_T_axis(train_images)   # [B,T,1,H,W] -> [B,T,H,W]
                    valid_images = fold_T_axis(valid_images)
                    # test_images was folded earlier
                for x,name in [(train_images,"train"), (valid_images,"valid")]:
                    assert x.dim() == 4, f"{name}_images should be [B,C,H,W], got {tuple(x.shape)}"
                mock_train = torch.zeros_like(train_images)
                mock_valid = torch.zeros_like(valid_images)
                train_dataset = TensorDataset(train_images, mock_train, train_labels)
                valid_dataset = TensorDataset(valid_images, mock_valid, valid_labels)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=False)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=False)

            if SHOWIMGS and lambda_generate not in [0, 8]: 
                if classifier in ['TinyCNN', 'SCNN', 'CNNSqueezeNet', 'Rustige', 'ScatterSqueezeNet', 'ScatterSqueezeNet2', 'Binary']:
                    #save_images_tensorboard(generated_images[:36], save_path=f"./classifier/{gen_model_name}_{galaxy_classes}_generated.png", nrow=6)
                    plot_histograms(pristine_train_images, valid_images, title1="Train images", title2="Valid images", imgs3=generated_images, imgs4=test_images, title3='Generated images', title4='Test images', save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_histograms.png")

            
            ###############################################
            ############# DEFINE MODEL ####################
            ###############################################
            
            if classifier == "Rustige":
                models = {"RustigeClassifier": {"model": RustigeClassifier(n_output_nodes=num_classes).to(DEVICE)}} 
            elif classifier == "SCNN":
                models = {"SCNN": {"model": SCNN(input_shape=tuple(valid_images.shape[1:]), num_classes=num_classes).to(DEVICE)}}
            elif classifier == "CNNSqueezeNet":
                models = {"CNNSqueezeNet": {"model": CNNSqueezeNet(input_shape=tuple(valid_images.shape[1:]), num_classes=num_classes).to(DEVICE)}}
            elif classifier == "DualCNNSqueezeNet":
                models = {"DualCNNSqueezeNet": {"model": DualCNNSqueezeNet(input_shape=tuple(valid_images.shape[1:]), num_classes=num_classes).to(DEVICE)}}
            elif classifier == "TinyCNN":
                models = {"TinyCNN": {"model": TinyCNN(input_shape=tuple(valid_images.shape[1:]), num_classes=num_classes).to(DEVICE)}} 
            elif classifier == 'CloudNet':
                cloud_model = CloudNet()  
                cloud_model.load_weights(os.path.join(sys.path[0],
                    'Cloud-Net_trained_on_38-Cloud_training_patches.h5'))
                cloud_model.trainable = True   # now you can fine-tune it on your radio maps
            elif classifier == "DANN":
                models = {"DANN": {"model": DANNClassifier(input_shape=tuple(valid_images.shape[1:]), num_classes=num_classes).to(DEVICE)}}
            elif classifier == "ScatterNet":
                models = {"ScatterNet": {"model": MLPClassifier(input_dim=int(np.prod(scatdim)), num_classes=num_classes).to(DEVICE)}}
            elif classifier == "ScatterResNet":
                models = {"ScatterResNet": {"model": ScatterResNet(scat_shape=scatdim, num_classes=num_classes).to(DEVICE)}}
            elif classifier == "ScatterSqueezeNet":
                if EXTRAVARS:
                    models = {"ScatterSqueezeNet": {"model": MetaWrapper(ScatterSqueezeNet(img_shape=tuple(valid_images.shape[1:]), scat_shape=scatdim, num_classes=num_classes), meta_dim=test_meta.shape[1]).to(DEVICE)}}
                else:
                    models = {"ScatterSqueezeNet": {"model": ScatterSqueezeNet(img_shape=tuple(valid_images.shape[1:]), scat_shape=scatdim, num_classes=num_classes).to(DEVICE)}}
            elif classifier == "ScatterSqueezeNet2":
                models = {"ScatterSqueezeNet2": {"model": ScatterSqueezeNet2(img_shape=tuple(valid_images.shape[1:]), scat_shape=scatdim, num_classes=num_classes).to(DEVICE)}}
            elif classifier == 'Binary':
                models = {"BinaryClassifier": {"model": BinaryClassifier(input_shape=tuple(valid_images.shape[1:])).to(DEVICE)}}
            else:
                raise ValueError("Model not found. Please select one of 'scatterMLP', 'smallSTMLP', or 'normalCNN'.")

            classifier_name, model_details = next(iter(models.items()))


            ###############################################
            ############### TRAINING LOOP #################
            ###############################################
            
            if weights is not None:
                print(f"Using class weights: {weights}")
                criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
            else:
                print("No class weighting")
                if len(galaxy_classes) == 2:
                    criterion = nn.BCEWithLogitsLoss()
                else:
                    criterion = nn.CrossEntropyLoss()
                

            optimizer = AdamW(models[classifier_name]["model"].parameters(), lr=lr, weight_decay=reg)
            if SCHEDULER:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=10*lr, 
                                        steps_per_epoch=len(train_loader), epochs=num_epochs)

            for classifier_name, model_details in models.items():
                model = model_details["model"].to(DEVICE)

                for subset_size in dataset_sizes[fold]:
                    if subset_size <= 0:
                        print(f"Skipping invalid subset size: {subset_size}")
                        continue
                    if subset_size not in training_times:
                        training_times[subset_size] = {}
                    if fold not in training_times[subset_size]:
                        training_times[subset_size][fold] = []

                    for experiment in range(num_experiments):
                        initialize_history(history, gen_model_name, subset_size, fold, experiment, lr, reg, lambda_generate)
                        initialize_metrics(metrics, gen_model_name, subset_size, fold, experiment, lr, reg, lambda_generate, crop_size, downsample_size, ver_key)
                        initialize_labels(all_true_labels, all_pred_labels, gen_model_name, subset_size, fold, experiment, lr, reg, lambda_generate)

                        start_time = time.time()
                        model.apply(reset_weights)

                        subset_indices = torch.randperm(len(train_dataset))[:subset_size].tolist() # Randomly select indices to include generated samples
                        subset_train_dataset = Subset(train_dataset, subset_indices)
                        eff_bs = max(2, min(batch_size, len(subset_train_dataset)))
                        subset_train_loader = DataLoader(subset_train_dataset, batch_size=eff_bs, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=True)

                        early_stopping = EarlyStopping(patience=patience, verbose=False) if ES else None

                        for epoch in tqdm(range(num_epochs), desc=f'Training {classifier}_{galaxy_classes}_{gen_model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}_{classifier}_lo{percentile_lo}_hi{percentile_hi}_cs{crop_size}'):
                            model.train()
                            total_loss = 0
                            total_images = 0

                            for images, scat, _rest in subset_train_loader:
                                if EXTRAVARS:
                                    meta, labels = _rest
                                    meta = meta.to(DEVICE)  # Send metadata to device
                                else:
                                    labels = _rest
                                images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE) # Send to device
                                optimizer.zero_grad()
                                if classifier == "DANN":
                                    # 1) forward pass: two heads
                                    class_logits, domain_logits = model(images, alpha=1.0)

                                    # 2) classification loss
                                    class_loss = criterion(class_logits, labels)

                                    # 3) domain loss (0=real, 1=fake)
                                    B = labels.size(0)
                                    domain_labels = torch.zeros(B, dtype=torch.long, device=DEVICE)
                                    # if you interleave generated samples in the same loader you need a flag per-sample;
                                    # for now this will assume your loader is real‐only, so all zeros
                                    domain_loss = nn.CrossEntropyLoss()(domain_logits, domain_labels)

                                    # 4) total loss
                                    loss = class_loss + 0.5 * domain_loss

                                    loss.backward()
                                    optimizer.step()
                                    if SCHEDULER:
                                        scheduler.step()

                                    total_loss += float(loss.item() * images.size(0))
                                    total_images += float(images.size(0))
                                else:
                                    if classifier in ["ScatterNet", "ScatterResNet"]:
                                        logits = model(scat)
                                    elif classifier in ["ScatterSqueezeNet", "ScatterSqueezeNet2"]:
                                        logits = model(images, scat)
                                    else:
                                        logits = model(images)

                                    # Collapse spatial maps to [B, C] if the model returns [B, C, H, W]
                                    if logits.ndim == 4:
                                        print(f"Collapsing logits from shape {logits.shape} to [B, C]")
                                        logits = torch.nn.functional.adaptive_avg_pool2d(logits, (1, 1)).squeeze(-1).squeeze(-1)
                                    elif logits.ndim == 3:  # rare: [B, C, H]
                                        print(f"Collapsing logits from shape {logits.shape} to [B, C]")
                                        logits = logits.mean(dim=-1)

                                    # Keep binary 2-logit shape for CE
                                    if logits.ndim == 1:
                                        print(f"Expanding logits from shape {logits.shape} to [B, C]")
                                        logits = logits.unsqueeze(1)
                                    if logits.shape[1] == 1 and num_classes == 2:
                                        print(f"Expanding logits from shape {logits.shape} to [B, 2]")
                                        logits = torch.cat([-logits, logits], dim=1)


                                    labels = labels.long()
                                    loss = criterion(logits, labels)

                                    loss.backward()
                                    optimizer.step()
                                    total_loss += float(loss.item() * images.size(0))
                                    total_images += float(images.size(0))

                            average_loss = total_loss / total_images
                            loss_key = f"{gen_model_name}_loss_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                            history[gen_model_name][loss_key].append(average_loss)

                            model.eval()
                            val_total_loss = 0
                            val_total_images = 0

                            with torch.no_grad(): # Validate on validation data
                                for i, (images, scat, _rest) in enumerate(valid_loader):
                                    if images is None or len(images) == 0:
                                        print(f"Empty batch at index {i}. Skipping...")
                                        continue
                                    if EXTRAVARS:
                                        meta, labels = _rest
                                        meta = meta.to(DEVICE)  # Send metadata to device
                                    else:
                                        labels = _rest
                                    images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)
                                    if classifier == "DANN":
                                        logits, _ = model(images, alpha=1.0)
                                    else:
                                        if classifier in ["ScatterNet", "ScatterResNet"]:
                                            logits = model(scat)
                                        elif classifier in ["ScatterSqueezeNet", "ScatterSqueezeNet2"]:
                                            logits = model(images, scat)
                                        else:
                                            logits = model(images)

                                    # Collapse spatial maps to [B, C] if the model returns [B, C, H, W]
                                    if logits.ndim == 4:
                                        logits = torch.nn.functional.adaptive_avg_pool2d(logits, (1, 1)).squeeze(-1).squeeze(-1)
                                    elif logits.ndim == 3:  # rare: [B, C, H]
                                        logits = logits.mean(dim=-1)

                                    # Keep binary 2-logit shape for CE
                                    if logits.ndim == 1:
                                        logits = logits.unsqueeze(1)
                                    if logits.shape[1] == 1 and num_classes == 2:
                                        logits = torch.cat([-logits, logits], dim=1)

                                    labels = labels.long()
                                    loss = criterion(logits, labels)
                                        
                                    # inside the training loop, just before loss = criterion(outputs, labels)
                                    assert labels.dtype == torch.long, f"labels dtype {labels.dtype} must be long"
                                    mn, mx = int(labels.min()), int(labels.max())
                                    assert 0 <= mn and mx < num_classes, f"label range [{mn},{mx}] not in [0,{num_classes-1}]"

                                    val_total_loss += float(loss.item() * images.size(0))
                                    val_total_images += float(images.size(0))

                            val_average_loss = val_total_loss / val_total_images if val_total_images > 0 else float('inf')
                            val_loss_key = f"{gen_model_name}_val_loss_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                            history[gen_model_name][val_loss_key].append(val_average_loss)
                            
                            if ES:
                                early_stopping(val_average_loss, model, f'./classifier/trained_models/{gen_model_name}_best_model.pth')
                                if early_stopping.early_stop:
                                    break

                        model.eval()
                        with torch.no_grad(): # Evaluate on test data
                            key = f"{gen_model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}"
                            all_pred_probs[key] = []
                            all_pred_labels[key] = []
                            all_true_labels[key] = []
                            mis_images = []
                            mis_trues  = []
                            mis_preds  = []
                            
                            for images, scat, _rest in test_loader:
                                if EXTRAVARS:
                                    meta, labels = _rest
                                    meta = meta.to(DEVICE)  # Send metadata to device
                                else:
                                    labels = _rest
                                images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)
                                if classifier == "DANN":
                                    logits, _ = model(images, alpha=1.0)
                                else:
                                    if classifier in ["ScatterNet", "ScatterResNet"]:
                                        logits = model(scat)
                                    elif classifier in ["ScatterSqueezeNet", "ScatterSqueezeNet2"]:
                                        logits = model(images, scat)
                                    else:
                                        logits = model(images)

                                # Collapse spatial maps to [B, C] if the model returns [B, C, H, W]
                                if logits.ndim == 4:
                                    logits = torch.nn.functional.adaptive_avg_pool2d(logits, (1, 1)).squeeze(-1).squeeze(-1)
                                elif logits.ndim == 3:  # rare: [B, C, H]
                                    logits = logits.mean(dim=-1)

                                # Keep binary 2-logit shape for CE
                                if logits.ndim == 1:
                                    logits = logits.unsqueeze(1)
                                if logits.shape[1] == 1 and num_classes == 2:
                                    logits = torch.cat([-logits, logits], dim=1)


                                pred_probs = torch.softmax(logits, dim=1).cpu().numpy()

                                true_labels = labels.cpu().numpy()
                                #true_labels = torch.argmax(labels, dim=1).cpu().numpy()
                                pred_labels = np.argmax(pred_probs, axis=1)
                                all_pred_probs[key].extend(pred_probs)
                                all_pred_labels[key].extend(pred_labels)
                                all_true_labels[key].extend(true_labels)

                                
                                if SHOWIMGS and experiment == num_experiments - 1:
                                    mask = pred_labels != true_labels
                                    mis_images.append(images.cpu()[mask])
                                    mis_trues .append(true_labels[mask])
                                    mis_preds .append(pred_labels[mask])
                                    
                                    if galaxy_classes == [52, 53]:
                                        cm = confusion_matrix(all_true_labels[key], all_pred_labels[key], labels=[0,1])
                                        plt.figure(figsize=(4,4))
                                        sns.heatmap(cm, annot=True, fmt='d',
                                                    xticklabels=['RH (52)','RR (53)'],
                                                    yticklabels=['RH (52)','RR (53)'])
                                        plt.xlabel('Predicted')
                                        plt.ylabel('True')
                                        plt.title(f'Confusion Matrix — {classifier_name}')
                                        plt.savefig(f"./{galaxy_classes}_{classifier_name}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_confusion_matrix.png", dpi=150)
                                        plt.close()

                            accuracy = accuracy_score(all_true_labels[key], all_pred_labels[key])
                            if len(galaxy_classes) == 2:
                                # For binary classification, pos_label=1 corresponds to the second class in sorted order
                                positive_class = 1 if 1 in np.unique(all_true_labels[key]) else 0
                                precision = precision_score(all_true_labels[key], all_pred_labels[key], pos_label=positive_class, average='binary', zero_division=0)
                                recall = recall_score(all_true_labels[key], all_pred_labels[key], pos_label=positive_class, average='binary', zero_division=0)
                                f1 = f1_score(all_true_labels[key], all_pred_labels[key], pos_label=positive_class, average='binary', zero_division=0)
                            else:
                                # For multi-class, use macro averaging
                                precision = precision_score(all_true_labels[key], all_pred_labels[key], average='macro', zero_division=0)
                                recall = recall_score(all_true_labels[key], all_pred_labels[key], average='macro', zero_division=0)
                                f1 = f1_score(all_true_labels[key], all_pred_labels[key], average='macro', zero_division=0)

                            update_metrics(metrics, gen_model_name, subset_size, fold, experiment, lr, reg, accuracy, precision, recall, f1, lambda_generate, crop_size, downsample_size, ver_key)

                            
                            # Print accuracy and other metrics
                            print(f"Fold {fold}, Experiment {experiment}, Subset Size {subset_size}, Classifier {classifier_name}, "
                                f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                                f"Recall: {recall:.4f}, F1 Score: {f1:.4f}, ")
        

                            if SHOWIMGS and mis_images and experiment == num_experiments - 1:
                                mis_images = torch.cat(mis_images, dim=0)[:36]
                                mis_trues  = np.concatenate(mis_trues)[:36]
                                mis_preds  = np.concatenate(mis_preds)[:36]

                                import matplotlib.pyplot as plt
                                fig, axes = plt.subplots(6, 6, figsize=(12, 12))
                                axes = axes.flatten()

                                for i, ax in enumerate(axes[:len(mis_images)]):
                                    img_tensor = mis_images[i]                           # shape is either (1,128,128) or (2,128,128)
                                    # pick the first channel if there are two, else drop the singleton channel
                                    img = img_tensor[0] if img_tensor.shape[0] > 1 else img_tensor.squeeze(0)
                                    ax.imshow(img.numpy(), cmap='viridis')
                                    ax.set_title(f"T={mis_trues[i]}, P={mis_preds[i]}")
                                    ax.axis('off')

                                for ax in axes[len(mis_images):]:
                                    ax.axis('off')

                                out_path = f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_misclassified.png"
                                fig.savefig(out_path, dpi=150, bbox_inches='tight')
                                plt.close(fig)


                    base = (
                        f"{gen_model_name}"
                        f"_ss{subset_size}"
                        f"_f{fold}"
                        f"_lr{lr}"
                        f"_reg{reg}"
                        f"_lam{lambda_generate}"
                        f"_cs{crop_size[0]}x{crop_size[1]}"
                        f"_ds{downsample_size[0]}x{downsample_size[1]}"
                        f"_ver{ver_key}"
                    )
                    mean_acc = float(np.mean(metrics[f"{base}_accuracy"])) if metrics[f"{base}_accuracy"] else float('nan')
                    mean_prec = float(np.mean(metrics[f"{base}_precision"])) if metrics[f"{base}_precision"] else float('nan')
                    mean_rec = float(np.mean(metrics[f"{base}_recall"])) if metrics[f"{base}_recall"] else float('nan')
                    mean_f1 = float(np.mean(metrics[f"{base}_f1_score"])) if metrics[f"{base}_f1_score"] else float('nan')
                    print(f"AVERAGE over {num_experiments} experiments — Accuracy: {mean_acc:.4f}, Precision: {mean_prec:.4f}, Recall: {mean_rec:.4f}, F1 Score: {mean_f1:.4f}")
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    #training_times[subset_size][fold].append(elapsed_time)
                    training_times.setdefault(fold, {}).setdefault(subset_size, []).append(elapsed_time)

                    if fold == folds[-1] and experiment == num_experiments - 1:
                        with open(log_path, 'w') as file:
                            file.write(f"Time taken to train the model on fold {fold}: {elapsed_time:.2f} seconds \n")
                
                generated_features = []
                with torch.no_grad():
                    for images, scat, _ in test_loader:
                        if EXTRAVARS:
                            meta, labels = _rest
                            meta = meta.to(DEVICE)  # Send metadata to device
                        else:
                            labels = _rest
                        images, scat, labels = images.to(DEVICE), scat.to(DEVICE), labels.to(DEVICE)
                        if classifier == "DANN":
                            class_logits, _ = model(images, alpha=1.0)
                            outputs = class_logits.cpu().detach().numpy()
                        elif classifier in ["ScatterNet", "ScatterResNet"]:
                            outputs = model(scat).cpu().detach().numpy()
                        elif classifier in ["ScatterSqueezeNet", "ScatterSqueezeNet2"]:
                            outputs = model(images, scat).cpu().detach().numpy()
                        else:
                            outputs = model(images).cpu().detach().numpy()

                        generated_features.append(outputs)

                generated_features = np.concatenate(generated_features, axis=0)
                cluster_error, cluster_distance, cluster_std_dev = cluster_metrics(generated_features, n_clusters=num_classes)
                with open(log_path, 'w') as file:
                    file.write(f"Results for fold {fold}, Classifier {classifier_name}, lr={lr}, reg={reg}, lambda_generate={lambda_generate}, percentile_lo={percentile_lo}, percentile_hi={percentile_hi}, crop_size={crop_size}, downsample_size={downsample_size}, STRETCH={STRETCH}, FILTERED={FILTERED}, TRAINONGENERATED={TRAINONGENERATED} \n")
                    file.write(f"Cluster Error: {cluster_error} \n")
                    file.write(f"Cluster Distance: {cluster_distance} \n")
                    file.write(f"Cluster Standard Deviation: {cluster_std_dev} \n")

                model_save_path = f'./classifier/trained_models/{gen_model_name}_model.pth'
                torch.save(model.state_dict(), model_save_path)
        import pickle
        from collections import defaultdict

        def dictify(x):
            """
            Recursively turn any defaultdict into a regular dict.
            Leave other types alone.
            """
            if isinstance(x, defaultdict):
                return {k: dictify(v) for k, v in x.items()}
            elif isinstance(x, dict):
                return {k: dictify(v) for k, v in x.items()}
            else:
                return x

        # … after all your training loops, right before saving:
        _clean = {
            "models": models,
            "history": dictify(history),
            "metrics": dictify(metrics),
            "metric_colors": metric_colors,
            "all_true_labels": dictify(all_true_labels),
            "all_pred_labels": dictify(all_pred_labels),
            "training_times": dictify(training_times),
            "all_pred_probs": dictify(all_pred_probs),
        }
        
        summary_path = f'./classifier/trained_models/{classifier}_{gen_model_names[0]}_summary_metrics.pkl'
        with open(summary_path, "wb") as f:
            pickle.dump(_clean, f)
        print(f"Summary metrics written to {summary_path}") 


