from utils.data_loader import load_galaxies, load_halos_and_relics, get_classes,  get_synthetic, augment_images, apply_formatting
from utils.classifiers import (
    RustigeClassifier, TinyCNN, MLPClassifier, SCNN, CNNSqueezeNet, ScatterResNet,
    DANNClassifier, BinaryClassifier, ScatterSqueezeNet, ScatterSqueezeNet2,
    DualCNNSqueezeNet, DualInputConvolutionalSqueezeNet, DISSN)
from utils.training_tools import EarlyStopping, reset_weights
from utils.calc_tools import cluster_metrics, normalise_images, check_tensor, fold_T_axis, compute_scattering_coeffs, custom_collate
from utils.plotting import plot_histograms, plot_images_by_class, plot_image_grid, plot_background_histogram
from pathlib import Path
from astropy.cosmology import Planck18 as COSMO
import astropy.units as u
import hashlib
import numpy as np
import torch, math, time, random
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from kymatio.torch import Scattering2D
from collections import defaultdict
from astropy.io import fits
from astropy.convolution import convolve_fft, Gaussian2DKernel
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter   # used in _load_fits_arrays_scaled()
from functools import lru_cache
from tqdm import tqdm
import itertools
from itertools import product
import seaborn as sns
import matplotlib 
import matplotlib.pyplot as plt
import sys, os, glob, re

######### CONTINUE FROM LINE 1983! 17.09.2025

FUDGE_GLOBAL = float(os.getenv("RT_FUDGE_SCALE", "1.00"))  # e.g. 1.05
os.environ.setdefault("RT_AUTO_FUDGE", "1")  # default is already "1"; this makes it explicit

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

print("Running ht3 with seed", SEED)


#############################################
################ CONFIGURATION ################
###############################################

galaxy_classes    = [50, 51]
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
                     "DualCNNSqueezeNet", # Dual mode Convolutional SqueezeNet
                     "DICSN",          # Dual-Input Convolutional SqueezeNet
                     "DISSN",          # Dual-Input Scatter SqueezeNet
                     "CloudNet",      # from SorourMo/Cloud-Net
                     "DANN",          # domain‐adversarial NN
                     "ScatterNet",
                     "ScatterSqueezeNet",
                     "ScatterSqueezeNet2",
                     "Binary",
                     "ScatterResNet"][-4]

# Define every value you want to try
param_grid = {
    'lr':            [1e-3],
    'reg':           [1e-3],
    'label_smoothing':[0.1],
    'J':             [2],
    'L':             [12],
    'order':         [2],
    'percentile_lo': [60],   
    'percentile_hi': [99], 
    'crop_size':     [(512,512)],
    'downsample_size':[(128,128)],
    'versions':       ['rt50', 'T50kpc']  # 'raw', 'T50kpc', ad hoc tapering: e.g. 'rt50'  strings in list → product() iterates them individually
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

_loader = load_halos_and_relics if galaxy_classes == [52, 53] else load_galaxies

# CSV with redshifts (slug,z)
CLUSTER_METADATA_CSV = f"{PSZ2_ROOT}/cluster_metadata.csv"

# --- optional RT knobs via env (keep default off to remain reproducible) ---
APPLY_UV_TAPER = os.getenv("RT_USE_UV_TAPER", "0") == "1"   # default off
UV_TAPER_FRAC  = float(os.getenv("RT_UV_TAPER_FRAC", "0.0"))  # e.g. 0.2 (20% of target FWHM)

# Align rt* to a fixed anchor when possible (keeps exact file parity)
ALIGN_RT_WITH_FIXED = True

if TRAINONGENERATED:
    lambda_values = [8]  # To identify and distinguish TRAINONGENERATED from other runs
    print("Using generated data for testing.")
    
if galaxy_classes == [52, 53]:
    # —— MULTI-LABEL SWITCH ——
    MULTILABEL = True     # predict presence of RH and/or RR (independent)
    LABEL_INDEX = {"RH": 0, "RR": 1}   # RH=52, RR=53 below
    THRESHOLD = 0.5       # sigmoid threshold at inference


########################################################################
##################### HELPER FUNCTIONS #################################
########################################################################

def debug_split_parity(label_tensor, filename_list, name="SPLIT"):
    import numpy as np, collections
    if filename_list is None:
        print(f"[{name}] no filenames")
        return
    idxs = (label_tensor if label_tensor.ndim==1 else as_index_labels(label_tensor)).cpu().numpy()
    cnt  = collections.Counter(idxs.tolist())
    print(f"[{name}] N={len(filename_list)}  class_counts={dict(cnt)}  unique_files={len(set(map(str,filename_list)))}")


def debug_background_sigma(slug_list, inner=64):
    import numpy as np
    from statistics import median
    vals_rt, vals_T = [], []
    Hout, Wout = int(downsample_size[-2]), int(downsample_size[-1])

    for slug in slug_list:
        out = _load_fits_arrays_scaled("rt50", slug, crop_ch=1, out_hw=(Hout, Wout))
        if out is None: continue
        _, rt_cut, raw_hdr, pix_eff_as, hdr_fft = out
        if rt_cut is None: continue
        # rt_cut can be [1,H,W] or [H,W]; keep it 2-D
        rt = np.asarray(rt_cut, dtype=np.float32)
        if rt.ndim == 3 and rt.shape[0] == 1:
            rt = rt[0]
        elif rt.ndim != 2:
            raise ValueError(f"Unexpected rt_cut shape {rt.shape}; expected [H,W] or [1,H,W]")


        base_dir = _find_base_dir(slug); base = os.path.basename(base_dir)
        _, _, _, t50_path = _fits_path_triplet(base_dir, base)[:4]
        if (t50_path is None) or (not os.path.exists(t50_path)):
            continue
        T  = np.squeeze(fits.getdata(t50_path)).astype(float)
        T  = reproject_like(T, fits.getheader(t50_path), hdr_fft)
        T  = np.nan_to_num(T)

        # match stretch & asinh
        t_rt = torch.from_numpy(rt).float().unsqueeze(0)
        t_T  = torch.from_numpy(T).float().unsqueeze(0)
        if STRETCH:
            t_rt = _per_image_percentile_stretch(t_rt.squeeze(0), percentile_lo, percentile_hi, USE_ASINH=True).unsqueeze(0)
            t_T  = _per_image_percentile_stretch(t_T.squeeze(0),  percentile_lo, percentile_hi, USE_ASINH=True).unsqueeze(0)
        RT, TT = t_rt.squeeze(0), t_T.squeeze(0)

        mask = _background_ring_mask(Hout, Wout, inner=inner)
        sig_rt = _robust_sigma(RT[mask]).item()
        sig_T  = _robust_sigma(TT[mask]).item()
        vals_rt.append(sig_rt); vals_T.append(sig_T)

    if vals_rt and vals_T:
        print(f"Background σ (median over {len(vals_rt)}):  rt50={median(vals_rt):.4f}  T50kpc={median(vals_T):.4f}")
    else:
        print("No valid pairs to measure background σ.")


def debug_uv_power(slug, nbins=48, outdir="./classifier/debug_uv"):
    import os, numpy as np, matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    Hout, Wout = int(downsample_size[-2]), int(downsample_size[-1])

    # Load rt50 and T50kpc on the same target header
    raw_cut, rt_cut, raw_hdr, pix_eff_as, hdr_fft = _load_fits_arrays_scaled("rt50", slug, crop_ch=1, out_hw=(Hout, Wout))
    if rt_cut is None:
        print(f"[uv] {slug}: no rt50"); return
    rt = rt_cut[0]

    # Try to load the on-disk T50kpc and reproject to the same FFT header for apples-to-apples
    base_dir = _find_base_dir(slug); base = os.path.basename(base_dir)
    _, _, _, t50_path = _fits_path_triplet(base_dir, base)[:4]
    if (t50_path is None) or (not os.path.exists(t50_path)):
        print(f"[uv] {slug}: no T50kpc"); return
    T = np.squeeze(fits.getdata(t50_path)).astype(float)
    T = reproject_like(T, fits.getheader(t50_path), hdr_fft)  # align to same grid/FOV
    T = np.nan_to_num(T)

    # stretch both the same way for display (FFT uses hdrs internally)
    t_rt = torch.from_numpy(rt).float().unsqueeze(0)
    t_T  = torch.from_numpy(T).float().unsqueeze(0)
    if STRETCH:
        t_rt = _per_image_percentile_stretch(t_rt.squeeze(0), percentile_lo, percentile_hi, USE_ASINH=True).unsqueeze(0)
        t_T  = _per_image_percentile_stretch(t_T.squeeze(0),  percentile_lo, percentile_hi, USE_ASINH=True).unsqueeze(0)
    
    RT = np.asarray(t_rt.squeeze(0).cpu().numpy())
    TT = np.asarray(t_T.squeeze(0).cpu().numpy())
    if RT.ndim != 2 or TT.ndim != 2:
        raise ValueError(f"Expected 2-D RT/TT, got RT={RT.shape}, TT={TT.shape}")

    # Belt & suspenders: make sure they’re 2-D
    if RT.ndim == 3 and RT.shape[0] == 1: RT = RT[0]
    if TT.ndim == 3 and TT.shape[0] == 1: TT = TT[0]
    assert RT.ndim == 2 and TT.ndim == 2, (RT.shape, TT.shape)
        
    U,V,_,A_rt = image_to_vis(RT, hdr_fft, beam_hdr=hdr_fft)
    _,_,_,A_T  = image_to_vis(TT, hdr_fft, beam_hdr=hdr_fft)
    r, m_rt = _radial_bin(U,V,A_rt, nbins=nbins, stat='median')
    _,  m_T = _radial_bin(U,V,A_T,  nbins=nbins, stat='median')

    fig, ax = plt.subplots(1,1,figsize=(5,4))
    ax.plot(r, m_rt, label="rt50 runtime")
    ax.plot(r, m_T,  label="T50kpc fixed", linestyle='--')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel("|uv| (λ)"); ax.set_ylabel("median |F|")
    ax.set_title(slug); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{slug}_uv_radial.png"), dpi=160)
    plt.close(fig)
    print(f"[uv] wrote {slug}_uv_radial.png")


def debug_compare_rt50_vs_T50(names, outdir="./classifier/debug_rt50_vs_T50", nmax=12):
    """
    For each source name (slug), show RAW (formatted), rt50 (runtime), T50kpc (fixed), and (rt50 - T50kpc).
    Also print quick stats & percentiles to console.
    """
    import os, numpy as np, matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)

    def _fmt_raw(name):
        try:
            # your helper, returns [1,H,W] after crop/resize+stretch+asinh
            return _format_raw_for_display(name).squeeze(0).cpu().numpy()
        except Exception as e:
            print(f"[raw] {name}: {e}")
            return None

    def _rt50(name, out_hw):
        try:
            raw_cut, rt_cut, raw_hdr, pix_eff_as, hdr_fft = _load_fits_arrays_scaled("rt50", name, crop_ch=1, out_hw=out_hw)
            if rt_cut is None:
                return None
            x = np.asarray(rt_cut, dtype=np.float32)
            return x[0] if x.ndim == 3 else x
        except Exception as e:
            print(f"[rt50] {name}: {e}")
            return None

    def _t50(name, out_hw):
        # Load on-disk T50kpc and push through same crop/resize+stretch path to match display
        try:
            base_dir = _find_base_dir(name); base = os.path.basename(base_dir)
            _, _, _, t50_path = _fits_path_triplet(base_dir, base)[:4]
            if (t50_path is None) or (not os.path.exists(t50_path)):
                return None
            arr = np.squeeze(fits.getdata(t50_path)).astype(float)
            t = torch.from_numpy(np.nan_to_num(arr)).float().unsqueeze(0)  # [1,H,W]
            img = apply_formatting(t,
                    crop_size=(1, crop_size[-2], crop_size[-1]),
                    downsample_size=(1, out_hw[0], out_hw[1])
                  ).squeeze(0)  # [1,H',W']
            if STRETCH:
                img = _per_image_percentile_stretch(img.squeeze(0), percentile_lo, percentile_hi, USE_ASINH=True).unsqueeze(0)
            return img.squeeze(0).cpu().numpy()
        except Exception as e:
            print(f"[T50kpc] {name}: {e}")
            return None

    Hout, Wout = int(downsample_size[-2]), int(downsample_size[-1])
    shown = 0
    for slug in names:
        if shown >= nmax: break
        raw_img = _fmt_raw(slug)
        rt_img  = _rt50(slug, (Hout, Wout))
        t50_img = _t50(slug, (Hout, Wout))

        if (rt_img is None) or (t50_img is None):
            continue

        diff = rt_img - t50_img
        print(f"\n[{slug}] stats:")
        for tag, arr in [("RAW", raw_img), ("rt50", rt_img), ("T50kpc", t50_img), ("rt50-T50", diff)]:
            if arr is None: 
                print(f"  {tag}: missing")
                continue
            p = np.percentile(arr, [0, 1, 25, 50, 75, 99, 100])
            print(f"  {tag}: min={arr.min():.4f} max={arr.max():.4f} med={np.median(arr):.4f} p1/99={p[1]:.4f}/{p[-2]:.4f}")

        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for ax, im, title in zip(axes, [raw_img, rt_img, t50_img, diff], ["RAW*", "rt50 (runtime)", "T50kpc (fixed)", "rt50 - T50"]):
            if im is None:
                ax.text(0.5, 0.5, "missing", ha='center', va='center'); ax.axis('off'); continue
            ax.imshow(im, origin='lower', cmap='viridis'); ax.set_title(title); ax.axis('off')
        fig.suptitle(slug, y=0.98)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{slug}_raw_rt50_t50_diff.png"), dpi=160)
        plt.close(fig)
        shown += 1

    print(f"[debug] wrote {shown} comparisons to {outdir}")


def get_z(name, hdr_primary):
    """Return a usable redshift for this source (CSV → header → siblings)."""
    import re, glob, os
    from astropy.io import fits

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
        print(f"[z] {name_base}: using z={z:.4f} from primary header")
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


# --- 3-column (RAW | rtXX | T50kpc) quicklook for TEST split ---
def _to_base_name(fn):
    return Path(str(fn)).stem.split('T', 1)[0]

def _img2np(img_t):
    x = img_t.detach().cpu()
    if x.ndim == 4: x = x[0]                 # [T/C,1,H,W] → [1,H,W]
    if x.ndim == 3: x = x[0] if x.shape[0] > 1 else x.squeeze(0)
    return x.numpy()

def _t50_path_for(base):
    base_dir = _first(f"{PSZ2_ROOT}/fits/{base}*") or f"{PSZ2_ROOT}/fits/{base}"
    p = (_first(f"{base_dir}/{base}T50kpc*.fits")
         or _first(f"{PSZ2_ROOT}/classified/T50kpc/*/{base}.fits")
         or _first(f"{PSZ2_ROOT}/classified/T50kpcSUB/*/{base}.fits"))
    return p

def _format_T50_for_display(base):
    """Load fixed T50 FITS and format like the pipeline (crop/downsample + stretch + asinh)."""
    p = _t50_path_for(base)
    if not p: 
        return None
    arr = np.squeeze(fits.getdata(p)).astype(float)
    arr = np.nan_to_num(arr, copy=False)
    t = torch.from_numpy(arr).float().unsqueeze(0)  # [1,H,W]
    formatted = apply_formatting(
        t,
        crop_size=(1, crop_size[-2], crop_size[-1]),
        downsample_size=(1, downsample_size[-2], downsample_size[-1])
    ).squeeze(0)  # [1,h,w]
    if STRETCH:
        formatted = _per_image_percentile_stretch(
            formatted.squeeze(0), percentile_lo, percentile_hi
        ).unsqueeze(0)
    img = torch.asinh(10.0 * formatted) / math.asinh(10.0)
    return img.squeeze(0).numpy()           # 2D numpy

def plot_before_after_rt_3col(raw_imgs, raw_fns, rt_imgs, rt_fns, tag='rt50',
                              outdir='./classifier/debug_rt_before_after', per_page=24):
    os.makedirs(outdir, exist_ok=True)
    bmap = { _to_base_name(fn): i for i, fn in enumerate(raw_fns or []) }
    amap = { _to_base_name(fn): i for i, fn in enumerate(rt_fns  or []) }
    common = sorted(set(bmap) & set(amap))
    if not common:
        print("[rt-debug] no overlap between RAW and RT filename sets.")
        return

    for page in range(0, len(common), per_page):
        chunk = common[page:page+per_page]
        n = len(chunk)
        fig, axes = plt.subplots(n, 3, figsize=(9.2, 3.0*n))
        if n == 1: axes = np.array([axes])
        for r, name in enumerate(chunk):
            i_raw, i_rt = bmap[name], amap[name]
            im_raw = _img2np(raw_imgs[i_raw])
            im_rt  = _img2np(rt_imgs[i_rt])
            im_t50 = _format_T50_for_display(name)

            axL, axM, axR = axes[r, 0], axes[r, 1], axes[r, 2]
            axL.imshow(im_raw, origin='lower', cmap='viridis'); axL.set_title(f"{name} — RAW");  axL.axis('off')
            axM.imshow(im_rt,  origin='lower', cmap='viridis'); axM.set_title(f"{name} — {tag}"); axM.axis('off')
            if im_t50 is not None:
                axR.imshow(im_t50, origin='lower', cmap='viridis'); axR.set_title(f"{name} — T50kpc"); axR.axis('off')
            else:
                axR.axis('off'); axR.text(0.5, 0.5, 'no T50', ha='center', va='center', transform=axR.transAxes)

        fig.tight_layout()
        out = os.path.join(outdir, f"test_before_after_{tag}_with_T50_p{page//per_page+1:02d}.png")
        fig.savefig(out, dpi=160, bbox_inches='tight'); plt.close(fig)
        print(f"[rt-debug] wrote {out}")


_anchor_versions = ('t25kpc', 't50kpc', 't100kpc')  # used by _has_rt_support()


def _find_base_dir(name: str):
    """
    Return the directory on disk that contains the FITS files for a source.
    Accepts a slug like 'PSZ2G181.06+48' (preferred) or a path.
    Searches under PSZ2_ROOT/fits/.
    """
    base = _name_base_from_fn(str(name))
    # direct path passed in?
    if os.path.isdir(name):
        return name
    # exact directory match
    d0 = os.path.join(PSZ2_ROOT, "fits", base)
    if os.path.isdir(d0):
        return d0
    # first prefix match under PSZ2_ROOT/fits/
    hits = sorted(glob.glob(os.path.join(PSZ2_ROOT, "fits", base + "*")))
    for h in hits:
        if os.path.isdir(h):
            return h
    return None  # caller will handle and skip

def _find_raw_path(base_dir: str, base: str):
    """
    Inside base_dir, return the path to the *RAW* image (not a T*kpc product).
    Prefers '<base>.fits'; otherwise the first .fits that isn't a tapered/aux file.
    """
    # exact preferred file
    p = os.path.join(base_dir, f"{base}.fits")
    if os.path.exists(p):
        return p

    # otherwise: any .fits that isn't a T*kpc or obvious aux (weight/primary/chandra)
    cand = sorted(glob.glob(os.path.join(base_dir, f"{base}*.fits")))
    def _is_raw(fp: str) -> bool:
        b = os.path.basename(fp).upper()
        bad = ("T25KPC" in b) or ("T50KPC" in b) or ("T100KPC" in b) \
              or ("WEIGHT" in b) or ("PRIMARY" in b) or ("CHANDRA" in b)
        return not bad
    cand = [fp for fp in cand if _is_raw(fp)]
    return cand[0] if cand else None


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

def _fits_path_triplet(base_dir, real_base):
    raw_path  = f"{base_dir}/{real_base}.fits"
    t25_path  = _first(f"{base_dir}/{real_base}T25kpc*.fits")
    t50_path  = _first(f"{base_dir}/{real_base}T50kpc*.fits")
    t100_path = _first(f"{base_dir}/{real_base}T100kpc*.fits")
    return raw_path, t25_path, t50_path, t100_path

def _load_fits_arrays_scaled(_gen_version, name, crop_ch=1, out_hw=(128,128)):
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

    # Choose what to build based on _gen_version
    if 'rt25' in _gen_version:
        t_hdr = _hdr_or_synth(t25_path, 25)
    elif 'rt50' in _gen_version:
        t_hdr = _hdr_or_synth(t50_path, 50)
    elif 'rt100' in _gen_version:
        t_hdr = _hdr_or_synth(t100_path, 100)
    elif 'rt' in _gen_version:
        t_hdr = _hdr_or_synth(t100_path, 100)  # default to 100kpc if no number

    ch, Hc_raw, Wc_raw = crop_ch, crop_size[-2], crop_size[-1]
    outH, outW = out_hw


    # Convolve RAW on its native grid to the target beam
    # Anti-alias only when we actually shrink the image
    ds = int(round(crop_size[-1] / out_hw[1]))  # e.g. 512→128 ⇒ ds≈4
    if ds > 1:
        raw_arr_prefiltered = gaussian_filter(raw_arr, sigma=0.5*ds, mode='nearest')
    else:
        raw_arr_prefiltered = raw_arr

    # 2a) convolve RAW → target restoring beam (always)
    r2_native  = convolve_to_target(raw_arr_prefiltered, raw_hdr, t_hdr)

    # 3) ➜ Reproject **convolved RAW** onto the *tapered* grids
    rt_on_t  = reproject_like(r2_native,  raw_hdr, t_hdr)  if t_hdr  is not None else None

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

    rt_cut    = _fmt(rt_on_t,   t_hdr)
    raw_cut = _fmt(raw_arr, raw_hdr)

    #if rt_cut is not None: check_tensor(f"rt_kpc {name}",   torch.tensor(rt_cut))
    #if raw_cut is not None: check_tensor(f"raw {name}", torch.tensor(raw_cut))

    ds_factor = (int(round(Hc_raw)) / outH) # e.g. 512/128=4
    pix_eff_arcsec = pix_native * ds_factor
    hdr_fft = make_fft_header(raw_hdr, pix_eff_arcsec, (outH, outW))

    return (raw_cut, rt_cut, raw_hdr, pix_eff_arcsec, hdr_fft)

def _stretch_from_p(arr, p_lo, p_hi, clip=False): # stretch to [0,1] using per-image percentiles
    if arr is None:
        return None
    y = (arr - p_lo) / (p_hi - p_lo + 1e-6)
    return np.clip(y, 0, 1) if clip else y

# --- DEBUG: save 5 examples in RAW and RT form, titled by source name ---
def _format_raw_for_display(fn):
    """Load the RAW FITS for a source name and format it like the pipeline (crop/resize + stretch)."""
    base = _name_base_from_fn(fn)
    _, _, _, _, _, raw_path = _headers_for_name(base)
    raw_native = np.squeeze(fits.getdata(raw_path)).astype(float)
    raw_native = np.nan_to_num(raw_native, copy=False)

    t = torch.from_numpy(raw_native).float().unsqueeze(0)  # [1,H,W]
    formatted = apply_formatting(
        t,
        crop_size=(1, crop_size[-2], crop_size[-1]),
        downsample_size=(1, downsample_size[-2], downsample_size[-1])
    ).squeeze(0)  # [1,H',W']

    img = _per_image_percentile_stretch(
        formatted.squeeze(0), percentile_lo, percentile_hi, USE_ASINH=True
    ).unsqueeze(0) if STRETCH else formatted

    return img  # [1,H',W']

def save_raw_and_rt_examples(fns, want_rt='rt50', n=5,
                             outdir='./classifier/debug_rt_examples'):
    """Save n sources in RAW and RT form. Titles = source names."""
    Hout, Wout = downsample_size[-2], downsample_size[-1]
    os.makedirs(os.path.join(outdir, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(outdir, want_rt), exist_ok=True)

    picked = []
    for fn in map(str, fns or []):
        if len(picked) >= n:
            break
        base = _name_base_from_fn(fn)
        try:
            # RAW (formatted like the pipeline)
            raw_img = _format_raw_for_display(base)  # [1,H,W]

            # RT (synthesized from RAW headers)
            dummy = torch.zeros((1, 1, Hout, Wout))  # content ignored; apply_taper_to_tensor reads FITS itself
            rt_img, keep_mask, kept_fns, skipped = apply_taper_to_tensor(
                dummy, want_rt, filenames=[base],
                crop_size=crop_size,
                downsample_size=downsample_size,
                percentile_lo=percentile_lo,
                percentile_hi=percentile_hi,
                do_stretch=STRETCH,
                require_fixed_header=False,
            )
            if rt_img.shape[0] == 0:
                continue  # skip if this source can't produce the requested RT

            # Save RAW
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(raw_img.squeeze(0).numpy(), origin='lower', cmap='viridis')
            ax.set_title(base)   # title = source name only
            ax.axis('off')
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, 'raw', f"{base}_raw{_versions_to_load}.png"), dpi=150)
            plt.close(fig)

            # Save RT
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(rt_img.squeeze(0).squeeze(0).cpu().numpy(), origin='lower', cmap='viridis')
            ax.set_title(base)   # title = source name only
            ax.axis('off')
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, want_rt, f"{base}_{want_rt}{_versions_to_load}.png"), dpi=150)
            plt.close(fig)

            picked.append(base)
        except Exception as e:
            print(f"[debug-rt] skipping {base}: {e}")

    print(f"[debug-rt] wrote {len(picked)} examples under {outdir}")
    return picked


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
    B = np.asarray(img, dtype=float)

    # Ensure 2-D
    if B.ndim == 3 and B.shape[0] == 1:
        B = B[0]
    if B.ndim != 2:
        raise ValueError(f"image_to_vis expects a 2-D image, got shape {B.shape}")

    A = np.where(np.isfinite(B), B, 0.0)
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


def shuffle_with_filenames(images, labels, filenames=None):
    perm = torch.randperm(images.size(0))
    images, labels = images[perm], labels[perm]
    if filenames is not None:
        filenames = [filenames[i] for i in perm.tolist()]
    return images, labels, filenames

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
    # accept [B,1,H,W] or [B,T,1,H,W] (T=1)
    if imgs.dim() == 5:
        # drop the version axis and keep the single channel
        if imgs.size(2) != 1:
            raise AssertionError(f"Expecting single-channel planes; got imgs.size(2)={imgs.size(2)}.")
        imgs = imgs[:, 0]  # now [B,1,H,W]

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
            do_stretch=STRETCH,
            require_fixed_header=False,  # allow synthetic headers if needed
        )

        kept_sets[gv] = set(kept_fns)
        removed_by_version[gv] = int(len(skipped))

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
    
    return imgs_out, labels_kept, fns_kept, info

def _first(pattern: str):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None

def _to_strs(x): 
    return [str(y) for y in x] if isinstance(x, (list, tuple)) else list(map(str, x))

def _filter_to_keep_set(imgs, labels, fns, keep_set):
    """Keep only samples whose filename is in keep_set; returns (imgs, labels, fns)."""
    if fns is None:
        raise RuntimeError("Filenames are required to filter by anchor T*kpc. Set PRINTFILENAMES=True.")
    keep_idx = [i for i, fn in enumerate(_to_strs(fns)) if str(fn) in keep_set]
    if not keep_idx:  # produce empty tensors with correct shape/device
        return imgs[:0], labels[:0], []
    imgs = imgs[keep_idx]
    labels = labels[keep_idx]
    fns = [fns[i] for i in keep_idx]
    return imgs, labels, fns

def _pixscale_arcsec(hdr):
    if 'CDELT1' in hdr:  # deg/pix
        return abs(hdr['CDELT1']) * 3600.0
    cd11 = hdr.get('CD1_1'); cd12 = hdr.get('CD1_2', 0.0)
    if cd11 is not None:
        return float(np.hypot(cd11, cd12)) * 3600.0
    raise KeyError("No CDELT* or CD* keywords in FITS header")

def collapse_logits(logits, num_classes, multilabel):
    # [B,C,H,W] → [B,C]
    if logits.ndim == 4:
        logits = torch.nn.functional.adaptive_avg_pool2d(logits, (1, 1)).squeeze(-1).squeeze(-1)
    elif logits.ndim == 3:
        logits = logits.mean(dim=-1)
    # ensure [B,C]
    if logits.ndim == 1:
        logits = logits.unsqueeze(1)
    if not multilabel and logits.shape[1] == 1 and num_classes == 2:
        logits = torch.cat([-logits, logits], dim=1)
    return logits

def compute_classification_metrics(y_true, y_pred, multilabel, num_classes):
    acc = accuracy_score(y_true, y_pred)
    if multilabel:
        avg = 'macro'
        return acc, precision_score(y_true, y_pred, average=avg, zero_division=0), \
                     recall_score(y_true, y_pred, average=avg, zero_division=0), \
                     f1_score(y_true, y_pred, average=avg, zero_division=0)
    if num_classes == 2:
        return acc, precision_score(y_true, y_pred, average='binary', zero_division=0), \
                     recall_score(y_true, y_pred, average='binary', zero_division=0), \
                     f1_score(y_true, y_pred, average='binary', zero_division=0)
    avg = 'macro'
    return acc, precision_score(y_true, y_pred, average=avg, zero_division=0), \
                 recall_score(y_true, y_pred, average=avg, zero_division=0), \
                 f1_score(y_true, y_pred, average=avg, zero_division=0)


def kernel_from_beams(raw_hdr, targ_hdr, fudge_scale=1.0):
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
    - Small negative eigenvalues (numerical noise) are clipped to zero.
    """

    def _beam_cov_radians(bmaj_as, bmin_as, pa_deg):
        sx = _sigma_from_fwhm_arcsec(bmaj_as)
        sy = _sigma_from_fwhm_arcsec(bmin_as)
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

_KER_CACHE = {}
def kernel_from_beams_cached(raw_hdr, targ_hdr, fudge_scale=1.0):
    key = (
        round(float(raw_hdr['BMAJ']),9), round(float(raw_hdr['BMIN']),9), round(float(raw_hdr.get('BPA',0.0)),6),
        round(float(targ_hdr['BMAJ']),9), round(float(targ_hdr['BMIN']),9), round(float(targ_hdr.get('BPA', raw_hdr.get('BPA',0.0))),6),
        tuple(round(float(x),12) for x in (
            raw_hdr.get('CD1_1', raw_hdr.get('CDELT1', 0.0)),
            raw_hdr.get('CD1_2', 0.0),
            raw_hdr.get('CD2_1', 0.0),
            raw_hdr.get('CD2_2', raw_hdr.get('CDELT2', 0.0)),
        )),
        round(float(fudge_scale),6),
    )
    ker = _KER_CACHE.get(key)
    if ker is None:
        ker = kernel_from_beams(raw_hdr, targ_hdr, fudge_scale=fudge_scale)
        _KER_CACHE[key] = ker
    return ker

def make_fft_header(raw_hdr, pix_eff_arcsec, out_hw):
    h = fits.Header()

    # --- orientation matrix from RAW ---
    if 'CD1_1' in raw_hdr:
        M = np.array([[raw_hdr['CD1_1'], raw_hdr.get('CD1_2', 0.0)],
                      [raw_hdr.get('CD2_1', 0.0), raw_hdr['CD2_2']]], float)
    else:
        pc11 = raw_hdr.get('PC1_1', 1.0); pc12 = raw_hdr.get('PC1_2', 0.0)
        pc21 = raw_hdr.get('PC2_1', 0.0); pc22 = raw_hdr.get('PC2_2', 1.0)
        cd1  = raw_hdr.get('CDELT1', -1.0); cd2 = raw_hdr.get('CDELT2', 1.0)
        M = np.array([[pc11, pc12],[pc21, pc22]], float) @ np.diag([cd1, cd2])  # deg/pix

    # unit-column orientation (preserve rotation/handedness), new isotropic step
    sx = abs(_cdelt_deg(raw_hdr, 1))
    sy = abs(_cdelt_deg(raw_hdr, 2))
    if sx <= 0 or sy <= 0 or not np.isfinite(sx*sy):
        s1 = np.sign(raw_hdr.get('CDELT1', raw_hdr.get('CD1_1', -1.0)) or -1.0)
        s2 = np.sign(raw_hdr.get('CDELT2', raw_hdr.get('CD2_2',  1.0)) or  1.0)
        step = pix_eff_arcsec/3600.0
        CD_new = np.array([[s1*step, 0.0],[0.0, s2*step]], float)
    else:
        R = M @ np.diag([1.0/sx, 1.0/sy])
        CD_new = R * (pix_eff_arcsec/3600.0)

    outH, outW = out_hw
    h['CD1_1'] = float(CD_new[0,0]); h['CD1_2'] = float(CD_new[0,1])
    h['CD2_1'] = float(CD_new[1,0]); h['CD2_2'] = float(CD_new[1,1])
    h['NAXIS1'] = int(outW); h['NAXIS2'] = int(outH)

    # put the reference pixel at the center of the new cutout
    h['CRPIX1'] = 0.5*(outW+1); h['CRPIX2'] = 0.5*(outH+1)

    # anchor CRVAL to the sky position that was at the RAW center
    try:
        w_raw = WCS(raw_hdr).celestial
        ra_c, dec_c = w_raw.wcs_pix2world([[raw_hdr['NAXIS1']/2.0, raw_hdr['NAXIS2']/2.0]], 0)[0]
        h['CRVAL1'] = float(ra_c); h['CRVAL2'] = float(dec_c)
    except Exception:
        for k in ('CRVAL1','CRVAL2'):
            if k in raw_hdr: h[k] = raw_hdr[k]

    # carry frame type strings (safe to copy verbatim)
    for k in ('CTYPE1','CTYPE2'):
        if k in raw_hdr: h[k] = raw_hdr[k]

    return h


def _beam_solid_angle_sr(hdr):
    """Gaussian beam solid angle in steradians; BMAJ/BMIN in degrees."""
    bmaj = float(hdr['BMAJ']) * (np.pi/180.0)
    bmin = float(hdr['BMIN']) * (np.pi/180.0)
    return (np.pi / (4.0*np.log(2.0))) * bmaj * bmin


@lru_cache(maxsize=None)
def _headers_for_name(base_name: str):
    """
    Return (raw_hdr, t25_hdr, t50_hdr, t100_hdr, pix_native_arcsec, raw_fits_path)
    Builds synthetic T*kpc headers from z if needed/possible.
    """
    base_dir = _first(f"{PSZ2_ROOT}/fits/{base_name}*") or f"{PSZ2_ROOT}/fits/{base_name}"
    raw_path = _first(f"{base_dir}/{os.path.basename(base_dir)}.fits") \
            or _first(f"{base_dir}/{os.path.basename(base_dir)}*.fits")
    if raw_path is None:
        raise FileNotFoundError(f"RAW FITS not found under {base_dir}")

    t25_path  = _first(f"{base_dir}/{base_name}T25kpc*.fits") \
             or _first(f"{PSZ2_ROOT}/classified/T25kpc/*/{base_name}.fits") \
             or _first(f"{PSZ2_ROOT}/classified/T25kpcSUB/*/{base_name}.fits")
    t50_path  = _first(f"{base_dir}/{base_name}T50kpc*.fits") \
             or _first(f"{PSZ2_ROOT}/classified/T50kpc/*/{base_name}.fits") \
             or _first(f"{PSZ2_ROOT}/classified/T50kpcSUB/*/{base_name}.fits")
    t100_path = _first(f"{base_dir}/{base_name}T100kpc*.fits") \
             or _first(f"{PSZ2_ROOT}/classified/T100kpc/*/{base_name}.fits") \
             or _first(f"{PSZ2_ROOT}/classified/T100kpcSUB/*/{base_name}.fits")

    raw_hdr = fits.getheader(raw_path)

    def _hdr_or_synth(tpath, kpc, ref_hdr=None, kpc_ref=None):
        if tpath:
            return fits.getheader(tpath)
        try:
            z = get_z(base_name, raw_hdr)
            return synth_taper_header_from_kpc(raw_hdr, z, kpc, mode="keep_ratio")
        except Exception:
            if (ref_hdr is not None) and (kpc_ref is not None):
                return synth_taper_header_from_ref(raw_hdr, ref_hdr, kpc, kpc_ref, mode="keep_ratio")
            return None

    # Prefer scaling from T50 if only one fixed exists
    t50_hdr  = _hdr_or_synth(t50_path,  50)
    t25_hdr  = _hdr_or_synth(t25_path,  25, ref_hdr=t50_hdr,  kpc_ref=50.0)
    t100_hdr = _hdr_or_synth(t100_path, 100, ref_hdr=t50_hdr, kpc_ref=50.0)

    pix_native = _pixscale_arcsec(raw_hdr)
    return raw_hdr, t25_hdr, t50_hdr, t100_hdr, pix_native, raw_path

def _has_anchors(fn: str, anchor_versions):
    base = _name_base_from_fn(fn)
    try:
        raw_hdr, t25_hdr, t50_hdr, t100_hdr, *_ = _headers_for_name(base)
    except Exception:
        return False
    need = {v.lower() for v in anchor_versions}
    def ok(v):
        if v == "t25kpc":  return t25_hdr  is not None
        if v == "t50kpc":  return t50_hdr  is not None
        if v == "t100kpc": return t100_hdr is not None
        if v == "raw":     return True
        return True
    return all(ok(v) for v in need)

def _has_rt_support(fn: str) -> bool:
    """
    True if we can make rt* at runtime. Prefer a real redshift; otherwise
    fall back to existing fixed anchors.
    """
    base = _name_base_from_fn(fn)
    try:
        raw_hdr, t25_hdr, t50_hdr, t100_hdr, *_ = _headers_for_name(base)
    except Exception:
        return False
    try:
        _ = get_z(base, raw_hdr)
        return True
    except Exception:
        return _has_anchors(fn, _anchor_versions)



def permute_like(x, perm):
    if x is None: return None
    idx = perm.cpu().tolist()
    if isinstance(x, torch.Tensor): return x[perm]
    if isinstance(x, np.ndarray):   return x[idx]
    if isinstance(x, (list, tuple)): return [x[i] for i in idx]
    return x

def relabel(y):
    """
    Convert raw single-class ids to 2-bit multi-label targets [RH, RR].
    RH (52) -> [1,0]
    RR (53) -> [0,1]
    If you ever have 'both', set both bits to 1 *upstream*.
    """
    y = y.long()
    out = torch.zeros((y.shape[0], 2), dtype=torch.float32, device=y.device)
    out[:, 0] = (y == 52).float()  # RH
    out[:, 1] = (y == 53).float()  # RR
    return out    

def _background_ring_mask(h, w, inner=64, pad=8):
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    cy, cx = h//2, w//2
    half = inner//2
    # guard band
    mask_c = (yy >= cy-(half+pad)) & (yy <= cy+(half+pad)) & (xx >= cx-(half+pad)) & (xx <= cx+(half+pad))
    return ~mask_c

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

def _robust_sigma(x2d):
    x = torch.as_tensor(x2d, dtype=torch.float32)
    med = x.median()
    return 1.4826 * (x - med).abs().median()

def _per_image_percentile_stretch(x2d, lo=60, hi=95, USE_ASINH=True, asin_scale=10.0):
    t = torch.as_tensor(x2d, dtype=torch.float32)
    pl = torch.quantile(t.reshape(-1), lo/100.0)
    ph = torch.quantile(t.reshape(-1), hi/100.0)
    y = (t - pl) / (ph - pl + 1e-6)
    if USE_ASINH:
        y = torch.asinh(asin_scale * y) /math.asinh(asin_scale)
    return y.clamp(0, 1)

def as_index_labels(y: torch.Tensor) -> torch.Tensor:
    return y.argmax(dim=1) if y.ndim > 1 else y

def convolve_to_target(raw_arr, raw_hdr, target_hdr, fudge_scale=1.0):
    """
    Same as convolve_to_target(); provided for convenience when the image is
    already on the RAW grid. Units in = Jy/beam_native; units out = Jy/beam_target.
    """
    ker = kernel_from_beams_cached(raw_hdr, target_hdr, fudge_scale=fudge_scale)
    out = convolve_fft(raw_arr, ker, boundary='fill', fill_value=np.nan,
                   nan_treatment='interpolate', normalize_kernel=True,
                   psf_pad=True, fft_pad=True, allow_huge=True)    
    try:
        out *= (_beam_solid_angle_sr(target_hdr) / _beam_solid_angle_sr(raw_hdr))
    except Exception:
        pass
    return out

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


def collapse_logits(logits, num_classes, multilabel):
    if logits.ndim == 4:
        logits = torch.nn.functional.adaptive_avg_pool2d(logits, (1, 1)).squeeze(-1).squeeze(-1)
    elif logits.ndim == 3:
        logits = logits.mean(dim=-1)
    if logits.ndim == 1:
        logits = logits.unsqueeze(1)
    if not multilabel and logits.shape[1] == 1 and num_classes == 2:
        logits = torch.cat([-logits, logits], dim=1)
    return logits

def reproject_like(arr, src_hdr, dst_hdr):
    """
    Reproject a 2-D image from src_hdr WCS to dst_hdr WCS.

    If the 'reproject' package is available and both headers contain a valid
    2-D celestial WCS, use bilinear interpolation. Otherwise fall back to a
    center-alignment translation (keeps shape, best-effort alignment).
    """
    import numpy as _np
    from astropy.wcs import WCS
    try:
        from reproject import reproject_interp
        HAVE_REPROJECT = True
    except Exception:
        HAVE_REPROJECT = False
    try:
        from scipy.ndimage import shift as _imgshift
    except Exception:
        _imgshift = None

    if arr is None or src_hdr is None or dst_hdr is None:
        return None

    # FAST PATH: identical grid → return as-is
    def _same_pixel_grid(h1, h2, atol=1e-12):
        for k in ('NAXIS1','NAXIS2','CTYPE1','CTYPE2'):
            if (h1.get(k) != h2.get(k)):
                return False
        def _cd(h):
            if 'CD1_1' in h:
                return (float(h['CD1_1']), float(h.get('CD1_2',0.0)),
                        float(h.get('CD2_1',0.0)), float(h['CD2_2']))
            pc11=h.get('PC1_1',1.0); pc12=h.get('PC1_2',0.0)
            pc21=h.get('PC2_1',0.0); pc22=h.get('PC2_2',1.0)
            c1=h.get('CDELT1',-1.0); c2=h.get('CDELT2', 1.0)
            M = _np.array([[pc11,pc12],[pc21,pc22]],float) @ _np.diag([c1,c2])
            return (float(M[0,0]), float(M[0,1]), float(M[1,0]), float(M[1,1]))
        if not _np.allclose(_cd(h1), _cd(h2), atol=atol): return False
        if not _np.allclose([h1.get('CRPIX1'),h1.get('CRPIX2')],
                            [h2.get('CRPIX1'),h2.get('CRPIX2')], atol=1e-9): return False
        if not _np.allclose([h1.get('CRVAL1'),h1.get('CRVAL2')],
                            [h2.get('CRVAL1'),h2.get('CRVAL2')], atol=1e-9): return False
        return True

    if _same_pixel_grid(src_hdr, dst_hdr):
        return _np.asarray(arr, float)

    # Try full WCS reprojection
    try:
        w_src = WCS(src_hdr).celestial
        w_dst = WCS(dst_hdr).celestial
    except Exception:
        w_src = w_dst = None

    if HAVE_REPROJECT and (w_src is not None) and (w_dst is not None):
        ny_out = int(dst_hdr['NAXIS2']); nx_out = int(dst_hdr['NAXIS1'])
        reproj, _ = reproject_interp((arr, w_src), w_dst,
                                     shape_out=(ny_out, nx_out),
                                     order='bilinear')
        return reproj.astype(float)

    # Fallback: align image centers via subpixel shift
    if (w_src is None) or (w_dst is None) or (_imgshift is None):
        return _np.asarray(arr, float)

    ny, nx = arr.shape
    (ra, dec) = w_src.wcs_pix2world([[nx/2.0, ny/2.0]], 0)[0]
    (x_dst, y_dst) = w_dst.wcs_world2pix([[ra, dec]], 0)[0]
    dx = (float(dst_hdr['NAXIS1'])/2.0) - x_dst
    dy = (float(dst_hdr['NAXIS2'])/2.0) - y_dst
    return _imgshift(arr, shift=(dy, dx), order=1, mode="nearest").astype(float)

def rt_only_from_raw(base_images, base_labels, base_fns, gen_versions,
                     *, crop_size, downsample_size, percentile_lo, percentile_hi,
                     do_stretch=True):
    """
    From RAW images + filenames, build ONLY the requested rt planes and return
    them stacked along channel dim (no RAW included).
    Returns: (rt_images[B,R,H,W], labels_out, fns_out) where R=len(gen_versions).
    """
    if not gen_versions:
        return base_images, base_labels, base_fns  # nothing to do

    if base_fns is None:
        raise ValueError("Filenames are required to synthesize rt* planes. Set PRINTFILENAMES=True.")

    fns = [str(fn) for fn in base_fns]

    # Ensure base is [B,1,H,W]
    x = base_images
    if x.dim() == 5:
        x = fold_T_axis(x)               # [B,T,H,W]
    if x.dim() != 4:
        raise AssertionError(f"Expecting [B,C,H,W]; got {tuple(x.shape)}")
    if x.size(1) != 1:
        x = x[:, :1]                     # keep one RAW channel

    # Build per-version planes, track filenames that succeeded
    planes_per_ver = {}
    keep_sets = []
    for gv in [str(v).lower() for v in gen_versions]:
        rt, keep_mask, kept_fns, skipped = apply_taper_to_tensor(
            x, gv, filenames=fns,
            crop_size=crop_size,
            downsample_size=downsample_size,
            percentile_lo=percentile_lo,
            percentile_hi=percentile_hi,
            do_stretch=do_stretch,
            require_fixed_header=False,
        )
        planes_per_ver[gv] = {str(fn): rt[i] for i, fn in enumerate(kept_fns)}
        keep_sets.append(set(map(str, kept_fns)))

    # Intersection across all requested rt versions
    keep_all = set(fns) if not keep_sets else set.intersection(*keep_sets)
    if not keep_all:
        # Return empty tensors with correct type/device
        empty_imgs = x[:0, :0]
        empty_lbls = base_labels[:0] if base_labels is not None else None
        return empty_imgs, empty_lbls, []

    keep_idx = [i for i, fn in enumerate(fns) if fn in keep_all]
    y_kept   = base_labels[keep_idx] if base_labels is not None else None
    fns_kept = [fns[i] for i in keep_idx]

    # Stack ONLY the rt planes (no RAW)
    rt_stacks = []
    for gv in [str(v).lower() for v in gen_versions]:
        stack = torch.stack([planes_per_ver[gv][fn] for fn in fns_kept], dim=0)  # [Bk,1,H,W]
        stack = stack.to(device=x.device, dtype=x.dtype)
        rt_stacks.append(stack)

    # Concatenate along C (channels) and squeeze the singleton per-plane channel
    rt_only = torch.cat(rt_stacks, dim=1)         # [B, R, H, W]  (R=len(gen_versions))
    return rt_only, y_kept, fns_kept


def auto_fudge_scale(raw_img, raw_hdr, targ_hdr, T_img, s_grid=None, nbins=36):
    import numpy as np
    if s_grid is None:
        s_grid = np.linspace(1.00, 1.20, 11)
    U,V,FT,AT = image_to_vis(T_img, targ_hdr, beam_hdr=targ_hdr)
    best_s, best_cost = 1.0, np.inf
    for s in s_grid:
        RT_native = convolve_to_target(raw_img, raw_hdr, targ_hdr, fudge_scale=s)
        RT_on_t   = reproject_like(RT_native, raw_hdr, targ_hdr)
        _,_,FR,AR = image_to_vis(RT_on_t, targ_hdr, beam_hdr=targ_hdr)
        # radial medians
        r, aT = _radial_bin(U,V,AT, nbins=nbins, stat='median')
        _, aR = _radial_bin(U,V,AR, nbins=nbins, stat='median')
        # simple coherence mask
        Rgrid = _np.sqrt(U*U + V*V)
        edges = _np.geomspace(_np.nanpercentile(Rgrid,1.0), _np.nanmax(Rgrid), nbins+1)
        coh = []
        for i in range(nbins):
            m = (Rgrid>=edges[i]) & (Rgrid<edges[i+1])
            if not _np.any(m): coh.append(_np.nan); continue
            num  = _np.nanmean(FT[m]*_np.conj(FR[m]))
            den1 = _np.nanmean(_np.abs(FT[m])**2)
            den2 = _np.nanmean(_np.abs(FR[m])**2)
            coh.append(_np.abs(num)/_np.sqrt(den1*den2))
        coh = _np.asarray(coh)
        good = (coh > 0.6) & _np.isfinite(aT) & _np.isfinite(aR)
        if not _np.any(good): 
            continue
        ratio = aT[good] / (aR[good] + 1e-12)
        cost = _np.nanmedian(_np.abs(_np.log(ratio)))
        if cost < best_cost:
            best_cost, best_s = float(cost), float(s)
    return best_s


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
    import os
    import numpy as _np
    import math as _math
    from astropy.io import fits
    from scipy.ndimage import gaussian_filter as _gauss
    import torch

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
            raw_native = _np.squeeze(fits.getdata(raw_path)).astype(float)
        except Exception:
            skipped.append(base); kept_flags.append(False); continue
        raw_native = _np.nan_to_num(raw_native, copy=False)

        # 2) Anti-alias (only if we’ll shrink to Hout×Wout)
        try:
            ds = int(round(crop_size[-2] / float(Hout)))
            if ds > 1:
                raw_pref = _gauss(raw_native, sigma=0.5*ds, mode='nearest')
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
                    T_img = _np.squeeze(fits.getdata(tpath)).astype(float)
                    fudge = auto_fudge_scale(
                        raw_pref, raw_hdr, targ_hdr_eff, T_img,
                        s_grid=_np.linspace(1.00, 1.20, 11), nbins=48
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
        t = torch.from_numpy(_np.nan_to_num(matched_on_t, copy=False)).float().unsqueeze(0)  # [1,H,W]
        crop_eff = _effective_crop_on_raw(raw_hdr, targ_hdr_eff, crop_size)
        formatted = apply_formatting(t, crop_size=crop_eff,
                                     downsample_size=(1, Hout, Wout)).squeeze(0)   # [1,Hout,Wout]

        # 7) Percentile stretch + optional asinh
        if do_stretch:
            stretched = _per_image_percentile_stretch(formatted.squeeze(0), percentile_lo, percentile_hi, USE_ASINH=False).unsqueeze(0)
        else:
            stretched = formatted
        if use_asinh:
            stretched = torch.asinh(10.0 * stretched) / _math.asinh(10.0)

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
            import matplotlib.pyplot as _plt
            os.makedirs(debug_dir, exist_ok=True)
            fig, ax = _plt.subplots(1,2, figsize=(6,3))
            ax[0].imshow(formatted.squeeze(0).cpu().numpy(), cmap='viridis', origin='lower'); ax[0].axis('off'); ax[0].set_title('PSF-matched')
            ax[1].imshow(stretched.squeeze(0).cpu().numpy(), cmap='viridis', origin='lower'); ax[1].axis('off'); ax[1].set_title('final')
            fig.suptitle(base); fig.tight_layout()
            tag = f"rt{want_kpc}"
            fig.savefig(os.path.join(debug_dir, f"{base}_{tag}{_versions_to_load}.png"), dpi=140)
            _plt.close(fig)

    keep_mask = torch.tensor(kept_flags, dtype=torch.bool)
    out = torch.stack(out, dim=0).to(device=device, dtype=dtype) if out else torch.empty((0,1,Hout,Wout), device=device, dtype=dtype)
    return out, keep_mask, kept_fns, skipped

def replicate_list(x, n):
    return [v for v in x for _ in range(int(n))]

def late_augment(images, labels, filenames=None, *, st_aug=False):
    """
    Apply your normal augmentations AFTER tapering.
    Returns (imgs_aug, labels_aug, filenames_aug).
    If images is empty, this is a no-op.
    """
    if images is None or (isinstance(images, torch.Tensor) and images.numel() == 0):
        return images, labels, filenames
    imgs_aug, labels_aug = augment_images(images, labels, ST_augmentation=st_aug)
    n_aug = imgs_aug.size(0) // max(1, images.size(0))
    if filenames is not None:
        filenames = replicate_list(filenames, n_aug)
    return imgs_aug, labels_aug, filenames


def late_augment_old(images, labels, filenames=None, *, st_aug=False):
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

# ---------- redshift + header helpers ----------

def _load_cluster_meta(csv_path):
    import csv, os
    d = {}
    if not os.path.exists(csv_path): 
        return d
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            slug = (row.get("slug") or "").strip()
            ztxt = (row.get("z") or "").strip()
            try:
                z = float(ztxt)
                if 0.0 < z < 5.0:
                    d[slug] = z
            except Exception:
                pass
    return d

CLUSTER_META = _load_cluster_meta(CLUSTER_METADATA_CSV)

def _truncate_slug(s: str) -> str:
    """
    Turn 'PSZ2G113.91-37.01' → 'PSZ2G113.91-37'
    Turn 'PSZ2G112.48+56.99' → 'PSZ2G112.48+56'
    """
    m = re.match(r'^(PSZ2G\d+\.\d+[+-]\d+)(?:\.\d+)?$', s)
    return m.group(1) if m else s

# Build alias map so truncated slugs also resolve
CLUSTER_META_ALIAS = {}
for k, z in CLUSTER_META.items():
    kt = _truncate_slug(k)
    if kt not in CLUSTER_META:
        # only add if unambiguous; if multiple long keys collapse to the same short key with *different* z,
        # prefer the one with more decimals (do nothing here).
        if kt not in CLUSTER_META_ALIAS:
            CLUSTER_META_ALIAS[kt] = z


def _name_base_from_fn(fn):
    stem = Path(str(fn)).stem
    return stem.split('T', 1)[0]

def _z_from_meta(name: str):
    # 1) exact
    z = CLUSTER_META.get(name)
    if z is not None:
        return z

    # 2) truncated alias (logger shows names like PSZ2G113.91-37)
    kt = _truncate_slug(name)
    z = CLUSTER_META.get(kt) or CLUSTER_META_ALIAS.get(kt)
    if z is not None:
        return z

    # 3) prefix match (handle cases where code passed an even shorter form)
    #    e.g., name='PSZ2G134.70+48' and CSV has 'PSZ2G134.70+48.91'
    for k, zv in CLUSTER_META.items():
        if k.startswith(name):
            return zv

    return None

def _coerce_float(v):
    try:
        x = float(v); 
        return x if np.isfinite(x) else None
    except Exception:
        pass
    if isinstance(v, (bytes, bytearray)):
        try: v = v.decode('utf-8', 'ignore')
        except Exception: return None
    if isinstance(v, str):
        m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', v)
        if m:
            try: return float(m.group(0))
            except Exception: return None
    return None


def kpc_to_arcsec(z, L_kpc):
    L = (L_kpc * u.kpc)
    DA = COSMO.angular_diameter_distance(float(z)).to(u.kpc)  # ensure same length unit
    theta = (L / DA) * u.rad                                  # ratio → radians
    return theta.to(u.arcsec).value

def _sigma_from_fwhm_arcsec(theta_as):
    return (theta_as * ARCSEC) / (2.0 * np.sqrt(2.0 * np.log(2.0)))

def _make_uv_gaussian_weight(nx, ny, dx, dy, theta_as):
    sigma_th = _sigma_from_fwhm_arcsec(theta_as)
    u = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    v = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    U, V = np.meshgrid(u, v)
    return np.exp(-2.0 * (np.pi**2) * (sigma_th**2) * (U*U + V*V))

def _cdelt_deg(hdr, axis: int) -> float:
    key = f"CDELT{axis}"
    if key in hdr and np.isfinite(hdr[key]): return float(abs(hdr[key]))
    if 'CD1_1' in hdr:
        row = (hdr['CD1_1'], hdr.get('CD1_2',0.0)) if axis==1 else (hdr.get('CD2_1',0.0), hdr['CD2_2'])
        return float(np.hypot(*row))
    pc11,pc12 = hdr.get('PC1_1',1.0), hdr.get('PC1_2',0.0)
    pc21,pc22 = hdr.get('PC2_1',0.0), hdr.get('PC2_2',1.0)
    cd1,cd2   = hdr.get('CDELT1',1.0), hdr.get('CDELT2',1.0)
    M = np.array([[pc11,pc12],[pc21,pc22]], float) @ np.diag([cd1,cd2])
    row = M[0] if axis==1 else M[1]
    return float(np.hypot(row[0], row[1]))

def _effective_crop_on_raw(raw_hdr, targ_hdr, crop_size):
    Hc_raw, Wc_raw = crop_size[-2], crop_size[-1]
    try:
        s_raw = _cdelt_deg(raw_hdr, 1)
        s_tgt = _cdelt_deg(targ_hdr, 1) if targ_hdr is not None else s_raw
        scale = s_raw / s_tgt if (s_tgt and np.isfinite(s_tgt)) else 1.0
    except Exception:
        scale = 1.0
    Hc = max(16, int(round(Hc_raw * scale)))
    Wc = max(16, int(round(Wc_raw * scale)))
    return (1, Hc, Wc)

def apply_uv_gaussian_taper(img, hdr_img, theta_as, pad_factor=2):
    A = np.where(np.isfinite(img), img, 0.0).astype(float)
    ny0, nx0 = A.shape
    if pad_factor > 1:
        ny, nx = int(ny0*pad_factor), int(nx0*pad_factor)
        py = (ny-ny0)//2; px = (nx-nx0)//2
        A = np.pad(A, ((py,ny-ny0-py),(px,nx-nx0-px)), mode='constant', constant_values=0.0)
        crop = (slice(py,py+ny0), slice(px,px+nx0))
    else:
        ny, nx = ny0, nx0
        crop = (slice(0,ny0), slice(0,nx0))
    dx = abs(_cdelt_deg(hdr_img, 1)) * np.pi/180.0
    dy = abs(_cdelt_deg(hdr_img, 2)) * np.pi/180.0
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A))) * (dx * dy)
    W = _make_uv_gaussian_weight(nx, ny, dx, dy, theta_as)
    At = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F*W))).real / (dx*dy)
    return At[crop]




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

FIRST = True # To debug the first iteration
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

    if any(cls in galaxy_classes for cls in [10, 11, 12, 13]):
        batch_size = 128
    else:
        batch_size = 16

    img_shape = downsample_size
    print("IMG SHAPE:", img_shape)

    # --- Parse requested versions and set runtime-rt behavior (no anchoring) ---
    def _split_versions(v):
        if not isinstance(v, (list, tuple)):
            v = [v]
        flat = []
        for x in v:
            if isinstance(x, (list, tuple)):
                flat.extend(list(x))
            else:
                flat.append(x)
        flat = [str(x).strip().lower() for x in flat]
        load, gen = [], []
        for x in flat:
            m = re.fullmatch(r'rt(\d+)(?:kpc)?', x)
            if m:
                gen.append(f"rt{m.group(1)}")
            else:
                load.append(x)
        return load, gen

    # Split the request (e.g. 'rt50' -> gen; nothing to load from disk except raw)
    _load_versions, _gen_versions = _split_versions(versions)
    print(f"Versions to LOAD: {_load_versions}, to GENERATE: {_gen_versions}")
    DO_RT = bool(_gen_versions)

    # When DO_RT is True, pass the rt token (e.g. 'rt50') to the loader.
    # Your loader should interpret 'rt50' to mean: "return RAW images but only for sources
    # with a valid redshift (so run-time taper is feasible)."
    _versions_to_load = ([_gen_versions[0]] if DO_RT else (_load_versions or ['raw']))

    # if we ask for any rt*, we need filenames to synthesize them → enable printing filenames
    # force apples-to-apples for this sweep:
    if vers != ['raw']:
        LATE_AUG = False  # <— set False to match the T50kpc run
        PRINTFILENAMES = True
    else:
        LATE_AUG = False

    EXTRAVARS = False

    print(f"LATE_AUG={LATE_AUG}, PRINTFILENAMES={PRINTFILENAMES}, EXTRAVARS={EXTRAVARS}")

    # —— MULTI-LABEL mode for RH/RR ——
    if galaxy_classes == [52, 53]:
        MULTILABEL = True
        LABEL_INDEX = {"RH": 0, "RR": 1}
        THRESHOLD = 0.5
        def relabel(y):
            y = y.long()
            out = torch.zeros((y.shape[0], 2), dtype=torch.float32, device=y.device)
            out[:, 0] = (y == 52).float()
            out[:, 1] = (y == 53).float()
            return out
    else:
        MULTILABEL = False
        base_cls = min(galaxy_classes)
        def relabel(y):
            return (y - base_cls).long()

    if set(galaxy_classes) & {18} or set(galaxy_classes) & {19}:
        galaxy_classes = [20, 21, 21, 22, 23, 24, 25, 26, 27, 28, 29]  # Include all digits if 18 or 19 is in target_classes
    else:
        galaxy_classes = galaxy_classes
    num_classes = len(galaxy_classes)
    
    def _verkey(v):
        if isinstance(v, (list, tuple)):
            return "+".join(map(str, v))
        return str(v)
    ver_key = _verkey(versions)

    ###############################################
    ########## INITIALIZE DICTIONARIES ############
    ###############################################

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

        with torch.inference_mode():
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
        
        # ---------------------- TEST split ----------------------
        # Ask the loader for RAW, gated by the rt* anchor if we want rt*
        # (your loader maps 'rt50' -> "load RAW but only for sources where T50/redshift is feasible")
        _versions_for_loader = ([_gen_versions[0]] if DO_RT else (_load_versions or ['raw']))

        _out = _loader(
            galaxy_classes=galaxy_classes,
            versions=_versions_for_loader,
            fold=max(folds),                       # your “test split” rule
            crop_size=crop_size,
            downsample_size=downsample_size,
            sample_size=max_num_galaxies,
            REMOVEOUTLIERS=FILTERED,
            BALANCE=BALANCE,
            STRETCH=STRETCH,
            percentile_lo=percentile_lo,
            percentile_hi=percentile_hi,
            AUGMENT=False                  # (late aug happens after rt*)
            NORMALISE=NORMALISEIMGS,
            NORMALISETOPM=NORMALISEIMGSTOPM,
            USE_GLOBAL_NORMALISATION=USE_GLOBAL_NORMALISATION,
            GLOBAL_NORM_MODE=GLOBAL_NORM_MODE,
            PRINTFILENAMES=True,                   # ensure filenames for TEST
            train=False
        )

        # Unpack
        if len(_out) == 4:
            train_images, train_labels, test_images, test_labels = _out
            train_fns = test_fns = None
        elif len(_out) == 6:
            train_images, train_labels, test_images, test_labels, train_fns, test_fns = _out
            
            # keep RAW snapshot for “before vs after rt*” plots
            test_images_before_rt = (test_images.clone()
                                    if isinstance(test_images, torch.Tensor) else test_images)
            test_fns_before_rt = list(test_fns) if test_fns is not None else None
            # use some of your TEST filenames to ensure parity; take the first 12
            if FIRST and (test_images_before_rt is not None) and (test_fns_before_rt is not None):
                    plot_before_after_rt_3col(
                        raw_imgs=test_images_before_rt,    # RAW snapshot
                        raw_fns=test_fns_before_rt,
                        rt_imgs=test_images,               # synthesized rt*
                        rt_fns=test_fns,
                        tag=_gen_versions[0] if _gen_versions else 'rt??',
                        outdir='./classifier/debug_rt_before_after',
                        per_page=24
                    )
                    FIRST = False

        else:
            raise ValueError(f"loader(train=False) returned {len(_out)} values, expected 4 or 6")

        # ---------------------- Build rt-only TEST tensors (no RAW) ----------------------
        def _parse_rt_targets(tokens):
            """
            Map tokens like ['rt25','rt50','rt100'] (case/format flexible, accepts 'rt50', 'rt50kpc', 't50kpc')
            → a sorted unique list of {25,50,100}.
            """
            out = []
            for t in (tokens or []):
                s = str(t).lower()
                if '25' in s: out.append(25)
                elif '50' in s: out.append(50)
                elif '100' in s: out.append(100)
                else:
                    raise ValueError(f"Unrecognised rt* token: {t!r} (expected one of rt25/rt50/rt100)")
            # keep stable order 25,50,100 without duplicates
            seen, order = set(), []
            for k in (25,50,100):
                if k in out and k not in seen:
                    seen.add(k); order.append(k)
            return order

        if DO_RT:
            # We must have filenames to synthesize rt* planes
            if test_fns is None:
                raise RuntimeError("Loader must return filenames (len=6) when rt* is requested. Set PRINTFILENAMES=True.")

            # Guard models that require RAW alongside a second branch
            if classifier in ['DISSN', 'DICSN']:
                raise ValueError(f"{classifier} expects RAW (and/or fixed T50) inputs. "
                                f"You requested rt-only; pick a single-branch classifier (e.g. TinyCNN/SCNN/Scatter*).")

            # Which rt targets are requested?
            rt_targets = _parse_rt_targets(_gen_versions)   # e.g. [50] or [25,50]

            # Output size for the cuts (match your pipeline)
            outH, outW = int(downsample_size[-2]), int(downsample_size[-1])

            # Build rt-only planes by calling the same routine as your plotting script
            rt_planes_per_source = []   # list of [R, H, W] arrays
            kept_labels = []
            kept_fns    = []

            for i, fn in enumerate(test_fns):
                # normalise to a clean base (the helper in your plotting script)
                name = _name_base_from_fn(str(fn))
                try:
                    raw_cut, rt_cut, raw_hdr, pix_eff_arcsec, hdr_fft = _load_fits_arrays_scaled(
                        _gen_versions[0],           # e.g. "rt50"
                        name,
                        crop_ch=1,
                        out_hw=(outH, outW)
                    )
                except FileNotFoundError as e:
                    print(f"⚠️ Skipping {name}: {e}")
                    continue
                except Exception as e:
                    print(f"⚠️ {name}: rt* synthesis failed ({e}). Skipping.")
                    continue

                if rt_cut is None:
                    print(f"⚠️ Skipping {name}: requested {_gen_versions[0]} unavailable.")
                    continue

                # ensure [H,W] → [1,H,W]
                rt_cut = np.asarray(rt_cut, dtype=np.float32)
                if rt_cut.ndim == 2:
                    rt_cut = rt_cut[None, ...]

                rt_planes_per_source.append(rt_cut)      # [1,H,W]
                kept_labels.append(test_labels[i])
                kept_fns.append(str(fn))

            if len(rt_planes_per_source) == 0:
                raise RuntimeError("No TEST samples survived rt-only synthesis. "
                                "Check redshifts/filenames and your rt* targets.")

            # [B,R,H,W] float32
            test_images = torch.from_numpy(np.stack(rt_planes_per_source, axis=0)).float()  # [B,1,H,W]
            test_labels = torch.as_tensor(np.array(kept_labels), dtype=test_labels.dtype)
            test_fns    = kept_fns

            # Optional stretch/normalisation at this stage (mirrors your existing flags)
            if STRETCH:
                test_images  = _per_image_percentile_stretch(test_images,  percentile_lo, percentile_hi, USE_ASINH=True)
            if NORMALISEIMGSTOPM:
                test_images = normalise_images(test_images, -1, 1) if NORMALISEIMGSTOPM else test_images
            if USE_GLOBAL_NORMALISATION and (GLOBAL_NORM_MODE is not None):
                test_images = apply_global_normalisation(test_images, mode=GLOBAL_NORM_MODE)

        # Final safety before scattering
        assert test_images.dim() in (4,5), f"Unexpected TEST shape {tuple(test_images.shape)}"
        assert test_images.size(1) >= 1, "TEST images have 0 channels."

        # Late augmentation (after rt* and normalization)
        if LATE_AUG:
            test_images, test_labels, test_fns = late_augment(test_images, test_labels, test_fns)
            print(f"[TEST] after late augmentation: {test_images.size(0)} images")

        # Relabel to local indices/one-hot as needed
        test_labels = relabel(test_labels)

        # Shuffle TEST set to avoid any ordering systematics
        perm_test = torch.randperm(test_images.size(0))
        test_images, test_labels = test_images[perm_test], test_labels[perm_test]

        # ----------------- build TEST dataset -----------------
        # merge T/version axis into channels only if it exists
        if test_images.dim() == 5:
            test_images = fold_T_axis(test_images)      # [B,T,1,H,W] -> [B,T,H,W]
        assert test_images.dim() == 4, f"TEST should be [B,C,H,W], got {tuple(test_images.shape)}"

        mock_tensor = torch.zeros_like(test_images)

        if classifier in ['ScatterNet','ScatterResNet','ScatterSqueezeNet','ScatterSqueezeNet2','DISSN']:
            # Guard again for DISSN in case DO_RT==False but missing channels
            if classifier == 'DISSN':
                vnames = [str(v).lower() for v in (_versions_for_loader or ['raw'])]
                raise ValueError("DISSN expects RAW + T50 inputs, which are not provided in rt-only mode.")

            # scattering branch on (rt-only) images
            test_scat_coeffs = compute_scattering_coeffs(test_images, scattering, batch_size=128, device='cpu')
            if test_scat_coeffs.dim() == 5:
                test_scat_coeffs = test_scat_coeffs.flatten(start_dim=1, end_dim=2)
            if NORMALISESCS or NORMALISESCSTOPM:
                test_scat_coeffs = normalise_images(test_scat_coeffs, -1, 1) if NORMALISESCSTOPM else normalise_images(test_scat_coeffs, 0, 1)
            scatdim = test_scat_coeffs.shape[1:]
            if classifier in ['ScatterNet','ScatterResNet']:
                test_dataset = TensorDataset(torch.zeros_like(test_images), test_scat_coeffs, test_labels)
            else:
                # ScatterSqueezeNet variants: pass images + scattering
                test_dataset = TensorDataset(test_images, test_scat_coeffs, test_labels)
        else:
            # plain CNN branches
            test_dataset = TensorDataset(test_images, mock_tensor, test_labels)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=0, collate_fn=custom_collate, drop_last=False)



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
                _versions_arg = _versions_to_load if isinstance(_versions_to_load, (list, tuple)) else [_versions_to_load]
                _out = _loader(
                    galaxy_classes=galaxy_classes,
                    versions=_versions_arg or ['raw'],
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
                valid_labels = relabel(valid_labels)
                perm_valid = torch.randperm(valid_images.size(0))
                valid_images, valid_labels = valid_images[perm_valid], valid_labels[perm_valid]
                if PRINTFILENAMES: valid_fns = permute_like(valid_fns, perm_valid)

            else:
                # ---------------------- TRAIN + VALID split ----------------------
                # Decide what to load vs. what to generate (rt*)
                _versions_to_load, _gen_versions = _split_versions(versions)   # e.g. (['raw'], ['rt50'])
                DO_RT = bool(_gen_versions)

                # Small helper (same as in TEST block)
                def _parse_rt_targets(tokens):
                    """
                    Map tokens like ['rt25','rt50','rt100'] (case/format flexible, accepts 'rt50', 'rt50kpc', 't50kpc')
                    → a sorted unique list of {25,50,100}.
                    """
                    out = []
                    for t in (tokens or []):
                        s = str(t).lower()
                        if '25'  in s: out.append(25)
                        elif '50' in s: out.append(50)
                        elif '100' in s: out.append(100)
                        else:
                            raise ValueError(f"Unrecognised rt* token: {t!r} (expected one of rt25/rt50/rt100)")
                    seen, order = set(), []
                    for k in (25, 50, 100):
                        if k in out and k not in seen:
                            seen.add(k); order.append(k)
                    return order

                # Ask the loader for a gated RAW list when doing rt*, otherwise load what was requested
                _versions_for_loader = ([_gen_versions[0]] if DO_RT else (_versions_to_load or ['raw']))

                _out = _loader(
                    galaxy_classes=galaxy_classes,
                    versions=_versions_for_loader,
                    fold=max(folds),
                    crop_size=crop_size,
                    downsample_size=downsample_size,
                    sample_size=max_num_galaxies,
                    REMOVEOUTLIERS=FILTERED,
                    BALANCE=BALANCE,
                    STRETCH=STRETCH,
                    percentile_lo=percentile_lo,
                    percentile_hi=percentile_hi,
                    AUGMENT=False,                  # late aug happens after rt* synth
                    NORMALISE=NORMALISEIMGS,
                    NORMALISETOPM=NORMALISEIMGSTOPM,
                    USE_GLOBAL_NORMALISATION=USE_GLOBAL_NORMALISATION,
                    GLOBAL_NORM_MODE=GLOBAL_NORM_MODE,
                    PRINTFILENAMES=(PRINTFILENAMES or DO_RT),  # ensure filenames if we need to synth rt*
                    train=True
                )

                if len(_out) == 4:
                    # No filenames came back; this is fine only when not generating rt*
                    if DO_RT:
                        raise RuntimeError("rt* requested but loader returned 4-tuple (no filenames). "
                                        "Set PRINTFILENAMES=True when requesting rt* so we can synthesize planes.")
                    train_images, train_labels, valid_images, valid_labels = _out
                    train_fns = valid_fns = None

                    # shuffle immediately
                    perm_train = torch.randperm(train_images.size(0))
                    train_images, train_labels = train_images[perm_train], train_labels[perm_train]

                    perm_valid = torch.randperm(valid_images.size(0))
                    valid_images, valid_labels = valid_images[perm_valid], valid_labels[perm_valid]

                elif len(_out) == 6:
                    train_images, train_labels, test_images, test_labels, train_fns, test_fns = _out
                    if DO_RT:
                        # Disallow models that require RAW + another branch when we're rt-only
                        if classifier in ['DISSN', 'DICSN']:
                            raise ValueError(f"{classifier} expects RAW (+ T50 for DISSN / dual branches for DICSN). "
                                            f"You requested rt-only; pick a single-branch classifier.")

                        rt_targets = _parse_rt_targets(_gen_versions)   # e.g. [50] or [25,50]
                        outH, outW = int(downsample_size[-2]), int(downsample_size[-1])

                        def _make_rt_cube_for_list(fns, labels):
                            rt_list, keep_idx = [], []
                            for i, fn in enumerate(fns or []):
                                name = _name_base_from_fn(str(fn))
                                try:
                                    raw_cut, rt_cut, raw_hdr, pix_eff_arcsec, hdr_fft = _load_fits_arrays_scaled(
                                        _gen_versions[0],           # e.g. "rt50"
                                        name,
                                        crop_ch=1,
                                        out_hw=(outH, outW)
                                    )
                                except FileNotFoundError as e:
                                    print(f"⚠️ Skipping {name}: {e}")
                                    continue
                                except Exception as e:
                                    print(f"⚠️ {name}: rt* synthesis failed ({e}). Skipping.")
                                    continue

                                if rt_cut is None:
                                    print(f"⚠️ Skipping {name}: requested {_gen_versions[0]} unavailable.")
                                    continue

                                p = np.asarray(rt_cut, dtype=np.float32)
                                if p.ndim == 2: 
                                    p = p[None, ...]       # [1,H,W]
                                rt_list.append(p)          # one channel per sample
                                keep_idx.append(i)

                            if len(rt_list) == 0:
                                return None, None, None

                            imgs = torch.from_numpy(np.stack(rt_list, axis=0)).float()  # [B,R,H,W]
                            labs = labels[keep_idx] if isinstance(labels, torch.Tensor) else np.asarray(labels)[keep_idx]
                            fsel = [str(fns[i]) for i in keep_idx]
                            return imgs, labs, fsel

                        # Build rt-only tensors for TRAIN and VALID
                        train_images_rt, train_labels_rt, train_fns = _make_rt_cube_for_list(train_fns, train_labels)
                        valid_images_rt, valid_labels_rt, valid_fns = _make_rt_cube_for_list(valid_fns, valid_labels)

                        if train_images_rt is None or valid_images_rt is None:
                            raise RuntimeError("No samples survived rt-only synthesis for train/valid. "
                                            "Check redshifts/headers and requested rt* targets.")

                        train_images, train_labels = train_images_rt, (torch.as_tensor(train_labels_rt) if not torch.is_tensor(train_labels_rt) else train_labels_rt)
                        valid_images, valid_labels = valid_images_rt, (torch.as_tensor(valid_labels_rt) if not torch.is_tensor(valid_labels_rt) else valid_labels_rt)

                        # Optional stretch/normalisation like your TEST path
                        if STRETCH:
                            train_images = _per_image_percentile_stretch(train_images, percentile_lo, percentile_hi, USE_ASINH=True)
                            valid_images = _per_image_percentile_stretch(valid_images, percentile_lo, percentile_hi, USE_ASINH=True)
                        if NORMALISEIMGSTOPM:
                            train_images = normalise_images(train_images, -1, 1)
                            valid_images = normalise_images(valid_images, -1, 1)
                        if USE_GLOBAL_NORMALISATION and (GLOBAL_NORM_MODE is not None):
                            train_images = apply_global_normalisation(train_images, mode=GLOBAL_NORM_MODE)
                            valid_images = apply_global_normalisation(valid_images, mode=GLOBAL_NORM_MODE)
                            
                    if train_images == None:
                        raise ValueError("train_images is None, cannot compute scattering coefficients")

                    # Late augmentation happens after rt* + normalisation
                    if LATE_AUG:
                        before = len(train_images)
                        train_images, train_labels, train_fns = late_augment(train_images, train_labels, train_fns)
                        valid_images, valid_labels, valid_fns = late_augment(valid_images, valid_labels, valid_fns)
                        n_aug = len(train_images) // max(1, before)
                        print(f"[AUG] late_augment replicated train by ~x{n_aug} ({before} → {len(train_images)})")

                    # shuffle AFTER late augmentation
                    perm_train = torch.randperm(train_images.size(0))
                    train_images, train_labels = train_images[perm_train], train_labels[perm_train]
                    if train_images == None:
                        raise ValueError("train_images is None, cannot compute scattering coefficients")
                    if PRINTFILENAMES and train_fns is not None:
                        train_fns = permute_like(train_fns, perm_train)

                    perm_valid = torch.randperm(valid_images.size(0))
                    valid_images, valid_labels = valid_images[perm_valid], valid_labels[perm_valid]
                    if PRINTFILENAMES and valid_fns is not None:
                        valid_fns = permute_like(valid_fns, perm_valid)

                    if EXTRAVARS:
                        print("Warning, EXTRAVARS path is lightly tested.")
                        train_data = train_fns
                        valid_data = valid_fns
                else:
                    raise ValueError(f"loader(train=True) returned {len(_out)} values, expected 4 or 6")


                def _desc(name, x):
                    import torch
                    if not isinstance(x, torch.Tensor):
                        x = torch.as_tensor(x)
                    shape = tuple(x.shape)
                    if x.numel() == 0:
                        print(f"[{name}] shape={shape}, EMPTY – skipping min/max")
                        return
                    # .amin/.amax read more cleanly than float(x.min())
                    print(f"[{name}] shape={shape}, min={x.amin().item():.3g}, max={x.amax().item():.3g}")


                _desc("TRAIN", train_images)
                _desc("VALID", valid_images)
                _desc("TEST",  test_images)

                dataset_sizes[fold] = [max(2, int(len(train_images) * p)) for p in dataset_portions]

                train_labels = relabel(train_labels)
                valid_labels = relabel(valid_labels) 
                
                debug_split_parity(train_labels, train_fns, "TRAIN")
                debug_split_parity(valid_labels, valid_fns, "VALID")
                debug_split_parity(test_labels,  test_fns,  "TEST")
                
                check_tensor(f"train_images for version: {ver_key}:", train_images)

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
                        train_images = generated_images.to(DEVICE, non_blocking=True) # Non_blocking makes things faster by overlapping data transfer and computation
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
                    cls_mask = (as_index_labels(train_labels) == (cls - min(galaxy_classes))) if train_labels.ndim > 1 else (train_labels == cls)
                    cls_images = train_images[cls_mask]
                    check_tensor(f"Generated images for class {cls} with model {gen_model_name}", cls_images)
                train_labels = relabel(train_labels)
                        
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
                if train_images == None:
                    raise ValueError("train_images is None, cannot compute scattering coefficients")
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
                    if train_images == None:
                        raise ValueError("train_images is None, cannot compute scattering coefficients")

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
            print(f"Fold {fold}, training on {len(train_images)} images, validating on {len(valid_images)} images.")
            
            if fold == folds[0] and SHOWIMGS and downsample_size == (1, 128, 128):
                print(f"Plotting sanity-check images for fold {fold} with {len(galaxy_classes)} classes: {galaxy_classes}")
                
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
                        save_path=f"./{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_histogram{_versions_to_load}.png"
                    )

                    plot_background_histogram(
                        train_images_cls1.cpu(),        # shape (936, 1, 128, 128)
                        train_images_cls2.cpu(),        # shape (720, 1, 128, 128)
                        img_shape=(1, 128, 128),
                        title="Background histograms",
                        save_path=f"./{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_background_hist{_versions_to_load}.png"
                    )

                    for cls in galaxy_classes:
                        orig_imgs = train_images[train_labels == (cls - min(galaxy_classes))][:36]
                        test_imgs = test_images[test_labels == (cls - min(galaxy_classes))][:36]
                                    
                        plot_image_grid(
                            orig_imgs.cpu(),
                            num_images=36,
                            save_path=f"./{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_{cls}_train_grid{_versions_to_load}.png"
                        )
                        plot_image_grid(
                            test_imgs.cpu(),
                            num_images=36,
                            save_path=f"./{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_{cls}_test_grid{_versions_to_load}.png"
                        )
                        
                        if lambda_generate not in [0, 8]:
                            gen_imgs = generated_by_class[cls][:36]

                            plot_image_grid(
                                gen_imgs,
                                num_images=36,
                                save_path=f"./{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_{cls}_generated_grid{_versions_to_load}.png"
                            )
                            plot_histograms(
                                gen_imgs,
                                orig_imgs.cpu(),
                                title1="Generated Images",
                                title2="Train Images",
                                save_path=f"./{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_{cls}_histogram{_versions_to_load}.png"
                            )
                            plot_background_histogram(
                                orig_imgs,
                                gen_imgs,
                                img_shape=(1, 128, 128),
                                title="Background histograms",
                                save_path=f"./{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_{cls}_background_hist{_versions_to_load}.png")
            
            if MULTILABEL:
                pos_counts = train_labels.sum(dim=0)                       # [2]
                neg_counts = train_labels.shape[0] - pos_counts            # [2]
                pos_counts = torch.clamp(pos_counts, min=1.0)
                pos_weight = (neg_counts / pos_counts).to(DEVICE)          # [2]
                weights = None
            else:
                if USE_CLASS_WEIGHTS:
                    unique, counts = np.unique(train_labels.cpu().numpy(), return_counts=True)
                    total_count = sum(counts)
                    class_weights = {i: total_count / count for i, count in zip(unique, counts)}
                    weights = torch.tensor([class_weights.get(i, 1.0) for i in range(len(galaxy_classes))],
                                        dtype=torch.float, device=DEVICE)
                else:
                    weights = None

            if fold in [0, 5] and SHOWIMGS:
                imgs = train_images.detach().cpu().numpy()
                lbls = (train_labels + min(galaxy_classes)).detach().cpu().numpy() # This is to match the original class labels
                plot_images_by_class(imgs, labels=lbls, num_images=5, save_path=f"./{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_example_train_data{_versions_to_load}.png")
            
            # Prepare input data
            mock_tensor = torch.zeros_like(train_images)
            valid_mock_tensor = torch.zeros_like(valid_images)
            if classifier in ['ScatterNet', 'ScatterResNet', 'ScatterSqueezeNet', 'ScatterSqueezeNet2', 'DISSN']:
                if train_images == None:
                    raise ValueError("train_images is None, cannot compute scattering coefficients")
                train_images = fold_T_axis(train_images)
                valid_images = fold_T_axis(valid_images)

                if classifier == 'DISSN':
                    vnames = [str(v).lower() for v in (_versions_to_load or ['raw'])]
                    assert 'raw' in vnames and 't50kpc' in vnames, \
                        f"DISSN requires versions to include 'raw' and 'T50kpc', got: {vnames}"

                    idx_raw = vnames.index('raw')
                    idx_t50 = vnames.index('t50kpc')

                    # images are RAW
                    raw_train = train_images[:, idx_raw:idx_raw+1]
                    raw_valid = valid_images[:, idx_raw:idx_raw+1]

                    # scattering on T50kpc only
                    t50_train = train_images[:, idx_t50:idx_t50+1].contiguous()
                    t50_valid = valid_images[:, idx_t50:idx_t50+1].contiguous()

                    train_scat_coeffs = compute_scattering_coeffs(t50_train, scattering, batch_size=128, device="cpu")
                    valid_scat_coeffs = compute_scattering_coeffs(t50_valid, scattering, batch_size=128, device="cpu")

                    if train_scat_coeffs.dim() == 5:
                        train_scat_coeffs = train_scat_coeffs.flatten(start_dim=1, end_dim=2)
                        valid_scat_coeffs = valid_scat_coeffs.flatten(start_dim=1, end_dim=2)

                    # (optional) joint normalization for stability
                    all_scat = torch.cat([train_scat_coeffs, valid_scat_coeffs], dim=0)
                    if NORMALISESCS or NORMALISESCSTOPM:
                        if NORMALISESCSTOPM:
                            all_scat = normalise_images(all_scat, -1, 1)
                        else:
                            all_scat = normalise_images(all_scat, 0, 1)
                    train_scat_coeffs = all_scat[:len(train_scat_coeffs)]
                    valid_scat_coeffs = all_scat[len(train_scat_coeffs):]

                    scatdim = train_scat_coeffs.shape[1:]

                    # Build datasets (img=RAW, scat=scattering(T50kpc))
                    train_dataset = TensorDataset(raw_train, train_scat_coeffs, train_labels)
                    valid_dataset = TensorDataset(raw_valid, valid_scat_coeffs, valid_labels)

                    # also keep these around for model shape inference below
                    train_images = raw_train
                    valid_images = raw_valid

                else:
                    # original behavior for the other scatter-based classifiers
                    mock_train = torch.zeros_like(train_images)
                    mock_valid = torch.zeros_like(valid_images)
                    if train_images == None:
                        raise ValueError("train_images is None, cannot compute scattering coefficients")
                    train_scat_coeffs = compute_scattering_coeffs(train_images, scattering, batch_size=128, device="cpu")
                    valid_scat_coeffs = compute_scattering_coeffs(valid_images, scattering, batch_size=128, device="cpu")
                    if train_scat_coeffs.dim() == 5:
                        train_scat_coeffs = train_scat_coeffs.flatten(start_dim=1, end_dim=2)
                        valid_scat_coeffs = valid_scat_coeffs.flatten(start_dim=1, end_dim=2)
                    all_scat = torch.cat([train_scat_coeffs, valid_scat_coeffs], dim=0)
                    if NORMALISESCS or NORMALISESCSTOPM:
                        if NORMALISESCSTOPM:
                            all_scat = normalise_images(all_scat, -1, 1)
                        else:
                            all_scat = normalise_images(all_scat, 0, 1)
                    train_scat_coeffs, valid_scat_coeffs = all_scat[:len(train_scat_coeffs)], all_scat[len(train_scat_coeffs):]
                    scatdim = train_scat_coeffs.shape[1:]
                    if classifier in ['ScatterNet', 'ScatterResNet']:
                        train_dataset = TensorDataset(mock_train, train_scat_coeffs, train_labels)
                        valid_dataset = TensorDataset(mock_valid, valid_scat_coeffs, valid_labels)
                    else:
                        train_dataset = TensorDataset(train_images, train_scat_coeffs, train_labels)
                        valid_dataset  = TensorDataset(valid_images, valid_scat_coeffs, valid_labels)

            else:
                if train_images.dim() == 5:
                    train_images = fold_T_axis(train_images)   # [B,T,1,H,W] -> [B,T,H,W]
                    valid_images = fold_T_axis(valid_images)
                    # test_images was folded earlier
                for x,name in [(train_images,"train"), (valid_images,"valid")]:
                    assert x.dim() == 4, f"{name}_images should be [B,C,H,W], got {tuple(x.shape)}"
                mock_train = torch.zeros_like(train_images)
                mock_valid = torch.zeros_like(valid_images)
                if classifier == "DICSN":
                    idx_v1 = 0
                    idx_v2 = 1
                    v1_train = train_images[:, idx_v1:idx_v1+1]
                    v2_train = train_images[:, idx_v2:idx_v2+1]
                    v1_valid = valid_images[:, idx_v1:idx_v1+1]
                    v2_valid = valid_images[:, idx_v2:idx_v2+1]
                    train_dataset = TensorDataset(v1_train, v2_train, train_labels)
                    valid_dataset = TensorDataset(v1_valid, v2_valid, valid_labels)
                else:
                    train_dataset = TensorDataset(train_images, mock_train, train_labels)
                    valid_dataset = TensorDataset(valid_images, mock_valid, valid_labels)


            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate, drop_last=False)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate, drop_last=False)

            if SHOWIMGS and lambda_generate not in [0, 8]: 
                if classifier in ['TinyCNN', 'SCNN', 'CNNSqueezeNet', 'Rustige', 'ScatterSqueezeNet', 'ScatterSqueezeNet2', 'Binary']:
                    #save_images_tensorboard(generated_images[:36], save_path=f"./{gen_model_name}_{galaxy_classes}_generated.png", nrow=6)
                    plot_histograms(pristine_train_images, valid_images, title1="Train images", title2="Valid images", imgs3=generated_images, imgs4=test_images, title3='Generated images', title4='Test images', save_path=f"./{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_{_versions_to_load}_histograms{_versions_to_load}.png")

            
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
            elif classifier == "DICSN":
                H, W = valid_images.shape[-2], valid_images.shape[-1]
                models = {"DICSN": {"model": DualInputConvolutionalSqueezeNet(input_shape=(1, H, W), num_classes=num_classes).to(DEVICE)}}
            elif classifier == "DISSN":
                models = {"DISSN": {"model": DISSN(img_shape=tuple(valid_images.shape[1:]), scat_shape=scatdim, num_classes=num_classes).to(DEVICE)}}
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
            
            if MULTILABEL:
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                if weights is not None:
                    print(f"Using class weights: {weights}")
                    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
                else:
                    print("No class weighting")
                    criterion = nn.BCEWithLogitsLoss() if len(galaxy_classes)==2 else nn.CrossEntropyLoss()

                

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
                        subset_train_loader = DataLoader(subset_train_dataset, batch_size=eff_bs, shuffle=False, num_workers=0, collate_fn=custom_collate, drop_last=True)

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
                                    elif classifier in ["ScatterSqueezeNet", "ScatterSqueezeNet2", "DICSN", "DISSN"]:
                                        logits = model(images, scat)
                                    else:
                                        logits = model(images)

                                    logits = collapse_logits(logits, num_classes, MULTILABEL)

                                    #labels = labels.long()
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

                            with torch.inference_mode(): # Validate on validation data
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
                                        elif classifier in ["ScatterSqueezeNet", "ScatterSqueezeNet2", "DICSN", "DISSN"]:
                                            logits = model(images, scat)
                                        else:
                                            logits = model(images)
                                            
                                    if galaxy_classes == [52, 53]:
                                        # If a model ever returns [B, C, H, W], keep your collapse:
                                        if logits.ndim == 4:
                                            logits = torch.nn.functional.adaptive_avg_pool2d(logits, (1, 1)).squeeze(-1).squeeze(-1)
                                        elif logits.ndim == 3:
                                            logits = logits.mean(dim=-1)

                                        # For BCE: labels must be float and same shape as logits
                                        labels = labels.float()

                                        
                                    else:

                                        # Collapse spatial maps to [B, C] if the model returns [B, C, H, W]
                                        if logits.ndim == 4:
                                            logits = torch.nn.functional.adaptive_avg_pool2d(logits, (1, 1)).squeeze(-1).squeeze(-1)
                                        elif logits.ndim == 3:  # rare: [B, C, H]
                                            logits = logits.mean(dim=-1)

                                        # Keep binary 2-logit shape for CE
                                        if logits.ndim == 1:
                                            logits = logits.unsqueeze(1)
                                        if logits.shape[1] == 1 and num_classes == 2 and galaxy_classes != [52, 53]:
                                            logits = torch.cat([-logits, logits], dim=1)

                                    #labels = labels.long()
                                    loss = criterion(logits, labels)
                                        
                                    # inside the training loop, just before loss = criterion(outputs, labels)
                                    if galaxy_classes != [52, 53]:
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
                        with torch.inference_mode(): # Evaluate on test data
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
                                    elif classifier in ["ScatterSqueezeNet", "ScatterSqueezeNet2", "DICSN", "DISSN"]:
                                        logits = model(images, scat)
                                    else:
                                        logits = model(images)
                                        
                                if MULTILABEL:
                                    probs = torch.sigmoid(logits).cpu().numpy()           # [B,2]
                                    preds = (probs >= THRESHOLD).astype(int)              # [B,2]
                                    trues = labels.cpu().numpy().astype(int)              # [B,2]
                                    all_pred_probs[key].extend(probs)
                                    all_pred_labels[key].extend(preds)
                                    all_true_labels[key].extend(trues)
                                else:
                                    pred_probs = torch.softmax(logits, dim=1).cpu().numpy()
                                    true_labels = labels.cpu().numpy()
                                    pred_labels = np.argmax(pred_probs, axis=1)
                                    all_pred_probs[key].extend(pred_probs)
                                    all_pred_labels[key].extend(pred_labels)
                                    all_true_labels[key].extend(true_labels)
                                    
                                # WITH these lines
                                if SHOWIMGS and experiment == num_experiments - 1:
                                    if MULTILABEL:
                                        batch_pred = preds          # shape [B, 2]
                                        batch_true = trues          # shape [B, 2]
                                        mask = (batch_pred != batch_true).any(axis=1)
                                    else:
                                        batch_pred = pred_labels    # shape [B]
                                        batch_true = true_labels    # shape [B]
                                        mask = batch_pred != batch_true

                                    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=images.device)
                                    mis_images.append(images.detach().cpu()[mask_t.cpu()])
                                    mis_trues.append(batch_true[mask])
                                    mis_preds.append(batch_pred[mask])
                                            
                            # --- metrics ---
                            y_true = np.array(all_true_labels[key])
                            y_pred = np.array(all_pred_labels[key])
                            accuracy, precision, recall, f1 = compute_classification_metrics(y_true, y_pred, multilabel=MULTILABEL, num_classes=num_classes)
                            update_metrics(metrics, gen_model_name, subset_size, fold, experiment, lr, reg, accuracy, precision, recall, f1, lambda_generate, crop_size, downsample_size, ver_key)
                            print(f"Fold {fold}, Experiment {experiment}, Subset Size {subset_size}, Classifier {classifier_name}, "
                                f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                                f"Recall: {recall:.4f}, F1 Score: {f1:.4f}")

                            if SHOWIMGS and mis_images and experiment == num_experiments - 1:
                                mis_images = torch.cat(mis_images, dim=0)[:36]
                                mis_trues  = np.concatenate(mis_trues)[:36]
                                mis_preds  = np.concatenate(mis_preds)[:36]
                                
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

                                out_path = f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_misclassified{_versions_to_load}.png"
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
                with torch.inference_mode():
                    for images, scat, _rest in test_loader:
                        if EXTRAVARS:
                            meta, labels = _rest
                            meta = meta.to(DEVICE)
                        else:
                            labels = _rest
                        images = images.to(DEVICE, non_blocking=True)
                        scat   = scat.to(DEVICE, non_blocking=True)
                        labels = labels.to(DEVICE, non_blocking=True)

                        
                        if classifier == "DANN":
                            class_logits, _ = model(images, alpha=1.0)
                            outputs = class_logits.cpu().detach().numpy()
                        elif classifier in ["ScatterNet", "ScatterResNet"]:
                            outputs = model(scat).cpu().detach().numpy()
                        elif classifier in ["ScatterSqueezeNet", "ScatterSqueezeNet2", "DICSN", "DISSN"]:
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

