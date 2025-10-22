import torch, math, time, random, re, sys, os, glob, pickle, itertools, matplotlib
import numpy as np
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
from tqdm import tqdm
from torch.optim import AdamW
from itertools import product
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

print("Running script 4.1 when busy Latest version with seed", SEED)

#Update the classifiers so that they are okay for just two classes
#EDIT: Allow for max_num_galaxies as None

###############################################
################ CONFIGURATION ################
###############################################

classes = get_classes()
galaxy_classes = [50, 51]  # Classes to classify
max_num_galaxies = 1000000  # Upper limit for the all-classes combined training data before classical augmentation
dataset_portions = [1]  # Portions of complete dataset for the accuracy vs dataset size
J, L, order = 2, 12, 2  # Scatter transform parameters
classifier = ["TinyCNN", # Very Simple CNN
              "Rustige", # Simple CNN from Rustige et al. 2023, https://github.com/floriangriese/wGAN-supported-augmentation/blob/main/src/Classifiers/SimpleClassifiers/Classifiers.py
              "SCNN", # Simple CNN similar to Rustige's
              "CNNSqueezeNet", # SCNN with Squeeze-and-Excitation blocks
              "DualCNNSqueezeNet", # Dual CNN with Squeeze-and-Excitation blocks
              "CloudNet", # From https://github.com/SorourMo/Cloud-Net-A-semantic-segmentation-CNN-for-cloud-detection/tree/master
              "DANN", # Domain-Adversarial Neural Network
              "ScatterNet", "ScatterSqueezeNet", "ScatterSqueezeNet2",
              "Binary", "ScatterResNet"][-4]
gen_model_names = ['DDPM'] #['ST', 'DDPM', 'wGAN', 'GAN', 'Dual', 'CNN', 'STMLP', 'lavgSTMLP', 'ldiffSTMLP'] # Specify the generative model_name
label_smoothing = 0  # Label smoothing for the classifier
num_experiments = 100
learning_rates = [1e-3]  # Learning rates
regularization_params = [1e-4]  # Regularisation parameters
percentile_lo = 60 # Percentile stretch lower bound
percentile_hi = 99  # Percentile stretch upper bound
versions = ['rt50kpc']  # any mix of loadable and runtime-tapered planes. 'rt50' or 'rt100' for tapering. Square brackets for stacking
folds = [5] # 0-4 for 5-fold cross validation, 5 for only one training
lambda_values = [0]  # Ratio between generated images and original images per class. 8 is reserfved for TRAINONGENERATED
num_epochs_cuda = 200
num_epochs_cpu = 100

FLUX_CLIPPING = False  # Clip the flux of the images
STRETCH = True  # Stretch the images with mathematical morphology
USE_GLOBAL_NORMALISATION = True           # single on/off switch . False - image-by-image normalisation 
GLOBAL_NORM_MODE = "percentile"           # "percentile" or "flux"
ES, patience = True, 10  # Use early stopping
SCHEDULER = False  # Use a learning rate scheduler
SHOWIMGS = True  # Show some generated images for each class (Tool for control)

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

#####################################################################################
########################### AUTOMATIC CONFIGURATION #################################
#####################################################################################
    
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

if TRAINONGENERATED:
    lambda_values = [8]  # To identify and distinguish TRAINONGENERATED from other runs
    print("Using generated data for testing.")

if torch.cuda.is_available():
    DEVICE = "cuda"
    num_epochs = num_epochs_cuda
    print(f"CUDA is available. Setting epochs to {num_epochs}.")
else:
    DEVICE = "cpu"
    num_epochs = num_epochs_cpu
    print(f"CUDA is not available. Setting epochs to {num_epochs}.")
    
ARCSEC = np.deg2rad(1/3600.0)
PSZ2_ROOT = "/users/mbredber/scratch/data/PSZ2"  # FITS root used below

# drive augmentation/filenames solely from presence of rt*
LATE_AUG = bool(_gen_versions) # True if any(v.startswith('rt') for v in _gen_versions)
PRINTFILENAMES = bool(_gen_versions)
EXTRAVARS = False  # Use extra features (redshift, mass, size) for the classifier. Will automatically be true if test_meta is not None.

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

########################################################################
##################### HELPER FUNCTIONS #################################
########################################################################


def _to_base_name(fn):
    return Path(str(fn)).stem.split('T', 1)[0]

def _first_recursive(pattern: str):
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[0] if hits else None

def plot_first_rows_by_source(images, filenames, versions, out_path, n_show=10):
    """
    Plot the first N rows titled by source name. If images are [B,T,1,H,W] or [B,T,H,W],
    show 2 columns (left/right = first/second plane). If single version, plot 1 column.
    """

    if isinstance(images, (list, tuple)):
        images = torch.stack([torch.as_tensor(x) for x in images], dim=0)

    # Normalize shape to convenient form
    if images.dim() == 5:                 # [B, T, 1, H, W]
        images = images.flatten(2, 3)     # [B, T, H, W]
    elif images.dim() == 4:               # [B, C, H, W]
        pass
    else:
        raise ValueError(f"Unsupported images ndim={images.ndim}")

    B = images.shape[0]
    n_show = min(n_show, B)

    # Row titles from filenames (strip trailing T*kpc or T*kpcSUB)
    names = (filenames[:n_show] if filenames else [f"idx_{i}" for i in range(n_show)])
    def _src_name(s):
        b = os.path.splitext(os.path.basename(str(s)))[0]
        return re.sub(r'(?:T\d+kpc(?:SUB)?)$', '', b)
    row_titles = [_src_name(s) for s in names]

    is_two_cols = images.shape[1] >= 2      # have at least two planes (e.g., RT/T)
    if is_two_cols:
        fig, axes = plt.subplots(n_show, 2, figsize=(5.4, 2.6*n_show), constrained_layout=True)
        if n_show == 1:
            axes = np.array([axes])
        for i in range(n_show):
            left  = images[i, 0].detach().cpu().numpy()
            right = images[i, 1].detach().cpu().numpy()
            axes[i, 0].imshow(left, cmap="viridis", origin="lower")
            axes[i, 0].set_title(f"{row_titles[i]} — {versions[0] if isinstance(versions,(list,tuple)) else 'v0'}")
            axes[i, 0].axis('off')
            axes[i, 1].imshow(right, cmap="viridis", origin="lower")
            axes[i, 1].set_title(f"{row_titles[i]} — {versions[1] if isinstance(versions,(list,tuple)) else 'v1'}")
            axes[i, 1].axis('off')
    else:
        fig, axes = plt.subplots(n_show, 1, figsize=(2.7, 2.6*n_show), constrained_layout=True)
        if n_show == 1:
            axes = [axes]
        for i in range(n_show):
            img = images[i, 0].detach().cpu().numpy()   # first/only channel
            ax = axes[i]
            ax.imshow(img, cmap="viridis", origin="lower")
            ax.set_title(f"{row_titles[i]} — {versions if not isinstance(versions,(list,tuple)) else versions[0]}")
            ax.axis('off')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[quicklook] wrote {out_path}")

def _first(pattern: str):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None

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

def permute_like_old(x, perm, SEED=SEED):
    if x is None: return None
    idx = perm.cpu().tolist()
    if isinstance(x, torch.Tensor): return x[perm]
    if isinstance(x, np.ndarray):   return x[idx]
    if isinstance(x, (list, tuple)): return [x[i] for i in idx]
    return x

def permute_like(x, perm):
    if x is None:
        return None

    # Torch tensors: deterministic gather via index_select and device-safe indices
    if isinstance(x, torch.Tensor):
        idx = perm.to(device=x.device, dtype=torch.long)
        return x.index_select(0, idx)

    # Prepare a plain integer index array once for non-tensor cases
    if isinstance(perm, torch.Tensor):
        idx = perm.detach().cpu().to(torch.long).tolist()
    elif isinstance(perm, np.ndarray):
        idx = [int(i) for i in perm.tolist()]
    else:
        idx = [int(i) for i in perm]

    if isinstance(x, np.ndarray):
        return x[np.asarray(idx, dtype=np.int64)]
    if isinstance(x, list):
        return [x[i] for i in idx]
    if isinstance(x, tuple):
        return tuple(x[i] for i in idx)
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

def _per_image_percentile_stretch(x2d, lo=60, hi=95):
    t = torch.as_tensor(x2d, dtype=torch.float32)
    pl = torch.quantile(t.reshape(-1), lo/100.0)
    ph = torch.quantile(t.reshape(-1), hi/100.0)
    y = (t - pl) / (ph - pl + 1e-6)
    return y.clamp(0, 1)

def as_index_labels(y: torch.Tensor) -> torch.Tensor:
    return y.argmax(dim=1) if y.ndim > 1 else y

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

def _name_base_from_fn(fn):
    stem = Path(str(fn)).stem
    return stem.split('T', 1)[0]

def _z_from_meta(name):                 # base like "PSZ2G192.18+56.12"
    return CLUSTER_META.get(name)

# put near your other helpers
def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
    """Make images [B,C,H,W]. If [B,T,1,H,W], fold T into C."""
    if x is None:
        return x
    if x.dim() == 5:
        # [B, T, 1, H, W]  ->  [B, T, H, W] -> treat T as channels
        return x.flatten(1, 2)  # fold_T_axis does the same; this is inline & fast
    if x.dim() == 3:
        return x.unsqueeze(1)
    return x  # already [B,C,H,W]

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
    metrics.setdefault(f"{key_base}_accuracy", [])
    metrics.setdefault(f"{key_base}_precision", [])
    metrics.setdefault(f"{key_base}_recall", [])
    metrics.setdefault(f"{key_base}_f1_score", [])

    metrics[f"{key_base}_f1_score"]   = []

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
    metrics.setdefault(f"{key_base}_accuracy", []).append(accuracy)
    metrics.setdefault(f"{key_base}_precision", []).append(precision)
    metrics.setdefault(f"{key_base}_recall", []).append(recall)
    metrics.setdefault(f"{key_base}_f1_score", []).append(f1)

    
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

all_true_labels = {}
all_pred_labels = {}
training_times = {}
all_pred_probs = {}
history = {} 
dataset_sizes = {}


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
                FLUX_CLIPPING=FLUX_CLIPPING,
                STRETCH=STRETCH,
                percentile_lo=percentile_lo,  # Percentile stretch lower bound
                percentile_hi=percentile_hi,  # Percentile stretch upper bound
                AUGMENT=not LATE_AUG,
                NORMALISE=NORMALISEIMGS,
                NORMALISETOPM=NORMALISEIMGSTOPM,
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
        if REPLACE_WITH_RT:
            test_images = test_images[:, 1:2]  # keep only the runtime-tapered plane
            print(f"[TEST] after replacing with RT: {test_images.size(0)} images")

        if LATE_AUG:
            test_images, test_labels, test_fns = late_augment(test_images, test_labels, test_fns)
            print(f"[TEST] after late augmentation: {test_images.size(0)} images")


    # ——— Data sanity checks ———
    for i, cls in enumerate(galaxy_classes):
        cls_mask = (train_labels == i)
        cls_images = train_images[cls_mask]
        check_tensor(f"Train images for class {cls} (idx={i})", cls_images)
        cls_mask = (test_labels == i)
        cls_images = test_images[cls_mask]
        check_tensor(f"Test images for class {cls} (idx={i})", cls_images)

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
                FLUX_CLIPPING=FLUX_CLIPPING,
                STRETCH=STRETCH,
                percentile_lo=percentile_lo,
                percentile_hi=percentile_hi,
                AUGMENT=not LATE_AUG,
                NORMALISE=NORMALISEIMGS,
                NORMALISETOPM=NORMALISEIMGSTOPM,
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
                    
                    train_images, train_labels, train_fns, info_tr = _append_rt_versions(train_images, train_fns, _gen_versions, labels=train_labels)
                    valid_images, valid_labels, valid_fns, info_va = _append_rt_versions(valid_images, valid_fns, _gen_versions, labels=valid_labels)
                    print(f"[TRAIN] dropped {info_tr['removed_total']} (kept {info_tr['kept']}/{info_tr['initial']})")
                    print(f"[VALID] dropped {info_va['removed_total']} (kept {info_va['kept']}/{info_va['initial']})")
                    
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
                    save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_histogram.pdf"
                )

                plot_background_histogram(
                    train_images_cls1.cpu(),        # shape (936, 1, 128, 128)
                    train_images_cls2.cpu(),        # shape (720, 1, 128, 128)
                    img_shape=(1, 128, 128),
                    title="Background histograms",
                    save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_background_hist.pdf"
                )

                for cls in galaxy_classes:
                    orig_imgs = train_images[train_labels == (cls - min(galaxy_classes))][:36]
                    test_imgs = test_images[test_labels == (cls - min(galaxy_classes))][:36]
                                
                    plot_image_grid(
                        orig_imgs.cpu(),
                        num_images=36,
                        save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_{cls}_train_grid.pdf"
                    )
                    plot_image_grid(
                        test_imgs.cpu(),
                        num_images=36,
                        save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_{cls}_test_grid.pdf"
                    )
                    tag_to_desc = { d["tag"]: d["description"] for d in get_classes() }
                    
                    # helper to plot summed‐intensity histogram with dynamic labels
                    def plot_intensity_histogram(tensor1, tensor2, label1, label2, save_path, bins=30):
                        vals1 = tensor1.sum(dim=(1,2,3)).cpu().numpy()
                        vals2 = tensor2.sum(dim=(1,2,3)).cpu().numpy()
                        plt.figure(figsize=(10,5))
                        plt.hist(vals1, bins=bins, alpha=0.6, label=label1, color='C1')
                        plt.hist(vals2, bins=bins, alpha=0.6, label=label2, color='C0')
                        plt.xlabel('Total Intensity')
                        plt.ylabel('Count')
                        plt.title(f'Total Intensity per Image: {label1} vs {label2}')
                        plt.legend()
                        plt.savefig(save_path)
                        plt.close()

                    # now call it using your class descriptions
                    plot_intensity_histogram(
                        train_images_cls1.cpu(),
                        train_images_cls2.cpu(),
                        label1=tag_to_desc[classes[galaxy_classes[0]]['tag']],
                        label2=tag_to_desc[classes[galaxy_classes[1]]['tag']],
                        save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_summed_intensity_histogram.pdf"
                    )
                    
        
                    if lambda_generate not in [0, 8]:
                        gen_imgs = generated_by_class[cls][:36]

                        plot_image_grid(
                            gen_imgs,
                            num_images=36,
                            save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_{cls}_generated_grid.pdf"
                        )
                        plot_histograms(
                            gen_imgs,
                            orig_imgs.cpu(),
                            title1="Generated Images",
                            title2="Train Images",
                            save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_{cls}_histogram.pdf"
                        )
                        plot_background_histogram(
                            orig_imgs,
                            gen_imgs,
                            img_shape=(1, 128, 128),
                            title="Background histograms",
                            save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_{cls}_background_hist.pdf")
        
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
            mock_train = torch.zeros_like(train_images)
            mock_valid = torch.zeros_like(valid_images)
            train_dataset = TensorDataset(train_images, mock_train, train_labels)
            valid_dataset = TensorDataset(valid_images, mock_valid, valid_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=False)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=False)

        if SHOWIMGS and lambda_generate not in [0, 8]: 
            if classifier in ['TinyCNN', 'SCNN', 'CNNSqueezeNet', 'Rustige', 'ScatterSqueezeNet', 'ScatterSqueezeNet2', 'Binary']:
                #save_images_tensorboard(generated_images[:36], save_path=f"./classifier/{gen_model_name}_{galaxy_classes}_generated.png", nrow=6)
                plot_histograms(pristine_train_images, valid_images, title1="Train images", title2="Valid images", imgs3=generated_images, imgs4=test_images, title3='Generated images', title4='Test images', save_path=f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_histograms.pdf")

        
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

        for classifier_name, model_details in models.items():
            if FIRSTTIME:
                print(f"Summary for {classifier_name}:")
                if classifier == "ScatterNet":
                    summary(model_details["model"], input_size=(int(np.prod(scatdim)),), device=DEVICE)
                elif classifier == "ScatterResNet":
                    summary(model_details["model"], input_size=scatdim, device=DEVICE)
                elif classifier == "ScatterSqueezeNet":
                    summary(model_details["model"], input_size=[valid_images.shape[1:], scatdim])
                elif classifier == "ScatterSqueezeNet2":
                    summary(model_details["model"], input_size=[valid_images.shape[1:], scatdim])
                else:
                    summary(model_details["model"], input_size=tuple(valid_images.shape[1:]), device=DEVICE)
            FIRSTTIME = False
            
   
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
                                    plt.savefig(f"./{galaxy_classes}_{classifier_name}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_confusion_matrix.pdf", dpi=150)
                                    plt.close()

                        accuracy = accuracy_score(all_true_labels[key], all_pred_labels[key])
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

                            out_path = f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_{dataset_sizes[folds[-1]][-1]}_misclassified.pdf"
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
                print(f"Fold {fold}, Subset Size {subset_size}, Classifier {classifier_name}, AVERAGE over {num_experiments} experiments — Accuracy: {mean_acc:.4f}, Precision: {mean_prec:.4f}, Recall: {mean_rec:.4f}, F1 Score: {mean_f1:.4f}")
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
    
    directory = './classifier/trained_models_filtered/' if FILTERED else './classifier/trained_models/'
    for fold, lr, reg, lambda_generate in param_combinations:
        for subset_size in dataset_sizes[fold]:
            for experiment in range(num_experiments):
                                
                metrics_save_path = f'{directory}{classifier}_{galaxy_classes}_{gen_model_name}_{subset_size}_{fold}_{experiment}_{lr}_{reg}_{lambda_generate}_metrics_data.pkl'

                # Build robust, per-setting summaries using empirical percentiles
                robust_summary = {}   # { base_key: {metric: {'n', 'p16','p50','p84','sigma68'} } }
                skip_keys = {"accuracy", "precision", "recall", "f1_score"}
                rows = []
                for key, values in metrics.items():
                    if key in skip_keys:
                        continue
                    if not isinstance(values, (list, tuple)) or len(values) == 0:
                        continue

                    vals = np.asarray(values, dtype=float)
                    p16, p50, p84 = np.percentile(vals, [16, 50, 84])
                    sigma68 = 0.5 * (p84 - p16)

                    base, metric_name = key.rsplit('_', 1)  # split "..._accuracy" → ("...", "accuracy")
                    robust_summary.setdefault(base, {})[metric_name] = {
                        "n": int(vals.size),
                        "p16": float(p16),
                        "p50": float(p50),   # median
                        "p84": float(p84),
                        "sigma68": float(sigma68)  # half-width of the central 68% interval
                    }
                    
                    # Histogram with percentile markers (bins span data range)
                    vmin, vmax = float(np.min(vals)), float(np.max(vals))
                    if vmin == vmax:  # guard against a degenerate run where all values are identical
                        eps = 1e-6
                        vmin, vmax = vmin - eps, vmax + eps
                    edges = np.linspace(vmin, vmax, 21)  # 20 bins across [min, max]

                    plt.figure(figsize=(5,3.2))
                    plt.hist(vals, bins=edges, edgecolor='none', alpha=0.8)
                    for x, style in [(p16, ':'), (p50, '-'), (p84, ':')]:
                        plt.axvline(x, linestyle=style)
                    plt.xlim(vmin, vmax)  # force axis to the observed range
                    plt.title(key)
                    plt.xlabel(metric_name)
                    plt.ylabel("count")
                    plt.tight_layout()

                    # save one file per metric so nothing gets overwritten
                    save_path_hist = (
                        f"./classifier/{galaxy_classes}_{classifier}_{gen_model_name}_"
                        f"{dataset_sizes[folds[-1]][-1]}_{metric_name}_histogram.pdf"
                    )
                    plt.savefig(save_path_hist, dpi=150)
                    plt.close()

                    rows.append({
                        "setting": base,
                        "metric": metric_name,
                        "n": int(vals.size),
                        "p16": float(p16),
                        "p50": float(p50),
                        "p84": float(p84),
                        "sigma68": float(sigma68)
                    })

                # Also write a tidy CSV so you can scan summaries quickly
                summary_csv = f'{directory}{classifier}_{galaxy_classes}_{gen_model_name}_percentile_summary.csv'
                pd.DataFrame(rows).to_csv(summary_csv, index=False)

                with open(metrics_save_path, 'wb') as f:
                    pickle.dump({
                        "models": models,
                        "history": history,
                        "metrics": metrics,                 # raw per-run values remain
                        "metric_colors": metric_colors,
                        "all_true_labels": all_true_labels,
                        "all_pred_labels": all_pred_labels,
                        "training_times": training_times,
                        "all_pred_probs": all_pred_probs,
                        "percentile_summary": robust_summary  # new robust summaries
                    }, f)
                    