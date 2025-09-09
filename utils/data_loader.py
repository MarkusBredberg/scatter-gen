import numpy as np
import h5py
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from utils.GAN_models import load_gan_generator
from utils.calc_tools import normalise_images, generate_from_noise, load_model, check_tensor
from firstgalaxydata import FIRSTGalaxyData
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.ndimage import label
import skimage
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import collections
from collections import Counter, defaultdict
import pandas as pd
from PIL import Image
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
import random
import math
import hashlib
import glob
import os

# For reproducibility
SEED = 42 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

######################################################################################################
################################### CONFIGURATION ####################################################
######################################################################################################

root_path =  '/users/mbredber/scratch/data/' #'/home/sysadmin/Scripts/data/'  #

######################################################################################################
################################### GENERAL STUFF ####################################################
######################################################################################################

import sys
sys.path.append(root_path)
#from MiraBest.MiraBest_F import MBFRConfident, MBFRUncertain, MBFRFull, MBRandom


def get_classes():
    return [
        # GALAXY10 size: 3x256x256
        {"tag": 0, "length": 1081, "description": "Disturbed Galaxies"},
        {"tag": 1, "length": 1853, "description": "Merging Galaxies"},
        {"tag": 2, "length": 2645, "description": "Round Smooth Galaxies"},
        {"tag": 3, "length": 2027, "description": "In-between Round Smooth Galaxies"},
        {"tag": 4, "length": 334, "description": "Cigar Shaped Smooth Galaxies"},
        {"tag": 5, "length": 2043, "description": "Barred Spiral Galaxies"},
        {"tag": 6, "length": 1829, "description": "Unbarred Tight Spiral Galaxies"},
        {"tag": 7, "length": 2628, "description": "Unbarred Loose Spiral Galaxies"},
        {"tag": 8, "length": 1423, "description": "Edge-on Galaxies without Bulge"},
        {"tag": 9, "length": 1873, "description": "Edge-on Galaxies with Bulge"},
        # FIRST size: 300x300
        {"tag": 10, "length": 395, "description": "FRI"},
        {"tag": 11, "length": 824, "description": "FRII"},
        {"tag": 12, "length": 291, "description": "Compact"},
        {"tag": 13, "length": 248, "description": "Bent"},
        # MIRABEST size: 150x150
        {"tag": 14, "length": 397, "description": "Confidently classified FRIs"},
        {"tag": 15, "length": 436, "description": "Confidently classified FRIIs"},
        {"tag": 16, "length": 591, "description": "FRI"},
        {"tag": 17, "length": 633, "description": "FRII"},
        # MNIST size: 1x28x28
        {"tag": 18, "length": 60000, "description": "All Digits"},
        {"tag": 19, "length": 60000, "description": "All Digits"},
        {"tag": 20, "length": 6000, "description": "Digit Zero"},
        {"tag": 21, "length": 6000, "description": "Digit One"},
        {"tag": 22, "length": 6000, "description": "Digit Two"},
        {"tag": 23, "length": 6000, "description": "Digit Three"},
        {"tag": 24, "length": 6000, "description": "Digit Four"},
        {"tag": 25, "length": 6000, "description": "Digit Five"},
        {"tag": 26, "length": 6000, "description": "Digit Six"},
        {"tag": 27, "length": 6000, "description": "Digit Seven"},
        {"tag": 28, "length": 6000, "description": "Digit Eight"},
        {"tag": 29, "length": 6000, "description": "Digit Nine"},
        # Radio Galaxy Zoo size: 1x132x132
        {"tag": 31, "length": 10, "description": "1_1"},
        {"tag": 32, "length": 15, "description": "1_2"},
        {"tag": 33, "length": 20, "description": "1_3"},
        {"tag": 34, "length": 12, "description": "2_2"},
        {"tag": 35, "length": 18, "description": "2_3"},
        {"tag": 36, "length": 25, "description": "3_3"},
        # MGCLS 1x1600x1600
        {"tag": 40, "length": 122, "description": "DE"}, # Diffuse Emission (only 52 unique sources)
        {"tag": 41, "length": 90, "description": "NDE"}, # Only 46 unique sources
        {"tag": 42, "length": 13, "description": "RH"}, # Radio Halo
        {"tag": 43, "length": 14, "description": "RR"}, # Radio Relic
        {"tag": 44, "length": 1, "description": "mRH"}, # Mini Radio Halo
        {"tag": 45, "length": 1, "description": "Ph"}, # Phoenix
        {"tag": 46, "length": 4, "description": "cRH"}, # Candidate Radio Halo
        {"tag": 47, "length": 16, "description": "cRR"}, # Candidate Radio Relic
        {"tag": 48, "length": 7, "description": "cmRH"}, # Candidate Mini Radio Halo
        {"tag": 49, "length": 2, "description": "cPh"}, # Candidate Phoenix
        # PSZ2 4x369x369
        {"tag": 50, "length": 62, "description": "DE"}, # RR + RH
        {"tag": 51, "length": 114, "description": "NDE"}, # No Diffuse Emission
        {"tag": 52, "length": 53, "description": "RH"}, # Radio Halo
        {"tag": 53, "length": 20, "description": "RR"}, # Radio Relic (Only 8 unique sources)
        {"tag": 54, "length": 19, "description": "cRH"}, # Candidate Radio Halo
        {"tag": 55, "length": 6, "description": "cRR"}, # Candidate Radio Relic
        {"tag": 56, "length": 24, "description": "cDE"}, # candidate Diffuse Emission
        {"tag": 57, "length": 47, "description": "U"}, # Uncertain
        {"tag": 58, "length": 40, "description": "unclassified"} # Unclassified
    ]


########################################################################################################
####################################### LOAD ARTIFICIAL DATA ###########################################
########################################################################################################

def get_synthetic(
    gen_model_name,
    cls,
    num_generate = 1000,
    batch_size_generate = 500,
    galaxy_classes = [10, 11, 12, 13],
    img_shape=(1, 128, 128),
    FILTERGEN=False,
    CLIPDDPM=True,
    model_kwargs = {},
    fold=0,
    device='cpu'
):
    """
    Returns two lists: generated_images_list and generated_labels_list
    following the GAN, DDPM or VAE branch & optional duplicate‑filtering.
    """
    generated_images_list = []
    generated_labels_list = []
    if gen_model_name == 'DDPM':
        ddpm_file_map = {
            #10: "new_generated_fr1.npy",  # Generated by starting with noisy image
            #11: "new_generated_fr2.npy",
            #12: "new_generated_compact.npy",
            #13: "new_generated_bent.npy"
            10: "generated_fr1_8k.npy",   # Generated by starting with pure noise
            11: "generated_fr2_8k.npy",
            12: "generated_compact_8k.npy",
            13: "generated_bent_8k.npy"
        }
        ddpm_path = f"/users/mbredber/data/DDPM/{ddpm_file_map[cls]}"
        print(f"Loading DDPM data for class {cls} from {ddpm_path}")
        ddpm_data = np.load(ddpm_path)  # shape assumed (N, 128, 128)
        ddpm_data = torch.from_numpy(ddpm_data).unsqueeze(1).float()  # (N, 1, 128, 128)
        n_here = min(ddpm_data.shape[0], num_generate)  # Use as many samples as num_generate, or all if fewer exist
        ddpm_data = ddpm_data[:n_here]
        if CLIPDDPM: # Clip to [0,1] range
            ddpm_data = ddpm_data.clamp(0, 1)

        # Append to the lists (do not overwrite!)
        generated_images_list.append(ddpm_data)
        generated_labels_list.append(torch.ones(n_here, dtype=torch.long) * (cls - min(galaxy_classes)))
        
    elif gen_model_name in ['ST', 'wGAN', 'WGAN']:
        if gen_model_name == 'ST':
            #pattern = f"/users/mbredber/scratch/ST_generation/*100_{cls}.npy"
            pattern = f"/users/mbredber/scratch/ST_generation/1to1_*_{cls}.npy"
        else:
            pattern = f"/users/mbredber/data/wGAN_FIRST/generated_images_class_{cls}.npy"
        candidates = sorted(glob.glob(pattern))
        if len(candidates) == 0:
            raise FileNotFoundError(f"No ST file found matching pattern {pattern}")
        st_path = candidates[0]
        print(f"Loading data for class {cls} from {st_path}")
        st_data = np.load(st_path)  # shape assumed (N, 128, 128)
        st_data = torch.from_numpy(st_data).unsqueeze(1).float()  # (N, 1, 128, 128)
        n_here = min(st_data.shape[0], num_generate)  # Use as many samples as num_generate, or all if fewer exist
        st_data = st_data[:n_here]
        if gen_model_name in ['wGAN', 'WGAN']: # Normalise from 0, 255 to [0, 1]
            st_data = st_data / 255.0
    
        generated_images_list.append(st_data)
        generated_labels_list.append(torch.ones(n_here, dtype=torch.long) * (cls - min(galaxy_classes)))
        
    elif gen_model_name == 'NOISE': # To troubleshoot, add noise as generated data
        noise = torch.randn(num_generate, *img_shape)
        noise = noise.to(device)
        generated_images_list.append(noise)
        generated_labels_list.append(torch.ones(num_generate, dtype=torch.long) * (cls - min(galaxy_classes)))

    else:
        if gen_model_name == 'GAN':
            gan_path = (
                f"./GAN/generators/generator_"
                f"{model_kwargs['gan_type']}_ss{model_kwargs['gan_sample_size']}_cl{cls}_lrgen{model_kwargs['lr_gen']}_"
                f"lrdisc{model_kwargs['lr_disc']}_gl{model_kwargs['gan_gen_loss']}_dl{model_kwargs['gan_disc_loss']}_"
                f"ls{model_kwargs['gan_label_smoothing']}_ld{model_kwargs['gan_lambda_div']}_dv{model_kwargs['gan_data_version']}_"
                f"ep{model_kwargs['gan_epoch']}_f{fold}.pth"
            )
            print(f"Loading GAN data for class {cls} from {gan_path}")
            model = load_gan_generator(
                gan_path,
                latent_dim=model_kwargs['gan_latent_dim']
            ).to(device)
            latent_dim = model_kwargs['gan_latent_dim']
        else: # VAE
            subname = gen_model_name.split('-',1)[1]             # drop the "VAE-" prefix
            model_dir = f"/users/mbredber/scratch/generator/VAE_{cls}_{model_kwargs['VAE_train_size']}_{subname}_{fold}"
            path      = os.path.join(model_dir, "model.pth")
            model     = load_model(
                path,
                scatshape=model_kwargs['scatshape'],
                hidden_dim1=model_kwargs['hidden_dim1'],
                hidden_dim2=model_kwargs['hidden_dim2'],
                latent_dim=model_kwargs['vae_latent_dim'],
                num_classes=len(galaxy_classes)
            )
            latent_dim = model_kwargs['vae_latent_dim']
            
        model.eval()
        num_batches = (num_generate + batch_size_generate - 1) // batch_size_generate
        total_generated = 0

        for batch_idx in range(num_batches):
            current_batch = min(batch_size_generate, num_generate - batch_idx * batch_size_generate)
            total_generated += current_batch

            generated_labels = torch.ones(current_batch, dtype=torch.long) * (cls - min(galaxy_classes))
            with torch.no_grad():
                generated_batch = generate_from_noise(
                    model,
                    latent_dim=latent_dim,
                    num_samples=current_batch,
                    DEVICE=device)
            # Ensure the tensor is [batch_size, channels, H, W]
            if generated_batch.shape[0] != current_batch:
                generated_batch = generated_batch.permute(1, 0, 2, 3)
                
            if gen_model_name == 'GAN': 
                generated_batch = normalise_images(generated_batch, 0, 1) # Normalise from [-1, 1] to [0, 1]

            if FILTERGEN and cls != 12:
                total_unique = 0
                tol = 1e-3
                # --------------------------
                # (1) Remove duplicates *within* the new batch
                # --------------------------
                flat = generated_batch.view(current_batch, -1)  # flatten each image
                dists = torch.cdist(flat, flat, p=2)
                dists.fill_diagonal_(1e6)  # ignore self-comparisons
                unique_mask = torch.ones(current_batch, dtype=torch.bool, device=generated_batch.device)
                for i in range(current_batch):
                    if unique_mask[i]:
                        # Mark subsequent images that are too close as duplicates
                        unique_mask[i+1:] &= ~(dists[i, i+1:] < tol)

                batch_unique_images = generated_batch[unique_mask]
                batch_unique_labels = generated_labels[unique_mask]
                n_unique = len(batch_unique_images)

                # --------------------------
                # (2) Remove duplicates *across* previously accepted images
                # --------------------------
                if len(generated_images_list) > 0:
                    # Flatten the newly filtered batch
                    new_flat = batch_unique_images.view(n_unique, -1)

                    # Flatten everything we've already accepted
                    prev_accepted = torch.cat(generated_images_list, dim=0)  # shape: [N_accepted, C, H, W]
                    prev_flat = prev_accepted.view(prev_accepted.size(0), -1)

                    # Compare new vs. all accepted
                    cross_dists = torch.cdist(new_flat, prev_flat, p=2)
                    cross_unique_mask = torch.ones(n_unique, dtype=torch.bool, device=batch_unique_images.device)
                    for i in range(n_unique):
                        # If the new image is within tol of *any* accepted image, exclude it
                        if torch.any(cross_dists[i] < tol):
                            cross_unique_mask[i] = False

                    batch_unique_images = batch_unique_images[cross_unique_mask]
                    batch_unique_labels = batch_unique_labels[cross_unique_mask]
                    n_unique = len(batch_unique_images)

                # Now everything in batch_unique_* is unique vs. each other & prior batches
                total_unique += n_unique
                
                # --------------------------
                # (3) Augment the unique images
                # --------------------------
                
                print("Shape of batch_unique_images: ", batch_unique_images.shape)
                generated_batch, generated_labels = augment_images(batch_unique_images, batch_unique_labels) # Produces 8 versions of each image
                
                removal_fraction = (total_generated - total_unique) / total_generated * 100
                print(f"Class {cls}: Removed {removal_fraction:.2f}% of images "
                f"(Generated: {total_generated}, Unique: {total_unique})")

            generated_images_list.append(generated_batch)
            generated_labels_list.append(generated_labels)
        

    generated_images = torch.cat(generated_images_list, dim=0)
    generated_labels = torch.cat(generated_labels_list, dim=0)

    return generated_images, generated_labels


######################################################################################################
################################### DATA SELECTION FUNCTIONS #########################################
######################################################################################################

def percentile_stretch(x, lo=30, hi=99):
    """
    x: Tensor of shape (B, C, H, W)
    Returns: same shape, linearly rescaled so that the lo-th percentile → 0 and hi-th → 1
    """
    if len(x.shape) == 2:  # If x is 2D, add a channel dimension twice
        x = x.unsqueeze(0).unsqueeze(0)
    elif len(x.shape) == 3:  # If x is 3D, add a channel dimension
        x = x.unsqueeze(0)
    elif len(x.shape) != 4:  # If x is not 4D, raise an error
        raise ValueError("Input tensor x must be of shape (B, C, H, W) or (H, W). Now it is: ", x.shape)

    #flat_all = x.view(-1)
    flat_all = x.reshape(-1)
    p_low  = flat_all.quantile(lo/100)
    p_high = flat_all.quantile(hi/100)
    
    # reshape to broadcast over (B,C,H,W)
    p_low  = p_low.view(1,1,1,1)
    p_high = p_high.view(1,1,1,1)
    
    y = (x - p_low) / (p_high - p_low + 1e-6)
    return y.clamp(0, 1)



def old_percentile_stretch(x, lo=2, hi=98):
    """
    x: Tensor of shape (B, C, H, W) with values in [0,1]
    Returns: same shape, linearly rescaled so that the lo-th percentile → 0 and hi-th → 1
    """
    if len(x.shape) == 2:  # If x is 2D, add a channel dimension twice
        x = x.unsqueeze(0).unsqueeze(0)
    if len(x.shape) == 3:  # If x is 3D, add a channel dimension
        x = x.unsqueeze(0)
    elif len(x.shape) == 5:  # If x is 5D, squeeze the channel dimension
        x = x.squeeze(0)
    elif len(x.shape) == 6:  # If x is 6D, squeeze the channel dimension twice
        x = x.squeeze(0).squeeze(0)
    elif len(x.shape) != 4:  # If x is not 4D, raise an error
        raise ValueError("Input tensor x must be of shape (B, C, H, W) or (H, W). Now it is: ", x.shape)
    B, C, H, W = x.shape
    flat = x.view(B, C, -1)
    p_low  = flat.quantile(lo/100, dim=2, keepdim=True).unsqueeze(-1)
    p_high = flat.quantile(hi/100, dim=2, keepdim=True).unsqueeze(-1)
    y = (x - p_low) / (p_high - p_low + 1e-6)
    return y.clamp(0,1)


def asinh_stretch(x, alpha=10):
    return torch.asinh(alpha * x) / math.asinh(alpha)

def log_stretch(x, alpha=10):
    return torch.log1p(alpha * x) / math.log1p(alpha)
    
def add_highpass(x, alpha=100):
    # asinh stretch
    y = torch.asinh(alpha * x) / math.asinh(alpha)
    # low-pass + subtract
    lp = torch.nn.functional.avg_pool2d(y, kernel_size=15, stride=1, padding=7)
    hp = torch.clamp(y - lp, min=0.)
    # normalize hp
    mn = hp.amin(dim=(2,3), keepdim=True)
    mx = hp.amax(dim=(2,3), keepdim=True)
    hp = (hp - mn) / (mx - mn + 1e-6)
    # stack channels

    return torch.cat([y, hp], dim=1)

def gamma_stretch(x, γ=0.5): # x in [0,1]
    return x.pow(γ)

def _to_2d_for_imshow(x, how="first"):
    """
    Return a (H, W) numpy array suitable for plt.imshow from a tensor/ndarray.

    Accepts shapes like:
      (H, W)
      (C, H, W)            or (H, W, C)
      (B, C, H, W)         or (T, C, H, W)
      (B, T, C, H, W)

    Parameters
    ----------
    x : torch.Tensor or np.ndarray
        Image-like object.
    how : {"first","mean","max"}
        How to reduce non-spatial/extra axes (channels, time, batch).
    """
    import numpy as np
    import torch

    def _reduce(a, axis=0):
        if how == "mean":
            return a.mean(axis=axis)
        if how == "max":
            return a.max(axis=axis)
        # "first"
        return np.take(a, 0, axis=axis)

    # ---- convert to numpy float32 without altering values ----
    if isinstance(x, torch.Tensor):
        a = x.detach().cpu().float().numpy()
    else:
        a = np.asarray(x, dtype=np.float32)

    # ---- peel dimensions until we have (H, W) ----
    if a.ndim == 2:
        img = a

    elif a.ndim == 3:
        # Heuristic: channels-first if first dim is small (<=4) and last isn't;
        # channels-last if last dim is small (<=4) and first isn't.
        c_first = (a.shape[0] in (1, 2, 3, 4)) and (a.shape[-1] not in (1, 2, 3, 4))
        c_last  = (a.shape[-1] in (1, 2, 3, 4)) and (a.shape[0]  not in (1, 2, 3, 4))

        if c_first:
            # (C, H, W)
            img = a[0] if a.shape[0] == 1 else _reduce(a, axis=0)
        elif c_last:
            # (H, W, C)
            img = a[..., 0] if a.shape[-1] == 1 else _reduce(a, axis=-1)
        else:
            # Ambiguous; take first plane along the leading axis.
            img = _reduce(a, axis=0)

    elif a.ndim == 4:
        # Assume leading axis is batch/time → reduce then recurse.
        img = _to_2d_for_imshow(_reduce(a, axis=0), how=how)

    elif a.ndim == 5:
        # (B, T, C, H, W) → reduce B and T, then recurse.
        a = _reduce(a, axis=0)
        a = _reduce(a, axis=0)
        img = _to_2d_for_imshow(a, how=how)

    else:
        # Fallback: keep reducing the first axis until 2D.
        while a.ndim > 2:
            a = _reduce(a, axis=0)
        img = a

    # Ensure float32 ndarray
    return np.asarray(img, dtype=np.float32)


def scale_weaker_region(image, adjustment, initial_threshold=0.9, step=0.01):
    """
    Identify two distinct peaks in the image by thresholding, continue until they merge into one,
    and apply a scaling adjustment to the weaker peak. The image is then normalized so that the
    maximum pixel intensity is 1.
    
    Args:
        image (torch.Tensor or np.ndarray): Input image to be processed.
        adjustment (float): Scaling factor to adjust the weaker peak region.
        initial_threshold (float): Starting threshold value to identify two separate regions.
        step (float): Step size to decrease the threshold until two regions merge.
        
    Returns:
        np.ndarray: Processed image with the weaker peak region scaled and normalized.
    """
    # Convert the image to NumPy if it's a torch tensor
    image_np = image.squeeze().cpu().numpy() if isinstance(image, torch.Tensor) else image.copy()
    
    # Normalize the image to have a peak intensity of 1
    image_np = image_np / np.max(image_np)
    
    # Start with an initial threshold and gradually decrease it to find two regions
    threshold = initial_threshold
    labeled_image = None
    num_features = 0
    prev_labeled_image = None
    found_two_regions = False
    drop_below_threshold_count = 0  # Counter to avoid premature break

    while threshold > 0:
        binary_image = image_np > threshold
        labeled_image, num_features = skimage.measure.label(binary_image, return_num=True)
        
        # If we've detected two regions, mark that as found and save the state
        if num_features >= 2:
            found_two_regions = True
            prev_labeled_image = labeled_image  # Store the last state with two distinct regions
            drop_below_threshold_count = 0  # Reset the drop counter when two regions are found
        # If we've previously found two regions but now drop to one, start counting to ensure stability before breaking
        elif num_features < 2 and found_two_regions:
            drop_below_threshold_count += 1
            if drop_below_threshold_count >= 3:  # Allow a few checks to avoid breaking too early
                break
        
        threshold -= step

    # If we never found two distinct regions, exit and print a warning
    if not found_two_regions or prev_labeled_image is None:
        print("Unable to find two distinct regions; no adjustment applied.")
        return image_np

    
    # Identify the two regions based on the peak intensities in the last two-region state
    region_intensities = []
    for region_label in range(1, np.max(prev_labeled_image) + 1):
        region_mask = (prev_labeled_image == region_label)
        region_intensity = np.max(image_np[region_mask])
        region_intensities.append((region_intensity, region_label))
    
    # Sort the regions by intensity and identify the second brightest
    region_intensities.sort(reverse=True, key=lambda x: x[0])
    brightest_region_label = region_intensities[0][1]
    second_brightest_region_label = region_intensities[1][1]
    weaker_region_mask = (prev_labeled_image == second_brightest_region_label)
    
    image_np[weaker_region_mask] *= (1 + adjustment) # Adjust brightness weaker peak region
    image_np = image_np / np.max(image_np) # Normalise image
    
    return image_np


def filter_by_peak_intensity(images, labels, threshold=0.6, region_size=(64, 64)):
    removed_by_peak = []
    filtered_images_by_peak = []
    filtered_labels_by_peak = []
    
    for image, lbl in zip(images, labels):
        # Convert image to numpy for processing, removing channel dimension if present
        image_np = image.squeeze().numpy() if image.dim() == 3 else image.numpy()
        
        # Determine the center of the image and extract the central region
        center_x, center_y = image_np.shape[1] // 2, image_np.shape[0] // 2
        central_region = image_np[center_y - region_size[0] // 2:center_y + region_size[0] // 2,
                                  center_x - region_size[1] // 2:center_x + region_size[1] // 2]
                
        # Convert central region back to a torch tensor before using torch.max
        central_region_tensor = torch.tensor(central_region)
        max_intensity = torch.max(central_region_tensor).item()
    
        # Filter images based on peak intensity
        if max_intensity <= threshold:
            removed_by_peak.append((image, lbl))
        else:
            filtered_images_by_peak.append(image)
            filtered_labels_by_peak.append(lbl)
        
    return filtered_images_by_peak, filtered_labels_by_peak, removed_by_peak


def filter_images_with_edge_emission(images, crop_size=128, threshold=0.2):
    """
    Filter out images based on the max and sum of edge pixel values from a hypothetical 128x128 central crop.
    :param images: List of image tensors
    :return: List of filtered images and a list of removed images based on edge emission
    """
    filtered_images = []
    removed_by_edge = []
        
    for image in images:
        
        if image.dim() == 3 and image.shape[0] == 1:  # If the first dimension is 1 (single-channel grayscale)
            image = image.squeeze(0)
            
        height, width = image.shape[-2], image.shape[-1]
        start_y = (height - crop_size) // 2
        start_x = (width - crop_size) // 2
        end_y = start_y + crop_size
        end_x = start_x + crop_size
                
        # Extract the edges of the hypothetical cropped 128x128 region
        top_edge = image[start_y, start_x:end_x]
        bottom_edge = image[end_y - 1, start_x:end_x]
        left_edge = image[start_y:end_y, start_x]
        right_edge = image[start_y:end_y, end_x - 1]
        
        # Find the max pixel value along these edges
        top_edge_max = torch.max(top_edge)
        bottom_edge_max = torch.max(bottom_edge)
        left_edge_max = torch.max(left_edge)
        right_edge_max = torch.max(right_edge)
        total_max_edge_value = torch.max(torch.stack([top_edge_max, bottom_edge_max, left_edge_max, right_edge_max])).item()
        
        # Calculate the sum of edge pixel values
        top_edge_sum = torch.sum(top_edge.abs()).item()
        bottom_edge_sum = torch.sum(bottom_edge.abs()).item()
        left_edge_sum = torch.sum(left_edge.abs()).item()
        right_edge_sum = torch.sum(right_edge.abs()).item()
        total_edge_sum = top_edge_sum + bottom_edge_sum + left_edge_sum + right_edge_sum
        
        if total_max_edge_value > threshold:
            removed_by_edge.append(image)
        else:
            filtered_images.append(image)
        
    return filtered_images, removed_by_edge


def calculate_outside_emission(image, region_size=(64, 64)):
    """Calculate the fraction of emission outside a central region and the total emission."""
    image_np = image.squeeze().numpy() if image.dim() == 3 else image.numpy()  # Squeeze to remove channel dimension if present
    image_np = image_np / np.max(image_np) if np.max(image_np) != 0 else image_np
    
    center_x, center_y = image_np.shape[1] // 2, image_np.shape[0] // 2
    central_region = image_np[center_y - region_size[0] // 2:center_y + region_size[0] // 2,
                                center_x - region_size[1] // 2:center_x + region_size[1] // 2]
    
    total_emission = np.sum(image_np)
    if total_emission == 0:
        return 0, total_emission
    
    central_emission = np.sum(central_region)
    outside_emission_fraction = (total_emission - central_emission) / total_emission        
    return outside_emission_fraction, total_emission


def count_emission_regions(image, threshold=0.1, region_size=(128, 128)):
    """Count the number of distinct emission regions using connected component analysis."""
    image_np = image.squeeze().numpy() if image.dim() == 3 else image.numpy()
    center_y, center_x = image_np.shape[0] // 2, image_np.shape[1] // 2
    half_height, half_width = region_size[0] // 2, region_size[1] // 2
    start_y, end_y = center_y - half_height, center_y + half_height
    start_x, end_x = center_x - half_width, center_x + half_width
    central_region = image_np[start_y:end_y, start_x:end_x]
    binary_image = central_region > threshold
    labeled_image, num_features = label(binary_image)
    
    return num_features



def plot_cut_flow_for_all_filters(images, labels, num_thresholds=11, region_size=(64, 64), save_path_prefix="cut_flow"):
    """
    Saves four cut-flow graphs showing the proportion of images removed depending on the threshold value for
    peak intensity, outside emission, total intensity, and emission regions.
    
    Args:
        images: List or tensor of images to filter.
        labels: List or tensor of corresponding labels.
        num_thresholds: Number of threshold values to test (will generate linearly spaced values between 0 and 1).
        region_size: Size of the central region for filtering functions.
        save_path_prefix: Prefix for the saved image file names.
    """
    thresholds = np.linspace(0, 1, num=num_thresholds)
    thresholds = [0, 0.005, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.995, 1]
    total_images = len(images)
    
    # Initialize lists to hold the proportion of images removed for each filtering method
    removed_proportion_peak = []
    removed_proportion_outside_emission = []
    removed_proportion_intensity = []
    removed_proportion_regions = []
    
    for threshold in thresholds:
        # Apply peak intensity filtering
        _, _, removed_by_peak = filter_by_peak_intensity(images, labels, threshold, region_size)
        removed_proportion_peak.append(len(removed_by_peak) / total_images)
        
        # Apply outside emission filtering
        removed_by_emission = []
        for image, lbl in zip(images, labels):
            outside_emission_fraction, _ = calculate_outside_emission(image, region_size)
            if outside_emission_fraction > threshold:
                removed_by_emission.append((image, lbl))
        removed_proportion_outside_emission.append(len(removed_by_emission) / total_images)
        
        # Apply intensity filtering
        removed_by_intensity = []
        for image, lbl in zip(images, labels):
            _, total_emission = calculate_outside_emission(image, region_size)
            if total_emission > threshold*1000:  # Assuming the threshold is for intensity
                removed_by_intensity.append((image, lbl))
        removed_proportion_intensity.append(len(removed_by_intensity) / total_images)
        
        # Apply emission regions filtering
        removed_by_regions = []
        for image, lbl in zip(images, labels):
            num_regions = count_emission_regions(image, threshold, region_size)
            if num_regions > 3:  # Assuming the required number of regions is 2
                removed_by_regions.append((image, lbl))
        removed_proportion_regions.append(len(removed_by_regions) / total_images)
    
    # Create and save cut-flow plots for each filtering method
    def save_cut_flow_plot(thresholds, removed_proportions, title, filename):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, removed_proportions, marker='o', linestyle='-', color='b')
        plt.xlabel('Threshold Value')
        plt.ylabel('Proportion of Images Removed')
        plt.title(title)
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    # Save the plot for each filter type
    save_cut_flow_plot(thresholds, removed_proportion_peak, 'Cut-Flow: Peak Intensity', f"{save_path_prefix}peak_intensity.png")
    save_cut_flow_plot(thresholds, removed_proportion_outside_emission, 'Cut-Flow: Outside Emission', f"{save_path_prefix}outside_emission.png")
    save_cut_flow_plot(thresholds, removed_proportion_intensity, 'Cut-Flow: Intensity Threshold', f"{save_path_prefix}intensity.png")
    save_cut_flow_plot(thresholds, removed_proportion_regions, 'Cut-Flow: Emission Regions', f"{save_path_prefix}regions.png")


def filter_away_faintest(images, labels, region=(128, 128), threshold=0.1, save_path="./bright_pixel_filtering.png"):
    """
    Filters out images (and their associated labels) where the brightest pixel in the specified
    central region is below the given threshold.

    Parameters:
        images (list of np.ndarray or torch.Tensor): List of 2D image arrays/tensors.
        labels (list): List of labels corresponding to each image.
        region (tuple): Size of the central region (height, width) to check for brightness.
        threshold (float): Minimum required brightness in the region.
        save_path (str): File path to save the plot of removed images.

    Returns:
        tuple: (filtered_images, filtered_labels), where both lists contain only the images
               and labels that meet or exceed the brightness threshold.
    """
    filtered_images = []
    filtered_labels = []
    
    # We store (image, label, max_val) in removed_images
    removed_images = []

    for img, label in zip(images, labels):
        # Compute central region coordinates
        x_start, x_end = (img.shape[1] - region[1]) // 2, (img.shape[1] + region[1]) // 2
        y_start, y_end = (img.shape[0] - region[0]) // 2, (img.shape[0] + region[0]) // 2
        roi = img[y_start:y_end, x_start:x_end]

        # Convert ROI to NumPy if it's a torch.Tensor
        if isinstance(roi, torch.Tensor):
            roi = roi.detach().cpu().numpy()

        # Determine max brightness in the ROI
        max_val = np.max(roi)

        # Filter logic
        if max_val < threshold:
            # Store (img, label, max_val) for plotting
            removed_images.append((img, label, max_val))
        else:
            filtered_images.append(img)
            filtered_labels.append(label)

    total_per_class = Counter(labels)
    removed_per_class = Counter(lbl for _, lbl, _ in removed_images)

    for cls, total_cnt in total_per_class.items():
        removed_cnt = removed_per_class.get(cls, 0)
        fraction = removed_cnt / total_cnt * 100 if total_cnt else 0
        #print(f"Class {cls}: removed {removed_cnt}/{total_cnt} images ({fraction:.2f}%)")

    removed_count = len(removed_images)

    # Plot removed images
    if removed_count > 0:
        plt.figure(figsize=(5 * removed_count, 5))
        for i, (img, label, max_val) in enumerate(removed_images):
            plt.subplot(1, removed_count, i + 1)
            arr = _to_2d_for_imshow(img, how="first")
            im = plt.imshow(arr, cmap='viridis')
            plt.colorbar(im, ax=plt.gca())
            plt.title(f"Removed Image {i+1}\nLabel: {label}, Max: {max_val:.3f}")
            plt.axis('off')
        plt.savefig(save_path)
        plt.close()
    else:
        print("No images were filtered away.")

    return filtered_images, filtered_labels


def remove_outliers(images, labels, threshold=0.1, peak_threshold=0.6, intensity_threshold=200.0, max_regions=3, region_size=(64, 64), v="training", PLOTFILTERED=False):
    """
    Remove images and their corresponding labels if:
    - The images have more than the threshold fraction of total emission outside a central region (size specified by region_size),
    - Their summed intensities exceed the specified threshold,
    - They have more than a certain number of emission regions (specified by region_threshold).
    
    :param images: List of image tensors
    :param labels: List of corresponding labels
    :param threshold: Fraction of emission allowed outside the central region
    :param peak_threshold: Threshold for peak intensity filtering
    :param intensity_threshold: Threshold for total intensity filtering
    :param region_threshold: Maximum number of distinct emission regions allowed
    :param region_size: Size of the central region (tuple of two ints, e.g., (64, 64) for a 64x64 region)
    :return: Filtered list of image tensors, corresponding labels, and the fraction of images removed
    """
    filtered_labels, removed_by_emission, removed_by_intensity, removed_by_peak_intensity, removed_by_regions = [], [], [], [], []
    filtered_images, removed_by_edge = filter_images_with_edge_emission(images) # Remove images with bright pixels at the edge
    
    for image, lbl in zip(filtered_images, labels):  # Renamed 'label' to 'lbl' to avoid conflict
        outside_emission_fraction, total_emission = calculate_outside_emission(image)
        num_regions = count_emission_regions(image, threshold=0.3)
        
        if outside_emission_fraction > threshold and total_emission <= intensity_threshold:
            removed_by_emission.append((image, lbl))  # Remove images with emission outside central region
        elif total_emission > intensity_threshold:
            removed_by_intensity.append((image, lbl))  # Remove images with total intensity above threshold
        elif num_regions > max_regions:
            removed_by_regions.append((image, lbl))  # Remove images with too many regions
        else:
            filtered_labels.append(lbl)
    
    # Remove images with peak intensity
    filtered_images, filtered_labels, removed_by_peak_intensity = filter_by_peak_intensity(filtered_images, filtered_labels, peak_threshold, region_size)

    # Compile the final filtered list of images and labels
    final_filtered_images = [img for img in filtered_images 
                            if not any(torch.equal(img, removed_img[0]) for removed_img in removed_by_peak_intensity)]
    final_filtered_labels = [lbl for img, lbl in zip(filtered_images, filtered_labels) 
                            if not any(torch.equal(img, removed_img[0]) for removed_img in removed_by_peak_intensity)]
    fraction_final_removed = 1 - len(final_filtered_images) / len(images)

    # Print statistics
    #print(f"Images removed by edge pixels: {len(removed_by_edge)}")
    #print(f"Images removed by intensity (> {intensity_threshold}): {len(removed_by_intensity)}")
    #print(f"Images removed by outside emission (> {threshold} fraction): {len(removed_by_emission)}")
    #print(f"Images removed by peak intensity (< {peak_threshold}): {len(removed_by_peak_intensity)}")
    #print(f"Images removed by emission regions (> {max_regions} regions): {len(removed_by_regions)}")
    print(f"Fraction of images removed from {v} set: {fraction_final_removed}")
    #print("Number of images removed in total:", len(images) - len(final_filtered_images))
    
    def plot_removed_images(removed_by_emission, removed_by_intensity, removed_by_peak_intensity, removed_by_regions, removed_by_edge):
        """Plot examples of removed images for each filtering mechanism with a tighter layout and no empty frames."""

        # Determine the number of columns per row based on the maximum number of images to display (up to 5)
        max_images_per_row = 5
        fig, axs = plt.subplots(5, max_images_per_row, figsize=(10, 10), 
                        gridspec_kw={'wspace': 0.05, 'hspace': 0.1}, 
                        constrained_layout=True)
        fig.suptitle("Images Removed by Various Filtering Criteria")

        # Helper function to display a set of images
        def display_images(removed_images, ax_row, title, add_green_square=False):
            if isinstance(removed_images, tuple):
                print("Unexpected tuple format in image:", removed_images)

            # Add a single title to the left of the row
            axs[ax_row, 0].text(-0.3, 0.5, title, va='center', ha='center', rotation=90, fontsize=12, transform=axs[ax_row, 0].transAxes)

            for i in range(max_images_per_row):
                if i < len(removed_images):
                    image_data = removed_images[i]
                    if isinstance(image_data, tuple):
                        image_data = image_data[0]

                    if isinstance(image_data, (float, int)):
                        print(f"Skipping invalid data: {image_data}")
                        continue

                    if isinstance(image_data, torch.Tensor):
                        image_data = image_data.numpy()

                    if isinstance(image_data, np.ndarray):
                        if image_data.ndim == 3 and image_data.shape[0] == 1:
                            image_data = image_data.squeeze(0)
                    else:
                        if image_data.dim() == 3 and image_data.shape[0] == 1:
                            image_data = image_data.squeeze(0)

                    arr = _to_2d_for_imshow(image_data, how="first")
                    axs[ax_row, i].imshow(arr, cmap='viridis')
                    axs[ax_row, i].axis('off')

                    # Add a red semi-transparent rectangle (128x128 central crop)
                    img_height, img_width = image_data.shape
                    crop_size = 128
                    red_rect = patches.Rectangle(
                        ((img_width - crop_size) / 2, (img_height - crop_size) / 2),
                        crop_size, crop_size,
                        linewidth=2, edgecolor='r', facecolor='none', alpha=0.5
                    )
                    axs[ax_row, i].add_patch(red_rect)

                    if add_green_square:
                        green_crop_size = 64
                        green_rect = patches.Rectangle(
                            ((img_width - green_crop_size) / 2, (img_height - green_crop_size) / 2),
                            green_crop_size, green_crop_size,
                            linewidth=2, edgecolor='g', facecolor='none', alpha=0.5
                        )
                        axs[ax_row, i].add_patch(green_rect)
                else:
                    # Remove the axes entirely if there's no data to show
                    fig.delaxes(axs[ax_row, i])

        # Display images removed by various filtering criteria
        display_images(removed_by_edge, 0, f"Edge Pixels \n {len(removed_by_edge)}")
        display_images(removed_by_emission, 1, f"Outside Emission \n {len(removed_by_emission)}", add_green_square=True)
        display_images(removed_by_intensity, 2, f"Intensity \n {len(removed_by_intensity)}")
        display_images(removed_by_peak_intensity, 3, f"Peak Intensity \n {len(removed_by_peak_intensity)}", add_green_square=True)
        display_images(removed_by_regions, 4, f"Region Count \n {len(removed_by_regions)}")

        plt.savefig('./generator/filtering/removed_by_filtering.png')
        plt.close()



    if PLOTFILTERED:
        plot_removed_images(removed_by_emission, removed_by_intensity, removed_by_peak_intensity, removed_by_regions, removed_by_edge)
    
    return final_filtered_images, final_filtered_labels

#######################################################################################################
################################### DATA AUGMENTATION FUNCTIONS #######################################
#######################################################################################################

def plot_class_images(images, labels, filenames=None, set_name='train'):
    # ensure labels are a plain list of ints
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()

    # if images have more than one channel (C, H, W), only use the first channel
    if isinstance(images, torch.Tensor) and images.ndim == 4:
        images = [img[0] for img in images]
    
    desc_map = {c['tag']: c['description'] for c in get_classes()}
    
    for cls in sorted(set(labels)):
        # collect up to 10 examples of this class
        idxs = [i for i,l in enumerate(labels) if l == cls][:10]
        if not idxs:
            continue
        
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        fig.suptitle(f"{set_name} images for class {cls} – {desc_map.get(cls, '')}", fontsize=12)
        
        for ax, idx in zip(axes.flat, idxs):
            img = images[idx]
            arr = img.squeeze().cpu().numpy() if isinstance(img, torch.Tensor) else np.squeeze(img)
            
            # If multiple channels take the first channel
            if arr.ndim == 3 and arr.shape[0] > 1:
                arr = arr[0]
            
            ax.imshow(arr, cmap='viridis', origin='lower')
            ax.axis('off')
            if filenames and idx < len(filenames):
                ax.set_title(filenames[idx], fontsize=8)
        
        # blank out any unused subplots
        for ax in axes.flat[len(idxs):]:
            ax.axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"./classifier/{cls}_{set_name}_images.png", dpi=300)
        plt.close(fig)


def balance_classes(images, labels):
    """
    Randomly down‐sample each class so they all have the same number of samples
    equal to the size of the smallest class.
    """
    # collect indices per class
    class_idxs = defaultdict(list)
    for i, lbl in enumerate(labels):
        class_idxs[lbl].append(i)
    # find smallest class size
    min_n = min(len(idxs) for idxs in class_idxs.values())
    # sample
    selected = []
    for idxs in class_idxs.values():
        selected.extend(random.sample(idxs, min_n))
    random.shuffle(selected)
    # return balanced lists
    return [images[i] for i in selected], [labels[i] for i in selected]


# Using systematic transformations instead of random choices in augmentation
def apply_transforms_with_config(image, config):
    preprocess = transforms.Compose([
        transforms.Lambda(lambda x: transforms.functional.hflip(x) if config['flip_h'] else x),  # Horizontal flip
        transforms.Lambda(lambda x: transforms.functional.vflip(x) if config['flip_v'] else x),  # Vertical flip
        transforms.Lambda(lambda x: transforms.functional.rotate(x, config['rotation']))  # Rotation by specific angle
    ])
    
    if image.dim() == 2:  # Ensure image is 3D
        image = image.unsqueeze(0)
    transformed_image = preprocess(image) 
    return transformed_image


def complex_apply_transforms_with_config(image, config, img_shape=(128, 128), initial_threshold=0.9, step=0.01, brightness_adjustment=0.0):
    # Apply weaker peak region scaling first
    image_np = image.squeeze().cpu().numpy() if isinstance(image, torch.Tensor) else image
    scaled_image_np = scale_weaker_region(image_np, brightness_adjustment, initial_threshold=initial_threshold, step=step)
    scaled_image = torch.tensor(scaled_image_np).unsqueeze(0)  # Convert back to tensor with correct dimensions
    
    # Existing transformation pipeline
    preprocess = transforms.Compose([
        transforms.CenterCrop((img_shape[-2], img_shape[-1])),
        transforms.Resize((img_shape[-2], img_shape[-1])),  # Resize images to the desired size
        transforms.Lambda(lambda x: transforms.functional.hflip(x) if config['flip_h'] else x),  # Horizontal flip
        transforms.Lambda(lambda x: transforms.functional.vflip(x) if config['flip_v'] else x),  # Vertical flip
        transforms.Lambda(lambda x: transforms.functional.rotate(x, config['rotation']))  # Rotation by specific angle
    ])
    
    if image.dim() == 2:  # Ensure image is 3D
        scaled_image = scaled_image.unsqueeze(0)
    transformed_image = preprocess(scaled_image)
    transformed_image = transformed_image.squeeze(1)
    
    return transformed_image

def apply_formatting(image: torch.Tensor,
                     crop_size: tuple = (1, 128, 128),
                     downsample_size: tuple = (1, 128, 128)
                    ) -> torch.Tensor:
    """
    Center-crop and resize a single-channel tensor without PIL.

    Args:
      image: Tensor of shape [C, H0, W0] or [1, H0, W0].
      crop_size:      (C,Hc,Wc) or (Hc,Wc) or (T,C,Hc,Wc) → will be canonicalized.
      downsample_size:(C,Ho,Wo) or (Ho,Wo) or (T,C,Ho,Wo) → will be canonicalized.

    Returns:
      Tensor of shape [C, Ho, Wo].
    """

    # Canonicalize sizes to (C,H,W)
    def _canon_size(sz):
        if len(sz) == 2:
            return (1, sz[0], sz[1])
        if len(sz) == 3:
            return sz
        if len(sz) == 4:
            return (sz[-3], sz[-2], sz[-1])
        raise ValueError(f"crop/downsample size must have 2, 3 or 4 dims, got {sz}")

    crop_size = _canon_size(crop_size)
    downsample_size = _canon_size(downsample_size)

    # Normalize image dims
    if image.dim() == 4 and image.size(0) == 1:
        image = image.squeeze(0)               # [1,H0,W0]
    if image.dim() == 3:
        C, H0, W0 = image.shape
        img = image
    elif image.dim() == 2:
        H0, W0 = image.shape
        img = image.unsqueeze(0)
        C = 1
    else:
        raise ValueError(f"Unexpected image dims: {image.shape}")

    # Grayscale handling based on canonicalized channel dim
    if crop_size[0] == 1 or downsample_size[0] == 1:
        img = img.mean(dim=0, keepdim=True)

    # Unpack sizes
    _, Hc, Wc = crop_size
    _, Ho, Wo = downsample_size
    
    # Center crop and resize
    # Debug: show input and requested sizes (first 20 calls to avoid flooding)
    if not hasattr(apply_formatting, "_printed"):
        apply_formatting._printed = 0
    if apply_formatting._printed < 20:
        print(f"[apply_formatting] input={H0}x{W0}, requested_crop={Hc}x{Wc}, output={Ho}x{Wo}")
        apply_formatting._printed += 1

    y0, x0 = H0 // 2, W0 // 2
    y1, x1 = y0 - Hc // 2, x0 - Wc // 2
    y2, x2 = y1 + Hc, x1 + Wc

    # clamp in case input smaller than crop (no padding here—just report sizes)
    y1, y2 = max(0, y1), min(H0, y2)
    x1, x2 = max(0, x1), min(W0, x2)

    crop = img[:, y1:y2, x1:x2].unsqueeze(0)   # [1,C,*,*]
    resized = F.interpolate(crop, size=(Ho, Wo), mode='bilinear', align_corners=False)
    return resized.squeeze(0)                   # [C,Ho,Wo]

def apply_formatting_old(image: torch.Tensor,
                     crop_size: tuple = (1, 1, 128, 128),
                     downsample_size: tuple = (1, 1, 128, 128)
                    ) -> torch.Tensor:
    """
    Center-crop and resize a single-channel tensor without PIL.

    Args:
      image: Tensor of shape [C, H0, W0] or [1, H0, W0].
      crop_size:  (C, Hc, Wc) crop dimensions.
      downsample_size:  (C, Ho, Wo) output dimensions.

    Returns:
      Tensor of shape [C, Ho, Wo].
    """
        
    if image.dim() == 4 and image.size(0) == 1:
        image = image.squeeze(0)               # now [1, H0, W0]
    if image.dim() == 3:
        C, H0, W0 = image.shape
        img = image
    elif image.dim() == 2:
        H0, W0 = image.shape
        img = image.unsqueeze(0)
        C = 1
    else:
        raise ValueError(f"Unexpected image dims: {image.shape}")
    
    if crop_size[-3]==1 or downsample_size[-3]==1: # if grayscale
        img = img.mean(dim=0, keepdim=True)

    # unpack sizes
    _, Hc, Wc = crop_size
    _, Ho, Wo = downsample_size

    # compute crop coords (centered)
    y0, x0 = H0 // 2, W0 // 2
    y1, y2 = y0 - Hc//2, y0 + Hc//2
    x1, x2 = x0 - Wc//2, x0 + Wc//2

    # clamp in case input smaller than crop
    y1, y2 = max(0, y1), min(H0, y2)
    x1, x2 = max(0, x1), min(W0, x2)

    crop = img[:, y1:y2, x1:x2].unsqueeze(0)       # [1,C,Hc,Wc]
    resized = F.interpolate(crop, size=(Ho, Wo),   # [1,C,Ho,Wo]
                             mode='bilinear',
                             align_corners=False)
    return resized.squeeze(0)                      # [C,Ho,Wo]

def img_hash(img: torch.Tensor) -> str:
    #print("Shape of image before hashing:", img.shape)
    arr = img.cpu().contiguous().numpy()
    returnval = hashlib.sha1(arr.tobytes()).hexdigest()
    #print("Shape of image after hashing:", arr.shape)
    return returnval

######################################################################################################
################################### DATA LOADING FUNCTION ############################################
######################################################################################################

def load_vae_data(runname, sample_size, existing_images, existing_labels, data_type='generated'):
    print(f"Loading VAE {data_type} data...")

    def load_single_run(run):
        vae_file_path = root_path + f"VAE/VAE_{run}_{data_type}_images.pt"
        try:
            vae_data = torch.load(vae_file_path)
            vae_images = vae_data['images']
            vae_labels = vae_data['labels']
            return vae_images, vae_labels
        except FileNotFoundError:
            print(f"File {vae_file_path} not found. Proceeding without VAE images.")
            return None, None

    if isinstance(runname, list):
        vae_images_list, vae_labels_list = [], []
        for run in runname:
            vae_images, vae_labels = load_single_run(run)
            if vae_images is not None and vae_labels is not None:
                vae_images_list.append(vae_images)
                vae_labels_list.append(vae_labels)

        if vae_images_list and vae_labels_list:
            vae_images = torch.cat(vae_images_list, dim=0)
            vae_labels = torch.cat(vae_labels_list, dim=0)
        else:
            return existing_images, existing_labels
    else:
        vae_images, vae_labels = load_single_run(runname)
        if vae_images is None or vae_labels is None:
            print("No VAE images found.")
            return existing_images, existing_labels

    combined_images = torch.cat((existing_images, vae_images[:sample_size - len(existing_images)]), dim=0)
    combined_labels = torch.cat((existing_labels, vae_labels[:sample_size - len(existing_labels)]), dim=0)
    print("Combined images shape:", combined_images.shape)
    return combined_images.tolist(), combined_labels.tolist()


def isolate_galaxy_batch(images, upper_intensity_threshold=500000, lower_intensity_threshold=1000):
    total_images = len(images)
    removed_images = 0
    accepted_images = []

    for image in images:
        # Assuming the input is a NumPy array and already in grayscale format (2D array)
        if image.shape[0] == 3:  # If it's a 3-channel image, convert it to grayscale
            gray_image = image.mean(axis=0)  # Mean over the channel dimension
        else:
            gray_image = image.squeeze()  # Already grayscale, just squeeze if needed

        gray_image = (gray_image * 255).astype(np.uint8)  # Scale to 8-bit image
        thresh_val = 60  # Threshold value
        _, binary_image = cv2.threshold(gray_image, thresh_val, 255, cv2.THRESH_BINARY)

        # Find the connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8, ltype=cv2.CV_32S)

        # Find the label of the component that includes the center pixel
        center_x, center_y = gray_image.shape[1] // 2, gray_image.shape[0] // 2
        center_label = labels[center_y, center_x]

        # Create a mask for the component that includes the center pixel
        mask = np.zeros_like(gray_image)
        mask[labels == center_label] = 255

        # Set all pixels outside the central source to black
        black_background_image = np.where(mask == 255, gray_image, 0)

        # Add channel dimension back if necessary
        isolated_image = np.expand_dims(black_background_image, axis=0)  

        # Check the sum of the intensity of the isolated image
        intensity_sum = np.sum(black_background_image)
        
        if intensity_sum > upper_intensity_threshold or intensity_sum < lower_intensity_threshold:
            # If intensity is below threshold or above, discard image
            removed_images += 1
        else:
            accepted_images.append(isolated_image / 255)  # Normalize back to [0, 1]

    # Calculate the fraction of images removed
    removed_fraction = removed_images / total_images

    # Print the results
    print(f"Images removed in isolate_galaxy_batch: {removed_images}")
    print(f"Fraction removed in isolate_galaxy_batch: {removed_fraction:.2f}")

    # Return the accepted images
    return accepted_images
        
def augment_images(
    images, labels, rotations=[0, 90, 180, 270],
    flips = [(False, False), (True, False)], mem_threshold=1000,
    #translations = [(10, 0), (-10, 0), (0, 10), (0, -10)], #[(5, 0), (-5, 0), (0, 5), (0, -5)],
    translations = [(0, 0)], 
    ST_augmentation=False, n_gen = 1):
    """
    General function to augment images in chunks with memory optimization.

    Args:
        images (list or tensor): List or tensor of input images.
        labels (list or tensor): Corresponding labels for the images.
        img_shape (tuple): Shape of the input images.
        rotations (list): List of rotation angles in degrees.
        flips (list of tuples): List of tuples specifying horizontal and vertical flips.
        brightness_adjustments (list, optional): List of brightness adjustment factors. Default is None.

    Returns:
        tuple: Augmented images and labels as tensors.
    """

    # — normalize all inputs to exactly 3D (C=1, H, W) —
    normed = []
    for img in images:
        if isinstance(img, torch.Tensor):
            # if someone passed a “batch” dim of 1, remove it
            if img.dim() == 4 and img.size(0) == 1:
                img = img.squeeze(0)
            # if they somehow gave you a plain 2D H×W, make it (1,H,W)
            if img.dim() == 2:
                img = img.unsqueeze(0)
        normed.append(img)
    images = normed

    # Initialize empty lists for results
    augmented_images, augmented_labels = [], []
    cumulative_augmented_images, cumulative_augmented_labels = [], []
    
    if ST_augmentation:
        # labels may be a tensor; make them plain ints
        lbl_list = [int(x) for x in (labels.tolist() if torch.is_tensor(labels) else labels)]
        for cls in sorted(set(lbl_list)):
            # don't depend on exact count in the filename; use a pattern
            pattern = f"/users/mbredber/scratch/ST_generation/1to{n_gen}_*_{cls}.npy"
            candidates = sorted(glob.glob(pattern))
            if not candidates:
                print(f"[augment_images] No ST file matching {pattern}; skipping for class {cls}.")
                continue
            st_images = np.load(candidates[0])
            st_images = torch.tensor(st_images).float().unsqueeze(1)
            images.extend(st_images)
            lbl_list.extend([cls]*len(st_images))
        labels = lbl_list
    
    for idx, image in enumerate(images):
        for rot in rotations:
            for flip_h, flip_v in flips:
                for translation in translations:
                    if translation != (0, 0):
                        image = transforms.functional.affine(
                            image, angle=0, translate=translation, scale=1.0, shear=0, fill=0
                        )
                    config = {'rotation': rot, 'flip_h': flip_h, 'flip_v': flip_v}
                    augmented_image = apply_transforms_with_config(image.clone().detach(), config)
                    augmented_images.append(augmented_image)
                    augmented_labels.append(labels[idx])  # Append corresponding label

                    # Memory check: Save and clear if too many augmentations are in memory
                    if len(augmented_images) >= mem_threshold:  # Threshold for saving (adjustable)
                        cumulative_augmented_images.extend(augmented_images)
                        cumulative_augmented_labels.extend(augmented_labels)
                        augmented_images, augmented_labels = [], []  # Reset batch

    # Extend cumulative lists with remaining augmented images from the chunk
    cumulative_augmented_images.extend(augmented_images)
    cumulative_augmented_labels.extend(augmented_labels)

    # Convert cumulative lists to tensors
    augmented_images_tensor = torch.stack(cumulative_augmented_images)
    augmented_labels_tensor = torch.tensor(cumulative_augmented_labels)

    return augmented_images_tensor, augmented_labels_tensor

        
def reduce_by_class(images, labels, sample_size, num_classes):
    class_data = collections.defaultdict(list)
    
    # Group images by class
    for img, lbl in zip(images, labels):
        class_data[lbl].append(img)
    
    reduced_images, reduced_labels = [], []
    for lbl, imgs in class_data.items():
        # Limit samples to sample_size or the available number of samples
        selected_imgs = imgs[:sample_size]
        reduced_images.extend(selected_imgs)
        reduced_labels.extend([lbl] * len(selected_imgs))
    
    return reduced_images, reduced_labels

import math
import hashlib
from collections import defaultdict

def redistribute_excess(train_images, train_labels,
                        eval_images,  eval_labels,
                        target_classes,
                        train_filenames=None, eval_filenames=None):

    # 1) Pool everything
    all_imgs = list(train_images) + list(eval_images)
    all_lbls = list(train_labels) + list(eval_labels)
    # only build filenames if provided, otherwise pad with None
    all_fnames = (list(train_filenames) if train_filenames else []) \
               + (list(eval_filenames)  if eval_filenames  else [])
    if not all_fnames:
        all_fnames = [None] * len(all_imgs)

    # 2) Group by class, keeping filenames together
    bins = defaultdict(list)
    for img, lbl, fname in zip(all_imgs, all_lbls, all_fnames):
        bins[int(lbl)].append((img, fname))
        
    # 3) Compute how many per class to put in eval:
    total = len(all_imgs)
    n_cls = len(target_classes)
    per_class = math.ceil((total * 0.10) / n_cls)
    per_class = min(per_class, min(len(bins[c]) for c in target_classes))
    
    # 4) Deterministically split each bin by hash, preserving both img+fname
    new_eval_imgs, new_eval_lbls, new_eval_fnames = [], [], []
    new_train_imgs, new_train_lbls, new_train_fnames = [], [], []
    for cls in target_classes:
        items = bins[cls]
        items_sorted = sorted(items, key=lambda x: img_hash(x[0]))
        ev = items_sorted[:per_class]
        tr = items_sorted[per_class:]
        new_eval_imgs   += [x[0] for x in ev]
        new_eval_lbls   += [cls]*len(ev)
        new_eval_fnames += [x[1] for x in ev]
        new_train_imgs   += [x[0] for x in tr]
        new_train_lbls   += [cls]*len(tr)
        new_train_fnames += [x[1] for x in tr]
        
    
    # 5) Convert back to original types
    # — Images —
    if isinstance(train_images, torch.Tensor):
        # preserve shape if no examples
        train_imgs2 = (torch.stack(new_train_imgs)
                       if new_train_imgs
                       else torch.empty((0,)+train_images.shape[1:]))
    else:
        train_imgs2 = new_train_imgs

    if isinstance(eval_images, torch.Tensor):
        eval_imgs2 = (torch.stack(new_eval_imgs)
                      if new_eval_imgs
                      else torch.empty((0,)+eval_images.shape[1:]))
    else:
        eval_imgs2 = new_eval_imgs

    # — Labels —
    if isinstance(train_labels, torch.Tensor):
        train_lbls2 = torch.tensor(new_train_lbls, dtype=train_labels.dtype)
    else:
        train_lbls2 = new_train_lbls

    if isinstance(eval_labels, torch.Tensor):
        eval_lbls2 = torch.tensor(new_eval_lbls, dtype=eval_labels.dtype)
    else:
        eval_lbls2 = new_eval_lbls
        
    return (
        train_imgs2, train_lbls2,
        new_train_fnames if train_filenames else [],
        eval_imgs2,  eval_lbls2,
        new_eval_fnames  if eval_filenames  else [],
    )

def bitold_redistribute_excess(train_images, train_labels, eval_images, eval_labels, target_classes):
    """
    Deterministically re‐split pooled train+eval so that:
      1) evaluation set is exactly balanced across target_classes
      2) |eval| ≥ 10% of total images
    """
    # 1) Pool everything
    all_imgs = list(train_images) + list(eval_images)
    all_lbls = list(train_labels) + list(eval_labels)

    # 2) Group by class
    bins = defaultdict(list)
    for img, lbl in zip(all_imgs, all_lbls):
        bins[int(lbl)].append(img)

    # 3) Compute how many per class to put in eval:
    total = len(all_imgs)
    n_cls = len(target_classes)
    per_class = math.ceil((total * 0.10) / n_cls)
    per_class = min(per_class, min(len(bins[c]) for c in target_classes))

    # 4) Deterministically split each bin by sorting on img_hash
    def img_hash(tensor):
        arr = tensor.cpu().numpy().tobytes()
        return hashlib.sha1(arr).hexdigest()

    new_eval_imgs, new_eval_lbls = [], []
    new_train_imgs, new_train_lbls = [], []

    for cls in target_classes:
        imgs = bins[cls]
        # sort by hash so that "first" per_class is reproducible
        imgs_sorted = sorted(imgs, key=img_hash)
        ev = imgs_sorted[:per_class]
        tr = imgs_sorted[per_class:]
        new_eval_imgs += ev
        new_eval_lbls += [cls] * len(ev)
        new_train_imgs += tr
        new_train_lbls += [cls] * len(tr)

    # 5) Convert back to original types
    # — Images —
    if isinstance(train_images, torch.Tensor):
        # preserve shape if no examples
        train_imgs2 = (torch.stack(new_train_imgs)
                       if new_train_imgs
                       else torch.empty((0,)+train_images.shape[1:]))
    else:
        train_imgs2 = new_train_imgs

    if isinstance(eval_images, torch.Tensor):
        eval_imgs2 = (torch.stack(new_eval_imgs)
                      if new_eval_imgs
                      else torch.empty((0,)+eval_images.shape[1:]))
    else:
        eval_imgs2 = new_eval_imgs

    # — Labels —
    if isinstance(train_labels, torch.Tensor):
        train_lbls2 = torch.tensor(new_train_lbls, dtype=train_labels.dtype)
    else:
        train_lbls2 = new_train_lbls

    if isinstance(eval_labels, torch.Tensor):
        eval_lbls2 = torch.tensor(new_eval_lbls, dtype=eval_labels.dtype)
    else:
        eval_lbls2 = new_eval_lbls

    return train_imgs2, train_lbls2, eval_imgs2, eval_lbls2



def old_redistribute_excess(train_images, train_labels, eval_images, eval_labels, target_classes):
    """
    Redistribute excess images from the training set to the evaluation set to balance the classes.
    """
    
    # 1) Split existing eval into target-class vs other
    target_eval = [(img, lbl) for img, lbl in zip(eval_images, eval_labels) if lbl in target_classes]
    other_eval  = [(img, lbl) for img, lbl in zip(eval_images, eval_labels) if lbl not in target_classes]

    # 2) Group target eval by class
    groups = collections.defaultdict(list)
    for img, lbl in target_eval:
        groups[lbl].append(img)

    if not groups:
        print("No valid classes found in validation data; skipping redistribution.")
        return train_images, train_labels, eval_images, eval_labels

    # 3) Find smallest class size in eval
    min_count = min(len(imgs) for imgs in groups.values())

    # 4) Keep only min_count per class; mark the rest to move back to train
    kept, moved_back = [], []
    for lbl, imgs in groups.items():
        kept_imgs   = imgs[:min_count]
        moved_imgs  = imgs[min_count:]
        kept    += [(img, lbl) for img in kept_imgs]
        moved_back += [(img, lbl) for img in moved_imgs]

    # 5) Rebuild balanced eval (kept + any non-target classes)
    final_eval = kept + other_eval
    eval_imgs2, eval_lbls2 = zip(*final_eval) if final_eval else ([], [])
    eval_imgs2, eval_lbls2 = list(eval_imgs2), list(eval_lbls2)

    # 6) Send all “moved_back” images into train (rest of train stays untouched)
    train_imgs2 = list(train_images) + [img for img, _ in moved_back]
    train_lbls2 = list(train_labels) + [lbl for _, lbl in moved_back]

    return train_imgs2, train_lbls2, eval_imgs2, eval_lbls2
        
##########################################################################################
################################## SPECIFIC DATASET LOADER ###############################
##########################################################################################


def load_galaxy10(path=root_path + 'Galaxy10.h5', sample_size=300, target_classes=[4], 
                  crop_size=(3, 256, 256), downsample_size=(3, 256, 256), fold=None, island=True, train=False):
    print("Loading Galaxy10 data...")

    try:
        with h5py.File(path, 'r') as F:
            images = np.array(F['images'])
            labels = np.array(F['ans']).astype(int)

            all_images, all_labels = [], []
            for cls in target_classes:
                class_indices = np.where(labels == cls)[0]
                print(f"Number of images in class {cls}: {len(class_indices)}")
                
                if len(class_indices) == 0:
                    continue

                selected_images = 2 * (images[class_indices].astype(np.float32) / 255.0 - .5)
                if crop_size[-3] == 1:  
                    selected_images = np.mean(selected_images, axis=-1, keepdims=True)

                filtered_images = isolate_galaxy_batch(selected_images)
                print(f"Number of images remaining after isolate_galaxy_batch: {len(filtered_images)} \n")

                class_images, class_labels = [], []
                selected_images = np.moveaxis(selected_images, -1, 1)

                for image in filtered_images:
                    image = torch.tensor(image, dtype=torch.float)
                    if crop_size != 256:
                        image = apply_formatting(image, crop_size, downsample_size)
                    class_images.append(image)
                    class_labels.append(cls)

                all_images.append(torch.stack(class_images))
                all_labels.append(torch.tensor(class_labels))

            if all_images:
                all_images = torch.cat(all_images).clone().detach()
                all_labels = torch.cat(all_labels).clone().detach()

            print(f"Total number of images: {len(all_images)}")
            print(f"Total number of labels: {len(all_labels)}")

            if len(all_images) == 0 or len(all_labels) == 0:
                raise ValueError("No data available after filtering.")

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

            if fold is None or fold < 0 or fold >= 5:
                raise ValueError("Fold must be an integer between 0 and 4")

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(all_images, all_labels)):
                if fold_idx == fold:
                    train_images = all_images[train_idx]
                    train_labels = all_labels[train_idx]
                    eval_images = all_images[test_idx]
                    eval_labels = all_labels[test_idx]
                    break

            # Ensure all images have consistent shape (cropping and resizing)
            train_images = [apply_formatting(img, crop_size, downsample_size) for img in train_images]
            eval_images = [apply_formatting(img, crop_size, downsample_size) for img in eval_images]
                
            # Limit by sample size
            if sample_size is not None and sample_size > 0:
                if len(train_images) > 10:
                    train_images, train_labels = reduce_by_class(train_images, train_labels, sample_size, len(target_classes))
                if len(eval_images) > 10:
                    eval_images, eval_labels = reduce_by_class(eval_images, eval_labels, int(sample_size*0.2), len(target_classes))

    except OSError as e:
        print(f"Failed to open the file: {e}")

    if not train:
        return train_images, train_labels
    
    return train_images, train_labels, eval_images, eval_labels


def load_FIRST(path=None, fold=0, target_classes=None, crop_size=(1, 300, 300), downsample_size=(300, 300), sample_size=300,
               island=False, REMOVEOUTLIERS=True, BALANCE=False, AUGMENT=False, train=True):
    print("Loading FIRST data...")
    
    class_mapping = {10: 'FRI', 11: 'FRII', 12: 'Compact', 13: 'Bent'}
    target_classes_names = [class_mapping.get(tc, tc) for tc in target_classes]
    if len(downsample_size) == 3:
        downsample_size = downsample_size[1:]
    if len(crop_size) == 3:
        crop_size = crop_size[1:]
    
    def get_data(data):
        images, labels = [], []
        for item in data:
            images.append(item[0])
            labels.append(item[1])
        return images, labels

    if fold is not None and fold >= 0 and fold < 6:
        if fold == 5:
            train_data = FIRSTGalaxyData(root="./.cache", selected_split="train", input_data_list=[f"galaxy_data_h5.h5"],
                                        selected_classes=target_classes_names, is_PIL=True, is_RGB=False, transform=transforms.ToTensor())
            if train:
                eval_data = FIRSTGalaxyData(root="./.cache", selected_split="valid", input_data_list=[f"galaxy_data_h5.h5"],
                                            selected_classes=target_classes_names, is_PIL=True, is_RGB=False, transform=transforms.ToTensor())
            else:
                eval_data = FIRSTGalaxyData(root="./.cache", selected_split="test", input_data_list=[f"galaxy_data_h5.h5"],
                                            selected_classes=target_classes_names, is_PIL=True, is_RGB=False, transform=transforms.ToTensor())
        else:
            train_data = FIRSTGalaxyData(root="./.cache", selected_split="train", input_data_list=[f"galaxy_data_crossvalid_{fold}_h5.h5"],
                                         selected_classes=target_classes_names, is_PIL=True, is_RGB=False, transform=transforms.ToTensor())
            if train:
                eval_data = FIRSTGalaxyData(root="./.cache", selected_split="valid", input_data_list=[f"galaxy_data_crossvalid_{fold}_h5.h5"],
                                            selected_classes=target_classes_names, is_PIL=True, is_RGB=False, transform=transforms.ToTensor())
            else:
                eval_data = FIRSTGalaxyData(root="./.cache", selected_split="test", input_data_list=["galaxy_data_crossvalid_test_h5.h5"],
                                            selected_classes=target_classes_names, is_PIL=True, is_RGB=False, transform=transforms.ToTensor()) # Smaller sample of test data

            
        train_images, train_labels = get_data(train_data)
        eval_images, eval_labels = get_data(eval_data)
        
        # Add 10 to the labels to match the class mapping
        train_labels = [lbl + 10 for lbl in train_labels]
        eval_labels  = [lbl + 10 for lbl in eval_labels]

        if len(eval_images) == 0:
            raise ValueError("No valid images found. Check the dataset and loading process.")

                
        train_images = torch.stack(train_images)
        eval_images = torch.stack(eval_images)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        eval_labels  = torch.tensor(eval_labels,  dtype=torch.long)

        return train_images, train_labels, eval_images, eval_labels

    else:
        raise ValueError("Fold must be an integer between 0 and 5")



def load_Mirabest(target_classes=[16], crop_size=(1, 150, 150), downsample_size=(1, 150, 150), sample_size=397, train=False):
    print("Loading MiraBest data...")

    # Select the dataset class based on target_classes
    if 16 in target_classes or 17 in target_classes:
        DatasetClass = MBFRFull
    elif 14 in target_classes or 15 in target_classes:
        DatasetClass = MBFRConfident
        print("Selecting confidently classified galaxies")
    elif 'uncertain' in target_classes:  # Reading of these two classes not implemented in this script
        DatasetClass = MBFRUncertain
    elif 'random' in target_classes:
        DatasetClass = MBRandom
    else:
        raise ValueError("Invalid target class or class combination provided.")

    # Determine if we are loading both training and testing data, or just training data
    splits = ["train", "test"] if train else ["train"]

    # Initialize lists to store images and labels for training and testing data
    all_train_images = []
    all_train_labels = []
    all_test_images = []
    all_test_labels = []

    # Iterate over the splits (train/test)
    for split in splits:
        # Load the dataset for the specified split (train or test)
        dataset = DatasetClass(root='./batches', train=(split == 'train'), download=True, transform=transforms.ToTensor())
        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        # Process images based on the target classes
        for target_class in target_classes:
            images = []
            labels = []
            original_images = []

            # Iterate over the data in the loader
            for batch_images, batch_labels in loader:
                for i, label in enumerate(batch_labels):
                    # Check if the label matches the target class
                    if target_class % 2 == label.item():  # 0:FRI, 1:FRII
                        image = batch_images[i].squeeze(0)
                        original_images.append(image.clone().detach())
                        # Apply transformations if required (e.g., resizing, cropping, noise addition)
                        if crop_size[1] != 150:
                            processed_image = apply_formatting(image, crop_size, downsample_size)
                        images.append(processed_image)
                        labels.append(label.item())
                        # Stop once we have enough samples
                        if len(images) >= sample_size:
                            break
                if len(images) >= sample_size:
                    break
            
            # Handle training data differently from testing/validation data
            if split == 'train':
                # Augment the training data if not enough images are available
                while len(images) < sample_size:
                    idx = np.random.randint(0, len(original_images))
                    if idx < len(images):  # Ensure the index is within bounds
                        augmented_image = apply_formatting(original_images[idx].clone().detach(), crop_size, downsample_size)
                        images.append(augmented_image)
                        labels.append(labels[idx])
                    else:
                        print(f"Index {idx} is out of bounds for the images array with length {len(images)}")

                # Add the processed images and labels to the training list
                all_train_images.extend(images)
                all_train_labels.extend(labels)
            else:
                # Apply the same transformations to the test/validation images
                for img in original_images:
                    preprocessed_image = apply_formatting(img, crop_size, downsample_size)
                    all_test_images.append(preprocessed_image)
                all_test_labels.extend(labels)

    # Convert the lists of training images and labels to tensors
    train_images = torch.stack(all_train_images).clone().detach()
    train_labels = torch.tensor(all_train_labels, dtype=torch.long)

    if not train:  # If not in training mode, return all images (combined train and test)
        all_images = train_images
        all_labels = train_labels
        if all_test_images and all_test_labels:
            # Combine the training and testing images and labels into a single dataset
            test_images = torch.stack(all_test_images).clone().detach()
            test_labels = torch.tensor(all_test_labels, dtype=torch.long)
            all_images = torch.cat((train_images, test_images), dim=0)
            all_labels = torch.cat((train_labels, test_labels), dim=0)
        return all_images, all_labels
    
    else:  # If in training mode, return separate training and testing datasets
        test_images = torch.stack(all_test_images).clone().detach()
        test_labels = torch.tensor(all_test_labels, dtype=torch.long)

        return train_images, train_labels, test_images, test_labels
    

def load_MNIST(target_classes=[19], fold=0, crop_size=(1, 28, 28), downsample_size=(1, 28, 28), sample_size=60000, train=False, batch_size=32):
    print("Loading MNIST data...")
    train_data = datasets.MNIST(root=root_path, train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, REMOVEOUTLIERS=True, batch_size=batch_size, shuffle=True)

    train_images = []
    train_labels = []

    if set(target_classes) & {18} & {19}:
        digit_classes = list(range(10))  # Include all digits if 18 or 19 is in target_classes
    else:
        digit_classes = [tc % 10 for tc in target_classes]  # Map each target_class to a digit

    for batch_images, batch_labels in train_loader:
        for i, label in enumerate(batch_labels):
            if label.item() in digit_classes:
                image = batch_images[i]
                if crop_size[1] != 28:
                    image = apply_formatting(image, crop_size, downsample_size)
                train_images.append(image)
                train_labels.append(label.item())
                if len(train_images) >= sample_size:
                    break
        if len(train_images) >= sample_size:
            break

    train_images = torch.stack(train_images).clone().detach()
    train_labels = torch.tensor(train_labels, dtype=torch.long)

    if not train:
        return train_images, train_labels
    else:
        test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        test_images = []
        test_labels = []

        for batch_images, batch_labels in test_loader:
            for i, label in enumerate(batch_labels):
                if label.item() in digit_classes:
                    image = batch_images[i]
                    if crop_size[1] != 28:
                        image = apply_formatting(image, crop_size, island=False)
                    test_images.append(image)
                    test_labels.append(label.item())
                    if len(test_images) >= int(0.15 * sample_size):
                        break
            if len(test_images) >= int(0.15 * sample_size):
                break

        test_images = torch.stack(test_images).clone().detach()
        test_labels = torch.tensor(test_labels, dtype=torch.long)

        return train_images, train_labels, test_images, test_labels
    
    
def load_RGZ10k(path=root_path + "RGZ_10k", fold=5, target_classes=None, crop_size=(1, 132, 132), sample_size=300, train=True, random_state=SEED):
    print("Loading rgz10k data...")
    """
    Returns splits from the RGZ10k dataset.

    All images (7,702 total) are now assumed to be in the path.
    The allowed labels are defined in the function.

    For fold values 0–4, a standard 5‑fold stratified split is performed:
      - The training set is the union of 4 folds,
      - The validation set is the remaining fold.

    For fold == 5, a final test split is created by removing a balanced test set,
    consisting of 150 images per allowed class, from the overall dataset.
    The remaining images form the training set.

    Additionally, if sample_size is provided (>0), the training set is reduced so that 
    each class contains at most sample_size images, and the evaluation set is reduced 
    to at most int(sample_size*0.2) images per class.

    Parameters:
      - fold: An integer between 0 and 5.
          * 0–4: 5-fold cross validation.
          * 5: Final test split with a balanced test set (150 images per allowed class).
      - path: Path to the folder containing all images and the subfolder "annotation".
      - random_state: For reproducible shuffling/sampling.
      - sample_size: Maximum number of images per class (for training; eval set uses sample_size*0.2).
      - REMOVEOUTLIERS: If True, remove outlier images similar to the FIRST function.
    
    Returns:
      - train_images: List of PyTorch tensors for training images.
      - train_labels: List of labels corresponding to train_images.
      - eval_images:  List of PyTorch tensors for evaluation images (validation or test set).
      - eval_labels:  List of labels corresponding to eval_images.
    """
    import re, random, xml.etree.ElementTree as ET
    # (Assumes that augment_images, reduce_by_class, remove_outliers, and plot_cut_flow_for_all_filters are imported)
    
    # Full set of allowed labels in the dataset.
    allowed_labels_all = ["1_1", "1_2", "1_3", "2_2", "2_3", "3_3"]

    # If target_classes is None, use default mapping (all classes).
    if target_classes is None:
        target_classes = [31, 32, 33, 34, 35, 36]
        
    mapping = {31: "1_1", 32: "1_2", 33: "1_3", 34: "2_2", 35: "2_3", 36: "3_3"}
    target_allowed_labels = {mapping[num] for num in target_classes}

    # Define folder path for annotations.
    annotation_folder = os.path.join(path, "annotation")
    
    # ------------------------------------------------------------
    # Step 1. Build a mapping from image filename (from XML) to its XML file.
    # ------------------------------------------------------------
    xml_mapping = {}  # key: image filename; value: XML file path
    xml_files = glob.glob(os.path.join(annotation_folder, "*.xml"))
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            filename_elem = root.find("filename")
            if filename_elem is None:
                continue
            img_filename = filename_elem.text.strip()
            if img_filename not in xml_mapping:
                xml_mapping[img_filename] = xml_file
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
    
    # ------------------------------------------------------------
    # Step 2. Define a helper to extract a label from an XML file.
    # (Return the first allowed label encountered.)
    # ------------------------------------------------------------
    def get_label_from_xml(xml_file):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall("object"):
                name_elem = obj.find("name")
                if name_elem is None:
                    continue
                label = name_elem.text.strip()
                if label in allowed_labels_all:
                    return label
            return None
        except Exception as e:
            print(f"Error processing XML {xml_file}: {e}")
            return None
    
    # ------------------------------------------------------------
    # Step 3. Process all PNG images in the path.
    # ------------------------------------------------------------
    def process_folder(folder):
        data = []  # Will hold tuples: (image_path, label)
        png_files = glob.glob(os.path.join(folder, "*.png"))
        for png_path in png_files:
            png_filename = os.path.basename(png_path)
            # Adjust the filename if needed (remove suffix variations).
            base_name = re.sub(r"(_logminmax|_infraredctmask)?\.png$", ".png", png_filename)
            xml_file = None
            if base_name in xml_mapping:
                xml_file = xml_mapping[base_name]
            elif png_filename in xml_mapping:
                xml_file = xml_mapping[png_filename]
            if not xml_file:
                continue  # No matching XML annotation found.
            label = get_label_from_xml(xml_file)
            if label is None:
                continue
            data.append((png_path, label))
        return data

    # ------------------------------------------------------------
    # Step 4: Process the entire dataset (all images are now in path).
    # ------------------------------------------------------------
    all_data = process_folder(path)
    if len(all_data) == 0:
        raise ValueError(f"No images found in path {path}. Check your folder and XML annotations.")


    # Filter the dataset to include only images whose label is in the target set.
    all_data = [(img, lbl) for img, lbl in all_data if lbl in target_allowed_labels]

    # Organize data by label.
    data_by_label = {label: [] for label in target_allowed_labels}
    
    # ------------------------------------------------------------
    # Step 5. Split the dataset.
    # ------------------------------------------------------------
    if fold == 5:
        # For fold==5, create a final test split by sampling 150 images per allowed class.
        # The remaining images will be used for training.
        for img_path, label in all_data:
            data_by_label[label].append((img_path, label))
        test_set = []
        random.seed(random_state)
        for label in target_allowed_labels:
            items = data_by_label[label]
            if len(items) < 150:
                raise ValueError(f"Not enough images for label {label}: need 150 but got {len(items)}")
            sampled = random.sample(items, 150)
            test_set.extend(sampled)
        test_paths = set(x[0] for x in test_set)
        train_set = [item for item in all_data if item[0] not in test_paths]
        
        train_images = [x[0] for x in train_set]
        train_labels  = [x[1] for x in train_set]
        eval_images = [x[0] for x in test_set]
        eval_labels  = [x[1] for x in test_set]
    elif 0 <= fold < 5:
        # Standard 5-fold cross validation using stratified splitting.
        X = [item[0] for item in all_data]
        y = [item[1] for item in all_data]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        splits = list(skf.split(X, y))
        try:
            train_idx, valid_idx = splits[fold]
        except IndexError:
            raise ValueError(f"Could not obtain fold {fold} from the dataset splits.")
        train_images = [X[i] for i in train_idx]
        train_labels  = [y[i] for i in train_idx]
        eval_images = [X[i] for i in valid_idx]
        eval_labels  = [y[i] for i in valid_idx]
    else:
        raise ValueError("fold must be an integer between 0 and 5 (0-4 for CV, 5 for final test).")
    
    # ------------------------------------------------------------
    # Step 6. Load and preprocess images.
    # ------------------------------------------------------------
    # Load and reshape train images.
    images = []
    for img_path in train_images:
        # Load the image in grayscale.
        img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_data is None:
            raise ValueError(f"Failed to load image: {img_path}")
        # Resize if needed.
        img_data = cv2.resize(img_data, (crop_size[-2], crop_size[-1]))
        # Reshape to the desired shape.
        img_data = img_data.reshape(crop_size)
        images.append(img_data)
    train_images = np.stack(images, axis=0)
    
    # Do the same for eval_images
    images = []
    for img_path in eval_images:
        img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_data is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img_data = cv2.resize(img_data, (crop_size[-2], crop_size[-1]))
        img_data = img_data.reshape(crop_size)
        images.append(img_data)
    eval_images = np.stack(images, axis=0)
    
    # Convert NumPy arrays to lists of PyTorch tensors.
    train_images = [torch.tensor(img, dtype=torch.float32) / 255.0 for img in train_images]
    eval_images = [torch.tensor(img, dtype=torch.float32) / 255.0 for img in eval_images]
    
    # Rename the labels to integer values.
    label_mapping = {label: idx for idx, label in enumerate(target_allowed_labels)}
    train_labels = [label_mapping[lbl] for lbl in train_labels]
    eval_labels = [label_mapping[lbl] for lbl in eval_labels]
    
    
    return train_images, train_labels, eval_images, eval_labels


def load_PSZ2(path=root_path + "PSZ2/classified/",
              fold=5,
              sample_size=300,
              target_classes=[53],
              crop_size=(1, 256, 256),
              downsample_size=(1, 256, 256),
              versions='T100kpcSUB',     
              USE_GLOBAL_NORMALISATION=False,
              GLOBAL_NORM_MODE='flux',
              percentile_lo=1, percentile_hi=99,
              target_fov_kpc=1000.0,   # <<< NEW: enforce same physical size (1 Mpc default)
              train=False):
    """
    PSZ2 loader with strict version alignment for tesseracts and optional physical FOV cropping.

    If target_fov_kpc is not None/False:
      - we compute a centered crop that spans target_fov_kpc (in kpc) using the FITS header
        pixel scale and the cluster redshift (from PSZ2/cluster_source_data.csv).
      - this happens both for multi-version 'CUBE' mode and single-version mode,
        so every image spans the same physical size on the sky before resizing to downsample_size.
    """

    # --- helper: pixel scale (deg/pixel) ---
    def _pixdeg(hdr):
        if 'CDELT1' in hdr:
            return abs(hdr['CDELT1'])
        cd11 = hdr.get('CD1_1'); cd12 = hdr.get('CD1_2', 0.0)
        if cd11 is not None:
            return float(np.hypot(cd11, cd12))
        raise KeyError("No CDELT* or CD* keywords in FITS header")

    # Canonicalize requested versions; decide whether we build a time cube
    def _canon_one(v):
        s  = str(v).strip()
        sl = s.lower()
        canonical = {
            'raw':'RAW','t25kpcsub':'T25kpcSUB','25kpcsub':'T25kpcSUB','t25kpc':'T25kpc','t25':'T25kpc','25kpc':'T25kpc',
            't50kpcsub':'T50kpcSUB','50kpcsub':'T50kpcSUB','t50kpc':'T50kpc','t50':'T50kpc','50kpc':'T50kpc',
            't100kpcsub':'T100kpcSUB','100kpcsub':'T100kpcSUB','t100kpc':'T100kpc','t100':'T100kpc','100kpc':'T100kpc',
        }
        return canonical.get(sl, s)

    if isinstance(versions, (list, tuple)):
        vf_list = [_canon_one(v) for v in versions]
    else:
        vf_list = [_canon_one(versions)]
    CUBE = len(vf_list) > 1

    # --- load cluster metadata if we will crop to a physical FOV ---
    TARGET_FOV_KPC = target_fov_kpc if target_fov_kpc else None
    meta_df = None; ZCOL = None
    if TARGET_FOV_KPC:
        meta_csv = os.path.join(root_path, "PSZ2/cluster_source_data.csv")
        if os.path.exists(meta_csv):
            meta_df = pd.read_csv(meta_csv)
            if 'slug' in meta_df.columns:
                meta_df = meta_df.rename(columns={'slug': 'base'})
            meta_df = meta_df.set_index('base')
            ZCOL = 'z' if 'z' in meta_df.columns else ('redshift' if 'redshift' in meta_df.columns else None)
            if ZCOL is None:
                print("[load_PSZ2] WARNING: metadata missing redshift column; disabling physical cropping.")
                TARGET_FOV_KPC = None
        else:
            print(f"[load_PSZ2] WARNING: {meta_csv} not found; disabling physical cropping.")
            TARGET_FOV_KPC = None

    images, labels, filenames = [], [], []
    classes_map = {c["tag"]: c["description"] for c in get_classes()}

    if CUBE:
        ref_version = 'RAW' if 'RAW' in {v.upper() for v in vf_list} else vf_list[0]

        def _list_fits_bases(version, class_folder):
            folder = os.path.join(path, version, class_folder)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Version folder missing: {folder}")
            bases = {os.path.splitext(fn)[0] for fn in os.listdir(folder)
                     if fn.lower().endswith(".fits")}
            if not bases:
                raise FileNotFoundError(f"No FITS files found in {folder}")
            return folder, bases

        for cls in target_classes:
            class_folder = classes_map.get(cls)
            if class_folder is None:
                continue
            label = cls

            folder_map, base_sets = {}, {}
            for vf in vf_list:
                folder, bases = _list_fits_bases(vf, class_folder)
                folder_map[vf] = folder
                base_sets[vf] = bases

            common_bases = sorted(set.intersection(*base_sets.values()))
            if not common_bases:
                print(f"[load_PSZ2] No common basenames for class '{class_folder}' across {vf_list}. Skipping.")
                continue

            Ho, Wo = downsample_size[-2:]
            Hc_ref, Wc_ref = crop_size[-2:]  # used if physical cropping is disabled

            for base in common_bases:
                ref_path = os.path.join(folder_map[ref_version], f"{base}.fits")
                ref_hdr  = fits.getheader(ref_path)
                ref_pixdeg = _pixdeg(ref_hdr)

                frames = []
                for vf in vf_list:
                    fpath = os.path.join(folder_map[vf], f"{base}.fits")
                    hdr   = fits.getheader(fpath)
                    arr   = np.squeeze(fits.getdata(fpath)).astype(np.float32)
                    if arr.ndim == 3:
                        arr = arr.mean(axis=0)
                    ten = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]

                    if TARGET_FOV_KPC and meta_df is not None and base in meta_df.index:
                        try:
                            z = float(meta_df.loc[base, ZCOL])
                            kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z).value / 60.0
                            arcsec_per_pix = _pixdeg(hdr) * 3600.0
                            side_pix = max(1, int(round(TARGET_FOV_KPC / (kpc_per_arcsec * arcsec_per_pix))))
                            frm = apply_formatting(ten, (1, side_pix, side_pix), (1, Ho, Wo))
                        except Exception as e:
                            print(f"[load_PSZ2] Physical crop failed for {base} ({e}); falling back to ratio.")
                            s = ref_pixdeg / _pixdeg(hdr)
                            Hc_v = max(1, int(round(Hc_ref * s)))
                            Wc_v = max(1, int(round(Wc_ref * s)))
                            frm = apply_formatting(ten, (1, Hc_v, Wc_v), (1, Ho, Wo))
                    else:
                        # old behavior: keep versions aligned by pixel-scale ratio
                        s = ref_pixdeg / _pixdeg(hdr)
                        Hc_v = max(1, int(round(Hc_ref * s)))
                        Wc_v = max(1, int(round(Wc_ref * s)))
                        frm = apply_formatting(ten, (1, Hc_v, Wc_v), (1, Ho, Wo))

                    frames.append(frm)

                cube = torch.stack(frames, dim=0)  # [T,1,Ho,Wo]
                images.append(cube)
                labels.append(label)
                filenames.append(base)

    else:
        Ho, Wo = downsample_size[-2:]

        for cls in target_classes:
            class_folder = classes_map.get(cls, None)
            if class_folder is None:
                continue
            label = cls
            folder_path = os.path.join(path, vf_list[0], class_folder)
            if not os.path.isdir(folder_path):
                continue

            for fname in os.listdir(folder_path):
                if not fname.lower().endswith(".fits"):
                    continue
                base, _ = os.path.splitext(fname)
                if base.endswith(vf_list[0]):
                    continue

                fits_path = os.path.join(folder_path, fname)
                hdr = fits.getheader(fits_path)
                arr = fits.getdata(fits_path).astype(float)
                arr2 = np.squeeze(arr)
                if arr2.ndim == 3:
                    arr2 = np.mean(arr2, axis=0)
                elif arr2.ndim != 2:
                    raise ValueError(f"Expected 2-D or 3-D stack, got {arr2.shape!r}")

                if USE_GLOBAL_NORMALISATION and GLOBAL_NORM_MODE == "flux":
                    fluxconv = hdr.get('FLUXCONV', 1.0)
                    src = torch.from_numpy(arr2 * fluxconv).float().unsqueeze(0)  # [1,H,W]
                    src = src.clamp(1e-10, 1e-5)
                    src = (src - 1e-10) / (1e-5 - 1e-10)
                else:
                    src = torch.from_numpy(arr2).float().unsqueeze(0)

                if TARGET_FOV_KPC and meta_df is not None and base in meta_df.index:
                    try:
                        z = float(meta_df.loc[base, ZCOL])
                        kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z).value / 60.0
                        arcsec_per_pix = _pixdeg(hdr) * 3600.0
                        side_pix = max(1, int(round(TARGET_FOV_KPC / (kpc_per_arcsec * arcsec_per_pix))))
                        frame = apply_formatting(src, (1, side_pix, side_pix), (1, Ho, Wo))
                    except Exception as e:
                        print(f"[load_PSZ2] Physical crop failed for {base} ({e}); using crop_size.")
                        frame = apply_formatting(src, crop_size=crop_size, downsample_size=downsample_size)
                else:
                    frame = apply_formatting(src, crop_size=crop_size, downsample_size=downsample_size)

                images.append(frame)
                labels.append(label)
                filenames.append(base)

    # Global percentile/flux normalization (unchanged)
    if USE_GLOBAL_NORMALISATION and GLOBAL_NORM_MODE == "percentile":
        print(f"[load_PSZ2] Applying global percentile normalisation ({percentile_lo}–{percentile_hi}%)")
        imgs_tensor = torch.stack(images) if isinstance(images, list) else images
        flat = imgs_tensor.reshape(-1).float()
        qlo = flat.quantile(percentile_lo/100)
        qhi = flat.quantile(percentile_hi/100)
        def _apply_fixed_pct_elem(x, lo=qlo, hi=qhi):
            return ((x - lo) / (hi - lo + 1e-6)).clamp(0, 1).to(x.dtype)
        if isinstance(images, list):
            images = [_apply_fixed_pct_elem(x) for x in images]
        else:
            images = _apply_fixed_pct_elem(images)

    if len(images) == 0:
        raise ValueError("No images loaded. Check the PSZ2 path/versions/classes.")
    assert len(images) == len(labels) == len(filenames)

    # RH vs RR conflicts + hash dedup (unchanged)
    if set(target_classes) == {52, 53}:
        base_to_labels = {}
        for lbl, base in zip(labels, filenames):
            base_to_labels.setdefault(base, set()).add(lbl)
        conflict_bases = {b for b, labs in base_to_labels.items() if len(labs) > 1}
        if conflict_bases:
            print(f"Excluding {len(conflict_bases)} ambiguous clusters present in both RH and RR.")
        filtered_triplets = [
            (img, lbl, base)
            for img, lbl, base in zip(images, labels, filenames)
            if base not in conflict_bases
        ]
        if filtered_triplets:
            images, labels, filenames = map(list, zip(*filtered_triplets))
        else:
            images, labels, filenames = [], [], []

        from collections import Counter
        hashes = []
        for img in images:
            arr = img.squeeze().numpy()
            hashes.append(hashlib.sha1(arr.tobytes()).hexdigest())
        counts = Counter(hashes)
        dup_hashes = {h for h, c in counts.items() if c > 1}
        if dup_hashes:
            keep = set(); seen = set()
            for i, h in enumerate(hashes):
                if h not in seen:
                    keep.add(i); seen.add(h)
            images  = [images[i]  for i in sorted(keep)]
            labels  = [labels[i]  for i in sorted(keep)]
            filenames = [filenames[i] for i in sorted(keep)]

    # Stratified split (unchanged)
    all_idx = list(range(len(images)))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=0.2, stratify=labels, random_state=SEED
    )
    train_images = [images[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    eval_images  = [images[i] for i in test_idx]
    eval_labels  = [labels[i] for i in test_idx]

    return train_images, train_labels, eval_images, eval_labels, \
           [filenames[i] for i in train_idx], [filenames[i] for i in test_idx]


def load_MGCLS(path='/users/mbredber/data/MGCLS/classified_crops_1600/',  # Path to MGCLS data
               fold=None,
               target_classes=[40],
               crop_size=(1600, 1600),
               downsample_size=(128, 128),
               sample_size=300,
               BALANCE=False, AUGMENT=False,
               train=False):


    print("Loading MGCLS diffuse data…")

    # build a map tag→folder
    classes_map = {c['tag']: c['description'] for c in get_classes()}

    images, labels, filenames = [], [], []
    for cls in target_classes:
        folder = classes_map.get(cls)
        if folder is None:
            continue
        class_dir = os.path.join(path, folder)
        for fn in os.listdir(class_dir):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                with Image.open(os.path.join(class_dir, fn)) as img:
                    pil_gray = img.convert("L")
                    tensor = transforms.ToTensor()(pil_gray)
                    tensor = apply_formatting(tensor, crop_size, downsample_size)
                images.append(tensor)
                labels.append(cls)
                filenames.append(fn)

    if not images:
        raise ValueError("No images loaded. Check the path and file extensions.")
    
    # throttle total loaded images
    if sample_size is not None and len(images) > sample_size:
        images, labels, filenames = (
            list(images)[:sample_size],
            list(labels)[:sample_size],
            list(filenames)[:sample_size],
        )

    # === GROUP-AWARE SPLIT ===
    # 1) extract group key (everything before the final underscore)
    groups = [fn.rsplit('_', 1)[0] for fn in filenames]

    # 2) unique groups and one label per group
    unique_groups, inverse_idxs = np.unique(groups, return_inverse=True)
    # pick the first occurrence of each group to get its label
    group_labels = [labels[groups.index(g)] for g in unique_groups]

    # 3) split groups, stratified by their class
    train_groups, test_groups = train_test_split(
        unique_groups,
        test_size=0.2,
        stratify=group_labels,
        random_state=SEED
    )

    # 4) assign each file to train/test based on its group
    train_idx = [i for i, g in enumerate(groups) if g in train_groups]
    test_idx  = [i for i, g in enumerate(groups) if g in test_groups]

    train_images = [images[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    eval_images  = [images[i] for i in test_idx]
    eval_labels  = [labels[i] for i in test_idx]

    return train_images, train_labels, eval_images, eval_labels

def load_galaxies(galaxy_classes, path=None, versions=None, fold=None, island=None, crop_size=None, downsample_size=None, sample_size=None, DEBUG=False,
                  REMOVEOUTLIERS=True, BALANCE=False, AUGMENT=False, USE_GLOBAL_NORMALISATION=False, GLOBAL_NORM_MODE="percentile", STRETCH=False, percentile_lo=30, percentile_hi=99,
                  NORMALISE=True, NORMALISETOPM=False, EXTRADATA=False, PRINTFILENAMES=False, SAVE_IMAGES=False, train=None, target_fov_kpc=1000.0): 
    """
    Master loader that delegates to specific dataset loaders and returns zero-based labels.
    """
    def get_max_class(galaxy_classes):
        if isinstance(galaxy_classes, list):
            return max(galaxy_classes)
        return galaxy_classes

    # Clean up kwargs to remove None values
    kwargs = {'path': path, 'versions': versions, 'sample_size': sample_size, 'fold':fold, 'train': train,
              'island': island, 'crop_size': crop_size, 'downsample_size': downsample_size, 'USE_GLOBAL_NORMALISATION': USE_GLOBAL_NORMALISATION,
              'GLOBAL_NORM_MODE': GLOBAL_NORM_MODE, 'percentile_lo': percentile_lo, 'percentile_hi': percentile_hi, 'target_fov_kpc': target_fov_kpc} 
    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}

    max_class = get_max_class(galaxy_classes)

    # Delegate to specific loaders based on class range
    if max_class < 10:
        path = root_path + 'Galaxy10.h5'
        target_classes = galaxy_classes if isinstance(galaxy_classes, list) else [galaxy_classes]
        data = load_galaxy10(path=path, target_classes=target_classes, **clean_kwargs)
    elif max_class < 14:
        if clean_kwargs.get('crop_size') is not None:
            clean_kwargs['crop_size'] = np.squeeze(clean_kwargs['crop_size'])
        path = root_path + 'firstgalaxydata/'
        target_classes = galaxy_classes if isinstance(galaxy_classes, list) else [galaxy_classes]
        data = load_FIRST(path=path, target_classes=target_classes, **clean_kwargs)
    elif max_class < 18:
        clean_kwargs['crop_size'] = np.squeeze(clean_kwargs.get('crop_size', None))
        target_classes = galaxy_classes if isinstance(galaxy_classes, list) else [galaxy_classes]
        data = load_Mirabest(target_classes=target_classes, **clean_kwargs)
    elif max_class < 30:
        target_classes = galaxy_classes if isinstance(galaxy_classes, list) else [galaxy_classes]
        data = load_MNIST(target_classes=target_classes, **clean_kwargs)
    elif max_class < 37:
        target_classes = galaxy_classes if isinstance(galaxy_classes, list) else [galaxy_classes]
        data = load_RGZ10k(target_classes=target_classes, **clean_kwargs)
    elif max_class < 50:
        target_classes = galaxy_classes if isinstance(galaxy_classes, list) else [galaxy_classes]
        data = load_MGCLS(target_classes=target_classes, **clean_kwargs)
    elif max_class <= 59:
        target_classes = galaxy_classes if isinstance(galaxy_classes, list) else [galaxy_classes]
        data = load_PSZ2(target_classes=target_classes, **clean_kwargs)
    else:
        raise ValueError("Invalid galaxy class provided.")
    
    if len(data) == 4:
        train_images, train_labels, eval_images, eval_labels = data
        train_filenames = eval_filenames = None  # No filenames returned
    elif len(data) == 6:
        train_images, train_labels, eval_images, eval_labels, train_filenames, eval_filenames = data
        overlap = set(train_filenames) & set(eval_filenames)
        assert not overlap, f"PSZ2 split error — these IDs are in both sets: {overlap}"
    else:
        raise ValueError("Data loader did not return the expected number of outputs.")

    # Check for overlap between train and test sets
    train_hashes = {img_hash(img) for img in train_images}
    eval_hashes = {img_hash(img) for img in eval_images}
    if len(eval_images) != 0:
        common = train_hashes & eval_hashes
        if common:
            overlap_hash = next(iter(common))
            train_idxs = [i for i,img in enumerate(train_images) if img_hash(img) == overlap_hash]
            test_idxs  = [i for i,img in enumerate(eval_images) if img_hash(img) == overlap_hash]
            print(f"🔍 Overlap hash {overlap_hash!r} found at train indices {train_idxs} and test indices {test_idxs}")
            assert False, f"Overlap detected: {len(common)} images appear in both train and test validation!"
            
    if isinstance(train_images, list):
        train_images = torch.stack(train_images)
    if isinstance(eval_images, list):
        eval_images  = torch.stack(eval_images)

    if BALANCE:
        train_images, train_labels = balance_classes(train_images, train_labels) # Remove excess images from the largest class
            
    sample_indices = {}
    # Convert labels to a list if they are tensors
    if isinstance(train_labels, torch.Tensor):
        train_labels = train_labels.tolist()
    for cls in sorted(set(train_labels)):
        idxs = [i for i, lbl in enumerate(train_labels) if lbl == cls]
        sample_indices[cls] = idxs[:10]
        
    if isinstance(eval_labels, torch.Tensor):
        eval_labels = eval_labels.tolist()
    for cls in sorted(set(eval_labels)):
        idxs = [i for i, lbl in enumerate(eval_labels) if lbl == cls]
        sample_indices[cls] = sample_indices.get(cls, []) + idxs[:10]
        
    if DEBUG:
        plot_class_images(train_images, train_labels, train_filenames, set_name='train_1.before_normalisation')
        plot_class_images(eval_images,  eval_labels,  eval_filenames,  set_name='eval_1.before_normalisation')
        check_tensor("train_images", train_images)
        check_tensor("eval_images",  eval_images)


    if NORMALISE:
        if isinstance(train_images, list):
            train_images = torch.stack(train_images)
        if isinstance(eval_images, list):
            eval_images = torch.stack(eval_images)
        all_images = torch.cat([train_images, eval_images], dim=0)
        #all_images = normalise_images(all_images, out_min=0, out_max=1)  
        if USE_GLOBAL_NORMALISATION: # Regular normalisation of all images to [0,1]
            if DEBUG:
                print("Applying global normalisation to [0,1]")
            all_images = normalise_images(all_images, out_min=0, out_max=1)
        else:  # Percentile stretch to [0,1]
            if DEBUG:
                print(f"Applying percentile stretch to [{percentile_lo},{percentile_hi}]%")
            
            def per_image_percentile_stretch(x, lo=80, hi=99):
                # x: [B, C, H, W]; returns same shape
                B = x.shape[0]
                out = x.clone()
                for i in range(B):
                    flat = out[i].reshape(-1)
                    p_low  = flat.quantile(lo/100)
                    p_high = flat.quantile(hi/100)
                    out[i] = ((out[i] - p_low) / (p_high - p_low + 1e-6)).clamp(0, 1)
                return out

            if all_images.ndim == 5:   # [B, T, C, H, W]
                for t in range(all_images.shape[1]):
                    all_images[:, t] = per_image_percentile_stretch(all_images[:, t], percentile_lo, percentile_hi)
            else:                      # [B, C, H, W]
                all_images = per_image_percentile_stretch(all_images, percentile_lo, percentile_hi)
        train_images = all_images[:len(train_images)]
        eval_images  = all_images[len(train_images):]
       
    if NORMALISETOPM:
        if DEBUG:
            print("Applying normalise to ±1")
        all_images = torch.cat([train_images, eval_images], dim=0)
        all_images = normalise_images(all_images, out_min=-1, out_max=1)
        train_images = all_images[:len(train_images)]
        eval_images  = all_images[len(train_images):]
    
    if STRETCH:
        if DEBUG:
            print("Applying asinh stretch")
        # Concatenate so train/eval get the exact same mapping
        alpha = 10.0
        all_images = torch.cat([train_images, eval_images], dim=0)

        # Asinh stretch (elementwise), preserves shape/device/dtype
        stretched = torch.asinh(all_images * alpha) / math.asinh(alpha)

        # Split back
        n_tr = train_images.shape[0]
        train_images = stretched[:n_tr]
        eval_images  = stretched[n_tr:]

    # Always redistribute excess images for the test data, for fair evaluation
    classes_present = torch.unique(torch.cat([torch.tensor(train_labels), torch.tensor(eval_labels)])).tolist()
    train_images, train_labels, train_filenames, eval_images, eval_labels, eval_filenames = redistribute_excess(train_images, train_labels, eval_images, eval_labels, classes_present, train_filenames, eval_filenames)
    
    if SAVE_IMAGES:
        for kind, imgs, lbls in (
            ('train', train_images, train_labels),
            ('eval',  eval_images,  eval_labels),
        ):
            for cls in target_classes:
                cls_imgs = [img for img, lbl in zip(imgs, lbls) if lbl == (cls-min(target_classes))]
                print(f"Length of {kind} images for class {cls}: {len(cls_imgs)}")
                np.save(f"{path}_{kind}_{cls}_{len(cls_imgs)}.npy", cls_imgs)
        
    if EXTRADATA and not PRINTFILENAMES:
        if DEBUG:
            print("Loading PSZ2 metadata")
        meta_df = pd.read_csv(os.path.join(root_path, "PSZ2/cluster_source_data.csv"))
        print("PSZ2 metadata columns:", meta_df.columns.tolist())
        meta_df.rename(columns={"slug": "base"}, inplace=True)
        meta_df.set_index("base", inplace=True)

        # build a list of metadata rows in the same order as your `filenames` list:
        train_data = [meta_df.loc[base].values for base in train_filenames]
        eval_data  = [meta_df.loc[base].values for base in eval_filenames]
        
    if AUGMENT:
        if DEBUG:
            print("Applying data augmentation…")
        train_images, train_labels = augment_images(train_images, train_labels, ST_augmentation=False)
        if not (galaxy_classes[0] == 52 and galaxy_classes[1] == 53): 
            eval_images, eval_labels = augment_images(eval_images, eval_labels, ST_augmentation=False) # Only augment if not RR and RH
        else:
            if len(eval_images.shape) == 3:
                eval_images = eval_images.unsqueeze(1)
            if isinstance(eval_images, (list, tuple)):
                eval_images = torch.stack(eval_images)
            if isinstance(eval_labels, (list, tuple)):
                eval_labels = torch.tensor(eval_labels, dtype=torch.long)
        if EXTRADATA and not PRINTFILENAMES:
                n_aug = 8  # default is 4*2 = 8
                train_data = [row for row in train_data for _ in range(n_aug)]
                if not (galaxy_classes[0] == 52 and galaxy_classes[1] == 53):
                    eval_data  = [row for row in eval_data  for _ in range(n_aug)]
        if PRINTFILENAMES:
            n_aug = 8  # default is 4*2 = 8
            train_filenames = [fname for fname in train_filenames for _ in range(n_aug)]
            if not (galaxy_classes[0] == 52 and galaxy_classes[1] == 53): 
                eval_filenames  = [fname for fname in eval_filenames  for _ in range(n_aug)]

    else:
        # Unsqueeze if the images are of shape (B, H, W) 
        if len(train_images.shape) == 3:
            train_images = train_images.unsqueeze(1)
        if len(eval_images.shape) == 3:
            eval_images = eval_images.unsqueeze(1)
        if isinstance(train_images, (list, tuple)):
            train_images = torch.stack(train_images)
        if isinstance(train_labels, (list, tuple)):
            train_labels = torch.tensor(train_labels, dtype=torch.long)
        if isinstance(eval_images, (list, tuple)):
            eval_images = torch.stack(eval_images)
        if isinstance(eval_labels, (list, tuple)):
            eval_labels = torch.tensor(eval_labels, dtype=torch.long)
            
    if PRINTFILENAMES:
        return train_images, train_labels, eval_images, eval_labels, train_filenames, eval_filenames
    elif EXTRADATA:
        train_data = torch.tensor(np.stack(train_data), dtype=torch.float32)
        eval_data  = torch.tensor(np.stack(eval_data),  dtype=torch.float32)
        return train_images, train_labels, eval_images, eval_labels, train_data, eval_data
    
    return train_images, train_labels, eval_images, eval_labels

def load_halos_and_relics(
    galaxy_classes,
    versions=('RAW',),
    fold=5,
    crop_size=(1, 128, 128),
    downsample_size=(1, 128, 128),
    sample_size=1_000_000,
    REMOVEOUTLIERS=True,
    BALANCE=False,
    STRETCH=False,
    percentile_lo=1,
    percentile_hi=99,
    AUGMENT=False,
    NORMALISE=True,
    NORMALISETOPM=False,
    USE_GLOBAL_NORMALISATION=False,
    GLOBAL_NORM_MODE='percentile',
    PRINTFILENAMES=False,
    train=True,
):
    """
    Unifies dataset loading so the training driver can call a single entry point.
    Returns:
      train=True:
        4-tuple: (train_images, train_labels, valid_images, valid_labels)
        6-tuple if PRINTFILENAMES: (..., train_fns, valid_fns)
      train=False:
        4-tuple: (empty_images, empty_labels, test_images, test_labels)
        6-tuple if PRINTFILENAMES: (..., test_fns)
    Notes:
      * Labels are left as original class tags (e.g. 52, 53). The driver relabels later.
      * If multiple `versions` are provided (e.g. ['RAW','T50kpc']), PSZ2 returns a tesseract [T,1,H,W] per sample.
    """

    # -------- 1) Load raw images/labels (+ optional filenames) ----------
    # Decide dataset family from class tags
    is_psz2  = any(50 <= int(c) <= 58 for c in galaxy_classes)
    is_first = any(10 <= int(c) <= 13 for c in galaxy_classes)

    images = labels = filenames = None

    if is_psz2:
        raw = load_PSZ2(
            fold=fold,
            sample_size=sample_size,
            target_classes=galaxy_classes,
            crop_size=crop_size,
            downsample_size=downsample_size,
            versions=versions,
            USE_GLOBAL_NORMALISATION=USE_GLOBAL_NORMALISATION,
            GLOBAL_NORM_MODE=GLOBAL_NORM_MODE,
            percentile_lo=percentile_lo,
            percentile_hi=percentile_hi,
            train=False,  # splitting handled below
        )
        # Accept (imgs,lbls,fnames) or (imgs,lbls)
        if isinstance(raw, tuple) and len(raw) == 3:
            images, labels, filenames = raw
        elif isinstance(raw, tuple) and len(raw) == 2:
            images, labels = raw
            filenames = [""] * len(labels)
        else:
            raise RuntimeError("load_PSZ2 returned unexpected shape")

    elif is_first:
        # FIRST already returns train/eval; we’ll unify shapes anyway.
        tr_imgs, tr_lbls, ev_imgs, ev_lbls = load_FIRST(
            fold=fold if fold in [0,1,2,3,4,5] else 5,
            target_classes=galaxy_classes,
            crop_size=crop_size,
            downsample_size=downsample_size,
            sample_size=sample_size,
            train=True,
        )
        images  = list(tr_imgs) + list(ev_imgs)
        labels  = list(tr_lbls.cpu().tolist()) + list(ev_lbls.cpu().tolist())
        filenames = [""] * len(labels)
    else:
        raise ValueError("load_galaxies: unsupported class set; add a branch for your dataset.")

    # Ensure list types
    if isinstance(images, torch.Tensor):
        images = [img for img in images]
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().tolist()

    # -------- 2) Optional image-level transforms (percentile stretch, asinh, normalise) ----------
    def _maybe_proc(img):
        x = img
        if STRETCH:
            # per-image percentile stretch then asinh, like the driver expects
            x = percentile_stretch(x, lo=percentile_lo, hi=percentile_hi)
            x = asinh_stretch(x, alpha=10)
        if NORMALISE:
            if NORMALISETOPM:
                x = normalise_images(x, -1, 1)
            else:
                x = normalise_images(x, 0, 1)
        return x

    images = [_maybe_proc(x) for x in images]

    # -------- 3) Optional class balancing (undersample to min class size) ----------
    if BALANCE:
        from collections import defaultdict
        byc = defaultdict(list)
        for i, lbl in enumerate(labels):
            byc[int(lbl)].append(i)
        min_n = min(len(v) for v in byc.values())
        keep = []
        for v in byc.values():
            keep.extend(v[:min_n])
        keep = sorted(keep)
        images   = [images[i] for i in keep]
        labels   = [labels[i] for i in keep]
        filenames = [filenames[i] for i in keep] if filenames else []

    # -------- 4) Optional augmentation (pre-split, like your current driver when LATE_AUG=False) ----------
    if AUGMENT:
        imgs_t = torch.stack(images)
        lbls_t = torch.tensor(labels, dtype=torch.long)
        imgs_t, lbls_t = augment_images(imgs_t, lbls_t)
        images = [imgs_t[i] for i in range(len(imgs_t))]
        labels = lbls_t.cpu().tolist()
        if filenames:
            # replicate filenames n_aug times
            n_aug = len(images) // max(1, len(filenames))
            filenames = [fn for fn in filenames for _ in range(n_aug)]

    # -------- 5) Stratified split into train/valid (or build test only) ----------
    y = np.array(labels)
    idx_all = np.arange(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(skf.split(idx_all, y))
    # Map fold==5 to "last split" for your driver’s convention
    split_idx = fold if fold in [0,1,2,3,4] else 4
    tr_idx, va_idx = splits[split_idx]

    def _take(idxs):
        ims = [images[i] for i in idxs]
        lbs = torch.tensor([labels[i] for i in idxs], dtype=torch.long)
        fns = [filenames[i] for i in idxs] if filenames else []
        # stack if 3D or 4D tensors, else leave as-is
        try:
            ims = torch.stack(ims)
        except Exception:
            pass
        return ims, lbs, fns

    train_images, train_labels, train_fns = _take(tr_idx)
    valid_images, valid_labels, valid_fns = _take(va_idx)

    # Optionally bound per-class sample sizes
    if isinstance(sample_size, int) and sample_size > 0:
        # limit train set per class
        cls_counts = {c:0 for c in sorted(set(labels))}
        keep = []
        for i, lbl in enumerate(train_labels.tolist()):
            if cls_counts[lbl] < sample_size:
                keep.append(i); cls_counts[lbl] += 1
        train_images = train_images[keep]
        train_labels = train_labels[keep]
        if train_fns:
            train_fns = [train_fns[i] for i in keep]

    if not train:
        empty_imgs = torch.empty((0,)+tuple(train_images.shape[1:])) if isinstance(train_images, torch.Tensor) else []
        empty_lbls = torch.empty((0,), dtype=torch.long)
        if PRINTFILENAMES:
            return empty_imgs, empty_lbls, valid_images, valid_labels, [], valid_fns
        return empty_imgs, empty_lbls, valid_images, valid_labels

    if PRINTFILENAMES:
        return train_images, train_labels, valid_images, valid_labels, train_fns, valid_fns
    return train_images, train_labels, valid_images, valid_labels
