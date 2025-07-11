import numpy as np
import h5py
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from utils.GAN_models import load_gan_generator
from utils.calc_tools import normalise_images, generate_from_noise, load_model
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
        {"tag": 53, "length": 20, "description": "RR"}, # Radio Relic
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
            ddpm_data = np.clip(ddpm_data, 0, 1)

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

def percentile_stretch(x, lo=2, hi=98):
    """
    x: Tensor of shape (B, C, H, W) with values in [0,1]
    Returns: same shape, linearly rescaled so that the lo-th percentile → 0 and hi-th → 1
    """
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
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            img = np.squeeze(img)
            # Use viridis colormap and show a colorbar
            im = plt.imshow(img, cmap='viridis')
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

                    axs[ax_row, i].imshow(image_data, cmap='viridis')
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
    # ensure shape [1, H0, W0]
    if image.dim() == 3:
        C, H0, W0 = image.shape
        img = image
    elif image.dim() == 2:
        H0, W0 = image.shape
        img = image.unsqueeze(0)
        C = 1
    else:
        raise ValueError(f"Unexpected image dims: {image.shape}")

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

def apply_formatting_old(image, crop_size=(1, 1, 128, 128), downsample_size=(1, 1, 128, 128)):
    to_gray = (crop_size[-3] == 1) or (downsample_size[-3] == 1)
    transforms_list = [transforms.ToPILImage()]
    if to_gray:
        transforms_list.append(transforms.Grayscale(num_output_channels=1))
    transforms_list += [
        transforms.CenterCrop((crop_size[-2], crop_size[-1])),
        transforms.Resize((downsample_size[-2], downsample_size[-1])),
        transforms.ToTensor(),
    ]
    preprocess = transforms.Compose(transforms_list)
    return preprocess(image)


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
        for cls in set(labels):  # Get unique classes from labels
            images_cls = [img for img, lbl in zip(images, labels) if lbl == cls]
            path = f'/users/mbredber/scratch/ST_generation/1to{n_gen}_{len(images_cls)}_{cls}.npy'
            st_images = np.load(path)

            st_images = torch.tensor(st_images).float().unsqueeze(1)
            images.extend(st_images)
            labels.extend([cls] * len(st_images))
    
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


def redistribute_excess(train_images, train_labels, eval_images, eval_labels, target_classes):
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
    train_imgs2 = train_images + [img for img, _ in moved_back]
    train_lbls2 = train_labels + [lbl for _, lbl in moved_back]

    return train_imgs2, train_lbls2, eval_imgs2, eval_lbls2
        
##########################################################################################
################################## SPECIFIC DATASET LOADER ###############################
##########################################################################################


def load_galaxy10(path=root_path + 'Galaxy10.h5', sample_size=300, target_classes=[4], 
                  crop_size=(3, 256, 256), downsample_size=(3, 256, 256), fold=None, island=True, REMOVEOUTLIERS=True, BALANCE=False, AUGMENT=False, train=False):
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
                
            # Remove outliers
            if REMOVEOUTLIERS: 
                plot_cut_flow_for_all_filters(train_images, train_labels, save_path_prefix="./generator/filtering/")
                train_images, train_labels = remove_outliers(train_images, train_labels, PLOTFILTERED=True)
                eval_images, eval_labels = remove_outliers(eval_images, eval_labels, v="validation")

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

    if BALANCE:
        train_images, train_labels = balance_classes(train_images, train_labels)
        
    train_images_augmented = None
    train_labels_augmented = None
    eval_images_augmented = None  
    eval_labels_augmented = None

    # Parameters for augmentation
    if AUGMENT:
        chunk_size = 64
        # Augmentation for training data
        for i in range(0, len(train_images), chunk_size):
            chunk_images = train_images[i:i + chunk_size]
            chunk_labels = train_labels[i:i + chunk_size]
            augmented_chunk_images, augmented_chunk_labels = augment_images(chunk_images, chunk_labels)
            # Convert augmented data to tensors
            augmented_chunk_images = torch.tensor(augmented_chunk_images)
            augmented_chunk_labels = torch.tensor(augmented_chunk_labels)
            
            # Concatenate tensors for the final dataset
            if train_images_augmented is None:
                train_images_augmented = augmented_chunk_images
                train_labels_augmented = augmented_chunk_labels
            else:
                train_images_augmented = torch.cat([train_images_augmented, augmented_chunk_images])
                train_labels_augmented = torch.cat([train_labels_augmented, augmented_chunk_labels])

        # Update original variables with augmented data
        train_images = train_images_augmented
        train_labels = train_labels_augmented
    
        ## Augmentation for validation data
        for i in range(0, len(eval_images), chunk_size):
            chunk_images = eval_images[i:i + chunk_size]
            chunk_labels = eval_labels[i:i + chunk_size]
            augmented_chunk_images, augmented_chunk_labels = augment_images(chunk_images, chunk_labels,  mem_threshold=1)
            
            # Convert augmented data to tensors
            augmented_chunk_images = torch.tensor(augmented_chunk_images)
            augmented_chunk_labels = torch.tensor(augmented_chunk_labels)
            
            # Concatenate tensors for the final dataset
            if  eval_images_augmented is None:
                eval_images_augmented = augmented_chunk_images
                eval_labels_augmented = augmented_chunk_labels
            else:
                eval_images_augmented = torch.cat([eval_images_augmented, augmented_chunk_images])
                eval_labels_augmented = torch.cat([eval_labels_augmented, augmented_chunk_labels])

        # Update original variables with augmented data
        eval_images = eval_images_augmented
        eval_labels = eval_labels_augmented
    
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

        
        # Ensure all images have consistent shape (cropping and resizing)
        train_images = [apply_formatting(img, crop_size, downsample_size) for img in train_images]
        eval_images = [apply_formatting(img, crop_size, downsample_size) for img in eval_images]
                
        if REMOVEOUTLIERS: # Remove outliers
            #plot_cut_flow_for_all_filters(train_images, train_labels, save_path_prefix="./generator/filtering/")
            #train_images, train_labels = remove_outliers(train_images, train_labels, PLOTFILTERED=True)
            #eval_images, eval_labels = remove_outliers(eval_images, eval_labels, v="validation")
            train_images, train_labels = filter_away_faintest(train_images, train_labels, threshold=0.1, save_path="./generator/filtering/faintest_train.png")
            eval_images, eval_labels = filter_away_faintest(eval_images, eval_labels, threshold=0.1, save_path="./generator/filtering/faintest_eval.png")

                
        # Limit by sample size
        if sample_size is not None and sample_size > 0:
            if len(train_images) > 10:
                train_images, train_labels = reduce_by_class(train_images, train_labels, sample_size, len(target_classes))
            if len(eval_images) > 10:
                eval_images, eval_labels = reduce_by_class(eval_images, eval_labels, int(sample_size*0.2), len(target_classes))

        train_images, train_labels, eval_images, eval_labels = redistribute_excess(
            train_images, train_labels, eval_images, eval_labels, target_classes)
        
        # Save the evaluation images as a npy file
        #for kind in ['train', 'eval']:
        #    for cls in target_classes:
        #        if kind == 'train':
        #            images_cls = [img for img, lbl in zip(train_images, train_labels) if lbl == cls]
        #            print("Length of", kind, "images for class", cls, ":", len(images_cls))
        #            np.save(f"/users/mbredber/data/FIRST/FIRST_{kind}_{cls}_f{fold}_{len(images_cls)}.npy", images_cls)

        if BALANCE:
            train_images, train_labels = balance_classes(train_images, train_labels)
                
        # Augment train images (converts to tensors)
        if AUGMENT:
            train_images, train_labels = augment_images(train_images, train_labels)
            eval_images, eval_labels = augment_images(eval_images, eval_labels)
        else:
            # Convert from list to tensor
            train_images = torch.stack(train_images)
            eval_images = torch.stack(eval_images)
            train_labels = torch.tensor(train_labels, dtype=torch.long)
            eval_labels  = torch.tensor(eval_labels,  dtype=torch.long)

        return train_images, train_labels, eval_images, eval_labels

    else:
        raise ValueError("Fold must be an integer between 0 and 5")



def load_Mirabest(target_classes=[16], crop_size=(1, 150, 150), downsample_size=(1, 150, 150), sample_size=397,
                  train=False, island=False, REMOVEOUTLIERS=True, BALANCE=False, AUGMENT=False):
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

        # Shuffle the combined dataset before returning
        shuffled_indices = torch.randperm(len(all_images))
        all_images = all_images[shuffled_indices]
        all_labels = all_labels[shuffled_indices]

        return all_images, all_labels
    else:  # If in training mode, return separate training and testing datasets
        test_images = torch.stack(all_test_images).clone().detach()
        test_labels = torch.tensor(all_test_labels, dtype=torch.long)

        # Shuffle the training dataset
        train_indices = torch.randperm(len(train_images))
        train_images = train_images[train_indices]
        train_labels = train_labels[train_indices]

        # Shuffle the test dataset
        test_indices = torch.randperm(len(test_images))
        test_images = test_images[test_indices]
        test_labels = test_labels[test_indices]

        return train_images, train_labels, test_images, test_labels
    

def load_MNIST(target_classes=[19], fold=0, crop_size=(1, 28, 28), downsample_size=(1, 28, 28), sample_size=60000,
               train=False, island=False, BALANCE=False, AUGMENT=False, batch_size=32):
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

        # Shuffle the training data
        train_indices = torch.randperm(len(train_images))
        train_images = train_images[train_indices]
        train_labels = train_labels[train_indices]

        # Shuffle the test data
        test_indices = torch.randperm(len(test_images))
        test_images = test_images[test_indices]
        test_labels = test_labels[test_indices]

        return train_images, train_labels, test_images, test_labels
    
    
def load_RGZ10k(path=root_path + "RGZ_10k", fold=5, target_classes=None, crop_size=(1, 132, 132), sample_size=300,
                island=False, BALANCE=False, AUGMENT=False, REMOVEOUTLIERS=True, train=True, random_state=SEED):
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
    
    # REST CAN BE MOVED TO A COMMON FUNCTION
        
    # ------------------------------------------------------------
    # Step 7. Remove outliers and reduce dataset size per class.
    # ------------------------------------------------------------
    if REMOVEOUTLIERS:
        plot_cut_flow_for_all_filters(train_images, train_labels, save_path_prefix="./generator/filtering/")
        train_images, train_labels = remove_outliers(train_images, train_labels, PLOTFILTERED=True)
        eval_images, eval_labels = remove_outliers(eval_images, eval_labels, v="validation")
    
    if sample_size is not None and sample_size > 0:
        train_images, train_labels = reduce_by_class(train_images, train_labels, sample_size, len(target_classes))
        eval_images, eval_labels = reduce_by_class(eval_images, eval_labels, int(sample_size * 0.2), len(target_classes))
    
    # ------------------------------------------------------------
    # Step 8. Augment images.
    # ------------------------------------------------------------
    flips = [(False, False)] if AUGMENT else [(False, False), (True, False), (False, True), (True, True)] 
    train_images, train_labels = augment_images(train_images, train_labels, flips=flips)
    eval_images, eval_labels = augment_images(eval_images, eval_labels, flips=flips)
    
    return train_images, train_labels, eval_images, eval_labels


def load_PSZ2(path=root_path + "PSZ2/classified/", fold=5, BALANCE=False, AUGMENT=False, sample_size=300, target_classes=[53], crop_size=(1, 256, 256), downsample_size=(1, 256, 256), island=True, REMOVEOUTLIERS=True, train=False, SAVE_IMAGES=False, EXTRADATA=False, STRETCH=True):
    print("Loading PSZ2 data...")

    # if you passed a 4‐tuple, use its first entry as the number of versions to load
    if len(crop_size) == 4:
        num_versions, ch_c, h_c, w_c = crop_size
        crop_size = (ch_c, h_c, w_c)
    elif len(crop_size) == 3:
        num_versions = None
        ch_c, h_c, w_c = crop_size
    elif len(crop_size) == 2:
        num_versions = None
        ch_c, h_c, w_c = 1, crop_size[0], crop_size[1]
    else:
        raise ValueError("crop_size must be a 2, 3, or 4-tuple (num_versions, channels, height, width)")
    if len(downsample_size) == 4:
        num_versions, ch_d, h_d, w_d = downsample_size
        downsample_size = (ch_d, h_d, w_d)
    elif len(downsample_size) == 3:
        num_versions = None
        ch_d, h_d, w_d = downsample_size
    elif len(downsample_size) == 2:
        num_versions = None
        ch_d, h_d, w_d = 1, downsample_size[0], downsample_size[1] 
    else:
        raise ValueError("downsample_size must be a 2, 3, or 4-tuple (num_versions, channels, height, width)")
    
    CUBE = True if num_versions is not None else False

    
    version = 'T100kpcSUB' # if not CUBE
    version_folders = ['T50kpcSUB', 'T100kpcSUB']

    #version_folders = [
    #    "T15arcsec","T15SUBarcsec","T25kpc","T25kpcSUB",
    #    "T30arcsec","T30SUBarcsec","T50kpc","T50kpcSUB",
    #    "T100kpc","T100kpcSUB","XMM","CHANDRA","R-1.25",
    #    "compact-model",""   # the version-free name
    #]

    images = []
    labels = []
    filenames = []

    for cls in target_classes:
        class_folder = next(c["description"] for c in get_classes() if c["tag"] == cls)
        subfolders = [class_folder]
        label = cls
        
        if CUBE:
            version = 'CUBE'
            print("Cube mode enabled. Loading images as a cube of channels.")
            # only keep as many version_folders as the user asked for:
            vf_list = version_folders[:num_versions] if num_versions else version_folders
            for subfolder in subfolders:
                base_names = set()
                for vf in vf_list:
                    # load only .npy files
                    npy_path = os.path.join(root_path, "PSZ2/classified", vf, subfolder, f"{base}.npy")
                    if not os.path.isfile(npy_path):
                        # fallback to zeros if missing
                        frame = torch.zeros(ch_c, h_c, w_c)
                    else:
                        arr = np.load(npy_path).astype(float)
                        # percentile stretch to [0,1]
                        p2, p98 = np.percentile(arr, (2,98))
                        stretched = np.clip((arr - p2)/(p98 - p2 + 1e-12), 0, 1)
                        # convert to tensor (1, H, W)
                        frame = transforms.ToTensor()(stretched)
                    versions.append(frame)
                for base in base_names:
                    versions = []
                    for vf in vf_list:
                        img_path = os.path.join(root_path, "PSZ2/classified", vf, subfolder, f"{base}.png")
                        if os.path.isfile(img_path):
                            with Image.open(img_path) as img:
                                frame = transforms.ToTensor()(img)

                            # drop only on HxW mismatch
                            # if a version frame has the wrong size, zero‐pad it instead of dropping the whole source
                            if frame.shape[-2:] != (369, 369):
                                print(f"Zero‐padding base={base}, version={vf}: got HxW={frame.shape[-2:]}, expected {(369, 369)}")
                                frame = torch.zeros(ch_c, h_c, w_c)

                            # HOPEFUL WAY - FILL AND CUT TO RESIZE AND USE (Not tried yet)
                            ## ensure frame is exactly (h_c, w_c) by center‐cropping or zero‐padding
                            #actual_h, actual_w = frame.shape[-2], frame.shape[-1]
    #
                            ## if too large, crop centrally
                            #if actual_h > h_c or actual_w > w_c:
                            #    top = (actual_h - h_c) // 2
                            #    left = (actual_w - w_c) // 2
                            #    frame = frame[..., top:top+h_c, left:left+w_c]
    #
                            ## if too small, pad equally on all sides
                            #elif actual_h < h_c or actual_w < w_c:
                            #    pad_h = h_c - actual_h
                            #    pad_w = w_c - actual_w
                            #    pad_top = pad_h // 2
                            #    pad_bottom = pad_h - pad_top
                            #    pad_left = pad_w // 2
                            #    pad_right = pad_w - pad_left
                            #    frame = F.pad(frame, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

                            # **convert to your single‐channel + resize here!**
                            if ch_c != 4 or ch_d != 4 or h_c != 369 or h_d != 369 or w_c != 369 or w_d != 369:
                                frame = apply_formatting(frame, crop_size, downsample_size)
                        else:
                            frame = torch.zeros(ch_d, h_d, w_d)  # zero‐pad if no image found

                        # now every frame is the same (C,H,W), so this will work
                        versions.append(frame)

                    if len(versions) != len(vf_list):
                        print(f"⚠️  base={base} only has {len(versions)} / {len(vf_list)} versions!")

                    cube = torch.stack(versions, dim=0)  # [T, C, H, W]
                    images.append(cube)
                    labels.append(label)
                    filenames.append(base)           # ←—— add this

                print("Number of cubes added:", len(images), "each of shape", cube.shape)


        else:
            # --- only load .npy files, stretch to [0,1], save as PNG and collect tensors ---
            for subfolder in subfolders:
                folder_path = os.path.join(path, version, subfolder)
                if not os.path.isdir(folder_path):
                    print(f"Warning: {folder_path} not found, skipping.")
                    continue
                for fname in os.listdir(folder_path):
                    if not fname.lower().endswith('.npy'):
                        continue
                    npy_path = os.path.join(folder_path, fname)
                    arr = np.load(npy_path).astype(float)

                    # collapse singleton dims, then if you still have a 3D stack, average into one 2D frame
                    arr2 = np.squeeze(arr)
                    if arr2.ndim == 3:
                        arr2 = np.mean(arr2, axis=0)
                    elif arr2.ndim != 2:
                        raise ValueError(f"Expected 2-D array or 3-D stack, got shape {arr2.shape!r}")

                    raw = torch.from_numpy(arr2).unsqueeze(0).float() 
                    tensor = apply_formatting(raw,
                        crop_size=crop_size,
                        downsample_size=downsample_size,
                    ) 
                    images.append(tensor) # shape [1, C, H, W]
                    labels.append(label) # Set label to the same for all images in this subfolder
                    filenames.append(fname[:-4])  # remove the .npy extension
        

    if len(images) == 0:
        raise ValueError("No images loaded. Check the path and file extensions.")
    
    assert len(images) == len(labels) == len(filenames), \
       f"mismatch: {len(images)} imgs, {len(labels)} labels, {len(filenames)} files"

    # --- dedupe block, pulling in filenames too ---
    seen = {}                # hash -> first‐seen filename
    unique_images, unique_labels, unique_filenames = [], [], []
    duplicates = defaultdict(list)

    for img, lbl, fname in zip(images, labels, filenames):
        # make sure we hash the exact same raw array you later feed into the network
        arr = img.squeeze().numpy()  
        h = hashlib.sha1(arr.tobytes()).hexdigest()
        if h in seen:
            # record the duplicate filename
            duplicates[h].append(fname)
            continue
        # first time we see this hash
        seen[h] = fname
        unique_images.append(img)
        unique_labels.append(lbl)
        unique_filenames.append(fname)

    # report
    if duplicates:
        print("Found duplicate images (by hash):")
        for h, dups in duplicates.items():
            original = seen[h]
            print(f"  hash {h}:")
            print(f"    original: {original}")
            for dup in dups:
                print(f"    duplicate: {dup}")
    else:
        print("No exact‐byte duplicates found.")

    # overwrite
    images, labels, filenames = unique_images, unique_labels, unique_filenames

    # now do your split
    all_idx = list(range(len(images)))
    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=0.2,
        stratify=labels,
        random_state=SEED
    )
    print("train_idx", "test_idx lengths:", len(train_idx), len(test_idx))
    train_images = [images[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    eval_images  = [images[i] for i in test_idx]
    eval_labels  = [labels[i] for i in test_idx]

    print("Shape of train_images before redistribution:", len(train_images), train_images[0].shape)

    
    # REST CAN BE MOVED TO A COMMON FUNCTION

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
    
    if STRETCH:
        # percentile stretch
        p2, p98 = np.percentile(images, (2, 98))
        stretched = np.clip((images - p2) / (p98 - p2 + 1e-12), 0, 1)

        # Arcsin stretch (convert to Tensor with batch & channel dims)
        st_tensor = torch.from_numpy(stretched).unsqueeze(0).unsqueeze(0).float()   # shape [1,1,H,W]---
        tensor = torch.from_numpy(stretched).unsqueeze(0).float()
        print("Shape of stretched array :", stretched.shape)
        

    train_images, train_labels, eval_images, eval_labels = redistribute_excess(
        train_images, train_labels, eval_images, eval_labels, target_classes)
    
    if SAVE_IMAGES:
        for kind, imgs, lbls in (
            ('train', train_images, train_labels),
            ('eval',  eval_images,  eval_labels),
        ):
            for cls in target_classes:
                cls_imgs = [img for img, lbl in zip(imgs, lbls) if lbl == cls]
                print(f"Length of {kind} images for class {cls}: {len(cls_imgs)}")
                np.save(f"{path}_{kind}_{cls}_{len(cls_imgs)}.npy", cls_imgs)

    # Format and optionally augment
    if BALANCE:
        print("Counts per class before balancing:", collections.Counter(train_labels))
        train_images, train_labels = balance_classes(train_images, train_labels)
        print("Counts per class after balancing:", collections.Counter(train_labels))
        
    if EXTRADATA:
        meta_df = pd.read_csv(os.path.join(root_path, "PSZ2/cluster_source_data.csv"))
        print("PSZ2 metadata columns:", meta_df.columns.tolist())
        meta_df.rename(columns={"slug": "base"}, inplace=True)
        meta_df.set_index("base", inplace=True)

        # build a list of metadata rows in the same order as your `filenames` list:
        all_data = [meta_df.loc[base].values for base in filenames]
        train_data = [ all_data[i] for i in train_idx ]
        eval_data  = [ all_data[i] for i in test_idx  ]

    if AUGMENT:
        train_images, train_labels = augment_images(train_images, train_labels, ST_augmentation=False)
        eval_images, eval_labels = augment_images(eval_images, eval_labels, ST_augmentation=False)
        if EXTRADATA:
            n_aug = 8  # default is 4*2 = 8
            train_data = [row for row in train_data for _ in range(n_aug)]
            eval_data  = [row for row in eval_data  for _ in range(n_aug)]
    else:
        train_images = torch.stack(train_images)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        eval_images  = torch.stack(eval_images)
        eval_labels  = torch.tensor(eval_labels,  dtype=torch.long)
        
    print("Shape of train_images returned from PSZ2:", train_images.shape)
    
    if EXTRADATA:
        # (optional) convert to a tensor if you want a single numeric array
        train_data = torch.tensor(np.stack(train_data), dtype=torch.float32)
        eval_data  = torch.tensor(np.stack(eval_data),  dtype=torch.float32)

        # Return images, labels, AND metadata arrays instead of just IDs
        return train_images, train_labels, eval_images, eval_labels, train_data, eval_data
    return train_images, train_labels, eval_images, eval_labels


def load_MGCLS(path='/users/mbredber/data/MGCLS/classified_crops_1600/',  # Path to MGCLS data
               fold=None,
               target_classes=[40],
               crop_size=(1600, 1600),
               downsample_size=(128, 128),
               sample_size=300,
               island=True,
               REMOVEOUTLIERS=True,
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
    
    
    ## REST CAN BE MOVED TO A COMMON FUNCTION
    
    # Rebalance any leftover via your redistribute_excess
    train_images, train_labels, eval_images, eval_labels = redistribute_excess(
        train_images, train_labels, eval_images, eval_labels, target_classes)
    
    print("MGCLS images shape before augmentation:", len(train_images), "x", train_images[0].shape if train_images else "N/A")

    # stack into tensors (and optionally augment)
        # enforce equal class sizes before augmentation
    if BALANCE:
        train_images, train_labels = balance_classes(train_images, train_labels)

    if AUGMENT:
        train_images, train_labels = augment_images(train_images, train_labels, ST_augmentation=False)
        eval_images, eval_labels = augment_images(eval_images, eval_labels, ST_augmentation=False)
    else:
        train_images = torch.stack(train_images)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        eval_images  = torch.stack(eval_images)
        eval_labels  = torch.tensor(eval_labels,  dtype=torch.long)

    return train_images, train_labels, eval_images, eval_labels


def load_galaxies(galaxy_class, path=None, fold=None, island=None, crop_size=None, downsample_size=None, sample_size=None, REMOVEOUTLIERS=True, BALANCE=False, AUGMENT=False, STRETCH=False, train=None):
    """
    Master loader that delegates to specific dataset loaders and returns zero-based labels.
    """
    def get_max_class(galaxy_class):
        if isinstance(galaxy_class, list):
            return max(galaxy_class)
        return galaxy_class

    # Clean up kwargs to remove None values
    kwargs = {'path': path, 'sample_size': sample_size, 'fold':fold, 'train': train,
              'island': island, 'REMOVEOUTLIERS': REMOVEOUTLIERS, 'BALANCE': BALANCE, 'AUGMENT': AUGMENT,
              'crop_size': crop_size, 'downsample_size': downsample_size}
    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}

    max_class = get_max_class(galaxy_class)

    # Delegate to specific loaders based on class range
    if max_class < 10:
        path = root_path + 'Galaxy10.h5'
        target_classes = galaxy_class if isinstance(galaxy_class, list) else [galaxy_class]
        data = load_galaxy10(path=path, target_classes=target_classes, **clean_kwargs)
    elif max_class < 14:
        if clean_kwargs.get('crop_size') is not None:
            clean_kwargs['crop_size'] = np.squeeze(clean_kwargs['crop_size'])
        path = root_path + 'firstgalaxydata/'
        target_classes = galaxy_class if isinstance(galaxy_class, list) else [galaxy_class]
        data = load_FIRST(path=path, target_classes=target_classes, **clean_kwargs)
    elif max_class < 18:
        clean_kwargs['crop_size'] = np.squeeze(clean_kwargs.get('crop_size', None))
        target_classes = galaxy_class if isinstance(galaxy_class, list) else [galaxy_class]
        data = load_Mirabest(target_classes=target_classes, **clean_kwargs)
    elif max_class < 30:
        target_classes = galaxy_class if isinstance(galaxy_class, list) else [galaxy_class]
        data = load_MNIST(target_classes=target_classes, **clean_kwargs)
    elif max_class < 37:
        target_classes = galaxy_class if isinstance(galaxy_class, list) else [galaxy_class]
        data = load_RGZ10k(target_classes=target_classes, **clean_kwargs)
    elif max_class < 50:
        target_classes = galaxy_class if isinstance(galaxy_class, list) else [galaxy_class]
        data = load_MGCLS(target_classes=target_classes, **clean_kwargs)
    elif max_class <= 59:
        target_classes = galaxy_class if isinstance(galaxy_class, list) else [galaxy_class]
        data = load_PSZ2(target_classes=target_classes, **clean_kwargs)
    else:
        raise ValueError("Invalid galaxy class provided.")

    # Check for overlap between train and test sets
    print("Data loaded:", len(data[0]), "train images,", len(data[2]) if len(data) > 2 else 0, "eval images.")
    train_hashes = {img_hash(img) for img in data[0]}
    eval_images = data[2] if len(data) > 2 else []
    if len(eval_images) != 0:
        test_hashes = {img_hash(img) for img in eval_images}
        common = train_hashes & test_hashes
        if common:
            overlap_hash = next(iter(common))
            train_idxs = [i for i,img in enumerate(data[0]) if img_hash(img) == overlap_hash]
            test_idxs  = [i for i,img in enumerate(data[2]) if img_hash(img) == overlap_hash]
            print(f"🔍 Overlap hash {overlap_hash!r} found at train indices {train_idxs} and test indices {test_idxs}")
            assert False, f"Overlap detected: {len(common)} images appear in both train and test validation!"

    return data