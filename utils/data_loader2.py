import numpy as np
import h5py
import torch
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from torchvision.io import read_image
from scipy.ndimage import label
from pathlib import Path
from torch.utils.data import DataLoader
import cv2
import random


######################################################################################################
################################### CONFIGURATION ####################################################
######################################################################################################

root_path =  '/users/mbredber/scratch/data/'   ## '/home/markus/Scripts/data/'
SEED = 42


######################################################################################################
################################### GENERAL STUFF ####################################################
######################################################################################################

import sys
sys.path.append(root_path)
from MiraBest.MiraBest_F import MBFRConfident, MBFRUncertain, MBFRFull, MBRandom
from firstgalaxydata import FIRSTGalaxyData

np.random.seed(SEED)
torch.manual_seed(SEED)

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
        {"tag": 18, "length": 6000, "description": "Digit Eight"},
        {"tag": 19, "length": 6000, "description": "Digit Nine"},
        {"tag": 20, "length": 60000, "description": "All Digits"},
        {"tag": 21, "length": 6000, "description": "Digit One"},
        {"tag": 22, "length": 6000, "description": "Digit Two"},
        {"tag": 23, "length": 6000, "description": "Digit Three"},
        {"tag": 24, "length": 6000, "description": "Digit Four"},
        {"tag": 25, "length": 6000, "description": "Digit Five"},
        {"tag": 26, "length": 6000, "description": "Digit Six"},
        {"tag": 27, "length": 6000, "description": "Digit Seven"},
        {"tag": 28, "length": 6000, "description": "Digit Eight"},
        {"tag": 29, "length": 6000, "description": "Digit Nine"}
    ]


######################################################################################################
################################### DATA SELECTION FUNCTIONS #########################################
######################################################################################################


def remove_outliers(images, labels, threshold=0.3, peak_threshold=0.6, intensity_threshold=200.0, required_regions=2, region_size=(64, 64)):
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
    
    def calculate_outside_emission(image):
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

    def count_emission_regions(image, emission_threshold=0.1):
        """Count the number of distinct emission regions using connected component analysis."""
        image_np = image.squeeze().numpy() if image.dim() == 3 else image.numpy()  # Squeeze to remove channel dimension if present
        binary_image = image_np > emission_threshold  # Apply a threshold to the image to create a binary map of emission regions
        labeled_image, num_features = label(binary_image)  # Label connected regions in the binary image
        return num_features

    filtered_images, filtered_labels = [], []
    removed_by_emission, removed_by_intensity, removed_by_peak_intensity, removed_by_regions = [], [], [], []
    
    for image, lbl in zip(images, labels):  # Renamed 'label' to 'lbl' to avoid conflict
        outside_emission_fraction, total_emission = calculate_outside_emission(image)
        num_regions = count_emission_regions(image)
        
        if outside_emission_fraction > threshold and total_emission <= intensity_threshold:
            removed_by_emission.append((image, lbl))  # Remove images with emission outside threshold
        elif total_emission > intensity_threshold:
            removed_by_intensity.append((image, lbl))  # Remove images with total intensity above threshold
        elif num_regions != required_regions:
            removed_by_regions.append((image, lbl))  # Remove images with too many regions
        else:
            filtered_images.append(image)  
            filtered_labels.append(lbl)  

    # Peak intensity filtering as before
    def filter_by_peak_intensity(images, labels, threshold):
        removed_by_peak = []
        filtered_images_by_peak = []
        filtered_labels_by_peak = []
        
        for image, lbl in zip(images, labels):
            center_x, center_y = image.shape[-2] // 2, image.shape[-1] // 2
            half_size = region_size[0] // 2  # Half of the central region size
            
            central_region = image[..., center_x-half_size:center_x+half_size, center_y-half_size:center_y+half_size]
            
            if torch.max(central_region) <= threshold:
                removed_by_peak.append((image, lbl))
            else:
                filtered_images_by_peak.append(image)
                filtered_labels_by_peak.append(lbl)
                
        return filtered_images_by_peak, filtered_labels_by_peak, removed_by_peak
    
    filtered_images, filtered_labels, removed_by_peak_intensity = filter_by_peak_intensity(filtered_images, filtered_labels, peak_threshold)

    # Compile the final filtered list of images and labels
    final_filtered_images = [img for img in filtered_images 
                            if not any(torch.equal(img, removed_img[0]) for removed_img in removed_by_peak_intensity)]
    final_filtered_labels = [lbl for img, lbl in zip(filtered_images, filtered_labels) 
                            if not any(torch.equal(img, removed_img[0]) for removed_img in removed_by_peak_intensity)]
    fraction_final_removed = 1 - len(final_filtered_images) / len(images)

    # Print statistics
    print(f"Total images: {len(images)}")
    print(f"Images removed by intensity (> {intensity_threshold}): {len(removed_by_intensity)}")
    print(f"Images removed by outside emission (> {threshold} fraction): {len(removed_by_emission)}")
    print(f"Images removed by peak intensity (< {peak_threshold}): {len(removed_by_peak_intensity)}")
    print(f"Images removed by emission regions (!= {required_regions} regions): {len(removed_by_regions)}")
    print("Fraction of images removed:", fraction_final_removed)
    print("Number of images removed in total:", len(images) - len(final_filtered_images))
    
    return final_filtered_images, final_filtered_labels


def generate_transformation_configs(n):
    """
    Generate a systematic list of transformations.
    Each transformation includes: rotation, flip, and affine transformations.
    """
    rotations = [0, 90, 180, 270]  # Predefined rotation angles
    flips = [(False, False), (True, False), (False, True), (True, True)]  # Predefined flip (horiz, vert)    
    transformation_configs = []
    
    # Create a combination of transformations by looping through predefined options
    for rot in rotations:
        for flip_h, flip_v in flips:
            transformation_configs.append({'rotation': rot, 'flip_h': flip_h, 'flip_v': flip_v})

    # Repeat the transformation configs if the number of samples exceeds the predefined configurations
    multiplier = (n // len(transformation_configs)) + 1
    return (transformation_configs * multiplier)[:n]  # Trim to n transformations

# Using systematic transformations instead of random choices in augmentation
def apply_transforms_with_config(image, config, img_shape=(128, 128)):
    preprocess = transforms.Compose([
        transforms.CenterCrop((img_shape[-2], img_shape[-1])),
        transforms.Resize((img_shape[-2], img_shape[-1])),  # Resize images to the desired size
        transforms.Lambda(lambda x: transforms.functional.hflip(x) if config['flip_h'] else x),  # Horizontal flip
        transforms.Lambda(lambda x: transforms.functional.vflip(x) if config['flip_v'] else x),  # Vertical flip
        transforms.Lambda(lambda x: transforms.functional.rotate(x, config['rotation']))  # Rotation by specific angle
    ])
    
    if image.dim() == 2:  # Ensure image is 3D
        image = image.unsqueeze(0)
    image = preprocess(image)
    
    return image

        

def apply_transforms(image, img_shape=(128, 128), island=False):

    class AddGaussianNoise(object):
        def __init__(self, mean=0., std=0.01):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            noise = torch.randn_like(tensor) * self.std + self.mean
            noised_image = tensor + noise
            noised_image_clipped = torch.clamp(noised_image, 0, 1)  # Clip the values to be within [0, 1]
            return noised_image_clipped

        def __repr__(self):
            return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

    preprocess = transforms.Compose([
        transforms.CenterCrop((img_shape[-2], img_shape[-1])),
        transforms.Resize((img_shape[-2], img_shape[-1])),  # Resize images to the desired size
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
        transforms.RandomVerticalFlip(p=0.5),  # Random vertical flip
        #transforms.RandomRotation(degrees=90),  # Random rotation by up to 90 degrees
        transforms.Lambda(lambda x: transforms.functional.rotate(x, random.choice([0, 90, 180, 270]))), # Random rotation by 0, 90, 180, or 270 degrees
        #transforms.RandomAffine(degrees=0, translate=(0.01, 0.1)),  # Random translation by up to 1% of the image size
        #transforms.ColorJitter(brightness=(1, 1.05)),  # Random brightness increase from 0% to 10%
        #AddGaussianNoise(0., 0.05)  # Add Gaussian noise with mean and std
    ])

    if image.dim() == 2:  # Ensure image is 3D
        image = image.unsqueeze(0)
    image = preprocess(image)
    
    return image


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
        vae_images_list = []
        vae_labels_list = []
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
    print(f"Total images: {total_images}")
    print(f"Images removed: {removed_images}")
    print(f"Fraction removed: {removed_fraction:.2f}")

    # Return the accepted images
    return accepted_images

def load_galaxy10(path=root_path + 'Galaxy10.h5', sample_size=300, target_classes=[4], 
                  img_shape=(3, 256, 256), process=True, fold=None, island=True, runname=None, 
                  generated=False, reconstructed=False, train=False):
    print("Loading Galaxy10 data...")

    try:
        with h5py.File(path, 'r') as F:
            images = np.array(F['images'])  # This step takes 10-15 seconds
            labels = np.array(F['ans']).astype(int)

            all_images = []
            all_labels = []

            # Filter for target classes
            for cls in target_classes:
                class_indices = np.where(labels == cls)[0]
                available_samples = len(class_indices)
                np.random.seed(SEED)
                torch.manual_seed(SEED)
                np.random.shuffle(class_indices)

                # Handle the case where available samples are fewer than requested
                if available_samples < sample_size:
                    class_indices = np.tile(class_indices, int(np.ceil(sample_size / available_samples)))[:sample_size]
                else:
                    class_indices = class_indices[:sample_size]

                selected_images = 2 * (images[class_indices].astype(np.float32) / 255.0 - .5)

                # Convert to grayscale if necessary
                if img_shape[0] == 1:  
                    selected_images = np.mean(selected_images, axis=-1, keepdims=True)

                class_images = []
                class_labels = []

                # Apply isolation or filtering to the selected images
                selected_images = np.moveaxis(selected_images, -1, 1)
                filtered_images = isolate_galaxy_batch(selected_images)

                # Ensure that the filtered images match the desired shape
                for image in filtered_images:
                    image = torch.tensor(image, dtype=torch.float)
                    if process or img_shape != 256:
                        image = apply_transforms(image, img_shape, island=island)

                    class_images.append(image)
                    class_labels.append(cls)

                # Shuffle class-specific images and labels
                if class_images:  
                    class_images_tensor = torch.stack(class_images)
                    class_labels_tensor = torch.tensor(class_labels, dtype=torch.long)

                    # Shuffle indices and update tensors
                    shuffled_indices = torch.randperm(len(class_images_tensor))
                    class_images_tensor = class_images_tensor[shuffled_indices]
                    class_labels_tensor = class_labels_tensor[shuffled_indices]

                    all_images.append(class_images_tensor)
                    all_labels.append(class_labels_tensor)

            # Concatenate all class-specific tensors
            if all_images:
                all_images = torch.cat(all_images).clone().detach()
                all_labels = torch.cat(all_labels).clone().detach()

            if runname is not None:
                if generated:
                    all_images, all_labels = load_vae_data(runname, sample_size, all_images, all_labels, data_type='generated')

            # Stratified k-fold cross-validation setup
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

            # Ensure that the fold parameter is valid
            print("fold", fold)
            if fold is None or fold < 0 or fold >= 5:
                raise ValueError("Fold must be an integer between 0 and 4")

            # Split the data into the appropriate fold
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(all_images, all_labels)):
                if fold_idx == fold:
                    train_images = all_images[train_idx]
                    train_labels = all_labels[train_idx]
                    test_images = all_images[test_idx]
                    test_labels = all_labels[test_idx]
                    break

            # Augment the training data if necessary
            while len(train_images) < sample_size:
                idx = np.random.randint(0, len(train_images))
                augmented_image = apply_transforms(train_images[idx], img_shape, island=True)
                train_images = torch.cat((train_images, augmented_image.unsqueeze(0)), dim=0)
                train_labels = torch.cat((train_labels, train_labels[idx].unsqueeze(0)), dim=0)

            # Shuffle the train and test sets
            train_indices = torch.randperm(len(train_images))
            train_images = train_images[train_indices]
            train_labels = train_labels[train_indices]

            test_indices = torch.randperm(len(test_images))
            test_images = test_images[test_indices]
            test_labels = test_labels[test_indices]

            if not train:
                return train_images, train_labels
            return train_images, train_labels, test_images, test_labels
    except OSError as e:
        print(f"Failed to open the file: {e}")


def load_FIRST(path, fold, target_classes=None, img_shape=(1, 300, 300), sample_size=300, process=False,
               island=False, runname=None, generated=False, train=False, reconstructed=False):
    print("Loading FIRST data...")
    print("fold", fold)
    
    class_mapping = {11: 'FRI', 12: 'FRII', 13: 'Compact', 14: 'Bent'}
    target_classes = [class_mapping.get(tc, tc) for tc in target_classes]
    
    def get_data(data):
        images, labels = [], []
        for item in data:
            images.append(item[0])
            labels.append(item[1])
        return images, labels

    if fold is not None and fold >= 0 and fold < 6:
        if fold == 5: 
            train_data = FIRSTGalaxyData(root="./", selected_split="train", input_data_list=[f"galaxy_data_h5.h5"],
                                         selected_classes=target_classes, is_PIL=True, is_RGB=False, transform=transforms.ToTensor())
            test_data = FIRSTGalaxyData(root="./", selected_split="test", input_data_list=[f"galaxy_data_h5.h5"],
                                        is_PIL=True, is_RGB=False, transform=transforms.ToTensor())
            train_images, train_labels = get_data(train_data)
            valid_images, valid_labels = get_data(test_data)
        else:
            train_data = FIRSTGalaxyData(root="./", selected_split="train", input_data_list=[f"galaxy_data_crossvalid_{fold}_h5.h5"],
                                         selected_classes=target_classes, is_PIL=True, is_RGB=False, transform=transforms.ToTensor())
            valid_data = FIRSTGalaxyData(root="./", selected_split="valid", input_data_list=[f"galaxy_data_crossvalid_{fold}_h5.h5"],
                                         selected_classes=target_classes, is_PIL=True, is_RGB=False, transform=transforms.ToTensor())
            train_images, train_labels = get_data(train_data)
            valid_images, valid_labels = get_data(valid_data)
        
        if len(valid_images) == 0:
            raise ValueError("No valid images found. Check the dataset and loading process.")
        
        train_images_tensor = torch.stack(train_images)
        print("Original training images shape before augmentation:", train_images_tensor.shape)
        valid_images_tensor = torch.stack(valid_images)
        print("Original valid images shape before augmentation:", valid_images_tensor.shape)
        
        train_images, train_labels = remove_outliers(train_images, train_labels)
        valid_images, valid_labels = remove_outliers(valid_images, valid_labels)

        # Ensure all images have consistent shape
        processed_train_images = [apply_transforms(img, img_shape) for img in train_images]
        processed_train_labels = train_labels[:]  # Copy the original labels
        
        processed_valid_images = [apply_transforms(img, img_shape) for img in valid_images]
        processed_valid_labels = valid_labels[:]  # Copy the original labels
        
        if False:
            # Augment train_images until reaching sample_size
            while len(processed_train_images) < sample_size:
                idx = np.random.randint(0, len(train_images))
                augmented_image = apply_transforms(train_images[idx].clone().detach(), img_shape, island)
                processed_train_images.append(augmented_image)
                processed_train_labels.append(train_labels[idx])  # Append the corresponding label
            
            # Ensure valid_images is at least 20% of the train_images
            while len(processed_valid_images) < max(20, int(0.2 * len(processed_train_images))):
                idx = np.random.randint(0, len(valid_images))
                augmented_image = apply_transforms(valid_images[idx].clone().detach(), img_shape, island)
                processed_valid_images.append(augmented_image)
                processed_valid_labels.append(valid_labels[idx])  # Append the corresponding label
                
        # Augmentation loop using systematic transformations
        else:  # Augment each image with unique transformations (rotation and flip)
            transformation_configs = generate_transformation_configs(len(train_images))
            complete_train_images, complete_train_labels = [], []  # List to store augmented images
            for idx, image in enumerate(train_images):
                rotations = [0, 90, 180, 270]
                flips = [(False, False), (True, False), (False, True), (True, True)]
                for rot in rotations:
                    for flip_h, flip_v in flips:
                        config = {'rotation': rot, 'flip_h': flip_h, 'flip_v': flip_v}
                        augmented_image = apply_transforms_with_config(image.clone().detach(), config, img_shape)
                        complete_train_images.append(augmented_image)
                        complete_train_labels.append(train_labels[idx])  # Append the corresponding label
            
            # Augment each image with unique transformations (rotation and flip)
            transformation_configs = generate_transformation_configs(len(valid_images))
            complete_valid_images, complete_valid_labels = [], []  # List to store augmented images
            for idx, image in enumerate(valid_images):
                rotations = [0, 90, 180, 270]
                flips = [(False, False), (True, False), (False, True), (True, True)]
                for rot in rotations:
                    for flip_h, flip_v in flips:
                        config = {'rotation': rot, 'flip_h': flip_h, 'flip_v': flip_v}
                        augmented_image = apply_transforms_with_config(image.clone().detach(), config, img_shape)
                        complete_valid_images.append(augmented_image)
                        complete_valid_labels.append(valid_labels[idx])  # Append the corresponding label

        #complete_train_images = processed_train_images
        #complete_valid_images = processed_valid_images
        #complete_train_labels = processed_train_labels
        #complete_valid_labels = processed_valid_labels


        # Shuffle the augmented images and corresponding labels
        shuffle_indices = torch.randperm(len(valid_images))  # Generate a random permutation of indices
        valid_images = valid_images[shuffle_indices] 
        valid_labels = valid_labels[shuffle_indices] 

        # Assert that the number of labels matches the number of images
        assert len(complete_valid_labels) == len(valid_images)
        
        if not train:
            return train_images, train_labels
        return train_images, train_labels, valid_images, valid_labels

    # If not FFCV, load the data as usual        
    data_splits = {}
    splits = ["train", "test"] if train else ["train"]

    for s in splits:
        images = []
        labels = []
        for idx, class_label in enumerate(target_classes):
            class_path = Path(path) / 'galaxy_data'/ s / class_label
            all_img_paths = list(class_path.glob('*.png'))

            if not all_img_paths:
                raise RuntimeError(f"No images found for class '{class_label}' in {class_path}")
            num_images = len(all_img_paths)

            if num_images < sample_size:
                # If there aren't enough images, augment the data
                np.random.seed(SEED)
                torch.manual_seed(SEED)
                img_paths = np.random.choice(all_img_paths, size=sample_size, replace=True)
            else:
                # Otherwise shuffle and slice to desired sample size
                np.random.seed(SEED)
                torch.manual_seed(SEED)
                np.random.shuffle(all_img_paths)
                img_paths = all_img_paths[:sample_size]

            class_images, class_labels = [], []
            for img_path in img_paths:
                image = read_image(str(img_path)).type(torch.FloatTensor) / 255.0
                class_images.append(image)
                class_labels.append(idx)

            if len(class_images) == 0:
                raise RuntimeError(f"Class '{class_label}' in {class_path} is empty")

            # Debug: Check if all images have the same shape
            image_shapes = [img.shape for img in class_images]
            if len(set(image_shapes)) != 1:
                raise ValueError(f"Inconsistent image shapes found: {image_shapes}")

            # Convert lists to tensors
            class_images_tensor = torch.stack(class_images)
            class_labels_tensor = torch.tensor(class_labels)

            images.append(class_images_tensor)
            labels.append(class_labels_tensor)

        # Concatenate all class-specific tensors
        images_tensor = torch.cat(images)
        labels_tensor = torch.cat(labels)

        # Shuffle the concatenated tensors
        shuffled_indices = torch.randperm(len(images_tensor))
        images_tensor = images_tensor[shuffled_indices]
        labels_tensor = labels_tensor[shuffled_indices]
        
        # Apply the outlier removal function
        images_tensor, fraction_removed = remove_outliers(images_tensor, labels_tensor, img_shape=img_shape[1:])
        print(f"Fraction of {s} images removed: {fraction_removed:.2f}")
        
        processed_images = []
        if process or img_shape[1] != 300:
            for image in images_tensor:
                processed_image = apply_transforms(image, img_shape, island)
                processed_images.append(processed_image)
            images_tensor = torch.stack(processed_images)
        else:
            images_tensor = torch.stack(images_tensor)
            
        data_splits[s] = {'images': images_tensor, 'labels': labels_tensor[:len(images_tensor)]}        

    if train:
        train_images, train_labels = data_splits['train']['images'], data_splits['train']['labels']
        test_images, test_labels = data_splits['test']['images'], data_splits['test']['labels']               
        return train_images, train_labels, test_images, test_labels
    else:
        images, labels = [], []
        for s in splits:
            images.extend(data_splits[s]['images'])
            labels.extend(data_splits[s]['labels'])
        images_tensor = torch.stack(images)
        labels_tensor = torch.cat(labels)
        return images_tensor, labels_tensor




def load_Mirabest(target_classes=[16], img_shape=(1, 150, 150), sample_size=397, process=False,
                  train=False, island=False, runname=None, generated=False, reconstructed=False):
    print("Loading MiraBest data...")

    # Select the dataset class based on `target_classes`
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
                        if process or img_shape[1] != 150:
                            processed_image = apply_transforms(image, img_shape, island)
                        images.append(processed_image)
                        labels.append(label.item())
                        # Stop once we have enough samples
                        if len(images) >= sample_size:
                            break
                if len(images) >= sample_size:
                    break
            
            # Handle training data differently from testing/validation data
            if split == 'train':
                if runname is not None:
                    # Load additional VAE-generated or reconstructed images if specified
                    if generated:
                        images, labels = load_vae_data(runname, sample_size, torch.stack(images), torch.tensor(labels), data_type='generated')
                    if reconstructed:
                        images, labels = load_vae_data(runname, sample_size, torch.stack(images), torch.tensor(labels), data_type='reconstructed')
                
                # Augment the training data if not enough images are available
                while len(images) < sample_size:
                    idx = np.random.randint(0, len(original_images))
                    if idx < len(images):  # Ensure the index is within bounds
                        augmented_image = apply_transforms(original_images[idx].clone().detach(), img_shape, island)
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
                    preprocessed_image = apply_transforms(img, img_shape, island)
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





def load_MNIST(target_classes=[20], img_shape=(1, 28, 28), sample_size=60000, process=False,
               train=False, island=False, runname=None, generated=False, reconstructed=False, batch_size=32):
    print("Loading MNIST data...")
    train_data = datasets.MNIST(root=root_path, train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    train_images = []
    train_labels = []

    if set(target_classes) & {20}:
        digit_classes = list(range(10))  # Include all digits if 20 is in target_classes
    else:
        digit_classes = [tc % 10 for tc in target_classes]  # Map each target_class to a digit

    for batch_images, batch_labels in train_loader:
        for i, label in enumerate(batch_labels):
            if label.item() in digit_classes:
                image = batch_images[i]
                if process or img_shape[1] != 28:
                    image = apply_transforms(image, img_shape, island)
                train_images.append(image)
                train_labels.append(label.item())
                if len(train_images) >= sample_size:
                    break
        if len(train_images) >= sample_size:
            break

    train_images = torch.stack(train_images).clone().detach()
    train_labels = torch.tensor(train_labels, dtype=torch.long)

    if runname is not None:
        if generated:
            train_images, train_labels = load_vae_data(runname, sample_size, train_images, train_labels, data_type='generated')
        if reconstructed:
            train_images, train_labels = load_vae_data(runname, sample_size, train_images, train_labels, data_type='reconstructed')

    # Augment the data if there still aren't enough images
    while len(train_images) < sample_size:
        idx = np.random.randint(0, len(train_images))
        augmented_image = apply_transforms(train_images[idx], img_shape, island)
        train_images = torch.cat((train_images, augmented_image.unsqueeze(0)), dim=0)
        train_labels = torch.cat((train_labels, train_labels[idx].unsqueeze(0)), dim=0)

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
                    if process or img_shape[1] != 28:
                        image = apply_transforms(image, img_shape, island=False)
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

classes = get_classes()

def load_galaxies(galaxy_class, fold=None, island=None, img_shape=None, sample_size=None, process=None, train=None, runname=None, generated=False, reconstructed=False):

    def get_max_class(galaxy_class):
        if isinstance(galaxy_class, list):
            return max(galaxy_class)
        return galaxy_class

    # Clean up kwargs to remove None values
    kwargs = {'sample_size': sample_size, 'img_shape': img_shape, 'fold':fold, 'process': process, 'train': train, 'island': island, 'runname': runname, 'generated': generated, 'reconstructed': reconstructed}
    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}

    max_class = get_max_class(galaxy_class)

    if max_class < 10:  # Galaxy10 optical galaxies
        path =  root_path +  'Galaxy10.h5'
        target_classes = galaxy_class if isinstance(galaxy_class, list) else [galaxy_class]
        data = load_galaxy10(path=path, target_classes=target_classes, **clean_kwargs)

    elif max_class < 14:  # FIRST radio galaxies
        if 'img_shape' in clean_kwargs:
            clean_kwargs['img_shape'] = np.squeeze(clean_kwargs['img_shape'])
        path =  root_path +  'firstgalaxydata/'
        target_classes = [classes[c]['description'] for c in galaxy_class] if isinstance(galaxy_class, list) else [classes[galaxy_class]['description']]
        data = load_FIRST(path=path, target_classes=target_classes, **clean_kwargs)

    elif max_class < 18:  # MiraBest radio galaxies
        if 'img_shape' in clean_kwargs:
            clean_kwargs['img_shape'] = np.squeeze(clean_kwargs['img_shape'])
        target_classes = galaxy_class if isinstance(galaxy_class, list) else [galaxy_class]
        data = load_Mirabest(target_classes=target_classes, **clean_kwargs)

    elif max_class < 30:  # MNIST digits
        target_classes = galaxy_class if isinstance(galaxy_class, list) else [galaxy_class]
        data = load_MNIST(target_classes=target_classes, **clean_kwargs)

    else:
        raise ValueError("Invalid galaxy class provided.")
    return data
