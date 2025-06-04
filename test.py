import numpy as np
from PIL import Image
from utils.data_loader import load_galaxies
import os

# —– PARAMETERS (edit these two lines) —–
noise_sigma_list = [0, 10, 30, 60, 90]  # σ of Gaussian noise in intensity units

train_imgs, train_lbls, test_images, test_labels  = load_galaxies(galaxy_class=11,
            fold=5, #Any fold other than 5 gives me the test data for the five fold cross validation
            crop_size=(128, 128),
            downsample_size=(128, 128),
            sample_size=100, 
            REMOVEOUTLIERS=True,
            AUGMENT=False,
            train=False)

output_dir = 'figure'
os.makedirs(output_dir, exist_ok=True)

# Convert tensor to NumPy and drop the singleton channel dim for PIL compatibility
print("shape of input image:", train_imgs[-1].shape)
arr = np.array(train_imgs[-1]).squeeze().astype(np.float32) * 255.0
print("Shape of array after squeezing:", arr.shape)

# Generate and save noisy layers
for idx, sigma in enumerate(noise_sigma_list, start=1):
    noise = np.random.normal(0, sigma, arr.shape)
    diffuse = np.clip(arr + noise, 0, 255).astype(np.uint8)
    Image.fromarray(diffuse).save(
        os.path.join(output_dir, f'layer{idx}_σ{sigma}.png')
    )
