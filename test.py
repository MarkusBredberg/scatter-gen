# Script that loads in data and saves it to a numpy ngz file
import numpy as np
from utils.data_loader import load_galaxies

# Constants
REMOVEOUTLIERS = True
image_size = (1, 128, 128)
galaxy_class = 10
fold = 5

data = load_galaxies(
        galaxy_class=galaxy_class,
        fold=fold,
        img_shape=image_size,
        sample_size=100000,
        REMOVEOUTLIERS=REMOVEOUTLIERS,
        train=True
    )
train_images, _, _, _ = data

# Save the data to a numpy file
np.savez(
    f"galaxies_{galaxy_class}_fold{fold}.npz",
    images=train_images,
)
