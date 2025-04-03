# Script that loads in data and saves it to a numpy ngz file
import numpy as np
from utils.data_loader import load_galaxies
from utils.plotting import plot_histograms

# Constants
REMOVEOUTLIERS = True
image_size = (1, 128, 128)
galaxy_class = 11
fold = 5

data = load_galaxies(
        galaxy_class=galaxy_class,
        fold=fold,
        img_shape=image_size,
        sample_size=100000,
        REMOVEOUTLIERS=REMOVEOUTLIERS,
        train=True
    )
train_images, _, test_images, _ = data

# Plot histograms of the data

plot_histograms(
    train_images, test_images,
    title1='Training input',
    title2='Test input',
    bins=50,
    main_title="Pixel Value Distribution",
    save_path=f'histogram_galaxies_{galaxy_class}_fold{fold}.png'
)


# Save the data to a numpy file
np.savez(
    f"galaxies_{galaxy_class}_fold{fold}.npz",
    images=train_images,
)
