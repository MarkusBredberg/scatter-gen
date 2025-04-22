import glob
import os
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

output_dir = "data/LOTSSDR2"
thumb_dir = os.path.join(output_dir, "thumbs")
os.makedirs(thumb_dir, exist_ok=True)

def make_thumbnail(fits_path, out_png, size=(256,256)):
    # start a fresh figure of the right pixel size
    plt.figure(figsize=(size[0]/100, size[1]/100))
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(float)
    # clip stretch
    vmax = np.percentile(data, 99)
    vmin = np.percentile(data, 1)
    plt.imshow(data, origin='lower', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()

for fits_path in glob.glob(os.path.join(output_dir, "*mosaic*.fits")):
    name = os.path.splitext(os.path.basename(fits_path))[0] + ".png"
    out_png = os.path.join(thumb_dir, name)
    make_thumbnail(fits_path, out_png)

print("All done!")
