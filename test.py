import os
import numpy as np
from astropy.io import fits

# Directory containing your FITS files
fits_dir = "/users/mbredber/scratch/create_image_sets_outputs/processed_psz2_fits"

# Loop through all FITS files
for fname in sorted(os.listdir(fits_dir)):
    if not fname.lower().endswith(".fits"):
        continue

    fpath = os.path.join(fits_dir, fname)
    try:
        with fits.open(fpath, memmap=False) as hdul:
            data = hdul[0].data
            if data is None:
                print(f"[EMPTY] {fname}")
                continue

            n_nans = np.isnan(data).sum()
            if n_nans > 0:
                print(f"[NaN found] {fname} — {n_nans} NaNs")
    except Exception as e:
        print(f"[ERROR] {fname}: {e}")
