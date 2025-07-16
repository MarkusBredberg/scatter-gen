import os
from astropy.io import fits
import numpy as np

directory = "/users/mbredber/scratch/data/PSZ2/classified/T100kpcSUB/DE"

for fname in sorted(os.listdir(directory)):
    path = os.path.join(directory, fname)
    if not os.path.isfile(path):
        continue

    size_bytes = os.path.getsize(path)
    try:
        if fname.lower().endswith('.fits'):
            arr = fits.getdata(path)
        elif fname.lower().endswith('.npy'):
            arr = np.load(path)
        else:
            print(f"{fname}: unsupported extension, size={size_bytes} bytes")
            continue
        print(f"{fname}: shape={arr.shape}, size={size_bytes} bytes")
    except Exception as e:
        print(f"{fname}: error loading ({e}), size={size_bytes} bytes")
