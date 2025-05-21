import os
from PIL import Image
import numpy as np

# point this at the folder containing your PNGs
folder = "/users/mbredber/scratch/data/PSZ2/classified/T100kpcSUB/DE"
for fname in os.listdir(folder):
    if fname.lower().endswith('.png'):
        path = os.path.join(folder, fname)
        arr = np.asarray(Image.open(path))
        print(f"{fname}: {arr.shape}")
