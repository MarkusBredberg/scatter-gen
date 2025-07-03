import os
from PIL import Image

folder = "/users/mbredber/scratch/data/PSZ2/classified/T100kpcSUB/DE"
for fname in os.listdir(folder):
    if fname.lower().endswith(('.png','.jpg','.jpeg')):
        path = os.path.join(folder, fname)
        with Image.open(path) as img:
            print(f"{fname}: mode={img.mode}, size={img.size}")