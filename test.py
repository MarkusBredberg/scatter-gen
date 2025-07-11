import os
import numpy as np
import hashlib
import matplotlib.pyplot as plt

# adjust these three to your setup
root_path   = '/users/mbredber/scratch/data/'
version     = 'T100kpcSUB'
class_name  = 'NDE'   # the subfolder for tag==53
hash_target = '1adc95bebe9eea8c112d40cd04ab7a8d75c4f961'

folder = os.path.join(root_path, 'PSZ2', 'classified', version, class_name)
group  = []

for fname in sorted(os.listdir(folder)):
    if not fname.endswith('.npy'):
        continue

    # 1) load the raw array
    arr = np.load(os.path.join(folder, fname)).squeeze()

    # 2) if it’s a stack of channels, collapse them
    if arr.ndim == 3:
        arr = arr.mean(axis=0)

    # 3) take the *central* 128×128 (just like apply_formatting with a centre-crop)
    h, w = arr.shape
    cy, cx = h//2, w//2
    crop = arr[cy-64:cy+64, cx-64:cx+64]

    # 4) hash the raw bytes
    hval = hashlib.sha1(crop.tobytes()).hexdigest()
    if hval == hash_target:
        group.append((fname, crop))

if not group:
    raise RuntimeError(f"No files hashed to {hash_target}; check version/class or your hash.")

# plot them side by side
n = len(group)
fig, axes = plt.subplots(1, n, figsize=(n*2, 2))
for ax, (name, im) in zip(axes, group):
    ax.imshow(im, cmap='gray')
    ax.set_title(name, fontsize=7)
    ax.axis('off')

out = os.path.join(root_path, f'duplicates_{hash_target}.png')
plt.tight_layout()
plt.savefig(out, dpi=150)
print("Wrote montage to:", out)
