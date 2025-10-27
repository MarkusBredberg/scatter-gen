# plot_ten_psz2.py
"""
Used to evaluate the data_loader.py PSZ2 loader.
"""

import matplotlib.pyplot as plt
from utils.data_loader import load_galaxies  # <-- replace 'your_module' with the actual module/file name (without .py)

def main():
    # Choose the PSZ2 class(es); 53 is radio halos in your setup
    galaxy_classes = [50, 51]

    # Keep versions single-frame to get [B, 1, H, W] tensors
    #versions = "RT50kpc"
    versions = ["RT50kpc", "T50kpc"]  # order defines left/right panels


    # Load (don’t pass sample_size; PSZ2 loader doesn’t accept it)
    train_images, train_labels, eval_images, eval_labels, train_fns, eval_fns = load_galaxies(
        galaxy_classes=galaxy_classes,
        versions=versions,
        fold=5,                             # your default “last split”
        crop_size=(1, 256, 256),
        downsample_size=(1, 256, 256),
        NORMALISE=True,
        USE_GLOBAL_NORMALISATION=False,     # percentile stretch from your loader
        percentile_lo=30,
        percentile_hi=99,
        AUGMENT=False,
        PRINTFILENAMES=True,
        EXTRADATA=False
    )

    # Pick first 10 from the training set (fallback to however many exist)
    n_show = min(10, len(train_images))   # train_images is a list of tensors
    imgs = train_images[:n_show]
    names = train_fns[:n_show] if train_fns else [f"idx_{i}" for i in range(n_show)]
    import os, re
    def _src_name(s):
        b = os.path.splitext(os.path.basename(str(s)))[0]
        return re.sub(r'(?:T\d+kpc(?:SUB)?)$', '', b)
    row_titles = [_src_name(s) for s in names]

    # Plot side-by-side
    # 2 columns: left = RT50, right = T50 (matches 'versions' order)
    # Plot: handle single-version ([1,H,W]) vs cube ([T,1,H,W] with T=2)
    is_cube = isinstance(versions, (list, tuple))

    if is_cube:
        # 2 columns: left = versions[0], right = versions[1]
        fig, axes = plt.subplots(n_show, 2, figsize=(5.4, 2.6*n_show), constrained_layout=True)
        if n_show == 1:
            axes = [axes]
        for i in range(n_show):
            cube = imgs[i]  # [T,1,H,W]
            left  = cube[0, 0].detach().cpu().numpy()
            right = cube[1, 0].detach().cpu().numpy()

            axes[i][0].imshow(left, cmap="viridis", origin="lower")
            axes[i][0].set_title(f"{row_titles[i]} — {versions[0]}")
            axes[i][0].set_xticks([]); axes[i][0].set_yticks([])

            axes[i][1].imshow(right, cmap="viridis", origin="lower")
            axes[i][1].set_title(f"{row_titles[i]} — {versions[1]}")
            axes[i][1].set_xticks([]); axes[i][1].set_yticks([])
    else:
        # Single version: one column
        fig, axes = plt.subplots(n_show, 1, figsize=(2.7, 2.6*n_show), constrained_layout=True)
        if n_show == 1:
            axes = [axes]
        for i in range(n_show):
            img = imgs[i][0].detach().cpu().numpy()  # [H,W]
            axes[i].imshow(img, cmap="viridis", origin="lower")
            axes[i].set_title(f"{row_titles[i]} — {versions}")
            axes[i].set_xticks([]); axes[i].set_yticks([])

    plt.savefig(f"/users/mbredber/scratch/psz2_{versions}.png", dpi=150)
    plt.close()
    print(f"Saved /users/mbredber/scratch/psz2_{versions}.png")

if __name__ == "__main__":
    main()
