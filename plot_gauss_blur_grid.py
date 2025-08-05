#!/usr/bin/env python3
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import torch
from utils.data_loader2 import load_galaxies

#def plot_galaxy_grid(images, filenames, sigmas=(1.0, 2.0, 3.0)):
def plot_galaxy_grid(images, filenames, labels, sigmas=(1.0, 2.0, 3.0)):
    """
    Plot a 6×6 grid: 
      rows = up to 6 sources
      cols = [RAW, T100kpc, T50kpc, blur1, blur2, blur3]
    """
    #n_rows, n_cols = 6, 6
    # 1) make a wide figure, reserving left margin=0.30 for filenames
    #fig, axes = plt.subplots(
    #    n_rows, n_cols,
    #    figsize=(12, 8),
    #    gridspec_kw={
    #        "left": 0.30,    # <-- 30% of fig width for filenames
    #        "right": 0.98,
    #        "top": 0.92,
    #        "bottom": 0.05,
    #        "wspace": 0.15,
    #        "hspace": 0.15,
    #    }
    #)
    # choose 3 DE (50) + 3 NDE (51) in that order
    de_idx  = [i for i, y in enumerate(labels) if int(y) == 50][:3]
    nde_idx = [i for i, y in enumerate(labels) if int(y) == 51][:3]
    order = de_idx + nde_idx
    if len(order) < 6:
        raise ValueError(f"Need at least 3 of each class; got {len(de_idx)} DE and {len(nde_idx)} NDE.")

    images = images[order]
    filenames = [filenames[i] for i in order]
    labels = [labels[i] for i in order]

    n_rows, n_cols = 6, 6
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(12, 8),
        gridspec_kw={"left": 0.30, "right": 0.98, "top": 0.92, "bottom": 0.05, "wspace": 0.15, "hspace": 0.15}
    )

    # 2) column titles
    col_titles = ["RAW", "T100 kpc", "T50 kpc"] + [f"Blur σ={s}" for s in sigmas]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=9, pad=8)

    # 3) plot each thumbnail
    for i in range(n_rows):
        cube = images[i]            # [T, C, H, W]
        raw, t100, t50 = (cube[idx].squeeze().detach().cpu().numpy()
                          for idx in (0, 1, 2))
        # first three versions
        for j, arr in enumerate((raw, t100, t50)):
            ax = axes[i, j]
            ax.imshow(arr, cmap="viridis", origin="lower")
            ax.axis("off")
        # blurs
        for j, sigma in enumerate(sigmas, start=3):
            ax = axes[i, j]
            ax.imshow(gaussian_filter(raw, sigma), cmap="viridis", origin="lower")
            ax.axis("off")

    # 4) draw the filenames in the left margin with fig.text
    #    y runs 0 (bottom) → 1 (top); we want each row centered in its subplot
    #for i, name in enumerate(filenames[:n_rows]):
    #    # compute y of the *center* of row i:
    #    # since top margin is .92 and bottom is .05, the height for rows = .92-.05
    #    row_height = 0.92 - 0.05
    #    y = 0.05 + row_height * (1 - (i + 0.5) / n_rows)
    #    fig.text(
    #        0.02,        # 2% in from left edge of figure
    #        y,           # computed y
    #        name,
    #        ha="left",
    #        va="center",
    #        fontsize=8,
    #    )

    # 4) draw the filenames + class in the left margin
    row_height = 0.92 - 0.05
    for i, (name, ycls) in enumerate(zip(filenames, labels)):
        y = 0.05 + row_height * (1 - (i + 0.5) / n_rows)
        tag = "DE" if int(ycls) == 50 else "NDE"
        fig.text(0.02, y, f"{name} ({tag})", ha="left", va="center", fontsize=8)


    # 5) save + close
    outname = f"gauss_blur_grid_{'_'.join(map(str, sigmas))}.png"
    plt.savefig(outname, dpi=300)
    plt.close(fig)
    print(f"Wrote {outname}")


if __name__ == "__main__":
    #result = load_galaxies(
    #    galaxy_class=[50,51],
    #    fold=5,
    #    crop_size=(3,1,256,256),
    #    downsample_size=(3,1,128,128),
    #    sample_size=500,
    #    REMOVEOUTLIERS=True,
    #    BALANCE=False,
    #    STRETCH=False,
    #    AUGMENT=False,
    #    NORMALISE=True,
    #    NORMALISETOPM=False,
    #    EXTRADATA=True,
    #    train=False,
    #)
    #_,_, eval_imgs,_,_, eval_fns = result
    #plot_galaxy_grid(eval_imgs, eval_fns, sigmas=(1.0,2.0,3.0))
   
    result = load_galaxies(
        galaxy_class=[50, 51],
        fold=5,
        crop_size=(3,1,512,512),
        downsample_size=(3,1,128,128),
        sample_size=500,
        REMOVEOUTLIERS=True,
        BALANCE=False,
        STRETCH=False,
        AUGMENT=False,
        NORMALISE=True,
        NORMALISETOPM=False,
        EXTRADATA=True,
        train=False,
    )
    eval_imgs   = result[2]
    eval_labels = result[3]  # eval_labels lives at index 3
    eval_fns    = result[5]
    plot_galaxy_grid(eval_imgs, eval_fns, eval_labels, sigmas=(1.0, 2.0, 3.0))
