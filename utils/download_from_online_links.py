import glob
import os
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

output_dir = "data/LOTSSDR2"
fullres_dir = os.path.join(output_dir, "fullres_pngs")
os.makedirs(fullres_dir, exist_ok=True)

def make_fullres_png(fits_path, out_png):
    print(f"Processing {fits_path}")
    # load first non‐empty HDU
    with fits.open(fits_path) as hdul:
        data = None
        for hdu in hdul:
            if hdu.data is not None:
                data = hdu.data.astype(float)
                break
    if data is None:
        print(f"  ⚠️ no image data, skipping")
        return

    # sanitize
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # compute stretch
    vals = data[np.isfinite(data) & (data > 0)]
    if vals.size == 0:
        vals = data.flatten()
    vmin, vmax = np.percentile(vals, [1, 99])
    print(f"  stretch vmin={vmin:.3g}, vmax={vmax:.3g}")

    # determine figure size so that figsize* dpi = original pixels
    h, w = data.shape
    dpi = 100
    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.imshow(data, origin="lower", vmin=vmin, vmax=vmax)
    ax.axis("off")
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"  ✅ saved {out_png}")

for fits_path in glob.glob(os.path.join(output_dir, "*mosaic*.fits")):
    base = os.path.splitext(os.path.basename(fits_path))[0]
    out_png = os.path.join(fullres_dir, f"{base}.png")
    make_fullres_png(fits_path, out_png)

print("All full‑resolution PNGs written!")
