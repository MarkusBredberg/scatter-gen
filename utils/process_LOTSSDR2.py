#!/usr/bin/env python3
"""
Script to iterate through every FITS in data/LOTSSDR2/fitsfiles,
cross-match to the Planck PSZ2 cluster catalogue,
detect diffuse emission, and write a 500×500 px crop around each pointing
into data/LOTSSDR2/crops, tagged as field/cluster/diffuse.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.nddata.utils import Cutout2D

from astroquery.vizier import Vizier
from scipy import ndimage

# — PATHS & PARAMETERS —
FITS_DIR        = os.path.abspath("data/LOTSSDR2/fitsfiles")
CROP_DIR        = os.path.abspath("data/LOTSSDR2/crops")
CLUSTER_RAD_DEG = 0.5            # deg
CROP_SIZE_PX    = (500, 500)     # width, height
THRESH_SIGMA    = 2.0            # σ for diffuse detection
AREA_MIN_PIX    = 1000           # min connected pixels

os.makedirs(CROP_DIR, exist_ok=True)
# — LOAD PSZ2 FROM VIZIER —
Vizier.ROW_LIMIT = -1
catalogs = Vizier.find_catalogs("PSZ2")
psz2_key = list(catalogs.keys())[0]
cat     = Vizier.get_catalogs(psz2_key)[0]
cluster_coords = SkyCoord(cat["RAJ2000"], cat["DEJ2000"], unit="deg")

def is_cluster_field(hdr):
    """True if field center is within CLUSTER_RAD_DEG of any PSZ2 entry."""
    center = SkyCoord(hdr["CRVAL1"], hdr["CRVAL2"], unit="deg")
    sep    = center.separation(cluster_coords)
    return (sep < CLUSTER_RAD_DEG * u.deg).any()

def detect_diffuse(data):
    """Simple diffuse detection: smooth → threshold → connected area."""
    data   = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    smooth = ndimage.gaussian_filter(data, sigma=3)
    thresh = smooth.mean() + THRESH_SIGMA * smooth.std()
    mask   = smooth > thresh
    labels, n = ndimage.label(mask)
    if n == 0:
        return False
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    return sizes.max() >= AREA_MIN_PIX

def crop_and_save(fits_path, out_png):
    """Cut out a box around FITS center and save to out_png."""
    with fits.open(fits_path) as hdul:
        hdr  = hdul[0].header
        data = hdul[0].data
    wcs    = WCS(hdr)
    center = SkyCoord(hdr["CRVAL1"], hdr["CRVAL2"], unit="deg")
    pix    = wcs.world_to_pixel(center)
    cut    = Cutout2D(data, position=pix, size=CROP_SIZE_PX, wcs=wcs)

    plt.figure(figsize=(5, 5))
    vmin, vmax = np.percentile(cut.data, [5, 95])
    plt.imshow(cut.data, origin="lower", vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_png, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close()

if __name__ == "__main__":
    fits_list = sorted(glob.glob(os.path.join(FITS_DIR, "*mosaic*.fits")))
    print(f"Found {len(fits_list)} FITS files in {FITS_DIR}")

    for fp in fits_list:
        base = os.path.splitext(os.path.basename(fp))[0]
        hdr  = fits.getheader(fp, 0)

        # classify
        if is_cluster_field(hdr):
            data0 = fits.getdata(fp, 0)
            label = "diffuse" if detect_diffuse(data0) else "cluster"
        else:
            label = "field"

        out_png = os.path.join(CROP_DIR, f"{base}_{label}.png")
        print(f"{base}: label={label} → {out_png}")
        crop_and_save(fp, out_png)
        
    # clean up crop filenames and tag full-res PNGs
    # 1) remove "_blanked_" in crops
    for fn in glob.glob(os.path.join(CROP_DIR, "*_blanked_*.png")):
        os.rename(fn, fn.replace("_blanked_", "_"))

    # 2) append _field or _cluster to every full-res PNG
    fullres_dir = os.path.abspath("data/LOTSSDR2/fullres_pngs")
    for fn in glob.glob(os.path.join(fullres_dir, "*mosaic*.png")):
        base = os.path.splitext(os.path.basename(fn))[0]
        hdr  = fits.getheader(os.path.join(FITS_DIR, base + ".fits"), 0)
        label = "cluster" if is_cluster_field(hdr) else "field"
        new_fn = os.path.join(fullres_dir, f"{base}_{label}.png")
        os.rename(fn, new_fn)

    print("Done.")
