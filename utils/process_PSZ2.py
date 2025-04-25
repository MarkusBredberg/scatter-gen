#!/usr/bin/env python3
import os, glob, shutil
import numpy           as np
import matplotlib.pyplot as plt

from astropy.io         import fits
from astropy.wcs        import WCS
from astropy.coordinates import SkyCoord
import astropy.units    as u
from astropy.nddata.utils import Cutout2D
from scipy               import ndimage
from astropy.table       import Table
import pandas as pd

# ─── CONFIG ────────────────────────────────────────────────────────────────
FITS_ROOT = "data/PSZ2/fits"
CROP_ROOT = "data/PSZ2/crops"
CROP_SZ   = (500, 500)
SIGMA     = 2.0
MIN_PIX   = 1000

META_FITS = "data/PSZ2/planck_dr2_metadata.fits"

# target directories for your classification scheme
OUT_BASE   = "data/PSZ2/classified"
BUCKETS    = {
    'RH':          'radio_halo',
    'RR':          'radio_relic',
    'NDE':         'no_diffuse',
    'cRH':         'uncertain',
    'cRR':         'uncertain',
    'U':           'uncertain',
    'N/A':         'unclassified'
}
for d in BUCKETS.values():
    os.makedirs(os.path.join(OUT_BASE, d), exist_ok=True)
# ────────────────────────────────────────────────────────────────────────────

# 1) Load metadata and decode the Name column
meta = Table.read(META_FITS).to_pandas()

# Name is bytes, decode to string
meta['Name_str'] = [n.decode('utf-8') if isinstance(n, (bytes,bytearray)) else str(n)
                    for n in meta['Name']]

# Build slug and re‐index
meta['slug'] = meta['Name_str'].str.replace(" ", "")
meta = meta.set_index('slug')

def detect_diffuse(data):
    data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)
    sm   = ndimage.gaussian_filter(data, sigma=3)
    thr  = sm.mean() + SIGMA*sm.std()
    mask = sm > thr
    labels, n = ndimage.label(mask)
    if n == 0: return False
    sizes = ndimage.sum(mask, labels, index=np.arange(1,n+1))
    return sizes.max() >= MIN_PIX

rows = []
for slug in os.listdir(FITS_ROOT):
    dpath = os.path.join(FITS_ROOT, slug)
    if not os.path.isdir(dpath): continue

    fits_path = os.path.join(dpath, f"{slug}.fits")
    if not os.path.exists(fits_path):
        print("❌ no FITS for", slug)
        continue

    # read
    with fits.open(fits_path) as hd:
        hdr  = hd[0].header
        data = hd[0].data

    # ─── FORCE A 2D IMAGE ───────────────────────────────────────────
    data = np.array(data)        # ensure it’s an ndarray
    data = np.squeeze(data)      # remove any singleton axes
    if data.ndim != 2:
        # if it’s still 3D (e.g. [nchan, y, x] or [npol, y, x]), pick the first plane
        data = data[0]
    # now data.ndim == 2

    
    # restrict to the 2D sky-only WCS (drop the frequency axis)
    wcs2d = WCS(hdr).celestial
    cen   = SkyCoord(hdr['CRVAL1'], hdr['CRVAL2'], unit="deg")
    pix_x, pix_y = wcs2d.world_to_pixel(cen) # now world_to_pixel expects only RA,Dec

    diff = detect_diffuse(data)
    label = 'diffuse' if diff else 'cluster'

    # make cutout
    cut = Cutout2D(data, (pix_x,pix_y), CROP_SZ, wcs=wcs2d).data
    out_png = os.path.join(CROP_ROOT, f"{slug}_{label}.png")
    vmin, vmax = np.percentile(cut, [5,95])
    plt.imsave(out_png, cut, origin='lower', cmap='gray',
               vmin=vmin, vmax=vmax)

    # lookup metadata
    if slug in meta.index:
        row = meta.loc[slug]
        cls = row['Classification'].decode('utf-8') if isinstance(row['Classification'], (bytes,bytearray)) else row['Classification']
    else:
        cls = 'N/A'

    # decide bucket
    bucket = BUCKETS.get(cls, 'unclassified')
    dest   = os.path.join(OUT_BASE, bucket, os.path.basename(out_png))
    shutil.copy(out_png, dest)

    rows.append({
        'slug': slug,
        'image': os.path.basename(out_png),
        'ps_class': cls,
        'bucket':   bucket,
        'detected_diffuse': diff
    })

# write a CSV of all images → their bucket → original PSZ2 classification
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_BASE, "train_labels.csv"), index=False)
print("Written", len(df), "entries to train_labels.csv")
