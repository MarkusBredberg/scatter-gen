#!/usr/bin/env python3

import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import pandas as pd

# ─── CONFIG ────────────────────────────────────────────────────────────────
FITS_ROOT       = "/users/mbredber/scratch/data/PSZ2/fits"
CROP_ROOT       = "/users/mbredber/scratch/data/PSZ2/crops"
META_FITS       = "/users/mbredber/scratch/data/PSZ2/planck_dr2_metadata.fits"
CROP_SIZE       = (128, 128)  # Set to None to use full size

FILENAME_SUFFIXES = ["T100kpcSUB"] # List of suffixes to use
# or to grab _all_ .fits variants in the first cluster folder:
if FILENAME_SUFFIXES == 'ALL':
    import glob
    suffixes = set()
    for slug in os.listdir(FITS_ROOT):
        dpath = os.path.join(FITS_ROOT, slug)
        if not os.path.isdir(dpath):
            continue
        pattern = os.path.join(dpath, f"{slug}*.fits")
        for fpath in glob.glob(pattern):
            base = os.path.splitext(os.path.basename(fpath))[0]
            suffixes.add(base.replace(slug, ""))

    FILENAME_SUFFIXES = sorted(suffixes)
    print("FILENAME_SUFFIXES:", FILENAME_SUFFIXES)


# ─── SETUP ─────────────────────────────────────────────────────────────────

BUCKETS = {
    'RH':   ['RH', 'DE'],
    'RR':   ['RR', 'DE'],
    'NDE':  ['NDE'],
    'cRH':  ['cRH', 'cDE'],
    'cRR':  ['cRR', 'cDE'],
    'U':    ['U'],
    'N/A':  ['unclassified']
}


os.makedirs(CROP_ROOT, exist_ok=True)

# ─── LOAD METADATA ─────────────────────────────────────────────────────────
meta = Table.read(META_FITS).to_pandas()
meta['Name_str'] = [n.decode('utf-8') if isinstance(n, (bytes, bytearray)) else str(n) for n in meta['Name']]
meta['slug'] = meta['Name_str'].str.replace(" ", "")
meta = meta.set_index('slug')
rows = []

for suffix in FILENAME_SUFFIXES:
    OUT_BASE = os.path.join("data/PSZ2/classified", suffix)
    os.makedirs(OUT_BASE, exist_ok=True)
    for dirs in BUCKETS.values():
        for d in dirs:
            os.makedirs(os.path.join(OUT_BASE, d), exist_ok=True)

    # ─── LOOP OVER CLUSTERS ────────────────────────────────────────────────────
    for slug in os.listdir(FITS_ROOT):
        dpath = os.path.join(FITS_ROOT, slug)
        fits_path = os.path.join(dpath, f"{slug}{suffix}.fits")
        if not os.path.exists(fits_path):
            print(f"❌ No {suffix} FITS for {slug}")
            continue

        fits_path = os.path.join(dpath, f"{slug}{suffix}.fits")
        print(f"  {fits_path}")
        if not os.path.exists(fits_path):
            print(f"❌ No FITS file for {slug}")
            continue

        try:
            hdu = fits.open(fits_path)[0]
            data = hdu.data
            data = data[0,0,:,:]
            
            plt.imshow(data, origin='lower', cmap='afmhot')
            plt.axis('off')
            plt.savefig(os.path.join(CROP_ROOT, f"{slug}.png"), bbox_inches='tight', pad_inches=0)
            plt.savefig("Last.png", bbox_inches='tight', pad_inches=0)
                
            # Crop the image if needed
            if CROP_SIZE is not None:
                center = np.array(data.shape) // 2
                half_size = CROP_SIZE // 2
                data = data[center[0]-half_size:center[0]+half_size, center[1]-half_size:center[1]+half_size]
                
            out_png = os.path.join(CROP_ROOT, f"{slug}.png")
            plt.imshow(data, origin='lower', cmap='afmhot')
            plt.axis('off')
            plt.savefig(out_png, bbox_inches='tight', pad_inches=0)
            plt.close()

            # Classification
            if slug in meta.index:
                cls = meta.loc[slug]['Classification']
                if isinstance(cls, (bytes, bytearray)):
                    cls = cls.decode('utf-8')
            else:
                cls = 'N/A'

            # if there are multiple labels (comma-separated), copy into each bucket
            labels = [s.strip() for s in cls.split(',')]
            for lab in labels:
                buckets = BUCKETS[lab] if lab in BUCKETS else ['unclassified']
                for bucket in buckets:
                    dest_dir = os.path.join(OUT_BASE, bucket)
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.copy(out_png, os.path.join(dest_dir, os.path.basename(out_png)))

            rows.append({
                'slug': slug,
                'image': os.path.basename(out_png),
                'class': cls
            })


            print(f"✅ Saved {slug}.png to {cls}. It has size {data.shape}.")

        except Exception as e:
            print(f"⚠️ Error processing {slug}: {e}")

# ─── WRITE CSV SUMMARY ─────────────────────────────────────────────────────
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_BASE, "train_labels.csv"), index=False)
print(f"📄 Written {len(df)} entries to train_labels.csv")
