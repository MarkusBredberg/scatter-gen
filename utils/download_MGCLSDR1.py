#!/usr/bin/env python3
import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config

print("Downloading MGCLSDR1 FITS files...")

# where to save
OUT_ROOT  = "/users/mbredber/scratch/data/MGCLS"
FITS_ROOT = os.path.join(OUT_ROOT, "fits")
os.makedirs(FITS_ROOT, exist_ok=True)

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

print("Downloading MGCLSDR1 FITS files…")

# where to save (unchanged)
OUT_ROOT  = "/users/mbredber/scratch/data/MGCLS"
FITS_ROOT = os.path.join(OUT_ROOT, "fits")
os.makedirs(FITS_ROOT, exist_ok=True)

# build and fetch the public HTML index
index_url = "https://archive-gw-1.kat.ac.za/public/repository/10.48479/7epd-w356/data/basic_products/index.html"
resp      = requests.get(index_url)
resp.raise_for_status()
soup      = BeautifulSoup(resp.text, 'html.parser')

# grab all .fits.gz links
fits_links = [
    a['href'] for a in soup.find_all('a', href=True)
    if a['href'].lower().endswith('.fits.gz')
]

print(f"Found {len(fits_links)} FITS files.")

# download each FITS via HTTP
for href in fits_links:
    url      = urljoin(index_url, href)
    filename = os.path.basename(href)
    outpath  = os.path.join(FITS_ROOT, filename)
    if os.path.exists(outpath):
        print(f"Skipping existing {filename}")
        continue

    print(f"Downloading {filename}…")
    r = requests.get(url); r.raise_for_status()
    with open(outpath, 'wb') as fd:
        fd.write(r.content)

print("All downloads complete.")

