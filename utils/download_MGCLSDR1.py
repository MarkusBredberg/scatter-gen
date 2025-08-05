import requests
from bs4          import BeautifulSoup
from urllib.parse import urljoin
import tarfile, os, csv

#BASE     = "https://lofar-surveys.org"
BASE = "https://archive-gw-1.kat.ac.za"
#INDEX    = BASE + "/public/repository/10.48479/7epd-w356/data/basic_products/index.html"
INDEX = "https://archive-gw-1.kat.ac.za/public/repository/10.48479/epd-w356/data/"
OUT_ROOT = "data/MGCLS"
FITS_ROOT= os.path.join(OUT_ROOT, "fits")
os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(FITS_ROOT, exist_ok=True)

# 1) hit the S3 list‑objects API (XML) rather than the HTML
r = requests.get(INDEX + "?list-type=2")
r.raise_for_status()

# 2) parse the XML keys and pick out only the FITS (or .tar.gz) files
soup = BeautifulSoup(r.text, "xml")
keys = [elt.text
        for elt in soup.find_all("Key")
        if elt.text.startswith("public/repository/10.48479/epd-w356/data/")
           and elt.text.lower().endswith((".fits", ".fits.gz", ".tar.gz"))]

# 3) build your download URLs
file_pages = ["https://archive-gw-1.kat.ac.za/" + key for key in keys]

# ─── then continue with your for‑loop over file_pages as before ───
for url in file_pages:
    filename = os.path.basename(url)
    out_path = os.path.join(OUT_ROOT, filename)
    print("⏬", filename)
    resp = requests.get(url); resp.raise_for_status()
    with open(out_path, "wb") as fd:
        fd.write(resp.content)
        
    print("✅ Downloaded:", filename)
