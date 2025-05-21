import requests
from bs4          import BeautifulSoup
from urllib.parse import urljoin
import tarfile, os

BASE     = "https://lofar-surveys.org"
INDEX    = BASE + "/planck_dr2.html"
OUT_ROOT = "data/PSZ2"
FITS_ROOT= os.path.join(OUT_ROOT, "fits")
os.makedirs(FITS_ROOT, exist_ok=True)

# 1) Download + parse the main table so we get per-cluster detail-page URLs
r = requests.get(INDEX); r.raise_for_status()
soup = BeautifulSoup(r.text, "html.parser")

# The first <table> on that page is the full‐sample PSZ2 list:
table = soup.find("table")
rows  = table.find_all("tr")[1:]  # skip header

cluster_pages = []
for tr in rows:
    # the "Name" column has <a href="clusters/PSZ2G023.17+86.71.html">
    a = tr.find("a", href=lambda h: h and "clusters/" in h)
    if a:
        cluster_pages.append(urljoin(BASE, a["href"]))

print(f"Found {len(cluster_pages)} cluster pages")

# 2) Loop over those URLs, grab each "Tarball with FITS images" link, download & extract
for page_url in cluster_pages:
    r = requests.get(page_url)
    if r.status_code != 200:
        print("✖️  Couldn’t fetch", page_url)
        continue
    soup = BeautifulSoup(r.text, "html.parser")

    #tar_a = soup.find("a", string="Tarball with FITS images")
    tar_a = soup.find("a", href=lambda h: h and h.endswith(".tar") or h.endswith(".tar.gz"))
    if not tar_a:
        print("⚠️  no tarball link on", page_url)
        continue

    tar_url = urljoin(page_url, tar_a["href"])
    slug    = os.path.basename(page_url).replace(".html", "")
    out_tar = os.path.join(OUT_ROOT, f"{slug}.tar.gz")

    print("⏬", slug)
    #tr = requests.get(tar_url); tr.raise_for_status()
    try:
        tr = requests.get(tar_url); tr.raise_for_status()
    except Exception as e:
        print(f"❌ failed to download tarball for {slug}: {e}")
        continue

    with open(out_tar, "wb") as fd:
        fd.write(tr.content)

    # Extract into data/PSZ2/fits/PSZ2G023.17+86.71/
    dest = os.path.join(FITS_ROOT, slug)
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(out_tar, "r:gz") as tf:
        tf.extractall(dest)
    extracted = os.listdir(dest)
    print(f"📁 Extracted files: {len(extracted)} → {[f for f in extracted if f.endswith('.fits')]}")
    os.remove(out_tar)
