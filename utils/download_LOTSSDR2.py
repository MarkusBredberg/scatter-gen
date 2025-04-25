import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin  # Use this to correctly join URLs
from urllib.parse import urlparse
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import os
output_dir = "data/LOTSSDR2"
os.makedirs(output_dir, exist_ok=True)


# Fetch the page HTML
page_url = "https://lofar-surveys.org/dr2_release.html"
response = requests.get(page_url)
response.raise_for_status()  # Abort if the page doesn't load

soup = BeautifulSoup(response.text, "html.parser")

# Find the table with the download links (assume first table is our target)
tables = soup.find_all("table")
if not tables:
    print("No table found on the page. Exiting.")
    exit()

table = tables[0]

# Collect all "Download" links from the table that end with .fits
download_links = []

rows = table.find_all("tr")[1:]  # Skip the header row
for row in rows:
    cells = row.find_all("td")
    if len(cells) < 4:
        continue  # Not enough columns in this row

    # The 4th cell contains the "Full_res mosaic" Download link
    link_tag = cells[8].find("a", href=True, string="Download")
    if link_tag:
        href = link_tag["href"].strip()
        full_url = urljoin("https://lofar-surveys.org/", href)
        if full_url.lower().endswith(".fits"):
            download_links.append(full_url)


# Download each file with a unique filename
for link in download_links:
    parsed_url = urlparse(link)
    # parsed_url.path would be something like: '/public/DR2/mosaics/P000+23/mosaic-blanked.fits'
    path_parts = parsed_url.path.split('/')
    if len(path_parts) >= 3:
        # Get the field name (second-to-last item) and the base filename
        field = path_parts[-2]
        base_filename = path_parts[-1]
        # Create a unique name by combining them
        savename = f"{field}-{base_filename}"
    else:
        savename = os.path.basename(link.split("?")[0])
    
    print(f"Downloading {savename} from {link}")
    r = requests.get(link)
    if r.status_code == 200:
        filepath = os.path.join(output_dir, savename)
        with open(filepath, "wb") as f:
            f.write(r.content)
        print(f"Saved {savename} to {filepath}")
    else:
        print(f"Failed to download {savename} (HTTP {r.status_code})")

print("All done!")
