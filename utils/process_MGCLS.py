from pathlib import Path
from PIL import Image
import pandas as pd
import re

print("Running MGCLS image processing script...")

# Define paths and crop size
csv_path =        Path("/users/mbredber/data/MGCLS/DE_table.csv")
source_dir =      Path("/users/mbredber/data/MGCLS/all_images")
target_base_dir = Path("/users/mbredber/data/MGCLS/classified_crops_512")
preferred_size = (512, 512)  # (width, height)

if "DE_table.csv" in csv_path.name:
    # Load and clean CSV
    df = pd.read_csv(csv_path)
    # ——— replace your df-filter + mapping with:
    df["Cluster name"] = (
        df["Cluster name"]
        .astype(str)
        .str.strip()
        # unify minus‐sign variants
        .str.replace("−", "-", regex=False)
    )
    df["D.E."] = df["D.E."].astype(str).str.strip()

    cluster_de_map = {}
    for name, alt, de in zip(df["Cluster name"], df["Alternate name"], df["D.E."]):
        label = "DE" if de == "Yes" else "NDE"
        print(f"Mapping {name} ({alt}) to {label}")
        cluster_de_map[name.replace(" ", "_")] = label
        if pd.notna(alt) and str(alt).strip():
            cluster_de_map[str(alt).replace(" ", "_")] = label

    # build a lowercase, underscore-stripped map for fuzzy matching
    cluster_de_map_norm = {}
    for key, label in cluster_de_map.items():
        norm_key = key.replace("_", "").lower()
        cluster_de_map_norm[norm_key] = label


    # Create DE/NDE directories
    for label in ["DE", "NDE"]:
        (target_base_dir / label).mkdir(parents=True, exist_ok=True)

    print("Cluster classification loaded from CSV:")
    for name, de in zip(df["Cluster name"], df["D.E."]):
        print(f"{name} → {de}")
        
else:
    # Load and clean CSV, forward-fill blank names
    df = pd.read_csv(csv_path)
    df["Cluster Name"] = (
        df["Cluster Name"]
        .fillna(method="ffill")
        .astype(str)
        .str.strip()
    )
    df["Morph"] = df["Morph"].astype(str).str.strip()

    # Build cluster→morph map (use first non-empty Morph for each cluster)
    cluster_de_map = {}
    for cluster, morph in zip(df["Cluster Name"], df["Morph"]):
        if morph:
            token = morph.split()[-1]
            label = token.replace("(", "").replace(")", "").replace("?", "")
            cluster_de_map[cluster.replace(" ", "_")] = label


    # Normalize for fuzzy lookup
    cluster_de_map_norm = {
        key.replace("_", "").lower(): label
        for key, label in cluster_de_map.items()
    }

    # Create one directory per unique morphology
    for label in set(cluster_de_map.values()):
        (target_base_dir / label).mkdir(parents=True, exist_ok=True)

# Crop and classify images
def process_images():
    for image_file in source_dir.glob("*"):
        if image_file.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        base_name = image_file.stem
 
        stem = image_file.stem
        base = re.sub(
            r'(_hi|_lo|\.fix_hi|\.fix_lo|\.fix2_hi|\.fix2_lo|B\.fx|\.phscal_lo||\.phscal_hi|\.phsc_hi|\.phsc_lo|\.apscal_hi|\.apscal_lo|\.apsc_hi|\.apsc_lo|\.redo_lo|\.redo_hi|\.orig_hi|\.orig_lo|\.dbeam|_a|_b|\.apsc(?:al)?)$',
            '',
            stem,
            flags=re.IGNORECASE
        )
        # lowercase, remove underscores, drop any trailing A/B
        norm_name = base.replace("_", "").lower()
        norm_name = re.sub(r'[ab]+$', '', norm_name)
        label = cluster_de_map_norm.get(norm_name)
       
        if label:
            with Image.open(image_file) as img:
                width, height = img.size
                left = (width - preferred_size[0]) / 2
                top = (height - preferred_size[1]) / 2
                right = (width + preferred_size[0]) / 2
                bottom = (height + preferred_size[1]) / 2
                cropped_img = img.crop((left, top, right, bottom))

                target_path = target_base_dir / label / image_file.name
                cropped_img.save(target_path)
        else: # Figure out why this is happening
            print(f"Warning: No label found for {image_file.name}. Skipping.")

process_images()
print("Image processing completed.")

