#!/usr/bin/env python3
"""
Find per-T-scale minima:
 - For each T{Y}kpc.fits scale found under ROOT, compute:
     * n_beams = min(FOV_x, FOV_y) / FWHM  (FOV and FWHM in arcsec)
     * FOV_side = min(FOV_x, FOV_y) in arcmin
 - Report the smallest n_beams and smallest FOV_side for each scale,
   and the source (file path) that produced that minimum.
 - Also print global minima across all scales.

Usage:
    python find_min_nbeams_fov.py /users/mbredber/scratch/data/PSZ2/fits
You can also pass a single file or directory that contains the example:
    /users/mbredber/scratch/data/PSZ2/fits/PSZ2G023.17+86.71/PSZ2G023.17+86.71T100kpc.fits
"""

from pathlib import Path
import re
import numpy as np
from astropy.io import fits

PAT_SCALE = re.compile(r"T([0-9]+(?:\.[0-9]+)?)kpc\.fits$", re.IGNORECASE)
ARCSEC_PER_RAD = 206264.80624709636

def _cd_matrix_rad(hdr):
    """Return 2x2 CD matrix in radians (like your script's _cd_matrix_rad)."""
    if 'CD1_1' in hdr:
        M = np.array([[hdr.get('CD1_1', 0.0), hdr.get('CD1_2', 0.0)],
                      [hdr.get('CD2_1', 0.0), hdr.get('CD2_2', 0.0)]], float)
    else:
        pc11=hdr.get('PC1_1',1.0); pc12=hdr.get('PC1_2',0.0)
        pc21=hdr.get('PC2_1',0.0); pc22=hdr.get('PC2_2',1.0)
        cd1 =hdr.get('CDELT1', 1.0); cd2 =hdr.get('CDELT2', 1.0)
        M = np.array([[pc11, pc12],[pc21, pc22]], float) @ np.diag([cd1, cd2])
    return M * (np.pi/180.0)

def arcsec_per_pix(hdr):
    J = _cd_matrix_rad(hdr)
    dx = float(np.hypot(J[0,0], J[1,0]))
    dy = float(np.hypot(J[0,1], J[1,1]))
    return dx*ARCSEC_PER_RAD, dy*ARCSEC_PER_RAD

def fwhm_major_as(hdr):
    # BMAJ/BMIN in degrees in many headers; convert to arcsec
    return max(float(hdr.get("BMAJ", 0.0)), float(hdr.get("BMIN", 0.0))) * 3600.0

def analyze_file(fpath: Path):
    """Return (scale_kpc as float or None, n_beams, fov_min_arcmin, details dict) or None on error."""
    m = PAT_SCALE.search(fpath.name)
    if not m:
        return None
    scale = float(m.group(1))
    try:
        with fits.open(fpath, memmap=False) as hdul:
            hdr = hdul[0].header
            # sanity checks
            nax1 = int(hdr.get('NAXIS1', 0))
            nax2 = int(hdr.get('NAXIS2', 0))
            if nax1 <= 0 or nax2 <= 0:
                raise RuntimeError("Invalid NAXIS")
            fwhm_as = fwhm_major_as(hdr)
            if fwhm_as <= 0:
                raise RuntimeError("Invalid BMAJ/BMIN")
            asx, asy = arcsec_per_pix(hdr)
            if asx <= 0 or asy <= 0:
                # fallback to CDELT in degrees -> arcsec (if present)
                asx = abs(hdr.get('CDELT1', 1.0)) * 3600.0
                asy = abs(hdr.get('CDELT2', 1.0)) * 3600.0

            fovx_as = nax1 * asx
            fovy_as = nax2 * asy
            fov_min_as = min(fovx_as, fovy_as)
            # n_beams uses FOV_min / FWHM (both in arcsec)
            n_beams = float(fov_min_as) / max(float(fwhm_as), 1e-12)

            details = dict(path=str(fpath), nax1=nax1, nax2=nax2,
                           asx=asx, asy=asy, fwhm_as=fwhm_as,
                           fovx_as=fovx_as, fovy_as=fovy_as)
            return scale, n_beams, (fov_min_as/60.0), details  # fov in arcmin
    except Exception as e:
        print(f"[WARN] failed to read {fpath}: {e}")
        return None

def find_all_T_files(root: Path):
    """Yield all files matching *T*kpc.fits under root (or yield file itself if root is a file)."""
    root = Path(root)
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*T*kpc.fits")):
        yield p

def main(root_dir):
    root = Path(root_dir)
    results_per_scale = {}  # scale -> list of (n_beams, fov_arcmin, details)
    n_checked = 0
    for f in find_all_T_files(root):
        res = analyze_file(f)
        if res is None: 
            continue
        scale, n_beams, fov_arcmin, details = res
        n_checked += 1
        results_per_scale.setdefault(scale, []).append((n_beams, fov_arcmin, details))

    if n_checked == 0:
        print("No T*kpc.fits files found under", str(root))
        return

    # For each scale, find minima
    print("\nPer-scale minima (scale_kpc):")
    header = f"{'scale(kpc)':>10}  {'min n_beams':>12}  {'source (min n_beams)':>45}  {'min FOV(arcmin)':>15}  {'source (min FOV)':>45}"
    print(header)
    print("-"*len(header))

    global_min_nbeams = (1e9, None, None)   # (value, scale, details)
    global_min_fov = (1e9, None, None)      # (value_arcmin, scale, details)

    for scale in sorted(results_per_scale.keys()):
        lst = results_per_scale[scale]
        # smallest n_beams
        smallest_n, _, _ = min(lst, key=lambda x: x[0])
        # pick the first file that attains this smallest value (stable)
        candidates_n = [t for t in lst if abs(t[0] - smallest_n) < 1e-12]
        min_n_entry = candidates_n[0]
        # smallest FOV (arcmin)
        smallest_fov, _, _ = min(lst, key=lambda x: x[1])[1], None, None  # get value
        # find the entry giving smallest fov
        min_fov_entry = min(lst, key=lambda x: x[1])

        n_val = min_n_entry[0]
        n_source = min_n_entry[2]['path']
        fov_val = min_fov_entry[1]
        fov_source = min_fov_entry[2]['path']

        print(f"{scale:10.3f}  {n_val:12.3f}  {n_source:45s}  {fov_val:15.3f}  {fov_source:45s}")

        if n_val < global_min_nbeams[0]:
            global_min_nbeams = (n_val, scale, min_n_entry[2])
        if fov_val < global_min_fov[0]:
            global_min_fov = (fov_val, scale, min_fov_entry[2])

    print("\nGlobal minima across all scales:")
    if global_min_nbeams[1] is not None:
        val, scale, det = global_min_nbeams
        print(f" - Smallest n_beams = {val:.3f} at scale {scale} kpc, source: {det['path']}")
        print(f"   FWHM (arcsec) = {det['fwhm_as']:.3f}, FOV_x (arcsec) = {det['fovx_as']:.1f}, FOV_y (arcsec) = {det['fovy_as']:.1f}")
    if global_min_fov[1] is not None:
        val, scale, det = global_min_fov
        print(f" - Smallest FOV (min side) = {val:.3f} arcmin at scale {scale} kpc, source: {det['path']}")
        print(f"   FWHM (arcsec) = {det['fwhm_as']:.3f}, FOV_x (arcsec) = {det['fovx_as']:.1f}, FOV_y (arcsec) = {det['fovy_as']:.1f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python find_min_nbeams_fov.py /path/to/root_or_file")
        sys.exit(1)
    main(sys.argv[1])
