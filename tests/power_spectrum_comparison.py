"""
Compare 2D power spectra of reference images before and after downsampling.

For a selection of sources, computes the azimuthally-averaged 1D power spectrum
of the original LoTSS FITS and the preprocessed 128x128 FITS, and plots them
together. The beam scale is marked to show which spatial frequencies are affected.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits

# ── paths ─────────────────────────────────────────────────────────────────────
RAW_DIR      = Path("/users/mbredber/scratch/data/PSZ2/fits")
PROC_DIR     = Path("/users/mbredber/scratch/data/PSZ2/beam_crop/circular/fits_files")
OUT_PATH     = Path("/users/mbredber/scratch/figures/processing/power_spectrum_comparison.png")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── source selection: one small-beam, one median-beam, one large-beam ─────────
# (slug, label) — edit to pick different sources
SOURCES = [
    "PSZ2G093.04-32.38",   # small beam  (~10th pct, BMAJ=10.5'')
    "PSZ2G133.60+69.04",   # median beam (~50th pct, BMAJ=13.5'')
    "PSZ2G172.63+35.15",   # large beam  (~90th pct, BMAJ=24.1'')
]


# ── helpers ───────────────────────────────────────────────────────────────────

def find_raw_fits(slug: str) -> Path | None:
    """Find the reference (non-tapered) radio FITS for a source.

    Prefers the exact-match file ({slug}.fits or {slug}R*.fits),
    so we never accidentally pick up X-ray (CHANDRA, XMM) or
    model/compact/tapered files.
    """
    src_dir = RAW_DIR / slug
    if not src_dir.exists():
        return None
    # First choice: exact slug match
    exact = src_dir / f"{slug}.fits"
    if exact.exists():
        return exact
    # Second choice: robust-weighting variant (e.g. PSZ2G...R-1.25.fits)
    robust = sorted(src_dir.glob(f"{slug}R*.fits"))
    if robust:
        return robust[0]
    # Fallback: any file that isn't a known non-radio product
    skip = {"kpc", "compact", "chandra", "xmm", "model", "sub"}
    candidates = [
        f for f in src_dir.glob("*.fits")
        if not any(s in f.name.lower() for s in skip)
    ]
    return candidates[0] if candidates else None


def find_proc_fits(slug: str) -> Path | None:
    candidates = sorted(PROC_DIR.glob(f"{slug}_RAW_fmt_*.fits"))
    return candidates[0] if candidates else None


def read_image(path: Path):
    """Return (2D array, pixel_arcsec, bmaj_arcsec, bmin_arcsec)."""
    with fits.open(path, memmap=False) as hdul:
        hdr  = hdul[0].header
        data = hdul[0].data
    # collapse degenerate axes (e.g. Stokes, frequency)
    while data.ndim > 2:
        data = data[0]
    pix_as = abs(hdr.get("CDELT2", hdr.get("CD2_2", 0))) * 3600.0
    bmaj   = hdr.get("BMAJ", np.nan) * 3600.0
    bmin   = hdr.get("BMIN", np.nan) * 3600.0
    return data.astype(np.float64), pix_as, bmaj, bmin


def azimuthal_power_spectrum(image: np.ndarray, pix_as: float):
    """
    Compute the azimuthally-averaged 1D power spectrum.

    Returns
    -------
    scales_as : angular scales in arcsec (1/frequency), length M
    power     : mean power per annulus, length M
    """
    img = image.copy()
    # replace NaNs with the image median so they don't dominate the FFT
    img[~np.isfinite(img)] = np.nanmedian(img)

    # subtract mean to remove DC spike
    img -= img.mean()

    # 2D Hann window to suppress edge ringing
    ny, nx = img.shape
    wy = np.hanning(ny)
    wx = np.hanning(nx)
    window = np.outer(wy, wx)
    img *= window

    # 2D FFT → power
    fft2  = np.fft.fft2(img)
    power = (np.abs(np.fft.fftshift(fft2)) ** 2) / (ny * nx)

    # frequency axes in cycles/arcsec
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=pix_as))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=pix_as))
    FX, FY = np.meshgrid(fx, fy)
    freq_map = np.sqrt(FX**2 + FY**2)          # cycles/arcsec

    # azimuthal average in frequency annuli
    f_max  = freq_map.max()
    n_bins = min(ny, nx) // 2
    f_bins = np.linspace(0, f_max, n_bins + 1)
    f_ctr  = 0.5 * (f_bins[:-1] + f_bins[1:])

    mean_power = np.array([
        power[(freq_map >= f_bins[i]) & (freq_map < f_bins[i + 1])].mean()
        if np.any((freq_map >= f_bins[i]) & (freq_map < f_bins[i + 1]))
        else np.nan
        for i in range(n_bins)
    ])

    # convert frequency → angular scale (arcsec); skip DC (f=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        scales_as = np.where(f_ctr > 0, 1.0 / f_ctr, np.nan)

    return scales_as, mean_power


# ── main ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, len(SOURCES), figsize=(5 * len(SOURCES), 4.5),
                         sharey=False)

for ax, slug in zip(axes, SOURCES):
    raw_path  = find_raw_fits(slug)
    proc_path = find_proc_fits(slug)

    if raw_path is None or proc_path is None:
        ax.set_title(f"{slug}\n(files not found)")
        continue

    img_raw,  pix_raw,  bmaj, bmin = read_image(raw_path)
    img_proc, pix_proc, _,    _    = read_image(proc_path)

    scales_raw,  power_raw  = azimuthal_power_spectrum(img_raw,  pix_raw)
    scales_proc, power_proc = azimuthal_power_spectrum(img_proc, pix_proc)

    # normalise both to their own peak so they're on the same scale
    norm_raw  = np.nanmax(power_raw)
    norm_proc = np.nanmax(power_proc)

    ax.loglog(scales_raw,  power_raw  / norm_raw,
              color="steelblue", lw=1.5, label=f"Original ({pix_raw:.1f}''/px)")
    ax.loglog(scales_proc, power_proc / norm_proc,
              color="darkorange", lw=1.5, ls="--",
              label=f"Downsampled ({pix_proc:.1f}''/px)")

    # mark beam scales
    ax.axvline(bmaj, color="steelblue",   ls=":", lw=1.2, alpha=0.8, label=f"BMAJ={bmaj:.1f}''")
    ax.axvline(bmin, color="steelblue",   ls="-.", lw=1.2, alpha=0.8, label=f"BMIN={bmin:.1f}''")
    # mark Nyquist limit of downsampled image
    nyquist_proc = 2 * pix_proc
    ax.axvline(nyquist_proc, color="darkorange", ls=":", lw=1.2, alpha=0.8,
               label=f"Nyquist ({nyquist_proc:.1f}'')")

    ax.set_xlabel("Angular scale (arcsec)")
    ax.set_ylabel("Normalised power")
    ax.set_title(f"{slug}\nBMAJ={bmaj:.1f}'', BMIN={bmin:.1f}''", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_xlim(1, 2000)
    ax.grid(True, which="both", ls=":", alpha=0.4)

fig.suptitle("Power spectrum: before vs after downsampling to 128×128 px", y=1.01)
fig.tight_layout()
fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
