#!/usr/bin/env python3
"""
Beam ratio sampler: split sources by Omega_blur/Omega_tap band and show
their processed images (RAW | T{kpc}kpc | Blur{kpc}kpc) to investigate
the origin of the bimodal distribution.

Usage:
    python scripts/beam_ratio_sampler.py [--scale 50] [--n 5]
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dcreclass.data import load_z_table, find_pairs_in_tree
from dcreclass.data.processing import circular_cov_kpc
from dcreclass.utils import beam_cov_world, beam_solid_angle_sr

FITS_ROOT = Path("/users/mbredber/scratch/data/PSZ2/fits")
PROC_ROOT = Path("/users/mbredber/scratch/data/PSZ2/beam_crop/circular/fits_files")
Z_CSV     = Path("/users/mbredber/scratch/data/PSZ2/cluster_source_data.csv")
OUT_DIR   = Path("/users/mbredber/scratch/figures/processing")
THRESHOLD = 0.57


# ── FITS helpers ──────────────────────────────────────────────────────────────

def load_header(path):
    with fits.open(path, memmap=False) as hdul:
        return hdul[0].header.copy()

def load_image(path):
    with fits.open(path, memmap=False) as hdul:
        return np.squeeze(np.array(hdul[0].data, dtype=float))

def omega_cov(C):
    """Gaussian beam solid angle from 2x2 world-coord covariance matrix (rad²)."""
    return 2.0 * np.pi * np.sqrt(max(0.0, np.linalg.det(C)))

def robust_vmin_vmax(img, plo=1, phi=99.5):
    finite = img[np.isfinite(img)]
    if len(finite) == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, plo)), float(np.percentile(finite, phi))


# ── ratio computation ─────────────────────────────────────────────────────────

def compute_ratios(scale_kpc, slug_to_z, use_actual_blur=False):
    kpc = int(scale_kpc)
    records = []
    for slug, raw_path, t_path, _, _ in find_pairs_in_tree(FITS_ROOT, desired_kpc=scale_kpc):
        z = slug_to_z.get(slug, np.nan)
        if not np.isfinite(z) or z <= 0:
            continue
        try:
            H_raw = load_header(raw_path)
            H_t   = load_header(t_path)
            omega_raw = beam_solid_angle_sr(H_raw)
            omega_tap = beam_solid_angle_sr(H_t)

            if use_actual_blur:
                blur_path = PROC_ROOT / f"{slug}_Blur{kpc}kpc_fmt_128x128_circular.fits"
                omega_blur = beam_solid_angle_sr(load_header(blur_path))
            else:
                C_raw      = beam_cov_world(H_raw)
                C_circ     = circular_cov_kpc(z, fwhm_kpc=scale_kpc)
                omega_blur = omega_cov(C_raw + C_circ)

            ratio = omega_blur / (omega_tap + 1e-40)
            records.append(dict(
                slug=slug, z=z,
                omega_raw=omega_raw, omega_tap=omega_tap, omega_blur=omega_blur,
                ratio=ratio,
            ))
        except Exception as e:
            print(f"  [skip] {slug}: {e}")
    return records


def sample_spread(records, n):
    """Pick n sources at evenly-spaced positions across the sorted ratio range."""
    records = sorted(records, key=lambda r: r['ratio'])
    if len(records) <= n:
        return records
    idx = np.round(np.linspace(0, len(records) - 1, n)).astype(int)
    return [records[i] for i in idx]


# ── plotting ──────────────────────────────────────────────────────────────────

def proc_path(slug, tag, scale_kpc):
    return PROC_ROOT / f"{slug}_{tag}{int(scale_kpc)}kpc_fmt_128x128_circular.fits"

def raw_proc_path(slug):
    return PROC_ROOT / f"{slug}_RAW_fmt_128x128_circular.fits"


def plot_grid(samples_lo, samples_hi, scale_kpc, out_path):
    kpc = int(scale_kpc)
    col_labels = ['RAW (reference)', f'T{kpc}kpc  (tapered)', f'Blur{kpc}kpc  (blurred)']
    col_keys   = [
        lambda s: raw_proc_path(s),
        lambda s: proc_path(s, 'T', scale_kpc),
        lambda s: proc_path(s, 'Blur', scale_kpc),
    ]

    bands = [
        ('LOW  (ratio < {:.2f})'.format(THRESHOLD), samples_lo, '#2166ac'),
        ('HIGH  (ratio ≥ {:.2f})'.format(THRESHOLD), samples_hi, '#d6604d'),
    ]

    n_lo, n_hi = len(samples_lo), len(samples_hi)
    n_rows = n_lo + n_hi
    n_cols = 3
    row_h  = 2.2
    label_w = 2.8   # inches reserved for left-side row labels

    fig_w = label_w + n_cols * row_h
    fig_h = n_rows  * row_h + 0.5   # +0.5 for column titles

    fig = plt.figure(figsize=(fig_w, fig_h))

    # GridSpec: one narrow label column + 3 image columns
    gs = plt.GridSpec(
        n_rows, n_cols + 1,
        figure=fig,
        width_ratios=[label_w / row_h] + [1.0] * n_cols,
        hspace=0.06,
        wspace=0.03,
        left=0.01, right=0.99,
        top=0.96,  bottom=0.01,
    )

    cmap = plt.cm.viridis.copy()
    cmap.set_bad('white', 1.0)

    row_idx = 0
    for band_label, samples, band_color in bands:
        for sample_idx, rec in enumerate(samples):
            slug = rec['slug']

            # --- label column ---
            ax_lbl = fig.add_subplot(gs[row_idx, 0])
            ax_lbl.axis('off')
            is_first_in_band = (sample_idx == 0)
            if is_first_in_band:
                ax_lbl.text(0.98, 0.98, band_label,
                            transform=ax_lbl.transAxes,
                            fontsize=8, fontweight='bold', color=band_color,
                            va='top', ha='right')
            label_txt = (f"{slug}\n"
                         f"z = {rec['z']:.3f}\n"
                         f"ratio = {rec['ratio']:.3f}")
            ax_lbl.text(0.98, 0.48, label_txt,
                        transform=ax_lbl.transAxes,
                        fontsize=7, va='center', ha='right',
                        family='monospace')

            # --- image columns ---
            for col_idx, (col_label, path_fn) in enumerate(zip(col_labels, col_keys)):
                ax = fig.add_subplot(gs[row_idx, col_idx + 1])

                # column header on very first row only
                if row_idx == 0:
                    ax.set_title(col_label, fontsize=9, fontweight='bold', pad=3)

                p = path_fn(slug)
                if p.exists():
                    img = load_image(p)
                    vmin, vmax = robust_vmin_vmax(img)
                    ax.imshow(img, origin='lower', cmap=cmap,
                              vmin=vmin, vmax=vmax, interpolation='nearest')
                else:
                    ax.text(0.5, 0.5, 'MISSING', transform=ax.transAxes,
                            ha='center', va='center', fontsize=8, color='red')

                # band-colour border on first image column
                if col_idx == 0 and is_first_in_band:
                    for spine in ax.spines.values():
                        spine.set_edgecolor(band_color)
                        spine.set_linewidth(2)
                        spine.set_visible(True)

                ax.axis('off')

            row_idx += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_ratio_histogram(records, samples_lo, samples_hi, scale_kpc, out_path):
    ratios = np.array([r['ratio'] for r in records])
    lo_ratios = [r['ratio'] for r in samples_lo]
    hi_ratios  = [r['ratio'] for r in samples_hi]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.linspace(ratios.min() * 0.95, min(ratios.max() * 1.05, 1.05), 40)
    ax.hist(ratios[ratios < THRESHOLD], bins=bins, color='#2166ac', alpha=0.7, label='LOW band')
    ax.hist(ratios[ratios >= THRESHOLD], bins=bins, color='#d6604d', alpha=0.7, label='HIGH band')
    ax.axvline(THRESHOLD, color='k', lw=1.5, ls='--', label=f'threshold {THRESHOLD}')

    for v in lo_ratios:
        ax.axvline(v, color='#2166ac', lw=1, ls=':')
    for v in hi_ratios:
        ax.axvline(v, color='#d6604d', lw=1, ls=':')

    ax.set_xlabel(r'$\Omega_\mathrm{blur}\ /\ \Omega_\mathrm{tap}$', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_title(f'Beam area ratio distribution — {int(scale_kpc)} kpc  (N={len(records)})',
                 fontsize=11)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', type=float, default=50.0,
                    help='Physical scale in kpc (default: 50)')
    ap.add_argument('--n', type=int, default=5,
                    help='Number of sources to sample per band (default: 5)')
    ap.add_argument('--actual-blur', action='store_true',
                    help='Use beam area from processed Blur FITS headers instead of theoretical values.')
    args = ap.parse_args()

    scale_kpc = args.scale
    tag = 'actual' if args.actual_blur else 'theoretical'
    print(f"Blur mode: {tag}")

    print(f"Loading redshift table...")
    slug_to_z = load_z_table(Z_CSV)

    print(f"Computing beam ratios for all sources (scale={int(scale_kpc)} kpc)...")
    records = compute_ratios(scale_kpc, slug_to_z, use_actual_blur=args.actual_blur)
    print(f"  {len(records)} sources with valid ratios")

    ratios = np.array([r['ratio'] for r in records])
    print(f"  ratio range: {ratios.min():.3f} – {ratios.max():.3f}")
    print(f"  median: {np.median(ratios):.3f}   threshold: {THRESHOLD}")

    lo = [r for r in records if r['ratio'] <  THRESHOLD]
    hi = [r for r in records if r['ratio'] >= THRESHOLD]
    print(f"  LOW band: {len(lo)} sources,  HIGH band: {len(hi)} sources")

    samples_lo = sample_spread(lo, args.n)
    samples_hi = sample_spread(hi, args.n)

    print("\nSampled sources:")
    for band, samples in [('LOW', samples_lo), ('HIGH', samples_hi)]:
        for r in samples:
            print(f"  [{band}]  ratio={r['ratio']:.3f}  z={r['z']:.3f}  {r['slug']}")

    kpc = int(scale_kpc)
    plot_ratio_histogram(
        records, samples_lo, samples_hi, scale_kpc,
        OUT_DIR / f"beam_ratio_histogram_{kpc}kpc_{tag}.pdf",
    )
    plot_grid(
        samples_lo, samples_hi, scale_kpc,
        OUT_DIR / f"beam_ratio_sampler_{kpc}kpc_{tag}.pdf",
    )


if __name__ == '__main__':
    main()
