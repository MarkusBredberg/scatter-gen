#!/usr/bin/env python3
"""
Reproduce Fig B.1 and Fig B.2 from the beam-area scatter analysis.

Fig B.1 — scatter + marginal histograms
    X : Ω_ref (native beam area, 10⁻¹⁰ sr)
    Y : Ω_T50 (tapered, circles) and Ω_blur (theoretical 50 kpc target, triangles), log scale
    C : Angular diameter distance D_A [Mpc]

Fig B.2 — ratio scatter + right-side histogram
    X : Ω_ref (10⁻¹⁰ sr)
    Y : Ω_blur / Ω_T50
    C : D_A [Mpc]
    Horizontal lines at median, 16th, 84th percentile

Usage:
    python scripts/plot_beam_scatter.py
"""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18 as COSMO
from astropy.io import fits
import astropy.units as u

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dcreclass.data import load_z_table, find_pairs_in_tree
from dcreclass.data.processing import circular_cov_kpc
from dcreclass.utils import beam_cov_world, beam_solid_angle_sr

FITS_ROOT = Path("/users/mbredber/scratch/data/PSZ2/fits")
PROC_ROOT = Path("/users/mbredber/scratch/data/PSZ2/beam_crop/circular/fits_files")
Z_CSV     = Path("/users/mbredber/scratch/data/PSZ2/cluster_source_data.csv")
OUT_DIR   = Path("/users/mbredber/scratch/figures/processing")

mpl.rcParams.update({
    "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
})


# ── data collection ───────────────────────────────────────────────────────────

def _load_header(path):
    with fits.open(path, memmap=False) as hdul:
        return hdul[0].header.copy()


def collect_beam_records(slug_to_z, fwhm_kpc=50.0, use_actual_blur=False):
    """Read FITS headers for all (RAW, T50kpc) pairs and compute beam quantities.

    use_actual_blur=False (default): blur beam area computed theoretically from
        cosmology + native beam (omega_tgt_circ = omega(C_raw + C_circ)).
    use_actual_blur=True: blur beam area read from the processed
        Blur{kpc}kpc FITS header in PROC_ROOT.
    """
    kpc = int(fwhm_kpc)
    records = []
    for slug, raw_path, t_path, _, _ in find_pairs_in_tree(FITS_ROOT, desired_kpc=fwhm_kpc):
        z = slug_to_z.get(slug, np.nan)
        if not np.isfinite(z) or z <= 0:
            continue
        try:
            H_raw = _load_header(raw_path)
            H_t   = _load_header(t_path)

            omega_raw = beam_solid_angle_sr(H_raw)
            omega_tap = beam_solid_angle_sr(H_t)

            C_raw  = beam_cov_world(H_raw)
            C_circ = circular_cov_kpc(z, fwhm_kpc=fwhm_kpc)

            _omega = lambda C: 2.0 * np.pi * np.sqrt(max(0.0, np.linalg.det(C)))
            # omega_rt_circ : pure theoretical target — always used for Fig B.1 triangles
            omega_rt_circ = _omega(C_circ)

            if use_actual_blur:
                blur_path = PROC_ROOT / f"{slug}_Blur{kpc}kpc_fmt_128x128_circular.fits"
                H_blur = _load_header(blur_path)
                omega_tgt_circ = beam_solid_angle_sr(H_blur)
            else:
                omega_tgt_circ = _omega(C_raw + C_circ)

            DA_Mpc = float(COSMO.angular_diameter_distance(z).to_value(u.Mpc))

            records.append(dict(
                slug=slug, z=z, DA_Mpc=DA_Mpc,
                omega_raw=omega_raw,
                omega_tap=omega_tap,
                omega_rt_circ=omega_rt_circ,
                omega_tgt_circ=omega_tgt_circ,
                ratio=omega_tgt_circ / (omega_tap + 1e-40),
            ))
        except Exception as e:
            print(f"  [skip] {slug}: {e}")
    return records


# ── plot helpers ──────────────────────────────────────────────────────────────

def _colored_x_hist(ax, x, c, bins, cmap, vmin, vmax):
    """Top marginal: linear bins, bars colored by mean C per bin."""
    counts, edges = np.histogram(x, bins=bins)
    for i, count in enumerate(counts):
        if count == 0:
            continue
        in_bin = (x >= edges[i]) & (x < edges[i + 1])
        mean_c = float(np.mean(c[in_bin])) if in_bin.any() else vmin
        col = cmap(np.clip((mean_c - vmin) / (vmax - vmin + 1e-30), 0, 1))
        ax.bar(edges[i], count, width=edges[i + 1] - edges[i],
               align='edge', color=col, alpha=0.85, linewidth=0)


def _stepped_outline_y_colored(ax, y, c, bins, linestyle, cmap, vmin, vmax):
    """Right marginal: stepped outline per log bin, colored by mean C per bin."""
    counts, edges = np.histogram(y, bins=bins)
    for i, (count, e0, e1) in enumerate(zip(counts, edges[:-1], edges[1:])):
        in_bin = (y >= e0) & (y < e1)
        if not in_bin.any():
            continue
        mean_c = float(np.mean(c[in_bin]))
        col = cmap(np.clip((mean_c - vmin) / (vmax - vmin + 1e-30), 0, 1))
        ax.plot([0, count, count, 0], [e0, e0, e1, e1],
                linestyle=linestyle, color=col, lw=1.4)


def _scatter_setup(fig, with_top_hist=True, scatter_width=4):
    """GridSpec: [top_hist?, scatter | right_hist | cbar]."""
    if with_top_hist:
        gs = fig.add_gridspec(2, 3,
                              width_ratios=[scatter_width, 0.5, 0.15],
                              height_ratios=[0.5, 4],
                              wspace=0.05, hspace=0.05)
        ax       = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        cax      = fig.add_subplot(gs[1, 2])
        return ax, ax_histx, ax_histy, cax
    else:
        gs = fig.add_gridspec(1, 3,
                              width_ratios=[scatter_width, 0.5, 0.15],
                              wspace=0.05)
        ax       = fig.add_subplot(gs[0, 0])
        ax_histy = fig.add_subplot(gs[0, 1], sharey=ax)
        cax      = fig.add_subplot(gs[0, 2])
        return ax, None, ax_histy, cax


def _finish_marginals(ax_histx, ax_histy):
    if ax_histx is not None:
        plt.setp(ax_histx.get_xticklabels(), visible=False)
        ax_histx.set_yticks([])
        ax_histx.tick_params(axis='x', length=0)
        for sp in ax_histx.spines.values():
            sp.set_visible(False)
    plt.setp(ax_histy.get_yticklabels(), visible=False)
    ax_histy.set_xticks([])
    ax_histy.xaxis.offsetText.set_visible(False)
    ax_histy.tick_params(axis='y', length=0)
    for sp in ax_histy.spines.values():
        sp.set_visible(False)


# ── Fig B.1 ──────────────────────────────────────────────────────────────────

def plot_fig_B1(records, out_path, use_actual_blur=False):
    """Scatter + marginal histograms: Ω_ref vs Ω_T50 / Ω_blur, colored by D_A."""
    X  = np.array([r['omega_raw']  for r in records]) * 1e10   # units: 10⁻¹⁰ sr
    Yt = np.array([r['omega_tap']      for r in records])   # tapered (circles)
    blur_key = 'omega_tgt_circ' if use_actual_blur else 'omega_rt_circ'
    Yb = np.array([r[blur_key] for r in records])           # blur beam (triangles)
    C  = np.array([r['DA_Mpc']    for r in records])

    mask = np.isfinite(X) & np.isfinite(Yt) & np.isfinite(Yb)
    X, Yt, Yb, C = X[mask], Yt[mask], Yb[mask], C[mask]

    vmin = float(np.nanpercentile(C, 2))
    vmax = float(np.nanpercentile(C, 98))
    cmap = plt.get_cmap('viridis')

    with mpl.rc_context({'font.size': 14, 'axes.labelsize': 14,
                         'xtick.labelsize': 13, 'ytick.labelsize': 13}):
        fig = plt.figure(figsize=(7.0, 5.5), layout='constrained')
        ax, ax_histx, ax_histy, cax = _scatter_setup(fig, with_top_hist=True)

        # Vertical connectors
        for xi, yt, yb in zip(X, Yt, Yb):
            if np.isfinite(yt) and np.isfinite(yb):
                ax.plot([xi, xi], [yt, yb], color='0.6', lw=0.6, zorder=1)

        sc = ax.scatter(X, Yt, c=C, s=24, alpha=0.85, cmap=cmap,
                        vmin=vmin, vmax=vmax, marker='o', zorder=3)
        ax.scatter(X, Yb, c=C, s=24, alpha=0.85, cmap=cmap,
                   vmin=vmin, vmax=vmax, marker='^', zorder=3)

        cb = fig.colorbar(sc, cax=cax)
        cb.set_label(r'$D_\mathrm{A}$ [Mpc]')

        ax.set_yscale('log')
        ax.set_xlabel(r'Reference beam area $\Omega_\mathrm{ref}$ [$10^{-10}$ sr]')
        ax.set_ylabel(r'Target beam area $\Omega$ [sr]')
        ax.grid(False)

        from matplotlib.legend_handler import HandlerTuple
        mid_c = cmap(0.55)
        tap_h  = (plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='0.5', ms=7),
                  plt.Line2D([0],[0], ls='-',  color=mid_c, lw=1.2))
        blur_h = (plt.Line2D([0],[0], marker='^', color='w', markerfacecolor='0.5', ms=7),
                  plt.Line2D([0],[0], ls='--', color=mid_c, lw=1.2))
        ax.legend(handles=[tap_h, blur_h],
                  labels=[r'$\Omega_\mathrm{T50}$ (tapered)',
                          r'$\Omega_\mathrm{blur}$ (blurred)'],
                  handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)},
                  frameon=False, fontsize=11)

        # Top histogram
        nbx = max(10, int(np.sqrt(X.size)))
        linx_bins = np.linspace(X.min(), X.max(), nbx)
        _colored_x_hist(ax_histx, X, C, linx_bins, cmap, vmin, vmax)

        # Right histogram
        both_Y = np.concatenate([Yt[Yt > 0], Yb[Yb > 0]])
        logy_bins = np.logspace(np.log10(np.nanmin(both_Y)),
                                np.log10(np.nanmax(both_Y)),
                                max(10, int(np.sqrt(X.size))))
        _stepped_outline_y_colored(ax_histy, Yt, C, logy_bins, '-',  cmap, vmin, vmax)
        _stepped_outline_y_colored(ax_histy, Yb, C, logy_bins, '--', cmap, vmin, vmax)

        _finish_marginals(ax_histx, ax_histy)

        # Median lines on right histogram
        for y_arr, ls in [(Yt, '-'), (Yb, '--')]:
            med = np.median(y_arr[y_arr > 0])
            ax_histy.axhline(med, color='0.3', lw=1.0, ls=ls)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Fig B.2 ──────────────────────────────────────────────────────────────────

def plot_fig_B2(records, out_path):
    """Ratio scatter + right histogram: Ω_blur/Ω_tap vs Ω_ref, colored by D_A."""
    X     = np.array([r['omega_raw'] for r in records]) * 1e10
    ratio = np.array([r['ratio']     for r in records])
    C     = np.array([r['DA_Mpc']   for r in records])

    mask = np.isfinite(X) & np.isfinite(ratio) & np.isfinite(C)
    X, ratio, C = X[mask], ratio[mask], C[mask]

    vmin = float(np.nanpercentile(C, 2))
    vmax = float(np.nanpercentile(C, 98))
    cmap = plt.get_cmap('viridis')

    with mpl.rc_context({'font.size': 14, 'axes.labelsize': 14,
                         'xtick.labelsize': 13, 'ytick.labelsize': 13}):
        fig = plt.figure(figsize=(7.0, 4.5), layout='constrained')
        ax, _, ax_histy, cax = _scatter_setup(fig, with_top_hist=False, scatter_width=2)

        sc = ax.scatter(X, ratio, c=C, s=24, alpha=0.85,
                        cmap=cmap, vmin=vmin, vmax=vmax, zorder=3)

        cb = fig.colorbar(sc, cax=cax)
        cb.set_label(r'$D_\mathrm{A}$ [Mpc]')

        ax.set_xlabel(r'Reference beam area $\Omega_\mathrm{ref}$ [$10^{-10}$ sr]')
        ax.set_ylabel(r'$\Omega_\mathrm{blur}\ /\ \Omega_\mathrm{T50}$')
        ax.grid(False)

        # Right histogram colored by mean D_A per ratio bin
        n_bins = max(20, int(np.sqrt(ratio.size)))
        r_bins = np.linspace(ratio.min() * 0.98, ratio.max() * 1.02, n_bins + 1)
        counts, edges = np.histogram(ratio, bins=r_bins)
        for i, count in enumerate(counts):
            if count == 0:
                continue
            in_bin = (ratio >= edges[i]) & (ratio < edges[i + 1])
            mean_c = float(np.mean(C[in_bin]))
            col = cmap(np.clip((mean_c - vmin) / (vmax - vmin + 1e-30), 0, 1))
            mid = 0.5 * (edges[i] + edges[i + 1])
            ax_histy.barh(mid, count, height=edges[i + 1] - edges[i],
                          color=col, alpha=0.85, linewidth=0)

        _finish_marginals(None, ax_histy)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--actual-blur', action='store_true',
                    help='Use beam area from processed Blur FITS headers instead of theoretical values.')
    args = ap.parse_args()

    tag = 'actual' if args.actual_blur else 'theoretical'
    print(f"Blur mode: {tag}")
    print("Loading redshift table...")
    slug_to_z = load_z_table(Z_CSV)

    print("Collecting beam records (reading FITS headers)...")
    records = collect_beam_records(slug_to_z, fwhm_kpc=50.0, use_actual_blur=args.actual_blur)
    print(f"  Collected {len(records)} records.")

    ratios = np.array([r['ratio'] for r in records])
    print(f"  ratio range: {ratios.min():.3f} – {ratios.max():.3f}")
    print(f"  median: {np.median(ratios):.3f}  "
          f"  16th: {np.percentile(ratios, 16):.3f}  "
          f"  84th: {np.percentile(ratios, 84):.3f}")

    print("\nPlotting Fig B.1...")
    plot_fig_B1(records, OUT_DIR / f"beam_scatter_fig_B1_{tag}.pdf",
                use_actual_blur=args.actual_blur)

    print("Plotting Fig B.2...")
    plot_fig_B2(records, OUT_DIR / f"beam_scatter_fig_B2_{tag}.pdf")

    print("\nDone.")


if __name__ == '__main__':
    main()
