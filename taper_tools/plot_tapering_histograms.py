#!/usr/bin/env python3
# hist_shapes_overlay_multiprocess.py
import os, time, numpy as np, multiprocessing as mp
from queue import Empty 
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

plt.ion()

from plot_tapered_beam_properties import load_z_table, pick_random_pairs, get_beam_info

mpl.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.titlesize": 13,
    "legend.fontsize": 11,
})

def common_log_edges(all_vals, bins=30, clip=(0.5, 99.5)):
    """Compute common log-spaced edges for multiple series."""
    pooled = np.concatenate([v for v in all_vals if v.size])
    pooled = pooled[np.isfinite(pooled) & (pooled > 0)]
    if pooled.size == 0:
        return None
    lo, hi = np.nanpercentile(pooled, list(clip))
    # fallback if degenerate
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0 or lo >= hi:
        lo, hi = np.nanmin(pooled), np.nanmax(pooled)
        lo = max(lo, np.nextafter(0.0, 1.0))
    return np.logspace(np.log10(lo), np.log10(hi), bins + 1)


def init_live_overlay(labels, bins=30):
    # accept list OR dict
    labels = list(labels.keys()) if isinstance(labels, dict) else list(labels)
    n = len(labels)                           # <-- define n

    fig = plt.figure(figsize=(8.4, 5.8), dpi=130)
    fig.set_facecolor("white")
    try:
        fig.canvas.manager.set_window_title("Histogram shapes (overlay)")
    except Exception:
        pass

    # leave extra bottom space so stacked x-axes fit under the plot
    left, right = 0.10, 0.88
    bottom_stack = 0.16 + 0.04 * max(0, n - 1)   # grow space with #axes
    top_margin   = 0.08

    base = fig.add_axes([left, bottom_stack, right - left, 1.0 - bottom_stack - top_margin],
                        frameon=True)
    base.set_xticks([]); base.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        base.spines[s].set_color("0.8")

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    axes = []

    # pixel offsets for overlaid spines (returned so update() can reuse)
    y_off_step = 30
    x_off_step = 30

    for i, q in enumerate(labels):
        axi = fig.add_axes(base.get_position(), label=f"ov_{i}", frameon=True)
        axi.patch.set_alpha(0.0)
        c = colors[i % len(colors)]

        axi.spines["left"].set_position(("outward",  y_off_step * i))
        # positive 'outward' moves the bottom spine DOWN (stack under)
        axi.spines["bottom"].set_position(("outward", x_off_step * i))
        for s in ("left", "bottom"):
            axi.spines[s].set_color(c)
        axi.spines["top"].set_visible(False)
        axi.spines["right"].set_visible(False)

        axi.tick_params(axis="y", colors=c, labelsize=10, pad=6)
        axi.tick_params(axis="x", colors=c, labelsize=10, pad=8)
        axes.append(axi)                      # <-- append to list

    handles = [Line2D([0], [0], color=colors[i % len(colors)], lw=2, label=q)
               for i, q in enumerate(labels)]
    fig.legend(handles, [h.get_label() for h in handles],
               loc="upper right", frameon=False, title="Quantities")
    fig.suptitle("Histogram shapes (overlay)  N=0", y=0.99)
    fig.canvas.draw_idle()
    plt.show(block=False)

    return fig, axes, colors, bins, labels, y_off_step, x_off_step

def update_live_overlay(fig, axes, colors, bins, labels, results, quant_funcs,
                        y_off_step, x_off_step, log_x=True):
    # ------- gather values per label -------
    vals_by_label = []
    for q in labels:
        v = []
        for r in results:
            try:
                v.append(float(quant_funcs[q](r)))
            except Exception:
                pass
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if log_x:
            v = v[v > 0]
        vals_by_label.append(v)

    # ------- common edges for fair comparison -------
    if log_x:
        edges = common_log_edges(vals_by_label, bins=bins, clip=(0.5, 99.5))
    else:
        pooled = np.concatenate([v for v in vals_by_label if v.size])
        lo, hi = np.nanpercentile(pooled, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = float(np.nanmin(pooled)), float(np.nanmax(pooled))
        edges = np.linspace(lo, hi, bins + 1)
    if edges is None:
        return  # nothing finite yet

    # ------- histograms + area-normalization -------
    series = []
    y_max = 0.0
    for v in vals_by_label:
        if v.size < 2:
            series.append(None); continue
        h, _ = np.histogram(v, bins=edges, density=True)
        # x for plotting the step midpoints
        x = 0.5*(edges[:-1] + edges[1:])
        # normalize the shape areas to 1 for direct shape comparison
        y = h / (np.trapz(h, x) + 1e-30)
        y_max = max(y_max, float(np.nanmax(y))) if y.size else y_max
        # summary stats
        med = np.median(v)
        lo68, hi68 = np.percentile(v, [16, 84])
        series.append((x, y, edges[0], edges[-1], med, lo68, hi68, v.size))

    if not np.isfinite(y_max) or y_max <= 0:
        y_max = 1.0


    # redraw
    for i, (ax, q, st) in enumerate(zip(axes, labels, series)):
        c = colors[i % len(colors)]
        ax.cla()
        ax.set_facecolor("none"); ax.patch.set_alpha(0.0); ax.set_zorder(i)
        ax.spines["left"].set_position(("outward",  y_off_step * i))
        ax.spines["bottom"].set_position(("outward", x_off_step * i))
        ax.spines["left"].set_color(c); ax.spines["bottom"].set_color(c)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", colors=c, labelsize=10, pad=6)
        ax.tick_params(axis="x", colors=c, labelsize=10, pad=8)
        if st is None:
            ax.set_ylim(0, y_max * 1.05)
            if log_x: ax.set_xscale("log")
            continue

        x, y, lo, hi, med, lo68, hi68, n = st
        if log_x:
            ax.set_xscale("log")
        ax.fill_between(x, 0.0, y, color=c, alpha=0.25, step="mid")
        ax.plot(x, y, color=c, lw=1.6, drawstyle="steps-mid")
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, y_max * 1.05)
        yt = ax.get_yticks()
        if len(yt) > 4:
            ax.set_yticks(yt[::2])

    fig.suptitle(f"Histogram shapes (overlay)  N={len(results)}", y=0.995)
    fig.canvas.draw_idle(); fig.canvas.flush_events()

def data_processor(pairs, slug_to_z, result_queue):
    """Background process that computes beam info and streams dicts back."""
    for RAW, T50 in pairs:
        slug = os.path.splitext(os.path.basename(RAW))[0]
        z = slug_to_z.get(slug, np.nan)
        if not np.isfinite(z):
            continue
        try:
            res = get_beam_info(RAW, T50_fits=T50, out_pdf=None, z=z, fwhm_kpc=50.0)
            result_queue.put(res)
        except Exception as e:
            print(f"[error] {slug}: {e}")
    result_queue.put(None)  # sentinel

if __name__ == "__main__":
    ROOT = "/home/markusbredberg/Scripts/data/PSZ2/fits"        
    Z_CSV = "/home/markusbredberg/Scripts/data/PSZ2/cluster_source_data.csv"

    QUANT_FUNCS = {
        "ratio_tgt_circ_over_tgt_raw":  lambda r: r["omega_tgt_circ"]  / (r["omega_tgt_raw"]  + 1e-30),
        # "ratio_tgt_circ_over_tgt_raw": lambda r: r["omega_tgt_circ"] / (r["omega_tgt_raw"]  + 1e-30),
        # "ratio_tgt_circ_over_tgt_T50": lambda r: r["omega_tgt_circ"] / (r["omega_tgt_T50"] + 1e-30),
        "ratio_tgt_T50_over_tgt_raw": lambda r: r["omega_tgt_T50"] / (r["omega_tgt_raw"]  + 1e-30),
    }

    # init plot
    fig, axes, colors, bins, labels, y_off_step, x_off_step = init_live_overlay(
        list(QUANT_FUNCS.keys()), bins=30
    )

    slug_to_z = load_z_table(Z_CSV)
    pairs = pick_random_pairs(ROOT, n=10**6, seed=42)

    result_queue = mp.Queue(maxsize=100)
    proc = mp.Process(target=data_processor, args=(pairs, slug_to_z, result_queue), daemon=True)
    proc.start()

    results = []
    UPDATE_EVERY = 10
    last_update_time = time.time()
    MIN_UPDATE_INTERVAL = 0.5

    try:
        while True:
            try:
                res = result_queue.get(timeout=0.1)
                if res is None:
                    break
                results.append(res)
                now = time.time()
                if len(results) % UPDATE_EVERY == 0 and (now - last_update_time) >= MIN_UPDATE_INTERVAL:
                    update_live_overlay(fig, axes, colors, bins, labels, results, QUANT_FUNCS,
                                        y_off_step, x_off_step)
                    last_update_time = now
            except Empty:
                # keep GUI responsive
                fig.canvas.flush_events()
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[interrupted]")
        proc.terminate()

    # final update & save
    update_live_overlay(fig, axes, colors, bins, labels, results, QUANT_FUNCS,
                        y_off_step, x_off_step)
    proc.join(timeout=2.0)
    if proc.is_alive():
        proc.terminate()

    outdir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "hist_shapes_overlay_live.png")
    fig.savefig(outfile, bbox_inches="tight")
    print(f"[plot] wrote {outfile} with {len(results)} sources")
    plt.ioff()
