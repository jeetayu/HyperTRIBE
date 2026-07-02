#!/usr/bin/env python3
"""
Plot mutation spectrum bar charts for TRIBE experiments.

Usage:
    python plot_mutation_spectrum.py \\
        --inputs  d210n=results/d210n/mutation_spectrum.tsv \\
                  zrsr2=results/zrsr2/mutation_spectrum.tsv \\
        --outdir  figures/

Generates:
    all_conversions.pdf       — all 12 conversion types, datasets overlaid
    a_dominant_only.pdf       — A>C, A>G, A>T only (most relevant for ADAR)
    per_dataset/              — one figure per dataset
    enrichment_over_bg.pdf    — A>G fold-enrichment over mean of A>C and A>T
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none',
})

# Ordered conversion types grouped by reference base
CONVERSION_ORDER = [
    "A>C", "A>G", "A>T",
    "C>A", "C>G", "C>T",
    "G>A", "G>C", "G>T",
    "T>A", "T>C", "T>G",
]

REF_COLORS = {
    "A": "#4393C3",   # blue
    "C": "#D6604D",   # red
    "G": "#74C476",   # green
    "T": "#FD8D3C",   # orange
}

ADAR_COLOR   = "#1A1A2E"   # dark navy for A>G
HIGHLIGHT_ALPHA = 1.0
NORMAL_ALPHA    = 0.55

# Hatch patterns by project — makes bars distinguishable in B&W / for colourblind readers
def _dataset_hatch(label: str) -> str:
    l = label.lower()
    if "noura" in l:
        return "////"
    if "yuxi" in l:
        return "xxxx"
    return ""   # Bolton and unknowns: solid


def load_spectrum(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["conversion"] = df["ref_base"] + ">" + df["alt_base"]
    return df.set_index("conversion")


def freq_col(df: pd.DataFrame) -> str:
    """Use global_freq as primary metric (total alt / total ref+alt)."""
    return "global_freq"


def plot_all_conversions(datasets: dict, outdir: Path):
    """Bar chart: all 12 conversions, one group per conversion, one bar per dataset."""
    fig, ax = plt.subplots(figsize=(13, 5))
    n_ds  = len(datasets)
    width = 0.8 / n_ds
    x     = np.arange(len(CONVERSION_ORDER))
    fc    = freq_col(next(iter(datasets.values())))

    for i, (label, df) in enumerate(datasets.items()):
        vals = [df.loc[c, fc] if c in df.index else 0.0 for c in CONVERSION_ORDER]
        colors = []
        for c in CONVERSION_ORDER:
            if c == "A>G":
                colors.append(ADAR_COLOR)
            else:
                ref = c[0]
                colors.append(REF_COLORS[ref])

        alphas = [HIGHLIGHT_ALPHA if c == "A>G" else NORMAL_ALPHA for c in CONVERSION_ORDER]
        hatch  = _dataset_hatch(label)
        for j, (c, v, col, alp) in enumerate(zip(CONVERSION_ORDER, vals, colors, alphas)):
            ax.bar(
                x[j] + (i - n_ds / 2 + 0.5) * width,
                v, width * 0.9,
                color=col, alpha=alp,
                hatch=hatch, edgecolor="white" if not hatch else "#555",
                linewidth=0.4,
                label=label if j == 0 else "_nolegend_",
                zorder=3,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(CONVERSION_ORDER, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Conversion frequency (alt / ref+alt)", fontsize=11)
    ax.set_title("Base conversion spectrum — all 12 types", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda y, _: f"{y*100:.3f}%"))

    # Legend: datasets (with hatch) + reference-base colour key
    ds_handles = [
        mpatches.Patch(label=l, facecolor="#888",
                       hatch=_dataset_hatch(l),
                       edgecolor="#555" if _dataset_hatch(l) else "#888")
        for l in datasets
    ]
    ref_handles = [mpatches.Patch(label=f"ref={b}", color=c, alpha=0.7)
                   for b, c in REF_COLORS.items()]
    adar_handle = mpatches.Patch(label="A>G (ADAR)", color=ADAR_COLOR)
    # Project hatch legend
    proj_handles = [
        mpatches.Patch(label="Bolton (solid)",  facecolor="#aaa", hatch="",     edgecolor="#aaa"),
        mpatches.Patch(label="Yuxi (xxxx)",     facecolor="#aaa", hatch="xxxx", edgecolor="#555"),
        mpatches.Patch(label="Noura (////)",    facecolor="#aaa", hatch="////", edgecolor="#555"),
    ]
    ax.legend(handles=ds_handles + [adar_handle] + proj_handles + ref_handles,
              frameon=False, fontsize=8, ncol=3, loc="upper right")

    ax.axvline(2.5, color="gray", lw=0.7, linestyle="--", alpha=0.4)
    ax.axvline(5.5, color="gray", lw=0.7, linestyle="--", alpha=0.4)
    ax.axvline(8.5, color="gray", lw=0.7, linestyle="--", alpha=0.4)

    fig.tight_layout()
    out = outdir / "all_conversions.pdf"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_a_dominant(datasets: dict, outdir: Path):
    """Focus plot: A>C, A>G, A>T with error bars (std across positions)."""
    a_conv = ["A>C", "A>G", "A>T"]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    n_ds   = len(datasets)
    width  = 0.7 / n_ds
    x      = np.arange(3)

    for i, (label, df) in enumerate(datasets.items()):
        vals = [df.loc[c, "global_freq"] if c in df.index else 0.0 for c in a_conv]
        errs = [df.loc[c, "std_pos_freq"] if c in df.index else 0.0 for c in a_conv]
        colors = [ADAR_COLOR if c == "A>G" else REF_COLORS["A"] for c in a_conv]
        alphas = [HIGHLIGHT_ALPHA if c == "A>G" else NORMAL_ALPHA for c in a_conv]

        for j, (v, e, col, alp) in enumerate(zip(vals, errs, colors, alphas)):
            offset = (i - n_ds / 2 + 0.5) * width
            ax.bar(x[j] + offset, v, width * 0.9, color=col, alpha=alp,
                   label=label if j == 1 else "_nolegend_", zorder=3)
            ax.errorbar(x[j] + offset, v, yerr=e, fmt="none",
                        color="black", capsize=3, linewidth=1.2, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(["A>C\n(background)", "A>G\n(ADAR)", "A>T\n(background)"],
                       fontsize=10)
    ax.set_ylabel("Conversion frequency (global)", fontsize=11)
    ax.set_title("A-dominant positions:\nADAR specificity vs. background", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda y, _: f"{y*100:.3f}%"))
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = outdir / "a_dominant_only.pdf"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


APOBEC_COLOR = "#6a3d9a"   # purple for C>T

def plot_enrichment(datasets: dict, outdir: Path):
    """Grouped bar chart: A>G (ADAR) and C>T (APOBEC) fold-enrichment per dataset."""
    n_ds  = len(datasets)
    width = 0.35
    x     = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(max(5, n_ds * 1.8), 4.5))

    ag_vals, ct_vals, labels = [], [], []
    for label, df in datasets.items():
        # A>G enrichment over mean(A>C, A>T)
        ag  = df.loc["A>G", "global_freq"] if "A>G" in df.index else 0.0
        a_bg = np.mean([
            df.loc["A>C", "global_freq"] if "A>C" in df.index else 0.0,
            df.loc["A>T", "global_freq"] if "A>T" in df.index else 0.0,
        ])
        ag_vals.append(ag / a_bg if a_bg > 0 else np.nan)

        # C>T enrichment over mean(C>A, C>G)  — APOBEC-like signal
        ct  = df.loc["C>T", "global_freq"] if "C>T" in df.index else 0.0
        c_bg = np.mean([
            df.loc["C>A", "global_freq"] if "C>A" in df.index else 0.0,
            df.loc["C>G", "global_freq"] if "C>G" in df.index else 0.0,
        ])
        ct_vals.append(ct / c_bg if c_bg > 0 else np.nan)
        labels.append(label)

    bars_ag = ax.bar(x - width / 2, ag_vals, width, color=ADAR_COLOR,
                     alpha=0.85, label="A>G / mean(A>C, A>T)  [ADAR]", zorder=3)
    bars_ct = ax.bar(x + width / 2, ct_vals, width, color=APOBEC_COLOR,
                     alpha=0.75, label="C>T / mean(C>A, C>G)  [APOBEC-like]", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Fold-enrichment over matched background", fontsize=11)
    ax.set_title("Editing specificity: A>G (ADAR) vs C>T (APOBEC-like)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(1, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(frameon=False, fontsize=9)

    for bar, val in zip(list(bars_ag) + list(bars_ct),
                        ag_vals + ct_vals):
        if not np.isnan(val) and val > 1:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(ag_vals + ct_vals, default=1) * 0.01,
                    f"{val:.1f}×", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out = outdir / "enrichment_over_bg.pdf"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_per_dataset(label: str, df: pd.DataFrame, outdir: Path):
    """One-panel version per dataset for supplementary figures."""
    sub_dir = outdir / "per_dataset"
    sub_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [3, 1]})

    # Left: all 12
    ax = axes[0]
    vals   = [df.loc[c, "global_freq"] if c in df.index else 0.0 for c in CONVERSION_ORDER]
    colors = [ADAR_COLOR if c == "A>G" else REF_COLORS[c[0]] for c in CONVERSION_ORDER]
    alphas = [HIGHLIGHT_ALPHA if c == "A>G" else NORMAL_ALPHA for c in CONVERSION_ORDER]
    bars   = ax.bar(range(12), vals, color=colors, zorder=3)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)
    ax.set_xticks(range(12))
    ax.set_xticklabels(CONVERSION_ORDER, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title(f"{label} — all 12 conversions", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda y, _: f"{y*100:.3f}%"))
    ax.axvline(2.5, color="gray", lw=0.6, ls="--", alpha=0.4)
    ax.axvline(5.5, color="gray", lw=0.6, ls="--", alpha=0.4)
    ax.axvline(8.5, color="gray", lw=0.6, ls="--", alpha=0.4)

    # Right: A>C, A>G, A>T zoom
    ax2 = axes[1]
    a_conv = ["A>C", "A>G", "A>T"]
    v2 = [df.loc[c, "global_freq"] if c in df.index else 0.0 for c in a_conv]
    e2 = [df.loc[c, "std_pos_freq"] if c in df.index else 0.0 for c in a_conv]
    c2 = [ADAR_COLOR if c == "A>G" else REF_COLORS["A"] for c in a_conv]
    a2 = [HIGHLIGHT_ALPHA if c == "A>G" else NORMAL_ALPHA for c in a_conv]
    for j, (v, e, col, alp) in enumerate(zip(v2, e2, c2, a2)):
        ax2.bar(j, v, 0.6, color=col, alpha=alp, zorder=3)
        ax2.errorbar(j, v, yerr=e, fmt="none", color="black", capsize=3, lw=1.2)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["A>C", "A>G", "A>T"], fontsize=9)
    ax2.set_title("A-dominant zoom", fontsize=10)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda y, _: f"{y*100:.3f}%"))

    fig.tight_layout()
    out = sub_dir / f"{label.replace(' ', '_')}_spectrum.pdf"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True,
                    metavar="LABEL=PATH",
                    help="label=path/to/mutation_spectrum.tsv pairs")
    ap.add_argument("--outdir", default="figures/mutation_spectrum")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    datasets = {}
    for item in args.inputs:
        if "=" not in item:
            print(f"ERROR: expected LABEL=PATH, got: {item}", file=sys.stderr)
            sys.exit(1)
        label, path = item.split("=", 1)
        datasets[label] = load_spectrum(path)
        print(f"Loaded: {label} ({path})")

    plot_all_conversions(datasets, outdir)
    plot_a_dominant(datasets, outdir)
    plot_enrichment(datasets, outdir)
    for label, df in datasets.items():
        plot_per_dataset(label, df, outdir)

    print(f"\nAll figures → {outdir}/")


if __name__ == "__main__":
    main()
