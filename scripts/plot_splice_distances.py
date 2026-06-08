#!/usr/bin/env python3
"""
Splice Site Distance Plots for HyperTRIBE Editing Sites
========================================================

Generates a multi-panel PDF summarising where intronic editing sites fall
relative to the splicing machinery:

  Panel 1 — Genomic feature breakdown (UTR / CDS_exon / intron / intergenic)
  Panel 2 — Intronic sub-region breakdown
             (5ss_proximal / ppt_region / 3ss_proximal / deep_intronic)
  Panel 3 — Distance-from-5'SS distribution (intronic sites, 0-300 nt)
             with the donor-proximal zone annotated
  Panel 4 — Distance-from-3'SS distribution (intronic sites, 0-300 nt)
             with PPT and acceptor-proximal zones annotated
  Panel 5 — (optional) Scatter of dist_5ss vs dist_3ss coloured by splice_region

If multiple input files are provided (e.g. WT, S161, S206) they are overlaid
on Panels 3/4 so you can compare conditions side by side.

Usage:
    python plot_splice_distances.py \\
        --input  results/puf60_wt/splice_annotated_editing_sites.bed \\
                 results/puf60_s161/splice_annotated_editing_sites.bed \\
        --labels WT S161 \\
        --output results/plots/splice_distance_comparison.pdf
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
})

# Colours (colourblind-friendly)
_REGION_COLORS = {
    'UTR':             '#3498db',
    'CDS_exon':        '#2ecc71',
    'intron':          '#e74c3c',
    'intergenic':      '#95a5a6',
    '5ss_proximal':    '#e74c3c',
    '3ss_proximal':    '#e67e22',
    'ppt_region':      '#f1c40f',
    'deep_intronic':   '#9b59b6',
}

_COND_PALETTE = ['#2980b9', '#e74c3c', '#27ae60', '#8e44ad', '#f39c12']

# Splice site proximity thresholds (must match annotate_splice_sites.py)
DONOR_PROXIMAL_NT    = 8
ACCEPTOR_PROXIMAL_NT = 3
PPT_MAX_NT           = 50
PPT_MIN_NT           = 4


def _load(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', comment='#', header=None,
                     low_memory=False)
    # Assign column names based on expected layout
    base_cols = [
        'chr', 'start', 'end', 'gene', 'edit_freq', 'strand',
        'control_cov', 'control_A', 'control_G',
        'treatment_cov', 'treatment_A', 'treatment_G',
        'fold_change', 'p_value', 'n_replicates',
        'gene_id', 'gene_type',
        'feature_type', 'dist_5ss', 'dist_3ss', 'splice_region',
    ]
    n = df.shape[1]
    df.columns = (base_cols + [f'extra_{i}' for i in range(n - len(base_cols))])[:n]

    # Coerce distances to numeric (NA → NaN)
    df['dist_5ss'] = pd.to_numeric(df['dist_5ss'], errors='coerce')
    df['dist_3ss'] = pd.to_numeric(df['dist_3ss'], errors='coerce')
    df['edit_freq'] = pd.to_numeric(df['edit_freq'], errors='coerce')
    df['label'] = label
    logger.info(f"  {label}: {len(df):,} sites loaded")
    return df


def _panel_feature_breakdown(ax, dfs: List[pd.DataFrame], labels: List[str]):
    """Grouped bar chart: feature_type fractions per condition."""
    categories = ['UTR', 'CDS_exon', 'intron', 'intergenic']
    x = np.arange(len(labels))
    width = 0.18
    offsets = np.linspace(-(len(categories)-1)*width/2,
                           (len(categories)-1)*width/2,
                           len(categories))

    for i, cat in enumerate(categories):
        pcts = []
        for df in dfs:
            n = len(df)
            pcts.append(100 * (df['feature_type'] == cat).sum() / n if n else 0)
        bars = ax.bar(x + offsets[i], pcts, width,
                      color=_REGION_COLORS.get(cat, '#aaa'),
                      label=cat, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('% of editing sites')
    ax.set_title('Genomic Feature Distribution')
    ax.legend(fontsize=9, frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 100)


def _panel_intronic_breakdown(ax, dfs: List[pd.DataFrame], labels: List[str]):
    """Stacked bar chart: splice region sub-categories for intronic sites."""
    categories = ['5ss_proximal', '3ss_proximal', 'ppt_region', 'deep_intronic']
    x = np.arange(len(labels))
    bottoms = np.zeros(len(labels))

    for cat in categories:
        pcts = []
        for df in dfs:
            intronic = df[df['feature_type'] == 'intron']
            n = len(intronic)
            pcts.append(100 * (intronic['splice_region'] == cat).sum() / n if n else 0)
        ax.bar(x, pcts, bottom=bottoms,
               color=_REGION_COLORS.get(cat, '#aaa'),
               label=cat, edgecolor='white', linewidth=0.5)
        bottoms += np.array(pcts)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('% of intronic sites')
    ax.set_title('Intronic Sub-Region (of intronic sites)')
    ax.legend(fontsize=9, frameon=True, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 100)


def _panel_distance_hist(ax, dfs: List[pd.DataFrame], labels: List[str],
                          col: str, title: str, max_dist: int = 300,
                          zones: Optional[list] = None):
    """
    Overlaid KDE/step histograms of dist_5ss or dist_3ss for intronic sites.
    `zones` is a list of (xmin, xmax, color, label) rectangles to shade.
    """
    bins = np.arange(0, max_dist + 10, 10)

    for df, label, color in zip(dfs, labels, _COND_PALETTE):
        sub = df[(df['feature_type'] == 'intron') & df[col].notna()
                 & (df[col] <= max_dist)]
        if sub.empty:
            continue
        ax.hist(sub[col], bins=bins, density=True,
                histtype='step', linewidth=1.8,
                color=color, label=f'{label} (n={len(sub):,})',
                alpha=0.85)
        # Add KDE overlay
        from scipy.stats import gaussian_kde
        vals = sub[col].values
        if len(vals) > 5:
            kde = gaussian_kde(vals, bw_method=0.15)
            xs = np.linspace(0, max_dist, 400)
            ax.plot(xs, kde(xs), color=color, linewidth=1.2, alpha=0.6,
                    linestyle='--')

    # Shade splice-element zones
    if zones:
        for xmin, xmax, zcolor, zlabel in zones:
            ax.axvspan(xmin, xmax, alpha=0.12, color=zcolor, zorder=0)
            ax.axvline(xmax, color=zcolor, linestyle=':', linewidth=1.0, alpha=0.8)

    ax.set_xlabel(f'Distance (nt)')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend(fontsize=9, frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max_dist)


def _panel_distance_full_log(ax, dfs: List[pd.DataFrame], labels: List[str], col: str, title: str):
    """Log-scale histogram of full distance range — captures deep intronic sites."""
    bins = np.logspace(0, 6, 60)
    for df, label, color in zip(dfs, labels, _COND_PALETTE):
        sub = df[(df['feature_type'] == 'intron') & df[col].notna() & (df[col] > 0)]
        if sub.empty:
            continue
        ax.hist(sub[col], bins=bins, histtype='step', linewidth=1.8,
                color=color, label=f'{label} (n={len(sub):,})',
                density=True, alpha=0.85)
    ax.set_xscale('log')
    ax.set_xlabel('Distance (nt, log scale)')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend(fontsize=9, frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _panel_scatter(ax, df: pd.DataFrame, label: str):
    """Scatter of dist_5ss vs dist_3ss, coloured by splice_region (first sample)."""
    intronic = df[(df['feature_type'] == 'intron')
                  & df['dist_5ss'].notna() & df['dist_3ss'].notna()]
    cap = 500
    intronic = intronic.copy()
    intronic['d5_capped'] = intronic['dist_5ss'].clip(upper=cap)
    intronic['d3_capped'] = intronic['dist_3ss'].clip(upper=cap)

    region_order = ['5ss_proximal', 'ppt_region', '3ss_proximal', 'deep_intronic']
    for reg in region_order:
        sub = intronic[intronic['splice_region'] == reg]
        ax.scatter(sub['d5_capped'], sub['d3_capped'],
                   s=12, alpha=0.55, label=reg,
                   color=_REGION_COLORS.get(reg, '#aaa'),
                   linewidths=0)

    # Add zone lines
    ax.axvline(DONOR_PROXIMAL_NT, color='#e74c3c', linestyle='--',
               linewidth=1.0, alpha=0.7)
    ax.axhline(ACCEPTOR_PROXIMAL_NT, color='#e67e22', linestyle='--',
               linewidth=1.0, alpha=0.7)
    ax.axhline(PPT_MAX_NT, color='#f1c40f', linestyle='--',
               linewidth=1.0, alpha=0.7)
    ax.text(DONOR_PROXIMAL_NT + 2, cap * 0.92, f'5\'SS ≤{DONOR_PROXIMAL_NT}nt',
            fontsize=8, color='#e74c3c')
    ax.text(5, PPT_MAX_NT + 8, f'PPT ≤{PPT_MAX_NT}nt', fontsize=8, color='#c0a000')

    ax.set_xlabel(f"dist_5'SS (nt, capped at {cap})")
    ax.set_ylabel(f"dist_3'SS (nt, capped at {cap})")
    ax.set_title(f"dist_5'SS vs dist_3'SS — {label}")
    ax.legend(fontsize=8, frameon=True, markerscale=1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot(input_files: List[str], labels: List[str], output: str):
    dfs = [_load(f, l) for f, l in zip(input_files, labels)]

    fig = plt.figure(figsize=(18, 22))
    gs = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.38)

    # Row 0: feature breakdown (col 0-1) + intronic sub-region (col 2)
    ax_feat  = fig.add_subplot(gs[0, :2])
    ax_intr  = fig.add_subplot(gs[0, 2])

    # Row 1: dist_5ss zoom (col 0-1) + dist_3ss zoom (col 2)
    ax_d5z   = fig.add_subplot(gs[1, :2])
    ax_d3z   = fig.add_subplot(gs[1, 2])

    # Row 2: log-scale dist_5ss (col 0-1) + log-scale dist_3ss (col 2)
    ax_d5l   = fig.add_subplot(gs[2, :2])
    ax_d3l   = fig.add_subplot(gs[2, 2])

    # Row 3: scatter (first condition only)
    ax_scat  = fig.add_subplot(gs[3, :])

    _panel_feature_breakdown(ax_feat, dfs, labels)
    _panel_intronic_breakdown(ax_intr, dfs, labels)

    _panel_distance_hist(
        ax_d5z, dfs, labels, 'dist_5ss',
        "Distance from 5' Splice Site (donor) — zoomed",
        max_dist=300,
        zones=[
            (0, DONOR_PROXIMAL_NT, '#e74c3c', f"5'SS ≤{DONOR_PROXIMAL_NT}nt"),
        ]
    )

    _panel_distance_hist(
        ax_d3z, dfs, labels, 'dist_3ss',
        "Distance from 3' Splice Site (acceptor) — zoomed",
        max_dist=300,
        zones=[
            (0, ACCEPTOR_PROXIMAL_NT, '#e67e22', f"3'SS ≤{ACCEPTOR_PROXIMAL_NT}nt"),
            (PPT_MIN_NT, PPT_MAX_NT, '#f1c40f', f"PPT {PPT_MIN_NT}–{PPT_MAX_NT}nt"),
        ]
    )

    _panel_distance_full_log(ax_d5l, dfs, labels, 'dist_5ss',
                              "Distance from 5'SS — full range (log)")
    _panel_distance_full_log(ax_d3l, dfs, labels, 'dist_3ss',
                              "Distance from 3'SS — full range (log)")

    _panel_scatter(ax_scat, dfs[0], labels[0])

    fig.suptitle("HyperTRIBE — Intronic Editing Site Proximity to Splice Elements",
                 fontsize=15, fontweight='bold', y=0.998)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output}")

    # Print summary table
    for df, label in zip(dfs, labels):
        intronic = df[df['feature_type'] == 'intron']
        n = len(intronic)
        logger.info(f"\n{label} — intronic sites: {n:,}")
        if n:
            for reg in ['5ss_proximal', '3ss_proximal', 'ppt_region', 'deep_intronic']:
                cnt = (intronic['splice_region'] == reg).sum()
                logger.info(f"  {reg:20s}: {cnt:4d}  ({100*cnt/n:.1f}%)")
            logger.info(f"  dist_5ss median : {intronic['dist_5ss'].median():.0f} nt")
            logger.info(f"  dist_3ss median : {intronic['dist_3ss'].median():.0f} nt")


def main():
    parser = argparse.ArgumentParser(
        description='Plot splice site distance distributions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input', nargs='+', required=True,
                        help='Splice-annotated BED file(s)')
    parser.add_argument('--labels', nargs='+',
                        help='Label(s) for each input (default: filename stems)')
    parser.add_argument('--output', required=True,
                        help='Output PDF path')
    args = parser.parse_args()

    labels = args.labels or [Path(f).stem.replace('_splice_annotated_editing_sites', '')
                              for f in args.input]
    if len(labels) != len(args.input):
        parser.error('--labels must have same count as --input')

    plot(args.input, labels, args.output)


if __name__ == '__main__':
    main()
