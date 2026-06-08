#!/usr/bin/env python3
"""
Splice Site Distance Plots for HyperTRIBE Editing Sites
========================================================

Generates metagene-style density plots showing where editing sites fall
relative to each splice element.  Each plot is centered at position 0
(the splice element itself) with:

    negative = upstream  (5' side, towards mRNA 5' cap)
    positive = downstream (3' side, towards mRNA 3' poly-A)

Panels:
  1 – Genomic feature breakdown (UTR / CDS_exon / intron / intergenic)
  2 – Intronic sub-region breakdown (5ss_proximal … deep_intronic)
  3 – Metagene centred on 5'SS donor
       Exonic sites cluster at negative positions (last exon bases),
       intronic sites at positive (first intron bases).
  4 – Metagene centred on 3'SS acceptor
       Intronic/PPT sites cluster at negative positions,
       exonic sites at positive.
  5 – Metagene centred on PPT centre (−27 nt from 3'SS)
       Shows whether editing enriches at the branch-point / PPT window.

Multiple input files are overlaid on panels 3–5 to compare conditions.

Usage:
    python plot_splice_distances.py \\
        --input wt/splice_annotated_editing_sites.bed \\
                s161/splice_annotated_editing_sites.bed \\
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
import seaborn as sns
from scipy.stats import gaussian_kde

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
})

_REGION_COLORS = {
    'UTR': '#3498db', 'CDS_exon': '#2ecc71',
    'intron': '#e74c3c', 'intergenic': '#95a5a6',
    '5ss_proximal': '#e74c3c', '3ss_proximal': '#e67e22',
    'ppt_region': '#f1c40f', 'deep_intronic': '#9b59b6',
}
_COND_PALETTE = ['#2980b9', '#e74c3c', '#27ae60', '#8e44ad', '#f39c12']

PPT_CENTER_FROM_3SS = 27    # midpoint of PPT window (4–50 nt from 3'SS)


def _load(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', comment='#', header=None, low_memory=False)
    base_cols = [
        'chr', 'start', 'end', 'gene', 'edit_freq', 'strand',
        'control_cov', 'control_A', 'control_G',
        'treatment_cov', 'treatment_A', 'treatment_G',
        'fold_change', 'p_value', 'n_replicates',
        'gene_id', 'gene_type',
        'feature_type', 'dist_5ss', 'dist_3ss', 'splice_region',
        'signed_5ss', 'signed_3ss',
    ]
    n = df.shape[1]
    df.columns = (base_cols + [f'extra_{i}' for i in range(n - len(base_cols))])[:n]
    for col in ('dist_5ss', 'dist_3ss', 'signed_5ss', 'signed_3ss', 'edit_freq'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['label'] = label
    logger.info(f"  {label}: {len(df):,} sites "
                f"({df['feature_type'].value_counts().to_dict()})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Breakdown bar charts
# ─────────────────────────────────────────────────────────────────────────────

def _panel_feature_breakdown(ax, dfs, labels):
    cats = ['UTR', 'CDS_exon', 'intron', 'intergenic']
    x = np.arange(len(labels))
    w = 0.18
    offsets = np.linspace(-(len(cats)-1)*w/2, (len(cats)-1)*w/2, len(cats))
    for i, cat in enumerate(cats):
        pcts = [100*(df['feature_type']==cat).sum()/len(df) for df in dfs]
        ax.bar(x+offsets[i], pcts, w, color=_REGION_COLORS[cat],
               label=cat, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('% of all editing sites'); ax.set_ylim(0, 100)
    ax.set_title('Genomic Feature Distribution')
    ax.legend(fontsize=9, frameon=True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)


def _panel_intronic_breakdown(ax, dfs, labels):
    cats = ['5ss_proximal', '3ss_proximal', 'ppt_region', 'deep_intronic']
    x = np.arange(len(labels)); bottoms = np.zeros(len(labels))
    for cat in cats:
        pcts = []
        for df in dfs:
            intr = df[df['feature_type']=='intron']
            pcts.append(100*(intr['splice_region']==cat).sum()/max(len(intr),1))
        ax.bar(x, pcts, bottom=bottoms, color=_REGION_COLORS[cat],
               label=cat, edgecolor='white', linewidth=0.5)
        bottoms += np.array(pcts)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('% of intronic sites'); ax.set_ylim(0, 100)
    ax.set_title('Intronic Sub-Region Breakdown')
    ax.legend(fontsize=9, frameon=True, loc='upper right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Metagene density plot helper
# ─────────────────────────────────────────────────────────────────────────────

def _metagene(ax, dfs: List[pd.DataFrame], labels: List[str],
              col: str, title: str,
              xmin: int, xmax: int,
              shade_zones: Optional[list] = None,
              vlines: Optional[list] = None):
    """
    Signed-distance metagene plot.
    col       : column name carrying the signed position values
    xmin/xmax : x-axis range (nt)
    shade_zones: list of (x0, x1, color, alpha, label)
    vlines    : list of (x, color, label) dashed vertical lines
    """
    bins = np.arange(xmin, xmax + 1, 5)

    all_counts = np.zeros(len(bins)-1)
    for df, label, color in zip(dfs, labels, _COND_PALETTE):
        vals = df[col].dropna()
        vals = vals[(vals >= xmin) & (vals <= xmax)]
        if len(vals) < 5:
            continue
        counts, _ = np.histogram(vals, bins=bins)
        all_counts += counts

        # Normalised step histogram
        ax.step(bins[:-1], counts / counts.sum() * 100,
                where='mid', color=color, linewidth=1.6,
                label=f'{label} (n={len(vals):,})', alpha=0.85)

        # KDE overlay
        kde = gaussian_kde(vals.values, bw_method=0.10)
        xs  = np.linspace(xmin, xmax, 600)
        kde_vals = kde(xs)
        ax.plot(xs, kde_vals / kde_vals.sum() * len(bins) * 100,
                color=color, linewidth=1.0, linestyle='--', alpha=0.5)

    # Shaded functional zones
    if shade_zones:
        for x0, x1, zc, za, zlabel in shade_zones:
            ax.axvspan(x0, x1, alpha=za, color=zc, zorder=0, label=zlabel)

    # Functional landmark lines
    ax.axvline(0, color='black', linewidth=1.8, linestyle='-', zorder=5, label='Splice site (0)')
    if vlines:
        for xv, vc, vl in vlines:
            ax.axvline(xv, color=vc, linewidth=1.0, linestyle=':', alpha=0.8, label=vl)

    ax.set_xlabel('Position relative to splice element (nt)')
    ax.set_ylabel('% of sites per 5-nt bin')
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=True, loc='upper right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_xlim(xmin, xmax)

    # Annotate axis sides
    ax.text(xmin * 0.92, ax.get_ylim()[1] * 0.92, '← upstream\n(5\' side)',
            fontsize=8, color='#555', ha='left', va='top')
    ax.text(xmax * 0.92, ax.get_ylim()[1] * 0.92, 'downstream →\n(3\' side)',
            fontsize=8, color='#555', ha='right', va='top')


# ─────────────────────────────────────────────────────────────────────────────
# Main plot
# ─────────────────────────────────────────────────────────────────────────────

def plot(input_files: List[str], labels: List[str], output: str,
         window_5ss: int = 300, window_3ss: int = 300, window_ppt: int = 60):

    dfs = [_load(f, l) for f, l in zip(input_files, labels)]

    # Compute PPT-centre column: signed_3ss + PPT_CENTER_FROM_3SS
    # → 0 = PPT centre, negative = further upstream, positive = toward 3'SS
    for df in dfs:
        df['signed_ppt'] = df['signed_3ss'] + PPT_CENTER_FROM_3SS

    fig = plt.figure(figsize=(18, 24))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(4, 2, figure=fig, hspace=0.52, wspace=0.35)

    ax_feat = fig.add_subplot(gs[0, 0])
    ax_intr = fig.add_subplot(gs[0, 1])
    ax_d5   = fig.add_subplot(gs[1, :])
    ax_d3   = fig.add_subplot(gs[2, :])
    ax_ppt  = fig.add_subplot(gs[3, :])

    _panel_feature_breakdown(ax_feat, dfs, labels)
    _panel_intronic_breakdown(ax_intr, dfs, labels)

    # ── 5'SS metagene ──────────────────────────────────────────────────
    _metagene(
        ax_d5, dfs, labels, 'signed_5ss',
        "Position Relative to 5' Splice Site (donor = 0)\n"
        "negative = exonic (upstream),  positive = intronic (downstream)",
        xmin=-window_5ss, xmax=window_5ss,
        shade_zones=[
            (-window_5ss, 0,     '#3498db', 0.06, 'exon'),
            (0,  8,              '#e74c3c', 0.12, f'5\'SS proximal (≤8 nt)'),
            (0,  window_5ss,     '#e74c3c', 0.04, 'intron'),
        ],
        vlines=[
            (-3, '#888', 'exon -3'),
            (8, '#e74c3c', '+8 nt (donor consensus end)'),
        ]
    )

    # ── 3'SS metagene ──────────────────────────────────────────────────
    _metagene(
        ax_d3, dfs, labels, 'signed_3ss',
        "Position Relative to 3' Splice Site (acceptor = 0)\n"
        "negative = intronic (upstream / PPT region),  positive = exonic (downstream)",
        xmin=-window_3ss, xmax=window_3ss,
        shade_zones=[
            (0,  window_3ss,     '#3498db', 0.06, 'exon'),
            (-window_3ss, 0,     '#e67e22', 0.04, 'intron'),
            (-50, -4,            '#f1c40f', 0.18, 'PPT (−50 to −4)'),
            (-3,  0,             '#e74c3c', 0.18, '3\'SS proximal (AG, ≤3 nt)'),
        ],
        vlines=[
            (-50, '#f39c12', 'PPT start (−50)'),
            (-27, '#c0a000', 'PPT centre (−27)'),
            (-4,  '#f39c12', 'PPT end (−4)'),
            (-3,  '#e74c3c', '3\'SS proximal (−3)'),
        ]
    )

    # ── PPT-centre metagene ────────────────────────────────────────────
    _metagene(
        ax_ppt, dfs, labels, 'signed_ppt',
        f"Position Relative to PPT Centre (−{PPT_CENTER_FROM_3SS} nt from 3'SS = 0)\n"
        "negative = upstream (toward 5'SS),  positive = downstream (toward 3'SS / AG)",
        xmin=-window_ppt, xmax=window_ppt,
        shade_zones=[
            (-23, +23, '#f1c40f', 0.18, 'PPT window (±23 nt from centre)'),
            (+23, window_ppt, '#e74c3c', 0.12, 'Toward AG'),
        ],
        vlines=[
            (+23, '#e74c3c', '3\'SS proximal zone'),
            (-23, '#f39c12', 'PPT upstream edge'),
        ]
    )

    fig.suptitle("HyperTRIBE — Editing Site Position Relative to Splice Elements",
                 fontsize=15, fontweight='bold', y=1.002)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output}")

    # Summary stats
    for df, label in zip(dfs, labels):
        n = len(df)
        logger.info(f"\n{label}  (n={n:,})")
        for feat in ('UTR', 'CDS_exon', 'intron', 'intergenic'):
            cnt = (df['feature_type']==feat).sum()
            logger.info(f"  {feat:12s}: {cnt:5d}  ({100*cnt/n:.1f}%)")
        intr = df[df['feature_type']=='intron']
        if len(intr):
            logger.info(f"  intronic signed_5ss median: {intr['signed_5ss'].median():.0f} nt")
            logger.info(f"  intronic signed_3ss median: {intr['signed_3ss'].median():.0f} nt")


def main():
    parser = argparse.ArgumentParser(
        description='Splice site metagene distance plots',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--labels', nargs='+')
    parser.add_argument('--output', required=True)
    parser.add_argument('--window-5ss', type=int, default=300)
    parser.add_argument('--window-3ss', type=int, default=300)
    parser.add_argument('--window-ppt', type=int, default=60)
    args = parser.parse_args()

    labels = args.labels or [Path(f).stem for f in args.input]
    if len(labels) != len(args.input):
        parser.error('--labels must have same count as --input')

    plot(args.input, labels, args.output,
         args.window_5ss, args.window_3ss, args.window_ppt)


if __name__ == '__main__':
    main()
