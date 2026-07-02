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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
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
# Intronic distance panels
# ─────────────────────────────────────────────────────────────────────────────

def _panel_ecdf(ax, dfs, labels):
    """ECDF of minimum distance to nearest splice site (intronic sites only)."""
    from matplotlib.lines import Line2D

    line_handles = []
    for df, label, color in zip(dfs, labels, _COND_PALETTE):
        intr = df[df['feature_type'] == 'intron']
        d5 = intr['signed_5ss'].dropna()          # positive for intronic
        d3 = (-intr['signed_3ss']).dropna()       # |signed_3ss|, positive
        idx = d5.index.intersection(d3.index)
        min_dist = pd.concat([d5.loc[idx], d3.loc[idx]], axis=1).min(axis=1)
        min_dist = min_dist[min_dist > 0].sort_values()
        ecdf = np.arange(1, len(min_dist) + 1) / len(min_dist)
        h, = ax.semilogx(min_dist.values, ecdf * 100, color=color,
                         linewidth=2.0, alpha=0.9,
                         label=f'{label} (n={len(min_dist):,})')
        line_handles.append(h)

    ref_lines = [(8, '5\'SS proximal\n(8 nt)', '#e74c3c', 20),
                 (50, 'PPT region\n(50 nt)', '#f39c12', 35),
                 (300, '300 nt', '#888', 50)]
    for d, lbl, lc, ypos in ref_lines:
        ax.axvline(d, color=lc, linewidth=1.0, linestyle='--', alpha=0.6)
        ax.text(d * 1.15, ypos, lbl, fontsize=7, color=lc, va='bottom')

    ax.set_xlabel('Distance to nearest splice site (nt, log scale)')
    ax.set_ylabel('Cumulative % of intronic sites')
    ax.set_title('ECDF: Distance to Nearest Splice Element\n(intronic sites only)')
    ax.legend(handles=line_handles, fontsize=9, frameon=True,
              title='Condition', title_fontsize=9)
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _panel_violin_3ss(ax, dfs, labels):
    """Violin of log10(|signed_3ss|) for intronic sites, one violin per condition."""
    from matplotlib.lines import Line2D

    log_data = []
    for df in dfs:
        intr = df[df['feature_type'] == 'intron']
        d3 = (-intr['signed_3ss']).dropna()
        d3 = d3[d3 > 0]
        log_data.append(np.log10(d3.values))

    parts = ax.violinplot(log_data, positions=range(len(labels)),
                          showmedians=True, showextrema=True, widths=0.65)
    for body, color in zip(parts['bodies'], _COND_PALETTE):
        body.set_facecolor(color)
        body.set_edgecolor('white')
        body.set_alpha(0.75)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2.5)
    for key in ('cmins', 'cmaxes', 'cbars'):
        parts[key].set_color('#999')
        parts[key].set_linewidth(1.0)

    # PPT landmark reference lines (log10 scale)
    landmarks = [(4,  'PPT end (4 nt)',    '#e74c3c'),
                 (50, 'PPT start (50 nt)', '#f39c12')]
    landmark_handles = []
    for d, lbl, lc in landmarks:
        ax.axhline(np.log10(d), color=lc, linewidth=1.2, linestyle='--', alpha=0.75)
        landmark_handles.append(Line2D([0], [0], color=lc, linewidth=1.2,
                                       linestyle='--', label=lbl))

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    nice = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000]
    ymax_data = max(d.max() for d in log_data)
    nice = [t for t in nice if np.log10(t) <= ymax_data + 0.1]
    ax.set_yticks([np.log10(t) for t in nice])
    ax.set_yticklabels([str(t) for t in nice])
    ax.set_ylabel("Distance to 3'SS acceptor (nt, log scale)")
    ax.set_title("Distance to 3'SS Acceptor\n(intronic sites only, log scale)")
    ax.legend(handles=landmark_handles, fontsize=8, frameon=True,
              title='Landmarks', title_fontsize=8, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _panel_relative_intronic_pos(ax, dfs, labels):
    """KDE of normalized intronic position: dist_5SS / (dist_5SS + dist_3SS)."""
    from scipy.stats import gaussian_kde
    from matplotlib.lines import Line2D

    line_handles = []
    for df, label, color in zip(dfs, labels, _COND_PALETTE):
        intr = df[df['feature_type'] == 'intron']
        d5 = intr['signed_5ss'].dropna()
        d3 = (-intr['signed_3ss']).dropna()
        idx = d5.index.intersection(d3.index)
        d5, d3 = d5.loc[idx], d3.loc[idx]
        mask = (d5 > 0) & (d3 > 0)
        d5, d3 = d5[mask], d3[mask]
        rel = d5 / (d5 + d3)

        xs = np.linspace(0, 1, 500)
        kde = gaussian_kde(rel.values, bw_method=0.08)
        ys = kde(xs)
        h, = ax.plot(xs, ys, color=color, linewidth=2.2, alpha=0.9,
                     label=f'{label} (n={len(rel):,})')
        line_handles.append(h)

    mid_handle = Line2D([0], [0], color='black', linewidth=1.0,
                        linestyle=':', label='midpoint (0.5)')
    ax.axvline(0.5, color='black', linewidth=1.0, linestyle=':', alpha=0.5)
    ax.set_xlabel("Relative intronic position  (0 = 5'SS donor  →  1 = 3'SS acceptor)")
    ax.set_ylabel('Density')
    ax.set_title('Normalized Position within Intron\n'
                 '(= dist_to_5SS / intron_length; removes intron-length bias)')
    ax.set_xlim(0, 1)
    leg1 = ax.legend(handles=line_handles, fontsize=9, frameon=True,
                     title='Condition', title_fontsize=9, loc='upper right')
    ax.add_artist(leg1)
    ax.legend(handles=[mid_handle], fontsize=8, frameon=False, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _panel_ppt_ecdf_zoom(ax, dfs, labels):
    """Zoomed ECDF of distance to 3'SS for intronic sites (0–200 nt, linear).

    Replaces the noisy step-histogram PPT panel with a smooth cumulative curve
    that directly answers: what fraction of intronic sites fall within the PPT
    or 3'SS-proximal zone?
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    xmax = 200

    line_handles = []
    for df, label, color in zip(dfs, labels, _COND_PALETTE):
        intr = df[df['feature_type'] == 'intron']
        d3 = (-intr['signed_3ss']).dropna()
        d3 = d3[d3 > 0].sort_values()
        total = len(d3)
        ecdf = np.arange(1, total + 1) / total
        mask = d3.values <= xmax
        h, = ax.plot(d3.values[mask], ecdf[mask] * 100, color=color,
                     linewidth=2.0, alpha=0.9,
                     label=f'{label} (n={total:,} intronic)')
        line_handles.append(h)

    # Shaded zones
    landmark_handles = []
    ax.axvspan(0,  4,  alpha=0.20, color='#e74c3c', zorder=0)
    ax.axvspan(4,  50, alpha=0.15, color='#f1c40f', zorder=0)
    landmark_handles.append(Patch(facecolor='#e74c3c', alpha=0.35,
                                  edgecolor='none', label="3'SS proximal (≤4 nt)"))
    landmark_handles.append(Patch(facecolor='#f1c40f', alpha=0.30,
                                  edgecolor='none', label='PPT (4–50 nt)'))

    for xv, lbl, lc in [(4,  '−4 nt',          '#e74c3c'),
                         (27, 'PPT centre (−27)', '#c0a000'),
                         (50, '−50 nt',           '#f39c12')]:
        ax.axvline(xv, color=lc, linewidth=1.2, linestyle='--', alpha=0.85)
        landmark_handles.append(Line2D([0], [0], color=lc, linewidth=1.2,
                                       linestyle='--', label=lbl))

    leg1 = ax.legend(handles=line_handles, fontsize=9, frameon=True,
                     title='Condition', title_fontsize=9, loc='lower right')
    ax.add_artist(leg1)
    ax.legend(handles=landmark_handles, fontsize=8, frameon=True,
              title='Landmarks', title_fontsize=8, loc='upper left', ncol=2)

    ax.set_xlabel("Distance to 3'SS acceptor (nt)")
    ax.set_ylabel("Cumulative % of all intronic sites")
    ax.set_title("PPT / 3'SS Proximity — Cumulative Fraction (intronic sites)\n"
                 "Zoom to 0–200 nt from 3'SS; y-axis is % of all intronic sites in each condition")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Metagene density plot helper
# ─────────────────────────────────────────────────────────────────────────────

def _metagene(ax, dfs: List[pd.DataFrame], labels: List[str],
              col: str, title: str,
              xmin: int, xmax: int,
              shade_zones: Optional[list] = None,
              vlines: Optional[list] = None,
              intron_only: bool = False):
    """
    Signed-distance metagene plot.
    col         : column name carrying the signed position values
    xmin/xmax   : x-axis range (nt)
    shade_zones : list of (x0, x1, color, alpha, label)
    vlines      : list of (x, color, label) dashed vertical lines
    intron_only : if True, filter to feature_type == 'intron' before plotting
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    bin_width = 2 if (xmax - xmin) <= 200 else 5
    bins = np.arange(xmin, xmax + 1, bin_width)

    line_handles = []
    for df, label, color in zip(dfs, labels, _COND_PALETTE):
        src = df[df['feature_type'] == 'intron'] if intron_only else df
        vals = src[col].dropna()
        vals = vals[(vals >= xmin) & (vals <= xmax)]
        if len(vals) < 5:
            continue
        counts, _ = np.histogram(vals, bins=bins)
        h, = ax.step(bins[:-1], counts / counts.sum() * 100,
                     where='mid', color=color, linewidth=2.0,
                     label=f'{label} (n={len(vals):,})', alpha=0.9)
        line_handles.append(h)

    # Shaded functional zones
    landmark_handles = []
    if shade_zones:
        for x0, x1, zc, za, zlabel in shade_zones:
            ax.axvspan(x0, x1, alpha=za, color=zc, zorder=0)
            landmark_handles.append(Patch(facecolor=zc, alpha=min(za + 0.15, 1.0),
                                          edgecolor='none', label=zlabel))

    # Functional landmark lines
    ax.axvline(0, color='black', linewidth=2.0, linestyle='-', zorder=5)
    landmark_handles.insert(0, Line2D([0], [0], color='black', linewidth=2.0,
                                      label='Splice site (0)'))
    if vlines:
        for xv, vc, vl in vlines:
            ax.axvline(xv, color=vc, linewidth=1.2, linestyle='--', alpha=0.85)
            landmark_handles.append(Line2D([0], [0], color=vc, linewidth=1.2,
                                           linestyle='--', label=vl))

    ylabel = (f'% of intronic sites per {bin_width}-nt bin' if intron_only
              else f'% of sites per {bin_width}-nt bin')
    ax.set_xlabel('Position relative to splice element (nt)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Two separate legends: conditions (top-right) + landmarks (lower-right)
    leg1 = ax.legend(handles=line_handles, fontsize=9, frameon=True,
                     loc='upper right', title='Condition', title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=landmark_handles, fontsize=8, frameon=True,
              loc='lower right', title='Landmarks', title_fontsize=8,
              ncol=2 if len(landmark_handles) > 4 else 1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(xmin, xmax)


# ─────────────────────────────────────────────────────────────────────────────
# Main plot
# ─────────────────────────────────────────────────────────────────────────────

def plot(input_files: List[str], labels: List[str], output: str,
         window_5ss: int = 300, window_3ss: int = 300, window_ppt: int = 60):

    dfs = [_load(f, l) for f, l in zip(input_files, labels)]

    fig = plt.figure(figsize=(18, 38))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(6, 2, figure=fig, hspace=0.55, wspace=0.35,
                  height_ratios=[1, 1, 1, 1.2, 1.2, 1.2])

    ax_feat   = fig.add_subplot(gs[0, 0])
    ax_intr   = fig.add_subplot(gs[0, 1])
    ax_ecdf   = fig.add_subplot(gs[1, 0])
    ax_violin = fig.add_subplot(gs[1, 1])
    ax_relpos = fig.add_subplot(gs[2, :])
    ax_d5     = fig.add_subplot(gs[3, :])
    ax_d3     = fig.add_subplot(gs[4, :])
    ax_ppt    = fig.add_subplot(gs[5, :])

    _panel_feature_breakdown(ax_feat, dfs, labels)
    _panel_intronic_breakdown(ax_intr, dfs, labels)
    _panel_ecdf(ax_ecdf, dfs, labels)
    _panel_violin_3ss(ax_violin, dfs, labels)
    _panel_relative_intronic_pos(ax_relpos, dfs, labels)

    # ── 5'SS metagene ──────────────────────────────────────────────────
    _metagene(
        ax_d5, dfs, labels, 'signed_5ss',
        "Distance to 5' Splice Site Donor\n"
        "negative = exonic  |  0 = donor  |  positive = intronic",
        xmin=-window_5ss, xmax=window_5ss,
        shade_zones=[
            (-window_5ss, 0,     '#3498db', 0.06, 'exon'),
            (0,  8,              '#e74c3c', 0.15, '5\'SS proximal (≤8 nt)'),
            (8,  window_5ss,     '#e74c3c', 0.04, 'intron'),
        ],
        vlines=[
            (-3, '#666', 'last exon codon (−3)'),
            (8,  '#e74c3c', 'donor consensus end (+8)'),
        ]
    )

    # ── 3'SS metagene ──────────────────────────────────────────────────
    _metagene(
        ax_d3, dfs, labels, 'signed_3ss',
        "Distance to 3' Splice Site Acceptor\n"
        "negative = intronic (PPT region)  |  0 = acceptor AG  |  positive = exonic",
        xmin=-window_3ss, xmax=window_3ss,
        shade_zones=[
            (0,   window_3ss,    '#3498db', 0.06, 'exon'),
            (-window_3ss, -50,   '#e67e22', 0.04, 'intron (deep)'),
            (-50, -4,            '#f1c40f', 0.22, 'PPT (−50 to −4)'),
            (-4,  0,             '#e74c3c', 0.22, '3\'SS proximal (−4 to 0)'),
        ],
        vlines=[
            (-50, '#f39c12', 'PPT start (−50)'),
            (-27, '#c0a000', 'PPT centre (−27)'),
            (-4,  '#e74c3c', 'PPT/AG boundary (−4)'),
        ]
    )

    # ── PPT / 3'SS proximity — zoomed cumulative fraction ─────────────
    _panel_ppt_ecdf_zoom(ax_ppt, dfs, labels)

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
