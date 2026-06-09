#!/usr/bin/env python3
"""
Plot Gene Biotype Distribution of Editing Sites
================================================

Two-panel figure:
  Panel A — horizontal bar chart of sites by gene biotype (top 15)
  Panel B — pie chart of the same data for proportional view

Requires the annotated BED (output of annotate_genes.py) which includes
gene_id (col 15) and gene_type (col 16).

Usage:
    python plot_biotype_distribution.py \\
        --input  results/annotated_editing_sites.bed \\
        --output results/plots/gene_biotype_distribution.pdf
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
})

_BASE_COLS = [
    'chr', 'start', 'end', 'gene', 'edit_freq', 'strand',
    'control_cov', 'control_A', 'control_G',
    'treatment_cov', 'treatment_A', 'treatment_G',
    'fold_change', 'p_value', 'n_replicates',
    'gene_id', 'gene_type',
]

# Biotype grouping: map verbose names to cleaner display labels
_BIOTYPE_LABELS = {
    'protein_coding':              'Protein-coding',
    'lncRNA':                      'lncRNA',
    'processed_pseudogene':        'Pseudogene',
    'unprocessed_pseudogene':      'Pseudogene',
    'transcribed_unprocessed_pseudogene': 'Pseudogene',
    'transcribed_processed_pseudogene':   'Pseudogene',
    'miRNA':                       'miRNA',
    'snRNA':                       'snRNA',
    'snoRNA':                      'snoRNA',
    'misc_RNA':                    'misc_RNA',
    'rRNA':                        'rRNA',
    'Mt_tRNA':                     'Mt_tRNA',
    'Mt_rRNA':                     'Mt_rRNA',
    '.':                           'Intergenic',
}

# Colorblind-friendly palette (12 distinct colors)
_PALETTE = [
    '#2980b9', '#27ae60', '#e74c3c', '#f39c12', '#8e44ad',
    '#16a085', '#d35400', '#2c3e50', '#7f8c8d', '#1abc9c',
    '#e91e63', '#607d8b',
]


def _load_bed(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', comment='#', header=None)
    n = df.shape[1]
    df.columns = (_BASE_COLS + [f'extra_{i}' for i in range(n - len(_BASE_COLS))])[:n]
    return df


def plot(input_file: str, output_file: str, top_n: int = 15) -> None:
    logger.info(f"Loading: {input_file}")
    df = _load_bed(input_file)
    logger.info(f"  {len(df):,} editing sites")

    if 'gene_type' not in df.columns:
        logger.error(
            "gene_type column not found — run annotate_genes.py first "
            "to add biotype information"
        )
        sys.exit(1)

    # Map to cleaner labels
    df['biotype_label'] = df['gene_type'].map(
        lambda b: _BIOTYPE_LABELS.get(b, b)
    )

    counts = df['biotype_label'].value_counts()
    logger.info(f"  {len(counts)} distinct gene biotypes")

    # Collapse "Other" if more than top_n categories
    top_cats = counts.head(top_n)
    if len(counts) > top_n:
        other_count = counts.iloc[top_n:].sum()
        top_cats['Other'] = other_count
    top_cats = top_cats.sort_values(ascending=True)

    colors = (_PALETTE * ((len(top_cats) // len(_PALETTE)) + 1))[:len(top_cats)]
    color_map = dict(zip(top_cats.index, colors))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ── Panel A: Horizontal bar chart ────────────────────────────────────────
    ax = axes[0]
    y_pos = np.arange(len(top_cats))
    ax.barh(y_pos, top_cats.values,
            color=[color_map[b] for b in top_cats.index],
            edgecolor='white', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_cats.index, fontsize=10)
    ax.set_xlabel('Number of Editing Sites')
    ax.set_title('Editing Sites by Gene Biotype')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate bars with percentage
    total = top_cats.sum()
    for i, (biotype, count) in enumerate(top_cats.items()):
        pct = count / total * 100
        ax.text(count + total * 0.005, i, f'{pct:.1f}%',
                va='center', fontsize=9, color='#444444')

    ax.set_xlim(right=top_cats.max() * 1.18)

    # ── Panel B: Pie chart ───────────────────────────────────────────────────
    ax2 = axes[1]
    sorted_cats = top_cats.sort_values(ascending=False)
    sorted_colors = [color_map[b] for b in sorted_cats.index]

    # Explode the largest slice slightly
    explode = [0.04 if i == 0 else 0 for i in range(len(sorted_cats))]

    wedges, texts, autotexts = ax2.pie(
        sorted_cats.values,
        labels=None,
        autopct=lambda p: f'{p:.1f}%' if p >= 2.0 else '',
        colors=sorted_colors,
        explode=explode,
        startangle=90,
        pctdistance=0.78,
        wedgeprops={'edgecolor': 'white', 'linewidth': 0.8},
    )
    for at in autotexts:
        at.set_fontsize(8)

    ax2.set_title('Proportional Biotype Breakdown')

    # Legend
    legend_handles = [
        mpatches.Patch(color=color_map[b], label=f'{b}  ({int(c):,})')
        for b, c in sorted_cats.items()
    ]
    ax2.legend(
        handles=legend_handles,
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        fontsize=9,
        frameon=True,
        title='Gene Biotype',
    )

    fig.suptitle('HyperTRIBE — Editing Sites by Gene Biotype',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_file}")

    logger.info("\nBiotype breakdown:")
    for biotype, count in sorted_cats.sort_values(ascending=False).items():
        logger.info(f"  {biotype}: {int(count):,}  ({count / total * 100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Plot gene biotype distribution of HyperTRIBE editing sites',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input', required=True,
                        help='Annotated editing sites BED file')
    parser.add_argument('--output', required=True,
                        help='Output plot file (.pdf or .png)')
    parser.add_argument('--top-n', type=int, default=15,
                        help='Number of biotype categories to show (default: 15)')
    args = parser.parse_args()
    plot(args.input, args.output, top_n=args.top_n)


if __name__ == '__main__':
    main()
