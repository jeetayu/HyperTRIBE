#!/usr/bin/env python3
"""
Plot Chromosome Distribution of Editing Sites
==============================================

Bar chart of editing site counts per chromosome, ordered by chromosome number
(chr1 → chrX → chrY → chrM). Includes a secondary axis showing sites per Mbp
to account for chromosome length differences.

Usage:
    python plot_chromosome_distribution.py \\
        --input  results/annotated_editing_sites.bed \\
        --output results/plots/chromosome_distribution.pdf
"""

import argparse
import logging
import re
import sys
from pathlib import Path

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
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
})

# hg38 chromosome lengths (bp) for density normalisation
HG38_CHR_LENGTHS = {
    'chr1': 248956422, 'chr2': 242193529, 'chr3': 198295559,
    'chr4': 190214555, 'chr5': 181538259, 'chr6': 170805979,
    'chr7': 159345973, 'chr8': 145138636, 'chr9': 138394717,
    'chr10': 133797422, 'chr11': 135086622, 'chr12': 133275309,
    'chr13': 114364328, 'chr14': 107043718, 'chr15': 101991189,
    'chr16': 90338345, 'chr17': 83257441, 'chr18': 80373285,
    'chr19': 58617616, 'chr20': 64444167, 'chr21': 46709983,
    'chr22': 50818468, 'chrX': 156040895, 'chrY': 57227415,
    'chrM': 16569,
}

_BASE_COLS = [
    'chr', 'start', 'end', 'gene', 'edit_freq', 'strand',
    'control_cov', 'control_A', 'control_G',
    'treatment_cov', 'treatment_A', 'treatment_G',
    'fold_change', 'p_value', 'n_replicates',
    'gene_id', 'gene_type',
]


def _chr_sort_key(name: str) -> tuple:
    """Sort chromosomes numerically, then X, Y, M."""
    name = name.replace('chr', '')
    if name.isdigit():
        return (0, int(name))
    return {'X': (1, 0), 'Y': (2, 0), 'M': (3, 0)}.get(name, (4, 0))


def _load_bed(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', comment='#', header=None)
    n = df.shape[1]
    df.columns = (_BASE_COLS + [f'extra_{i}' for i in range(n - len(_BASE_COLS))])[:n]
    return df


def plot(input_file: str, output_file: str) -> None:
    logger.info(f"Loading: {input_file}")
    df = _load_bed(input_file)
    logger.info(f"  {len(df):,} editing sites")

    # Keep only standard chromosomes
    standard = [f'chr{i}' for i in list(range(1, 23)) + ['X', 'Y', 'M']]
    df_std = df[df['chr'].isin(standard)].copy()
    n_other = len(df) - len(df_std)
    if n_other:
        logger.info(f"  Excluded {n_other:,} sites on non-standard chromosomes")

    counts = df_std['chr'].value_counts()
    chrs_present = sorted(counts.index.tolist(), key=_chr_sort_key)
    counts = counts.reindex(chrs_present)

    # Compute density (sites per Mbp)
    lengths_mbp = pd.Series({
        c: HG38_CHR_LENGTHS.get(c, np.nan) / 1e6 for c in chrs_present
    })
    density = counts / lengths_mbp

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={'height_ratios': [3, 2], 'hspace': 0.05})

    x = np.arange(len(chrs_present))
    bar_color = '#2980b9'
    density_color = '#27ae60'

    # ── Panel A: Raw counts ───────────────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(x, counts.values, color=bar_color, edgecolor='white',
                  linewidth=0.5, width=0.7)
    ax.set_ylabel('Number of Editing Sites')
    ax.set_title('Editing Site Distribution Across Chromosomes')
    # Annotate top bars
    top_n = min(5, len(chrs_present))
    top_chrs = counts.nlargest(top_n).index
    for bar, chrom in zip(bars, chrs_present):
        if chrom in top_chrs:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + counts.max() * 0.01,
                    f'{int(bar.get_height()):,}',
                    ha='center', va='bottom', fontsize=8, color='#333333')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Panel B: Density (sites per Mbp) ─────────────────────────────────────
    ax2 = axes[1]
    ax2.bar(x, density.values, color=density_color, edgecolor='white',
            linewidth=0.5, width=0.7, alpha=0.85)
    ax2.set_ylabel('Sites per Mbp')
    ax2.set_xticks(x)
    ax2.set_xticklabels(chrs_present, rotation=45, ha='right', fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlabel('Chromosome')

    fig.suptitle('HyperTRIBE — Editing Site Chromosomal Distribution (hg38)',
                 fontsize=14, fontweight='bold', y=1.01)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_file}")

    logger.info("\nChromosome summary (top 10 by site count):")
    for chrom in counts.nlargest(10).index:
        logger.info(f"  {chrom}: {int(counts[chrom]):,} sites  "
                    f"({density[chrom]:.2f} per Mbp)")


def main():
    parser = argparse.ArgumentParser(
        description='Plot chromosome distribution of editing sites',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input', required=True,
                        help='Annotated editing sites BED file')
    parser.add_argument('--output', required=True,
                        help='Output plot file (.pdf or .png)')
    args = parser.parse_args()
    plot(args.input, args.output)


if __name__ == '__main__':
    main()
