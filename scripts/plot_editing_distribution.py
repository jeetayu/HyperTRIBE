#!/usr/bin/env python3
"""
Plot A-to-G Editing Frequency Distribution
===========================================

Produces a two-panel figure:
  Panel A — histogram of editing frequencies with median/mean lines
  Panel B — cumulative distribution function (CDF)

Saves to the extension specified by --output (.pdf for publication, .png for reports).

Usage:
    python plot_editing_distribution.py \\
        --input  results/annotated_editing_sites.bed \\
        --output results/plots/editing_frequency_distribution.pdf
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
import matplotlib.ticker as mticker
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
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Column layout from call_editing_sites_parallel.py + annotate_genes.py
_BASE_COLS = [
    'chr', 'start', 'end', 'gene', 'edit_freq', 'strand',
    'control_cov', 'control_A', 'control_G',
    'treatment_cov', 'treatment_A', 'treatment_G',
    'fold_change', 'p_value', 'n_replicates',
    'gene_id', 'gene_type',
]


def _load_bed(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', comment='#', header=None)
    n = df.shape[1]
    df.columns = (_BASE_COLS + [f'extra_{i}' for i in range(n - len(_BASE_COLS))])[:n]
    df['edit_freq'] = pd.to_numeric(df['edit_freq'], errors='coerce')
    return df


def plot(input_file: str, output_file: str) -> None:
    logger.info(f"Loading: {input_file}")
    df = _load_bed(input_file)
    freqs = df['edit_freq'].dropna()
    logger.info(f"  {len(freqs):,} sites with valid editing frequencies")

    if len(freqs) == 0:
        logger.error("No valid edit_freq values — cannot generate plot")
        sys.exit(1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel A: Histogram ────────────────────────────────────────────────────
    ax = axes[0]
    n_bins = min(60, max(20, len(freqs) // 80))
    ax.hist(freqs, bins=n_bins, color='#2980b9', edgecolor='white',
            linewidth=0.4, alpha=0.85)
    median_val = freqs.median()
    mean_val = freqs.mean()
    ax.axvline(median_val, color='#c0392b', linestyle='--', linewidth=1.8,
               label=f'Median  {median_val:.1f}%')
    ax.axvline(mean_val, color='#e67e22', linestyle=':', linewidth=1.8,
               label=f'Mean  {mean_val:.1f}%')
    ax.set_xlabel('Editing Frequency (%)')
    ax.set_ylabel('Number of Sites')
    ax.set_title('A→G Editing Frequency Distribution')
    ax.legend(framealpha=0.9, fontsize=10)
    ax.text(0.97, 0.97, f'n = {len(freqs):,}', transform=ax.transAxes,
            ha='right', va='top', fontsize=10, color='#555555')

    # ── Panel B: CDF ─────────────────────────────────────────────────────────
    ax2 = axes[1]
    sorted_f = np.sort(freqs)
    cdf = np.arange(1, len(sorted_f) + 1) / len(sorted_f)
    ax2.plot(sorted_f, cdf, color='#2980b9', linewidth=2.2)
    ax2.fill_between(sorted_f, cdf, alpha=0.12, color='#2980b9')
    ax2.axvline(median_val, color='#c0392b', linestyle='--', linewidth=1.8,
                label=f'Median  {median_val:.1f}%')
    # Quartile shading
    q25, q75 = float(np.percentile(sorted_f, 25)), float(np.percentile(sorted_f, 75))
    ax2.axvspan(q25, q75, alpha=0.08, color='#2980b9', label=f'IQR  [{q25:.1f}–{q75:.1f}%]')
    ax2.set_xlabel('Editing Frequency (%)')
    ax2.set_ylabel('Cumulative Fraction of Sites')
    ax2.set_title('Cumulative Distribution of Editing Frequencies')
    ax2.set_xlim(left=0)
    ax2.set_ylim(0, 1.0)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax2.legend(framealpha=0.9, fontsize=10)

    fig.suptitle('HyperTRIBE — A-to-G Editing Frequency Analysis',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_file}")

    # Key stats to stdout for report generation
    logger.info(f"\nEditing frequency summary:")
    logger.info(f"  Min:    {freqs.min():.2f}%")
    logger.info(f"  Q1:     {q25:.2f}%")
    logger.info(f"  Median: {median_val:.2f}%")
    logger.info(f"  Mean:   {mean_val:.2f}%")
    logger.info(f"  Q3:     {q75:.2f}%")
    logger.info(f"  Max:    {freqs.max():.2f}%")
    logger.info(f"  Std:    {freqs.std():.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Plot A-to-G editing frequency distribution',
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
