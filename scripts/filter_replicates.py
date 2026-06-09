#!/usr/bin/env python3
"""
Filter Editing Sites by Replicate Reproducibility
==================================================

Raw editing sites from call_editing_sites_parallel.py contain one row per
(genomic position, treatment replicate) pair that passed initial thresholds.
This script groups rows by position, filters by how many replicates called the
site, and merges per-replicate rows into one consensus row with pooled statistics.

This mirrors the logic of Bullseye's summarize_sites.pl --MinRep and the
hyperTRIBER restrict_data() function.

Usage:
    python filter_replicates.py \\
        --input  results/raw_editing_sites.bed \\
        --min-replicates 2 \\
        --output results/filtered_editing_sites.bed
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Column names matching call_editing_sites_parallel.py output header
COLUMNS = [
    'chr', 'start', 'end', 'gene', 'edit_freq', 'strand',
    'control_cov', 'control_A', 'control_G',
    'treatment_cov', 'treatment_A', 'treatment_G',
    'fold_change', 'p_value'
]


def load_bed(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', comment='#', header=None)
    if df.shape[1] < 14:
        raise ValueError(
            f"Expected ≥14 columns in {path}, found {df.shape[1]}. "
            "Is this the raw editing sites BED from call_editing_sites_parallel.py?"
        )
    df = df.iloc[:, :14].copy()
    df.columns = COLUMNS
    return df


def _pooled_pvalue(
    total_treat_G: int, total_treat_A: int,
    ctrl_G: int, ctrl_A: int,
    stat_test: str
) -> float:
    """Compute pooled p-value using the same test as the per-site caller."""
    if stat_test == 'none':
        return 0.0
    elif stat_test == 'hypergeometric':
        M = ctrl_A + ctrl_G
        n = ctrl_G
        N = total_treat_A + total_treat_G
        k = total_treat_G
        try:
            if k == 0:
                return 1.0
            if N <= M and M > 0:
                return float(stats.hypergeom.sf(k - 1, M, n, N))
            else:
                bg_rate = n / M if M > 0 else 0.0
                return float(stats.binom.sf(k - 1, N, bg_rate)) if bg_rate > 0 else 1.0
        except Exception:
            return 1.0
    else:  # fisher
        try:
            _, p_value = stats.fisher_exact(
                [[total_treat_G, total_treat_A],
                 [ctrl_G, ctrl_A]],
                alternative='greater'
            )
            return p_value
        except Exception:
            return 1.0


def merge_replicate_group(group: pd.DataFrame, stat_test: str = 'fisher') -> dict:
    """
    Merge per-replicate rows at the same genomic position into one consensus row.

    Control counts are identical across replicates (merged control). Treatment
    counts are summed across replicates to produce pooled statistics, and the
    significance test is re-run on the pooled table using the same method
    specified by stat_test ('fisher', 'hypergeometric', or 'none').
    """
    first = group.iloc[0]

    total_treat_A = int(group['treatment_A'].sum())
    total_treat_G = int(group['treatment_G'].sum())
    total_treat_cov = int(group['treatment_cov'].sum())

    ctrl_A = int(first['control_A'])
    ctrl_G = int(first['control_G'])
    ctrl_cov = int(first['control_cov'])

    denominator = total_treat_A + total_treat_G
    edit_freq = (total_treat_G / denominator * 100.0) if denominator > 0 else 0.0

    ctrl_rate = ctrl_G / ctrl_A if ctrl_A > 0 else 0.0
    treat_rate = total_treat_G / total_treat_A if total_treat_A > 0 else 0.0
    fold_change = treat_rate / ctrl_rate if ctrl_rate > 0 else float('inf')

    p_value = _pooled_pvalue(total_treat_G, total_treat_A, ctrl_G, ctrl_A, stat_test)

    return {
        'chr':          first['chr'],
        'start':        first['start'],
        'end':          first['end'],
        'gene':         first['gene'],
        'edit_freq':    round(edit_freq, 4),
        'strand':       first['strand'],
        'control_cov':  ctrl_cov,
        'control_A':    ctrl_A,
        'control_G':    ctrl_G,
        'treatment_cov': total_treat_cov,
        'treatment_A':  total_treat_A,
        'treatment_G':  total_treat_G,
        'fold_change':  round(fold_change, 4),
        'p_value':      p_value,
        'n_replicates': len(group),
    }


def filter_replicates(
    input_file: str, min_replicates: int, output_file: str,
    stat_test: str = 'fisher'
) -> None:
    logger.info(f"Loading raw editing sites: {input_file}")
    df = load_bed(input_file)
    logger.info(f"  {len(df):,} raw site-replicate rows loaded")

    if len(df) == 0:
        logger.warning("Input file is empty — writing empty output")
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as fh:
            fh.write('#' + '\t'.join(COLUMNS + ['n_replicates']) + '\n')
        return

    # Count unique replicates per genomic position
    rep_counts = df.groupby(['chr', 'start']).size().rename('n_replicates')
    n_unique = len(rep_counts)
    logger.info(f"  {n_unique:,} unique genomic positions across all replicates")

    # Replicate count distribution
    for n in sorted(rep_counts.unique()):
        count = (rep_counts == n).sum()
        logger.info(f"    Sites in exactly {n} replicate(s): {count:,}")

    # Apply filter
    passing_coords = rep_counts[rep_counts >= min_replicates].index
    n_passing = len(passing_coords)
    n_removed = n_unique - n_passing
    logger.info(
        f"  Keeping sites in ≥{min_replicates} replicates: "
        f"{n_passing:,} retained, {n_removed:,} removed"
    )

    if n_passing == 0:
        logger.warning(
            "No sites passed the replicate filter. "
            "Try lowering --min-replicates or check alignment quality."
        )
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as fh:
            fh.write('#' + '\t'.join(COLUMNS + ['n_replicates']) + '\n')
        return

    # Merge per-replicate rows for retained positions
    df_keep = df.set_index(['chr', 'start']).loc[passing_coords].reset_index()

    merged_rows = []
    for (chrom, start), group in df_keep.groupby(['chr', 'start'], sort=False):
        merged_rows.append(merge_replicate_group(group, stat_test=stat_test))

    out = pd.DataFrame(merged_rows)
    out = out.sort_values(['chr', 'start']).reset_index(drop=True)

    # Write output with header comment
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    header_line = '#' + '\t'.join(COLUMNS + ['n_replicates']) + '\n'
    with open(output_file, 'w') as fh:
        fh.write(header_line)
        out.to_csv(fh, sep='\t', index=False, header=False, float_format='%.6g')

    logger.info(f"Wrote {len(out):,} filtered sites → {output_file}")

    # Summary statistics
    logger.info(f"\nFiltered sites summary:")
    logger.info(f"  Mean editing frequency:   {out['edit_freq'].mean():.2f}%")
    logger.info(f"  Median editing frequency: {out['edit_freq'].median():.2f}%")
    logger.info(f"  Mean fold change:         {out['fold_change'].replace(float('inf'), np.nan).mean():.2f}x")
    logger.info(f"  Sites with p < 0.05:      {(out['p_value'] < 0.05).sum():,}")
    logger.info(f"  Unique genes targeted:    {out['gene'].replace('.', np.nan).dropna().nunique():,}")


def main():
    parser = argparse.ArgumentParser(
        description='Filter HyperTRIBE editing sites by replicate reproducibility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input', required=True,
                        help='Raw editing sites BED from call_editing_sites_parallel.py')
    parser.add_argument('--min-replicates', type=int, default=2,
                        help='Minimum number of replicates a site must appear in (default: 2)')
    parser.add_argument('--output', required=True,
                        help='Output filtered BED file')
    parser.add_argument('--stat-test', default='fisher',
                        choices=['fisher', 'hypergeometric', 'none'],
                        help='Statistical test for pooled significance (must match '
                             'the test used in call_editing_sites_parallel.py; default: fisher)')
    args = parser.parse_args()

    logger.info('=' * 60)
    logger.info('HyperTRIBE Replicate Filtering')
    logger.info('=' * 60)
    logger.info(f'Input:           {args.input}')
    logger.info(f'Min replicates:  {args.min_replicates}')
    logger.info(f'Stat test:       {args.stat_test}')
    logger.info(f'Output:          {args.output}')

    filter_replicates(args.input, args.min_replicates, args.output, args.stat_test)

    logger.info('=' * 60)
    logger.info('Replicate filtering complete!')
    logger.info('=' * 60)


if __name__ == '__main__':
    main()
