#!/usr/bin/env python3
"""
Target Gene Venn Diagram and GO Enrichment / Depletion Analysis
================================================================

Compares target gene sets across HyperTRIBE conditions (e.g. WT, S161, S206 PUF60)
and produces:

  1. Three-way Venn diagram of target gene overlap
  2. UpSet plot showing intersection sizes (more informative for 3+ sets)
  3. GO term enrichment (g:Profiler API) for:
     - genes shared by all three conditions (core targets)
     - genes unique to each individual condition
     - genes in each pairwise-exclusive intersection

Requires internet access for g:Profiler queries.

Usage:
    python plot_target_overlap.py \\
        --input wt/annotated_editing_sites.bed \\
                s161/annotated_editing_sites.bed \\
                s206/annotated_editing_sites.bed \\
        --labels WT S161 S206 \\
        --output results/target_overlap.pdf \\
        --go-output results/go_enrichment.tsv
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib_venn import venn3, venn3_circles

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

_PALETTE = ['#2980b9', '#e74c3c', '#27ae60', '#8e44ad']


# ─────────────────────────────────────────────────────────────────────────────
# Load target genes from annotated BED
# ─────────────────────────────────────────────────────────────────────────────

def _load_genes(path: str) -> Set[str]:
    genes = set()
    with open(path) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            fields = line.split('\t')
            # col 3 = gene name (gene_name/symbol)
            if len(fields) > 3 and fields[3] not in ('.', ''):
                genes.add(fields[3])
    return genes


# ─────────────────────────────────────────────────────────────────────────────
# Venn diagram (3-way)
# ─────────────────────────────────────────────────────────────────────────────

def _panel_venn(ax, sets: List[Set[str]], labels: List[str]):
    colors = _PALETTE[:len(sets)]
    v = venn3(sets, set_labels=labels, ax=ax,
              set_colors=colors, alpha=0.55)
    if v:
        for text in v.set_labels:
            if text:
                text.set_fontsize(12)
        for text in v.subset_labels:
            if text:
                text.set_fontsize(10)
    c = venn3_circles(sets, ax=ax, linewidth=1.2)
    ax.set_title('Target Gene Overlap', fontsize=13, fontweight='bold')


# ─────────────────────────────────────────────────────────────────────────────
# Intersection bar chart (UpSet-style, simple version)
# ─────────────────────────────────────────────────────────────────────────────

def _panel_upset(ax, sets: List[Set[str]], labels: List[str]):
    """Simple intersection size bar chart as an UpSet substitute."""
    n = len(sets)
    # All 2^n - 1 non-empty subsets (as frozensets of indices)
    from itertools import combinations

    intersections = {}
    for r in range(n, 0, -1):
        for combo in combinations(range(n), r):
            key = frozenset(combo)
            # Intersection of chosen sets minus union of unchosen sets
            inter = sets[combo[0]].copy()
            for i in combo[1:]:
                inter &= sets[i]
            excluded = set()
            for i in range(n):
                if i not in combo:
                    excluded |= sets[i]
            exclusive = inter - excluded
            if exclusive:
                name = ' ∩ '.join(labels[i] for i in sorted(combo))
                if r < n:
                    name += ' only'
                intersections[name] = len(exclusive)

    names = list(intersections.keys())
    sizes = [intersections[k] for k in names]
    order = np.argsort(sizes)[::-1]
    names = [names[i] for i in order]
    sizes = [sizes[i] for i in order]

    colors_map = {labels[i]: _PALETTE[i] for i in range(n)}
    bar_colors = []
    for name in names:
        for lab in labels:
            if lab in name:
                bar_colors.append(colors_map[lab])
                break
        else:
            bar_colors.append('#888')

    bars = ax.bar(range(len(names)), sizes, color=bar_colors,
                  edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Number of target genes')
    ax.set_title('Target Gene Intersections', fontsize=13, fontweight='bold')
    ax.bar_label(bars, padding=2, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# GO enrichment via g:Profiler
# ─────────────────────────────────────────────────────────────────────────────

def _run_go(gene_set: Set[str], label: str, background: Set[str]) -> pd.DataFrame:
    if len(gene_set) < 5:
        logger.warning(f"  {label}: too few genes ({len(gene_set)}) — skipping GO")
        return pd.DataFrame()
    try:
        from gprofiler import GProfiler
        gp = GProfiler(return_dataframe=True)
        # Use genome-wide background (no custom background) so that enrichment
        # is relative to all human genes — more sensitive than restricting to
        # the target gene union, which would bias against core targets.
        result = gp.profile(
            organism='hsapiens',
            query=sorted(gene_set),
            sources=['GO:BP', 'GO:MF', 'GO:CC', 'KEGG', 'REAC'],
            significance_threshold_method='fdr',
            user_threshold=0.05,
            no_iea=False,   # include IEA for sensitivity
        )
        if result.empty:
            logger.info(f"  {label}: no significant GO terms")
            return pd.DataFrame()
        result['query_set'] = label
        logger.info(f"  {label}: {len(result)} significant terms")
        return result
    except Exception as e:
        logger.warning(f"  {label}: g:Profiler error — {e}")
        return pd.DataFrame()


def run_go_analysis(sets: List[Set[str]], labels: List[str]) -> pd.DataFrame:
    logger.info("Running GO enrichment analysis via g:Profiler...")
    background = sets[0] | sets[1] | sets[2]

    gene_groups = {}
    # All three
    core = sets[0] & sets[1] & sets[2]
    if core:
        gene_groups['Core (all 3)'] = core

    # Unique to each
    for i, label in enumerate(labels):
        others = set()
        for j in range(len(sets)):
            if j != i:
                others |= sets[j]
        unique = sets[i] - others
        if unique:
            gene_groups[f'{label} unique'] = unique

    # Pairwise exclusive (in exactly 2)
    n = len(sets)
    from itertools import combinations
    for i, j in combinations(range(n), 2):
        pair = sets[i] & sets[j]
        for k in range(n):
            if k != i and k != j:
                pair -= sets[k]
        if pair:
            gene_groups[f'{labels[i]}+{labels[j]} only'] = pair

    all_results = []
    for group_label, genes in gene_groups.items():
        logger.info(f"  Querying: {group_label} ({len(genes)} genes)")
        df = _run_go(genes, group_label, background)
        if not df.empty:
            all_results.append(df)
        time.sleep(0.5)  # be polite to the API

    if not all_results:
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# GO bubble plot
# ─────────────────────────────────────────────────────────────────────────────

def _panel_go(ax, go_df: pd.DataFrame, group: str, n_top: int = 15):
    sub = go_df[go_df['query_set'] == group].copy()
    if sub.empty:
        ax.text(0.5, 0.5, f'No significant GO terms\nfor {group}',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.axis('off')
        return

    sub = sub.nsmallest(n_top, 'p_value')
    sub['-log10p'] = -np.log10(sub['p_value'].clip(lower=1e-300))
    sub = sub.sort_values('-log10p')

    source_colors = {'GO:BP': '#3498db', 'GO:MF': '#e74c3c',
                     'GO:CC': '#2ecc71', 'KEGG': '#e67e22', 'REAC': '#9b59b6'}
    colors = [source_colors.get(s, '#888') for s in sub['source']]

    bars = ax.barh(range(len(sub)), sub['-log10p'], color=colors,
                   edgecolor='white', linewidth=0.4)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(
        [f"{n[:55]}…" if len(n) > 55 else n for n in sub['name']],
        fontsize=8
    )
    ax.set_xlabel('−log₁₀(FDR)')
    ax.set_title(f'GO Enrichment — {group}', fontsize=11, fontweight='bold')
    ax.axvline(x=-np.log10(0.05), color='#e74c3c', linestyle='--',
               linewidth=0.8, alpha=0.8, label='FDR 0.05')

    # Color legend for ontology source
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=s)
                       for s, c in source_colors.items() if s in sub['source'].values]
    ax.legend(handles=legend_elements, fontsize=7, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Edit count comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────

def _panel_editcount(ax, sets: List[Set[str]], labels: List[str], edit_counts: List[Dict]):
    """Bar chart of editing site counts per gene for shared vs unique genes."""
    categories = ['All targets', 'Core (all 3)', 'Unique']
    x = np.arange(len(labels))
    w = 0.28
    totals = [len(s) for s in sets]
    core = sets[0] & sets[1] & sets[2]
    core_count = len(core)

    bar1 = ax.bar(x - w, totals, w, color=[_PALETTE[i] for i in range(len(labels))],
                  alpha=0.85, label='All targets', edgecolor='white')
    core_vals = [core_count] * len(labels)
    bar2 = ax.bar(x, core_vals, w, color='#2c3e50', alpha=0.8,
                  label=f'Core (all 3,  n={core_count})', edgecolor='white')

    unique_vals = []
    for i in range(len(sets)):
        others = set()
        for j in range(len(sets)):
            if j != i:
                others |= sets[j]
        unique_vals.append(len(sets[i] - others))
    bar3 = ax.bar(x + w, unique_vals, w, color=[_PALETTE[i] for i in range(len(labels))],
                  alpha=0.4, label='Unique to condition', edgecolor='white', hatch='//')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Number of target genes')
    ax.set_title('Target Gene Set Sizes', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, frameon=True)
    ax.bar_label(bar1, padding=2, fontsize=8)
    ax.bar_label(bar3, padding=2, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Main plot
# ─────────────────────────────────────────────────────────────────────────────

def plot(input_files: List[str], labels: List[str], output: str,
         go_output: str, skip_go: bool = False):

    sets = [_load_genes(f) for f in input_files]
    for label, s in zip(labels, sets):
        logger.info(f"  {label}: {len(s):,} target genes")

    core = sets[0] & sets[1] & sets[2]
    logger.info(f"  Core (all 3): {len(core):,} genes")

    # GO analysis
    go_df = pd.DataFrame()
    if not skip_go:
        go_df = run_go_analysis(sets, labels)
        if not go_df.empty and go_output:
            Path(go_output).parent.mkdir(parents=True, exist_ok=True)
            go_df.to_csv(go_output, sep='\t', index=False)
            logger.info(f"GO table saved: {go_output}")

    # ── Layout ───────────────────────────────────────────────────────────
    n_go_panels = 0
    if not go_df.empty:
        groups_in_go = go_df['query_set'].unique()
        n_go_panels = min(len(groups_in_go), 4)

    n_rows = 2 + (n_go_panels + 1) // 2 if n_go_panels else 2
    fig = plt.figure(figsize=(18, 6 * n_rows))
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.55, wspace=0.38)

    ax_venn  = fig.add_subplot(gs[0, 0])
    ax_sizes = fig.add_subplot(gs[0, 1])
    ax_upset = fig.add_subplot(gs[1, :])

    _panel_venn(ax_venn, sets, labels)
    _panel_editcount(ax_sizes, sets, labels, [{}] * len(sets))
    _panel_upset(ax_upset, sets, labels)

    # GO panels
    if not go_df.empty:
        groups_in_go = list(go_df['query_set'].unique())
        for idx, group in enumerate(groups_in_go[:n_go_panels]):
            row = 2 + idx // 2
            col = idx % 2
            ax_go = fig.add_subplot(gs[row, col])
            _panel_go(ax_go, go_df, group)

    fig.suptitle("HyperTRIBE — Target Gene Overlap and GO Enrichment\n"
                 f"{' vs '.join(labels)}",
                 fontsize=15, fontweight='bold', y=1.002)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output}")

    # Print core gene list
    logger.info(f"\nCore target genes shared by all conditions ({len(core)}):")
    for g in sorted(core)[:50]:
        logger.info(f"  {g}")
    if len(core) > 50:
        logger.info(f"  ... and {len(core) - 50} more (see GO output)")


def main():
    parser = argparse.ArgumentParser(
        description='Target gene Venn diagram and GO enrichment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input', nargs='+', required=True,
                        help='Annotated editing site BED files (one per condition)')
    parser.add_argument('--labels', nargs='+',
                        help='Labels for each condition (default: file stems)')
    parser.add_argument('--output', required=True,
                        help='Output PDF path')
    parser.add_argument('--go-output', default='',
                        help='Output TSV path for GO enrichment table')
    parser.add_argument('--skip-go', action='store_true',
                        help='Skip GO enrichment (faster, no internet needed)')
    args = parser.parse_args()

    labels = args.labels or [Path(f).stem for f in args.input]
    if len(labels) != len(args.input):
        parser.error('--labels must have same count as --input')
    if len(args.input) != 3:
        parser.error('Exactly 3 input files required for 3-way Venn')

    plot(args.input, labels, args.output, args.go_output, args.skip_go)


if __name__ == '__main__':
    main()
