#!/usr/bin/env python3
"""
plot_editing_sites_gene.py
==========================
Per-gene editing site lollipop visualization across three PUF60 HyperTRIBE conditions.

For each gene produces a single multi-panel figure:
  Row 0: Gene model (canonical transcript: UTRs / CDS exons / introns with direction)
  Rows 1-3: Lollipop tracks — WT PUF60, S161 PUF60, S206 PUF60
             X = genomic coordinate, Y = editing frequency (%)
             Circle area ∝ treatment coverage, background shading by feature type.

Usage:
    python plot_editing_sites_gene.py \\
        --wt-splice   wt_splice_annotated.bed \\
        --s161-splice s161_splice_annotated.bed \\
        --s206-splice s206_splice_annotated.bed \\
        --gtf         genes.gtf \\
        --genes       CBL FOXO3 HMGCR TET2 ASXL1 \\
        --outdir      figures/gene_editing/
"""

import argparse
import logging
import sys
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# ── Colour / style constants ──────────────────────────────────────────────────
COND_COLORS = {
    'WT':   '#2980b9',   # blue
    'S161': '#e74c3c',   # red
    'S206': '#27ae60',   # green
}
FEAT_FACE = {
    'UTR':      '#fff3cd',   # pale amber
    'CDS_exon': '#d4edda',   # pale green
    'intron':   '#ffffff',   # white (no shade)
}
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
})


# ─────────────────────────────────────────────────────────────────────────────
# GTF parsing
# ─────────────────────────────────────────────────────────────────────────────

def _attr(attr_str: str, key: str) -> Optional[str]:
    m = re.search(rf'{key}\s+"([^"]+)"', attr_str)
    return m.group(1) if m else None


def parse_gtf_gene(gtf_path: str, gene_name: str) -> Dict:
    """
    Return a dict describing the canonical transcript for gene_name:
        {chrom, strand, gene_start, gene_end,
         transcripts: {tx_id: {exons: [(s,e)], cds: [(s,e)], utrs: [(s,e)]}}}
    Returns empty dict if gene not found.
    """
    logger.info(f"  Parsing GTF for {gene_name}...")
    records = []
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 9:
                continue
            feat = parts[2]
            if feat not in ('gene', 'transcript', 'exon', 'CDS', 'UTR',
                            'start_codon', 'stop_codon'):
                continue
            attrs = parts[8]
            gname = _attr(attrs, 'gene_name')
            if gname != gene_name:
                continue
            records.append({
                'chrom':  parts[0],
                'start':  int(parts[3]) - 1,   # 0-based
                'end':    int(parts[4]),
                'strand': parts[6],
                'feat':   feat,
                'tx_id':  _attr(attrs, 'transcript_id') or '',
                'attrs':  attrs,
            })

    if not records:
        logger.warning(f"  {gene_name} not found in GTF")
        return {}

    gene_recs = [r for r in records if r['feat'] == 'gene']
    if gene_recs:
        gr = gene_recs[0]
        chrom, strand = gr['chrom'], gr['strand']
        gene_start, gene_end = gr['start'], gr['end']
    else:
        chrom  = records[0]['chrom']
        strand = records[0]['strand']
        gene_start = min(r['start'] for r in records)
        gene_end   = max(r['end']   for r in records)

    # Group by transcript
    tx_data: Dict[str, Dict] = defaultdict(lambda: {'exons': [], 'cds': [], 'utrs': []})
    for r in records:
        tid = r['tx_id']
        if r['feat'] == 'exon':
            tx_data[tid]['exons'].append((r['start'], r['end']))
        elif r['feat'] == 'CDS':
            tx_data[tid]['cds'].append((r['start'], r['end']))
        elif r['feat'] == 'UTR':
            tx_data[tid]['utrs'].append((r['start'], r['end']))

    # Pick canonical = longest total CDS; fall back to longest exon span
    def tx_score(tid):
        c = tx_data[tid]['cds']
        if c:
            return sum(e - s for s, e in c)
        e = tx_data[tid]['exons']
        return sum(en - s for s, en in e) if e else 0

    canonical = max(tx_data.keys(), key=tx_score) if tx_data else ''

    return {
        'chrom':      chrom,
        'strand':     strand,
        'gene_start': gene_start,
        'gene_end':   gene_end,
        'canonical':  canonical,
        'transcripts': tx_data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BED loading
# ─────────────────────────────────────────────────────────────────────────────

SPLICE_COLS = [
    'chr', 'start', 'end', 'gene', 'edit_freq', 'strand',
    'ctrl_cov', 'ctrl_A', 'ctrl_G',
    'treat_cov', 'treat_A', 'treat_G',
    'fold_change', 'p_value', 'n_reps',
    'gene_id', 'gene_type', 'feature_type',
    'dist_5ss', 'dist_3ss', 'splice_region',
    'signed_5ss', 'signed_3ss',
]


def load_splice_bed(path: str, gene: str) -> pd.DataFrame:
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 18:
                continue
            if parts[3] != gene:
                continue
            rows.append(parts[:len(SPLICE_COLS)])
    if not rows:
        return pd.DataFrame(columns=SPLICE_COLS)
    df = pd.DataFrame(rows, columns=SPLICE_COLS[:len(rows[0])])
    df['start']     = df['start'].astype(int)
    df['edit_freq'] = pd.to_numeric(df['edit_freq'], errors='coerce')
    df['treat_cov'] = pd.to_numeric(df['treat_cov'], errors='coerce').fillna(20)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Gene model drawing
# ─────────────────────────────────────────────────────────────────────────────

def _draw_gene_model(ax, gene_info: Dict, xlim: Tuple[int, int], y0: float = 0.5):
    """Draw canonical transcript on ax. y0 is the midline y position (0-1)."""
    tx_id    = gene_info['canonical']
    tx       = gene_info['transcripts'].get(tx_id, {'exons': [], 'cds': [], 'utrs': []})
    strand   = gene_info['strand']
    x0, x1  = xlim

    exons = sorted(tx['exons'])
    cdss  = sorted(tx['cds'])
    utrs  = sorted(tx['utrs'])

    span = x1 - x0

    def xfrac(pos):
        return (pos - x0) / span

    # Intron backbone
    ax.axhline(y0, color='#555', lw=1.2, zorder=1)

    # Direction arrows along intron (every ~5% of span)
    arrow_step = max(1, span // 20)
    arrow_dir  = '>' if strand == '+' else '<'
    arrow_pos  = np.arange(x0 + arrow_step, x1, arrow_step)
    for xpos in arrow_pos:
        # Only draw in intron gaps (not inside exons)
        in_exon = any(s <= xpos <= e for s, e in exons)
        if not in_exon:
            ax.text(xfrac(xpos), y0, arrow_dir, ha='center', va='center',
                    fontsize=6, color='#999', transform=ax.transAxes, zorder=2)

    # UTRs (thin rect, height 0.20)
    utr_h = 0.20
    for s, e in utrs:
        xs = max(s, x0); xe = min(e, x1)
        if xs >= xe:
            continue
        rect = mpatches.FancyBboxPatch(
            (xfrac(xs), y0 - utr_h / 2), xfrac(xe) - xfrac(xs), utr_h,
            boxstyle='square,pad=0', facecolor='#95a5a6', edgecolor='#7f8c8d',
            linewidth=0.6, transform=ax.transAxes, zorder=3
        )
        ax.add_patch(rect)

    # CDS exons (thick rect, height 0.45)
    cds_h = 0.45
    for s, e in cdss:
        xs = max(s, x0); xe = min(e, x1)
        if xs >= xe:
            continue
        rect = mpatches.FancyBboxPatch(
            (xfrac(xs), y0 - cds_h / 2), xfrac(xe) - xfrac(xs), cds_h,
            boxstyle='square,pad=0', facecolor='#2c3e50', edgecolor='#1a252f',
            linewidth=0.6, transform=ax.transAxes, zorder=4
        )
        ax.add_patch(rect)

    # If no CDS annotation, draw exons as medium rects
    if not cdss and exons:
        ex_h = 0.35
        for s, e in exons:
            xs = max(s, x0); xe = min(e, x1)
            if xs >= xe:
                continue
            rect = mpatches.FancyBboxPatch(
                (xfrac(xs), y0 - ex_h / 2), xfrac(xe) - xfrac(xs), ex_h,
                boxstyle='square,pad=0', facecolor='#2c3e50', edgecolor='#1a252f',
                linewidth=0.6, transform=ax.transAxes, zorder=4
            )
            ax.add_patch(rect)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Strand label
    strand_txt = f"({strand}) strand"
    ax.text(1.002, y0, strand_txt, va='center', fontsize=7, color='#666',
            transform=ax.transAxes)


# ─────────────────────────────────────────────────────────────────────────────
# Lollipop track
# ─────────────────────────────────────────────────────────────────────────────

def _draw_lollipop(ax, df: pd.DataFrame, xlim: Tuple[int, int],
                   color: str, label: str, y_max: float,
                   show_xaxis: bool = False):
    """Draw lollipop plot on ax. df must have start, edit_freq, treat_cov, feature_type."""
    x0, x1 = xlim
    span = x1 - x0

    # Background shading by feature-type intervals
    if not df.empty:
        feat_df = df.dropna(subset=['feature_type'])
        for _, row in feat_df.iterrows():
            ft = str(row.get('feature_type', 'intron'))
            face = FEAT_FACE.get(ft, '#ffffff')
            if face != '#ffffff':
                ax.axvspan(row['start'] - 0.5, row['start'] + 0.5,
                           alpha=0.15, color=face, linewidth=0, zorder=0)

    # Coverage → dot size (scaled to 20–200 pt²)
    if not df.empty:
        cov = df['treat_cov'].clip(lower=1)
        cov_norm = (np.log1p(cov) - np.log1p(1)) / (np.log1p(cov.max()) - np.log1p(1) + 1e-9)
        sizes = 15 + 120 * cov_norm

        for (_, row), sz in zip(df.iterrows(), sizes):
            xpos = row['start']
            ypos = row['edit_freq']
            if pd.isna(ypos) or xpos < x0 or xpos > x1:
                continue
            # Stem
            ax.plot([xpos, xpos], [0, ypos], color=color, lw=0.6, alpha=0.6, zorder=2)
            # Circle
            ft = str(row.get('feature_type', 'intron'))
            edge = '#e74c3c' if ft == 'UTR' else '#2c3e50' if ft == 'CDS_exon' else '#aaa'
            ax.scatter(xpos, ypos, s=sz, color=color, edgecolors=edge,
                       linewidths=0.7, zorder=3, alpha=0.85)

    ax.set_xlim(x0, x1)
    ax.set_ylim(0, max(y_max * 1.1, 5))
    ax.set_ylabel(f'{label}\nedit %', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=7)

    # Site count annotation
    n_sites = len(df)
    ax.text(0.01, 0.92, f'n={n_sites} sites', transform=ax.transAxes,
            fontsize=7, va='top', color=color, fontweight='bold')

    if show_xaxis:
        # Format x-axis with Mb scale
        ax.tick_params(axis='x', labelsize=7)
        xticks = ax.get_xticks()
        ax.set_xticklabels([f'{x/1e6:.3f}' for x in xticks], fontsize=7)
        ax.set_xlabel('Genomic position (Mb)', fontsize=8)
    else:
        ax.tick_params(axis='x', labelbottom=False, bottom=False)


# ─────────────────────────────────────────────────────────────────────────────
# Per-gene figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_gene(gene: str, gene_info: Dict,
              dfs: Dict[str, pd.DataFrame],
              outdir: Path):
    """Generate the 4-panel figure for one gene."""

    chrom = gene_info['chrom']
    all_pos = []
    for df in dfs.values():
        if not df.empty:
            all_pos.extend(df['start'].tolist())
    if not all_pos:
        logger.warning(f"  {gene}: no editing sites in any condition — skipping")
        return

    # Genomic window: span of editing sites + 5% padding each side
    g_start = min(all_pos)
    g_end   = max(all_pos) + 1
    pad = max(int((g_end - g_start) * 0.05), 500)
    xlim = (g_start - pad, g_end + pad)

    # Global y_max for shared scale
    y_max = max(
        (df['edit_freq'].max() if not df.empty else 0)
        for df in dfs.values()
    )
    y_max = max(y_max, 10)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(
        4, 1, figure=fig,
        height_ratios=[1, 2, 2, 2],
        hspace=0.06
    )

    ax_gene  = fig.add_subplot(gs[0])
    ax_wt    = fig.add_subplot(gs[1])
    ax_s161  = fig.add_subplot(gs[2])
    ax_s206  = fig.add_subplot(gs[3])

    # Gene model
    _draw_gene_model(ax_gene, gene_info, xlim)
    ax_gene.set_title(
        f'{gene}  —  chr{chrom.lstrip("chr")}:{xlim[0]:,}–{xlim[1]:,}\n'
        f'Editing sites in WT={len(dfs["WT"])}  S161={len(dfs["S161"])}  S206={len(dfs["S206"])}',
        fontsize=12, fontweight='bold', pad=6
    )

    # Lollipop tracks
    cond_order = [('WT', ax_wt), ('S161', ax_s161), ('S206', ax_s206)]
    for i, (cond, ax) in enumerate(cond_order):
        show_x = (i == len(cond_order) - 1)
        _draw_lollipop(ax, dfs[cond], xlim,
                       color=COND_COLORS[cond],
                       label=f'{cond} PUF60',
                       y_max=y_max,
                       show_xaxis=show_x)

    # Shared x sync
    for _, ax in cond_order:
        ax.set_xlim(*xlim)

    # Legend for edge colours (feature type)
    legend_patches = [
        mpatches.Patch(facecolor=COND_COLORS['WT'],   label='WT PUF60'),
        mpatches.Patch(facecolor=COND_COLORS['S161'],  label='S161F PUF60'),
        mpatches.Patch(facecolor=COND_COLORS['S206'],  label='S206L PUF60'),
        mpatches.Patch(facecolor='#95a5a6', edgecolor='#7f8c8d', lw=0.6, label='UTR (edge)'),
        mpatches.Patch(facecolor='#2c3e50', edgecolor='#1a252f', lw=0.6, label='CDS exon (edge)'),
    ]
    fig.legend(handles=legend_patches, loc='upper right',
               bbox_to_anchor=(0.99, 0.98), fontsize=8,
               frameon=True, framealpha=0.85, ncol=2)

    out = outdir / f'{gene}_editing_sites.pdf'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--wt-splice',   required=True, help='WT splice-annotated BED')
    parser.add_argument('--s161-splice', required=True, help='S161 splice-annotated BED')
    parser.add_argument('--s206-splice', required=True, help='S206 splice-annotated BED')
    parser.add_argument('--gtf',         required=True, help='Annotation GTF')
    parser.add_argument('--genes', nargs='+', required=True,
                        help='Gene names to plot')
    parser.add_argument('--outdir', required=True,
                        help='Output directory for PDFs')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bed_paths = {
        'WT':   args.wt_splice,
        'S161': args.s161_splice,
        'S206': args.s206_splice,
    }

    for gene in args.genes:
        logger.info(f"Processing {gene}...")

        gene_info = parse_gtf_gene(args.gtf, gene)
        if not gene_info:
            logger.warning(f"  Skipping {gene} (not in GTF)")
            continue

        dfs = {}
        for cond, path in bed_paths.items():
            dfs[cond] = load_splice_bed(path, gene)
            logger.info(f"  {cond}: {len(dfs[cond])} sites")

        # Skip if no condition has >1 site
        total = sum(len(df) for df in dfs.values())
        if total < 3:
            logger.warning(f"  {gene}: too few sites ({total}) — skipping")
            continue

        plot_gene(gene, gene_info, dfs, outdir)

    logger.info("Done.")


if __name__ == '__main__':
    main()
