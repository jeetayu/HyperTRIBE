#!/home/biswasj/miniconda3/envs/hypertribe/bin/python3
"""
Gene Track Visualizer for HyperTRIBE
======================================
Renders a genome-browser-style figure for a gene of interest with:
  - Gene model (exons, UTRs, intron arrows, strand)
  - Editing site lollipops per condition (height = edit frequency %)
  - Per-replicate RPM coverage filled-area tracks

Usage
-----
python3 plot_gene_tracks.py \\
    --gene QKI \\
    --gtf /data1/abdelwao/jeet/annotations/.../genes.gtf \\
    --bam "Control 1:control:/path/cherry-nls-1.nodup.bam" \\
    --bam "Control 2:control:/path/cherry-nls-2.nodup.bam" \\
    --bam "WT rep1:wt:/path/wt-puf60-1.nodup.bam" \\
    --bam "WT rep2:wt:/path/wt-puf60-2.nodup.bam" \\
    --editing-sites "WT:/path/wt/annotated_editing_sites.bed" \\
    --editing-sites "S161:/path/s161/annotated_editing_sites.bed" \\
    --editing-sites "S206:/path/s206/annotated_editing_sites.bed" \\
    --output QKI_tracks.pdf

Each --bam argument is "Label:Group:/path/to.bam".
Each --editing-sites argument is "Label:/path/to/annotated.bed".
Group names that match a known condition key get automatic colors;
unknown groups cycle through a default palette.

Alternatively, specify a genomic region directly:
    --region chr6:163500000-163650000
"""

import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pysam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow
import matplotlib.ticker as mticker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
})

# ── Color palette ─────────────────────────────────────────────────────────────
_GROUP_COLORS = {
    'control': '#7f8c8d',
    'ctrl':    '#7f8c8d',
    'wt':      '#2980b9',
    'wt_puf60':'#2980b9',
    's161':    '#e74c3c',
    's206':    '#27ae60',
    'dhx15':   '#8e44ad',
}
_DEFAULT_CYCLE = ['#2980b9', '#e74c3c', '#27ae60', '#8e44ad',
                  '#e67e22', '#16a085', '#d35400', '#2c3e50']

_EXON_COLOR    = '#2c3e50'
_UTR_COLOR     = '#7f8c8d'
_INTRON_COLOR  = '#bdc3c7'
_ARROW_COLOR   = '#95a5a6'


# ── GTF parsing ───────────────────────────────────────────────────────────────

def _attr(field: str, key: str) -> str:
    m = re.search(rf'{key}\s+"([^"]+)"', field)
    return m.group(1) if m else ''


def parse_gene_region(
    gtf_path: str, gene_name: str
) -> Tuple[str, int, int, str, dict]:
    """
    Return (chrom, gene_start, gene_end, strand, transcripts_dict).

    transcripts_dict: {tx_id: {'exons': [(s,e)], 'utrs': [(s,e)], 'strand': str}}
    Coordinates are 0-based half-open [start, end).
    """
    logger.info(f"Parsing GTF for gene: {gene_name}")
    transcripts: dict = {}
    gene_chrom = gene_start = gene_end = gene_strand = None

    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 9:
                continue
            chrom, _, feature, start, end, _, strand, _, attrs = parts
            gname = _attr(attrs, 'gene_name') or _attr(attrs, 'gene_id')
            if gname != gene_name:
                continue
            s, e = int(start) - 1, int(end)   # GTF 1-based → 0-based half-open

            if feature == 'gene':
                gene_chrom, gene_start, gene_end, gene_strand = chrom, s, e, strand

            elif feature in ('transcript', 'mRNA'):
                tx_id = _attr(attrs, 'transcript_id')
                if tx_id not in transcripts:
                    transcripts[tx_id] = {'exons': [], 'utrs': [], 'strand': strand,
                                          'start': s, 'end': e}

            elif feature == 'exon':
                tx_id = _attr(attrs, 'transcript_id')
                if tx_id not in transcripts:
                    transcripts[tx_id] = {'exons': [], 'utrs': [], 'strand': strand,
                                          'start': s, 'end': e}
                transcripts[tx_id]['exons'].append((s, e))

            elif feature in ('UTR', 'three_prime_utr', 'five_prime_utr',
                              '3UTR', '5UTR'):
                tx_id = _attr(attrs, 'transcript_id')
                if tx_id not in transcripts:
                    transcripts[tx_id] = {'exons': [], 'utrs': [], 'strand': strand,
                                          'start': s, 'end': e}
                transcripts[tx_id]['utrs'].append((s, e))

    if not transcripts:
        raise ValueError(
            f"Gene '{gene_name}' not found in GTF. "
            "Check gene name spelling (case-sensitive) or use --region."
        )

    # Derive gene boundaries from transcript extents if gene feature absent
    all_starts = [t['start'] for t in transcripts.values()]
    all_ends   = [t['end']   for t in transcripts.values()]
    if gene_start is None:
        gene_start = min(all_starts)
        gene_end   = max(all_ends)
        gene_chrom = next(iter(transcripts.values())).get('chrom', None)
        # chrom from line — re-parse first transcript line if needed
        if gene_chrom is None:
            with open(gtf_path) as fh:
                for line in fh:
                    if line.startswith('#'):
                        continue
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) < 9:
                        continue
                    chrom, _, feature, _, _, _, strand, _, attrs = parts
                    if _attr(attrs, 'gene_name') == gene_name or \
                       _attr(attrs, 'gene_id') == gene_name:
                        gene_chrom  = chrom
                        gene_strand = strand
                        break

    logger.info(
        f"  {gene_name}: {gene_chrom}:{gene_start}-{gene_end} ({gene_strand}) "
        f"| {len(transcripts)} transcripts"
    )
    return gene_chrom, gene_start, gene_end, gene_strand, transcripts


def parse_region(region_str: str) -> Tuple[str, int, int]:
    """Parse 'chr:start-end' or 'chr:start:end' into (chrom, start, end)."""
    m = re.match(r'^(\S+)[:\s](\d[\d,]*)[-:\s](\d[\d,]*)$', region_str)
    if not m:
        raise ValueError(f"Cannot parse region: '{region_str}'. Use chr:start-end")
    chrom = m.group(1)
    start = int(m.group(2).replace(',', ''))
    end   = int(m.group(3).replace(',', ''))
    return chrom, start, end


# ── Coverage from BAM ─────────────────────────────────────────────────────────

def get_coverage(
    bam_path: str, chrom: str, start: int, end: int,
    min_mapq: int = 10, bin_size: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (positions, coverage_array) for the region.
    Coverage is computed as total depth (sum of A+C+G+T counts) at each position,
    then binned to bin_size resolution.
    """
    with pysam.AlignmentFile(bam_path, 'rb') as bam:
        cov = bam.count_coverage(
            chrom, start, end,
            quality_threshold=20,
            read_callback=lambda r: (
                not r.is_unmapped and
                not r.is_duplicate and
                r.mapping_quality >= min_mapq
            )
        )
    # cov is a tuple of 4 arrays: (A, C, G, T)
    total = np.array(cov[0]) + np.array(cov[1]) + np.array(cov[2]) + np.array(cov[3])

    if bin_size > 1:
        n_bins = len(total) // bin_size
        total = total[:n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)
        positions = np.arange(start, start + n_bins * bin_size, bin_size)
    else:
        positions = np.arange(start, start + len(total))

    return positions, total


def get_total_mapped(bam_path: str) -> int:
    """Return total mapped read count from BAM index statistics."""
    with pysam.AlignmentFile(bam_path, 'rb') as bam:
        return bam.mapped


# ── Editing sites ─────────────────────────────────────────────────────────────

def load_editing_sites(
    bed_path: str, chrom: str, start: int, end: int
) -> List[dict]:
    """Return list of site dicts in the region from an annotated BED file."""
    sites = []
    with open(bed_path) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            f = line.rstrip('\n').split('\t')
            if len(f) < 14:
                continue
            if f[0] != chrom:
                continue
            pos = int(f[1])
            if pos < start or pos >= end:
                continue
            try:
                sites.append({
                    'pos':        pos,
                    'gene':       f[3],
                    'edit_freq':  float(f[4]),
                    'strand':     f[5],
                    'fold_change': float(f[12]),
                    'p_value':    float(f[13]),
                })
            except (ValueError, IndexError):
                continue
    return sites


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_gene_model(
    ax, transcripts: dict, region_start: int, region_end: int, strand: str,
    max_tx: int = 5
):
    """Draw gene model tracks (exons, UTRs, intron lines with arrows)."""
    ax.set_xlim(region_start, region_end)
    ax.set_ylim(-0.5, max_tx - 0.5)
    ax.axis('off')

    # Select transcripts to display: pick up to max_tx longest
    sorted_tx = sorted(
        transcripts.items(),
        key=lambda kv: sum(e - s for s, e in kv[1]['exons']),
        reverse=True
    )[:max_tx]

    n = len(sorted_tx)
    for row, (tx_id, tx) in enumerate(sorted_tx):
        y = (n - 1 - row)   # top row = longest transcript
        exons = sorted(tx['exons'])
        utrs  = set()
        for us, ue in tx.get('utrs', []):
            for pos in range(us, ue):
                utrs.add(pos)

        if not exons:
            continue

        tx_s = exons[0][0]
        tx_e = exons[-1][1]

        # Intron backbone line
        ax.hlines(y, max(tx_s, region_start), min(tx_e, region_end),
                  colors=_INTRON_COLOR, linewidths=1.0, zorder=1)

        # Strand direction arrows on introns
        span = region_end - region_start
        arrow_spacing = max(span // 20, 500)
        arrow_dx = span // 100
        for intron_pos in range(
            max(tx_s, region_start) + arrow_spacing // 2,
            min(tx_e, region_end),
            arrow_spacing
        ):
            # Only draw if position is actually intronic (not in any exon)
            in_exon = any(s <= intron_pos < e for s, e in exons)
            if in_exon:
                continue
            dx = arrow_dx if strand == '+' else -arrow_dx
            ax.annotate(
                '', xy=(intron_pos + dx, y), xytext=(intron_pos, y),
                arrowprops=dict(
                    arrowstyle='->', color=_ARROW_COLOR,
                    lw=0.8, mutation_scale=6
                ),
                zorder=2
            )

        # Exon rectangles
        exon_h = 0.35
        utr_h  = 0.20
        for es, ee in exons:
            # clip to region
            es = max(es, region_start)
            ee = min(ee, region_end)
            if es >= ee:
                continue
            # Check if fully UTR
            exon_positions = set(range(es, ee))
            is_utr = exon_positions.issubset(utrs) and len(utrs) > 0
            h = utr_h if is_utr else exon_h
            col = _UTR_COLOR if is_utr else _EXON_COLOR
            rect = mpatches.FancyBboxPatch(
                (es, y - h / 2), ee - es, h,
                boxstyle='square,pad=0',
                facecolor=col, edgecolor='white', linewidth=0.3,
                zorder=3
            )
            ax.add_patch(rect)

        # Transcript label (truncated)
        label = tx_id.split('.')[0][-20:]
        ax.text(max(tx_s, region_start), y + exon_h / 2 + 0.05,
                label, fontsize=6, va='bottom', ha='left',
                color='#555', clip_on=True)


def _draw_editing_sites(
    ax, sites_by_condition: Dict[str, List[dict]],
    region_start: int, region_end: int,
    condition_colors: Dict[str, str]
):
    """
    Draw one lollipop row per condition.
    Stick height = edit_freq%; dot size encodes fold_change.
    """
    conditions = list(sites_by_condition.keys())
    n = len(conditions)
    row_height = 1.0
    ax.set_xlim(region_start, region_end)
    ax.set_ylim(-0.3, n * row_height)
    ax.axis('off')

    if n == 0:
        ax.text(0.5, 0.5, 'No editing sites in region',
                transform=ax.transAxes, ha='center', va='center',
                color='#888', fontsize=9)
        return

    # Y scale: edit_freq 0–100 mapped to 0–row_height * 0.8
    max_freq = max(
        (s['edit_freq'] for sites in sites_by_condition.values() for s in sites),
        default=100
    )
    max_freq = max(max_freq, 10)

    for row_i, cond in enumerate(conditions):
        y_base = row_i * row_height
        sites = sites_by_condition[cond]
        color = condition_colors.get(cond, _DEFAULT_CYCLE[row_i % len(_DEFAULT_CYCLE)])

        # Row background label
        ax.text(region_start, y_base + row_height * 0.5,
                f' {cond}', fontsize=8, va='center', ha='left',
                color=color, fontweight='bold', clip_on=True)

        # Baseline
        ax.hlines(y_base, region_start, region_end,
                  colors='#ecf0f1', linewidths=0.5, zorder=1)

        for site in sites:
            x   = site['pos']
            h   = (site['edit_freq'] / max_freq) * row_height * 0.85
            dot_sz = max(10, min(80, site['fold_change'] * 8))

            # Lollipop stick
            ax.vlines(x, y_base, y_base + h,
                      colors=color, linewidths=0.9, alpha=0.8, zorder=2)
            # Dot
            ax.scatter(x, y_base + h, s=dot_sz,
                       c=color, edgecolors='white', linewidths=0.4,
                       zorder=3, alpha=0.9)

    # Y-scale annotation (right side)
    ax.text(region_end, 0, f' {max_freq:.0f}%',
            fontsize=6, va='bottom', ha='left', color='#888')
    ax.text(region_end, 0, ' 0%',
            fontsize=6, va='top', ha='left', color='#888')


def _draw_coverage(
    ax, positions: np.ndarray, coverage: np.ndarray,
    color: str, label: str, total_mapped: int,
    rpm: bool = True, ymax: Optional[float] = None
):
    """Draw a single filled-area coverage track."""
    if rpm and total_mapped > 0:
        cov = coverage / total_mapped * 1e6
        ylabel = 'RPM'
    else:
        cov = coverage.astype(float)
        ylabel = 'Depth'

    ax.fill_between(positions, cov, color=color, alpha=0.65, linewidth=0)
    ax.plot(positions, cov, color=color, linewidth=0.6, alpha=0.9)
    ax.set_xlim(positions[0], positions[-1])

    if ymax:
        ax.set_ylim(0, ymax)
    else:
        ax.set_ylim(0, max(cov.max() * 1.1, 1))

    ax.set_ylabel(label, fontsize=7, rotation=0, labelpad=2,
                  va='center', ha='right')
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    ax.tick_params(axis='y', labelsize=6)
    ax.tick_params(axis='x', labelbottom=False, length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['right'].set_color('#bbb')

    # Max coverage annotation
    peak = cov.max()
    ax.text(0.99, 0.92, f'{peak:.1f}', transform=ax.transAxes,
            fontsize=6, ha='right', va='top', color=color)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Genome-browser-style track plot for HyperTRIBE sites of interest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    loc = parser.add_mutually_exclusive_group(required=True)
    loc.add_argument('--gene', help='Gene name (looked up in GTF)')
    loc.add_argument('--region', help='Genomic region, e.g. chr6:163500000-163650000')

    parser.add_argument('--gtf', help='GTF annotation file (required with --gene)')
    parser.add_argument(
        '--bam', action='append', dest='bams', metavar='LABEL:GROUP:PATH',
        help='BAM file with label and group. Repeat for each sample. '
             'Group controls color (control/wt/s161/s206/dhx15 get fixed colors).'
    )
    parser.add_argument(
        '--editing-sites', action='append', dest='editing_sites',
        metavar='LABEL:PATH',
        help='Annotated editing sites BED for one condition. Repeat per condition.'
    )
    parser.add_argument('--output', required=True, help='Output PDF (or PNG/SVG) path')
    parser.add_argument('--padding', type=int, default=500,
                        help='bp padding around gene (default: 500)')
    parser.add_argument('--no-rpm', action='store_true',
                        help='Show raw depth instead of RPM-normalized coverage')
    parser.add_argument('--ymax', type=float, default=None,
                        help='Fix y-axis max for all coverage tracks (default: auto per track)')
    parser.add_argument('--bin-size', type=int, default=0,
                        help='Bin coverage into N-bp bins for speed (0 = auto-select)')
    parser.add_argument('--max-transcripts', type=int, default=3,
                        help='Maximum transcripts to show in gene model (default: 3)')
    parser.add_argument('--min-mapq', type=int, default=10,
                        help='Minimum mapping quality for coverage (default: 10)')
    args = parser.parse_args()

    # ── Resolve region ────────────────────────────────────────────────────────
    if args.gene:
        if not args.gtf:
            parser.error('--gtf is required when using --gene')
        chrom, g_start, g_end, strand, transcripts = parse_gene_region(
            args.gtf, args.gene
        )
        region_start = max(0, g_start - args.padding)
        region_end   = g_end + args.padding
        title_gene   = args.gene
    else:
        chrom, region_start, region_end = parse_region(args.region)
        strand = '.'
        transcripts = {}
        title_gene = args.region
        if args.gtf:
            # Try to annotate nearby gene if GTF provided
            try:
                # We don't know gene name; skip model
                pass
            except Exception:
                pass

    region_span = region_end - region_start
    logger.info(f"Region: {chrom}:{region_start}-{region_end} ({region_span:,} bp)")

    # ── Auto bin size ─────────────────────────────────────────────────────────
    bin_size = args.bin_size
    if bin_size == 0:
        if region_span <= 5_000:
            bin_size = 1
        elif region_span <= 50_000:
            bin_size = 5
        elif region_span <= 200_000:
            bin_size = 20
        else:
            bin_size = 50
    logger.info(f"Coverage bin size: {bin_size} bp")

    # ── Parse BAM specs ───────────────────────────────────────────────────────
    bam_specs = []
    group_color_map: Dict[str, str] = {}
    default_idx = 0
    for spec in (args.bams or []):
        parts = spec.split(':', 2)
        if len(parts) != 3:
            parser.error(f"--bam must be 'Label:Group:Path', got: {spec!r}")
        label, group, path = parts
        group_lc = group.lower().replace('-', '_').replace(' ', '_')
        if group_lc not in group_color_map:
            color = _GROUP_COLORS.get(group_lc,
                                       _DEFAULT_CYCLE[default_idx % len(_DEFAULT_CYCLE)])
            group_color_map[group_lc] = color
            if group_lc not in _GROUP_COLORS:
                default_idx += 1
        bam_specs.append({'label': label, 'group': group_lc,
                          'path': path, 'color': group_color_map[group_lc]})
        if not Path(path).exists():
            logger.warning(f"BAM not found: {path}")

    # ── Parse editing site specs ──────────────────────────────────────────────
    editing_specs = []
    for spec in (args.editing_sites or []):
        parts = spec.split(':', 1)
        if len(parts) != 2:
            parser.error(f"--editing-sites must be 'Label:Path', got: {spec!r}")
        label, path = parts
        editing_specs.append({'label': label, 'path': path})

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading coverage from BAMs...")
    coverage_data = []
    for spec in bam_specs:
        logger.info(f"  {spec['label']}: {spec['path']}")
        try:
            pos, cov = get_coverage(
                spec['path'], chrom, region_start, region_end,
                min_mapq=args.min_mapq, bin_size=bin_size
            )
            total_mapped = get_total_mapped(spec['path'])
            coverage_data.append({**spec, 'positions': pos, 'coverage': cov,
                                   'total_mapped': total_mapped})
            logger.info(f"    {total_mapped:,} mapped reads, "
                        f"peak coverage {cov.max():.0f}")
        except Exception as e:
            logger.warning(f"    Failed: {e}")
            coverage_data.append({**spec, 'positions': None, 'coverage': None,
                                   'total_mapped': 0})

    logger.info("Loading editing sites...")
    sites_by_condition: Dict[str, List[dict]] = {}
    cond_colors: Dict[str, str] = {}
    default_idx = 0
    for spec in editing_specs:
        cond = spec['label']
        cond_lc = cond.lower().replace(' ', '_').replace('-', '_')
        sites = load_editing_sites(spec['path'], chrom, region_start, region_end)
        sites_by_condition[cond] = sites
        color = _GROUP_COLORS.get(cond_lc,
                                   _DEFAULT_CYCLE[default_idx % len(_DEFAULT_CYCLE)])
        cond_colors[cond] = color
        default_idx += 1
        logger.info(f"  {cond}: {len(sites)} sites in region")

    # ── Build figure layout ───────────────────────────────────────────────────
    has_model   = bool(transcripts)
    has_editing = bool(sites_by_condition)
    n_cov       = len(coverage_data)
    n_editing   = len(sites_by_condition)

    height_ratios = []
    panel_names   = []

    if has_model:
        n_tx = min(len(transcripts), args.max_transcripts)
        height_ratios.append(max(0.8, n_tx * 0.35))
        panel_names.append('model')

    if has_editing:
        height_ratios.append(max(1.0, n_editing * 0.6))
        panel_names.append('editing')

    for _ in coverage_data:
        height_ratios.append(1.0)
        panel_names.append('coverage')

    # X-axis tick panel at the bottom
    height_ratios.append(0.25)
    panel_names.append('xaxis')

    total_height = sum(height_ratios) * 1.8 + 1.0
    fig = plt.figure(figsize=(14, max(4, total_height)))
    gs = gridspec.GridSpec(
        len(height_ratios), 1,
        figure=fig, hspace=0.04,
        height_ratios=height_ratios
    )

    axes = [fig.add_subplot(gs[i]) for i in range(len(height_ratios))]

    # Share x-axis across all panels
    for ax in axes[1:]:
        ax.sharex(axes[0])

    # ── Draw panels ───────────────────────────────────────────────────────────
    ax_idx = 0

    if has_model:
        _draw_gene_model(
            axes[ax_idx], transcripts,
            region_start, region_end, strand,
            max_tx=args.max_transcripts
        )
        axes[ax_idx].set_ylabel('Gene\nmodel', fontsize=7, rotation=0,
                                 labelpad=30, va='center')
        ax_idx += 1

    if has_editing:
        _draw_editing_sites(
            axes[ax_idx], sites_by_condition,
            region_start, region_end, cond_colors
        )
        axes[ax_idx].set_ylabel('Editing\nsites', fontsize=7, rotation=0,
                                 labelpad=30, va='center')
        ax_idx += 1

    # ── Shared ymax for coverage ───────────────────────────────────────────────
    if args.ymax is None and not args.no_rpm:
        # Auto-shared ymax: set all coverage tracks to same scale within groups
        # (keep auto-per-track by default for better visibility)
        shared_ymax = None
    else:
        shared_ymax = args.ymax

    for cd in coverage_data:
        ax = axes[ax_idx]
        if cd['coverage'] is not None:
            _draw_coverage(
                ax, cd['positions'], cd['coverage'],
                color=cd['color'], label=cd['label'],
                total_mapped=cd['total_mapped'],
                rpm=not args.no_rpm,
                ymax=shared_ymax
            )
            # Group separator line
            ax.axhline(0, color='#ddd', linewidth=0.5)
        else:
            ax.text(0.5, 0.5, f"{cd['label']}: no data",
                    transform=ax.transAxes, ha='center', va='center',
                    color='#aaa', fontsize=8)
            ax.set_yticks([])
        ax_idx += 1

    # ── X-axis ticks ──────────────────────────────────────────────────────────
    ax_x = axes[ax_idx]
    ax_x.set_xlim(region_start, region_end)
    ax_x.set_ylim(0, 1)
    ax_x.axis('off')
    # Draw custom tick labels
    ticks = np.linspace(region_start, region_end, 6)
    for t in ticks:
        ax_x.text(t, 0.8, f'{int(t):,}', fontsize=7,
                  ha='center', va='top', color='#555')
    ax_x.text(
        (region_start + region_end) / 2, 0.1,
        f'{chrom}  ({region_span:,} bp)',
        ha='center', va='bottom', fontsize=8, color='#777'
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = []
    seen = set()
    for cd in coverage_data:
        key = (cd['label'], cd['color'])
        if key not in seen:
            legend_patches.append(
                mpatches.Patch(facecolor=cd['color'], label=cd['label'], alpha=0.8)
            )
            seen.add(key)
    if legend_patches:
        axes[0].legend(
            handles=legend_patches, loc='upper right',
            fontsize=7, framealpha=0.85, ncol=min(len(legend_patches), 4),
            handlelength=1.2
        )

    # ── Editing site legend ───────────────────────────────────────────────────
    if has_editing:
        editing_legend = [
            mpatches.Patch(facecolor=cond_colors[c], label=c, alpha=0.85)
            for c in sites_by_condition
        ]
        # Add dot size legend note
        from matplotlib.lines import Line2D
        editing_legend += [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#555',
                   markersize=4, label='dot size ∝ fold change')
        ]
        panel_edit_idx = 1 if has_model else 0
        axes[panel_edit_idx].legend(
            handles=editing_legend, loc='upper right',
            fontsize=7, framealpha=0.85, ncol=min(len(editing_legend), 4)
        )

    # ── Title ──────────────────────────────────────────────────────────────────
    strand_str = f' ({strand})' if strand != '.' else ''
    fig.suptitle(
        f'{title_gene}{strand_str}   {chrom}:{region_start:,}–{region_end:,}',
        fontsize=12, fontweight='bold', y=1.002
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
