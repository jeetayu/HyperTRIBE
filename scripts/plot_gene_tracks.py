#!/home/biswasj/miniconda3/envs/hypertribe/bin/python3
"""
Gene Track Visualizer for HyperTRIBE
======================================
Renders a genome-browser-style figure for a gene of interest with:
  - Gene model (exons, UTRs, intron arrows, strand)
  - Editing site lollipops per condition (height = edit frequency %)
  - Per-replicate RPM coverage filled-area tracks

Introns are compressed to --intron-scale fraction of their actual size
(default 0.05 = 5%) to reduce whitespace. Compressed regions are shaded
light gray. Use --intron-scale 1.0 to disable compression.

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

import gzip
import numpy as np
import pysam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

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
_COMPRESS_COLOR = '#f4f4f4'   # shading for compressed intron regions


# ── Coordinate transform (intron compression) ─────────────────────────────────

def _merge_intervals(intervals):
    if not intervals:
        return []
    sorted_ivs = sorted(intervals)
    merged = [list(sorted_ivs[0])]
    for s, e in sorted_ivs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


class CoordTransform:
    """
    Maps genomic coordinates to compressed display coordinates.
    Exonic regions render at 1:1 scale; intronic/intergenic regions are
    compressed to max(min_intron_bp, genomic_width * intron_scale) display units.
    """

    def __init__(self, exon_intervals, region_start, region_end,
                 intron_scale=0.05, min_intron_bp=300):
        # Clip exons to region and merge
        clipped = [
            (max(s, region_start), min(e, region_end))
            for s, e in exon_intervals
            if s < region_end and e > region_start
        ]
        exons = _merge_intervals(clipped)

        # Build alternating gap/exon segments spanning [region_start, region_end]
        segs = []   # (gstart, gend, is_exon)
        pos = region_start
        for es, ee in exons:
            if es > pos:
                segs.append((pos, es, False))
            if ee > es:
                segs.append((es, ee, True))
            pos = ee
        if pos < region_end:
            segs.append((pos, region_end, False))

        # Compute display widths
        d_widths = []
        for gs, ge, is_exon in segs:
            w = ge - gs
            if is_exon:
                d_widths.append(float(w))
            else:
                d_widths.append(float(max(min_intron_bp, w * intron_scale)))

        # Cumulative display positions
        self._segs = segs
        self._d_starts = []
        cum = 0.0
        for w in d_widths:
            self._d_starts.append(cum)
            cum += w
        self._d_ends = [s + w for s, w in zip(self._d_starts, d_widths)]
        self.total = cum
        self.xlim = (0.0, cum)

        # Store compressed segment display ranges for shading
        self.compressed_segs = [
            (self._d_starts[i], self._d_ends[i])
            for i, (_, _, is_exon) in enumerate(segs) if not is_exon
        ]

        self._intron_scale = intron_scale
        self._region_start = region_start
        self._region_end = region_end

    def to_display(self, gpos):
        gpos = float(gpos)
        for i, (gs, ge, _) in enumerate(self._segs):
            if gs <= gpos <= ge:
                span = ge - gs
                frac = (gpos - gs) / span if span > 0 else 0.0
                return self._d_starts[i] + frac * (self._d_ends[i] - self._d_starts[i])
        # Clamp to edges
        if gpos <= self._region_start:
            return 0.0
        return self.total

    def to_display_arr(self, arr):
        out = np.zeros(len(arr), dtype=float)
        for i, (gs, ge, _) in enumerate(self._segs):
            mask = (arr >= gs) & (arr <= ge)
            if not mask.any():
                continue
            span = ge - gs
            frac = (arr[mask] - gs) / span if span > 0 else np.zeros(mask.sum())
            out[mask] = self._d_starts[i] + frac * (self._d_ends[i] - self._d_starts[i])
        return out

    def to_genomic(self, dpos):
        dpos = float(dpos)
        for i in range(len(self._segs)):
            ds, de = self._d_starts[i], self._d_ends[i]
            if ds <= dpos <= de:
                gs, ge, _ = self._segs[i]
                span = de - ds
                frac = (dpos - ds) / span if span > 0 else 0.0
                return int(gs + frac * (ge - gs))
        return int(dpos)

    def tick_positions(self, n=7):
        """Return (display_pos, genomic_label_str) for sensible x-axis ticks."""
        exon_segs = [
            (self._d_starts[i], self._d_ends[i], gs, ge)
            for i, (gs, ge, is_exon) in enumerate(self._segs) if is_exon
        ]
        ticks = []
        if exon_segs:
            # Evenly sample exon centers
            step = max(1, len(exon_segs) // n)
            for ds, de, gs, ge in exon_segs[::step][:n]:
                ticks.append(((ds + de) / 2, f'{int((gs + ge) / 2):,}'))
        if len(ticks) < 3:
            # Fall back: uniform in display space
            for d in np.linspace(0, self.total, n):
                ticks.append((d, f'{self.to_genomic(d):,}'))
        return ticks[:n]


def _tx(val, transform):
    """Apply coord transform to a scalar or numpy array; identity if transform is None."""
    if transform is None:
        return val
    if isinstance(val, np.ndarray):
        return transform.to_display_arr(val)
    return transform.to_display(val)


def _xlim(region_start, region_end, transform):
    return transform.xlim if transform else (region_start, region_end)


def _shade_compressed(axes_list, transform):
    """Draw light gray background on all panels for compressed intron regions."""
    if transform is None:
        return
    for ax in axes_list:
        ylim = ax.get_ylim()
        for ds, de in transform.compressed_segs:
            ax.axvspan(ds, de, ymin=0, ymax=1,
                       color=_COMPRESS_COLOR, zorder=0, linewidth=0)
        ax.set_ylim(ylim)


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

    # Derive gene boundaries from exon extents (handles GTFs without gene/transcript features)
    # Use exon lists rather than t['start']/t['end'], which only reflect the first exon seen.
    all_starts = []
    all_ends   = []
    for t in transcripts.values():
        if t['exons']:
            all_starts.append(min(e[0] for e in t['exons']))
            all_ends.append(max(e[1] for e in t['exons']))
        else:
            all_starts.append(t['start'])
            all_ends.append(t['end'])

    if gene_start is None:
        gene_start = min(all_starts)
        gene_end   = max(all_ends)
        gene_chrom = next(iter(transcripts.values())).get('chrom', None)
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
    else:
        # Even when gene feature exists, extend to exon extremes in case they exceed gene record
        if all_starts:
            gene_start = min(gene_start, min(all_starts))
            gene_end   = max(gene_end,   max(all_ends))

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


# ── eCLIP peaks ───────────────────────────────────────────────────────────────

def load_eclip_peaks(
    tsv_path: str, chrom: str, start: int, end: int,
    q_cutoff: float = 0.05, min_l2or: float = 2.0
) -> List[dict]:
    """Return eCLIP peaks in the region from a Yeo-lab reproducible_enriched_windows TSV.gz."""
    peaks = []
    opener = gzip.open if tsv_path.endswith('.gz') else open
    with opener(tsv_path, 'rt') as fh:
        header = fh.readline().rstrip('\n').split('\t')
        idx = {c: i for i, c in enumerate(header)}
        for line in fh:
            f = line.rstrip('\n').split('\t')
            if f[idx['chr']] != chrom:
                continue
            ps, pe = int(f[idx['start']]), int(f[idx['end']])
            if pe < start or ps >= end:
                continue
            try:
                q = float(f[idx['q_min']])
                l2or = float(f[idx['enrichment_l2or_mean']])
                if q >= q_cutoff or l2or < min_l2or:
                    continue
                peaks.append({
                    'start':  max(ps, start),
                    'end':    min(pe, end),
                    'strand': f[idx.get('strand', 5)],
                    'gene':   f[idx['gene_name']],
                    'l2or':   l2or,
                    'q':      q,
                })
            except (ValueError, IndexError):
                continue
    return peaks


def _draw_eclip_peaks(
    ax, peaks_by_dataset: Dict[str, List[dict]],
    region_start: int, region_end: int,
    dataset_colors: Dict[str, str],
    transform=None
):
    """Draw one row per eCLIP dataset; peaks as colored rectangles, height = l2or (capped at 8)."""
    datasets = list(peaks_by_dataset.keys())
    n = len(datasets)
    row_height = 1.0
    xl0, xl1 = _xlim(region_start, region_end, transform)
    ax.set_xlim(xl0, xl1)
    ax.set_ylim(-0.1, n * row_height)
    ax.axis('off')

    if n == 0:
        ax.text(0.5, 0.5, 'No eCLIP peaks in region',
                transform=ax.transAxes, ha='center', va='center',
                color='#888', fontsize=9)
        return

    MAX_L2OR = 8.0

    for row_i, ds in enumerate(datasets):
        y_base = row_i * row_height
        peaks = peaks_by_dataset[ds]
        color = dataset_colors.get(ds, _DEFAULT_CYCLE[row_i % len(_DEFAULT_CYCLE)])

        ax.text(xl0, y_base + row_height * 0.5,
                f' {ds}', fontsize=8, va='center', ha='left',
                color=color, fontweight='bold', clip_on=True)

        ax.hlines(y_base, xl0, xl1,
                  colors='#ecf0f1', linewidths=0.5, zorder=1)

        for pk in peaks:
            ps_d = _tx(pk['start'], transform)
            pe_d = _tx(pk['end'], transform)
            h = min(pk['l2or'], MAX_L2OR) / MAX_L2OR * row_height * 0.85
            rect = mpatches.FancyBboxPatch(
                (ps_d, y_base), pe_d - ps_d, h,
                boxstyle='square,pad=0',
                facecolor=color, edgecolor='none',
                alpha=0.75, zorder=2
            )
            ax.add_patch(rect)

        if peaks:
            max_l2or = max(p['l2or'] for p in peaks)
            ax.text(xl1, y_base + row_height * 0.85,
                    f' max l2or={max_l2or:.1f}',
                    fontsize=6, va='top', ha='left', color=color)

    ax.text(xl1, 0, f' l2or={MAX_L2OR:.0f}',
            fontsize=6, va='bottom', ha='left', color='#888')


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_gene_model(
    ax, transcripts: dict, region_start: int, region_end: int, strand: str,
    max_tx: int = 5, transform=None
):
    """Draw gene model tracks (exons, UTRs, intron lines with arrows)."""
    xl0, xl1 = _xlim(region_start, region_end, transform)
    ax.set_xlim(xl0, xl1)
    ax.set_ylim(-0.5, max_tx - 0.5)
    ax.axis('off')

    # Select transcripts to display: pick up to max_tx longest
    sorted_tx = sorted(
        transcripts.items(),
        key=lambda kv: sum(e - s for s, e in kv[1]['exons']),
        reverse=True
    )[:max_tx]

    n = len(sorted_tx)
    # Fixed display-space arrow parameters
    display_span = xl1 - xl0
    arrow_dx_d = display_span / 120.0

    for row, (tx_id, tx) in enumerate(sorted_tx):
        y = (n - 1 - row)
        exons = sorted(tx['exons'])
        utrs  = set()
        for us, ue in tx.get('utrs', []):
            for pos in range(us, ue):
                utrs.add(pos)

        if not exons:
            continue

        tx_s = exons[0][0]
        tx_e = exons[-1][1]

        # Intron backbone line in display coords
        line_s = _tx(max(tx_s, region_start), transform)
        line_e = _tx(min(tx_e, region_end), transform)
        ax.hlines(y, line_s, line_e,
                  colors=_INTRON_COLOR, linewidths=1.0, zorder=1)

        # Strand direction arrows — evenly spaced in display space
        arrow_spacing_d = max(display_span / 20.0, arrow_dx_d * 2)
        for d_arrow in np.arange(line_s + arrow_spacing_d / 2, line_e, arrow_spacing_d):
            g_arrow = transform.to_genomic(d_arrow) if transform else int(d_arrow)
            in_exon = any(s <= g_arrow < e for s, e in exons)
            if in_exon:
                continue
            dx = arrow_dx_d if strand == '+' else -arrow_dx_d
            ax.annotate(
                '', xy=(d_arrow + dx, y), xytext=(d_arrow, y),
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
            es_c = max(es, region_start)
            ee_c = min(ee, region_end)
            if es_c >= ee_c:
                continue
            es_d = _tx(es_c, transform)
            ee_d = _tx(ee_c, transform)
            exon_positions = set(range(es_c, min(ee_c, es_c + 10000)))
            is_utr = exon_positions.issubset(utrs) and len(utrs) > 0
            h   = utr_h if is_utr else exon_h
            col = _UTR_COLOR if is_utr else _EXON_COLOR
            rect = mpatches.FancyBboxPatch(
                (es_d, y - h / 2), ee_d - es_d, h,
                boxstyle='square,pad=0',
                facecolor=col, edgecolor='white', linewidth=0.3,
                zorder=3
            )
            ax.add_patch(rect)

        # Transcript label
        label = tx_id.split('.')[0][-20:]
        ax.text(_tx(max(tx_s, region_start), transform), y + exon_h / 2 + 0.05,
                label, fontsize=6, va='bottom', ha='left',
                color='#555', clip_on=True)


def _draw_editing_sites(
    ax, sites_by_condition: Dict[str, List[dict]],
    region_start: int, region_end: int,
    condition_colors: Dict[str, str],
    transform=None
):
    """
    Draw one lollipop row per condition.
    Stick height = edit_freq%; dot size encodes fold_change.
    """
    conditions = list(sites_by_condition.keys())
    n = len(conditions)
    row_height = 1.0
    xl0, xl1 = _xlim(region_start, region_end, transform)
    ax.set_xlim(xl0, xl1)
    ax.set_ylim(-0.3, n * row_height)
    ax.axis('off')

    if n == 0:
        ax.text(0.5, 0.5, 'No editing sites in region',
                transform=ax.transAxes, ha='center', va='center',
                color='#888', fontsize=9)
        return

    max_freq = max(
        (s['edit_freq'] for sites in sites_by_condition.values() for s in sites),
        default=100
    )
    max_freq = max(max_freq, 10)

    for row_i, cond in enumerate(conditions):
        y_base = row_i * row_height
        sites = sites_by_condition[cond]
        color = condition_colors.get(cond, _DEFAULT_CYCLE[row_i % len(_DEFAULT_CYCLE)])

        ax.text(xl0, y_base + row_height * 0.5,
                f' {cond}', fontsize=8, va='center', ha='left',
                color=color, fontweight='bold', clip_on=True)

        ax.hlines(y_base, xl0, xl1,
                  colors='#ecf0f1', linewidths=0.5, zorder=1)

        for site in sites:
            x_d = _tx(site['pos'], transform)
            h   = (site['edit_freq'] / max_freq) * row_height * 0.85
            dot_sz = max(10, min(80, site['fold_change'] * 8))

            ax.vlines(x_d, y_base, y_base + h,
                      colors=color, linewidths=0.9, alpha=0.8, zorder=2)
            ax.scatter(x_d, y_base + h, s=dot_sz,
                       c=color, edgecolors='white', linewidths=0.4,
                       zorder=3, alpha=0.9)

    ax.text(xl1, 0, f' {max_freq:.0f}%',
            fontsize=6, va='bottom', ha='left', color='#888')
    ax.text(xl1, 0, ' 0%',
            fontsize=6, va='top', ha='left', color='#888')


def _draw_coverage(
    ax, positions: np.ndarray, coverage: np.ndarray,
    color: str, label: str, total_mapped: int,
    rpm: bool = True, ymax: Optional[float] = None,
    transform=None, pre_normalized: bool = False
):
    """Draw a single filled-area coverage track."""
    if pre_normalized:
        cov = coverage.astype(float)
        ylabel = 'RPM'
    elif rpm and total_mapped > 0:
        cov = coverage / total_mapped * 1e6
        ylabel = 'RPM'
    else:
        cov = coverage.astype(float)
        ylabel = 'Depth'

    pos_d = _tx(positions, transform)

    ax.fill_between(pos_d, cov, color=color, alpha=0.65, linewidth=0)
    ax.plot(pos_d, cov, color=color, linewidth=0.6, alpha=0.9)
    ax.set_xlim(pos_d[0], pos_d[-1])

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
    parser.add_argument(
        '--eclip-peaks', action='append', dest='eclip_peaks',
        metavar='LABEL:PATH',
        help='eCLIP reproducible_enriched_windows TSV(.gz) for one dataset. Repeat per dataset.'
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
    parser.add_argument('--average-replicates', action='store_true',
                        help='Average RPM coverage across replicates within each group '
                             'to show one track per condition instead of one per BAM.')
    parser.add_argument('--eclip-min-l2or', type=float, default=2.0,
                        help='Minimum enrichment_l2or_mean to display eCLIP peaks (default: 2.0)')
    parser.add_argument('--fig-width', type=float, default=24.0,
                        help='Figure width in inches for Illustrator export (default: 24)')
    parser.add_argument('--fig-height-per-track', type=float, default=1.8,
                        help='Height multiplier per track unit (default: 1.8)')
    parser.add_argument(
        '--intron-scale', type=float, default=0.05,
        help='Fraction of actual intron size to display (default: 0.05 = 5%%). '
             'Use 1.0 to disable compression.'
    )
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

    region_span = region_end - region_start
    logger.info(f"Region: {chrom}:{region_start}-{region_end} ({region_span:,} bp)")

    # ── Build coordinate transform ─────────────────────────────────────────────
    transform = None
    if args.intron_scale < 1.0 and transcripts:
        all_exons = [ex for tx in transcripts.values() for ex in tx['exons']]
        if all_exons:
            transform = CoordTransform(
                all_exons, region_start, region_end,
                intron_scale=args.intron_scale,
                min_intron_bp=300
            )
            logger.info(
                f"Intron compression: scale={args.intron_scale}, "
                f"genomic span={region_span:,} bp → "
                f"display span={transform.total:,.0f} units "
                f"({transform.total/region_span*100:.1f}% of original)"
            )

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

    # ── Average replicates within each group (one track per condition) ───────────
    if args.average_replicates:
        group_order: list = []
        group_meta: dict = {}
        for cd in coverage_data:
            g = cd['group']
            if g not in group_meta:
                group_order.append(g)
                group_meta[g] = {
                    'color': cd['color'],
                    'positions': cd['positions'],
                    'rpms': [],
                }
            if cd['coverage'] is not None and cd['total_mapped'] > 0:
                group_meta[g]['rpms'].append(
                    cd['coverage'] / cd['total_mapped'] * 1e6
                )

        averaged: list = []
        for g in group_order:
            gm = group_meta[g]
            avg_cov = np.mean(gm['rpms'], axis=0) if gm['rpms'] else None
            # Build a clean display label from the group key
            display_label = g.upper() if len(g) <= 4 else g.capitalize()
            averaged.append({
                'label': display_label,
                'group': g,
                'color': gm['color'],
                'positions': gm['positions'],
                'coverage': avg_cov,
                'total_mapped': 1,
                'pre_normalized': True,
            })
        coverage_data = averaged
        logger.info(f"  Averaged into {len(coverage_data)} group tracks: "
                    f"{[cd['label'] for cd in coverage_data]}")

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

    # ── Parse eCLIP specs ─────────────────────────────────────────────────────
    eclip_specs = []
    for spec in (args.eclip_peaks or []):
        parts = spec.split(':', 1)
        if len(parts) != 2:
            parser.error(f"--eclip-peaks must be 'Label:Path', got: {spec!r}")
        eclip_specs.append({'label': parts[0], 'path': parts[1]})

    logger.info("Loading eCLIP peaks...")
    # Cell lines: warm reds/oranges; MCF10A normal: gray; V5 constructs: blues
    _eclip_colors = ['#c0392b', '#e67e22', '#8e44ad', '#95a5a6', '#2471a3', '#16a085']
    peaks_by_dataset: Dict[str, List[dict]] = {}
    eclip_ds_colors: Dict[str, str] = {}
    for i, spec in enumerate(eclip_specs):
        ds = spec['label']
        peaks = load_eclip_peaks(spec['path'], chrom, region_start, region_end,
                                  min_l2or=args.eclip_min_l2or)
        peaks_by_dataset[ds] = peaks
        eclip_ds_colors[ds] = _eclip_colors[i % len(_eclip_colors)]
        logger.info(f"  {ds}: {len(peaks)} peaks in region (l2or≥{args.eclip_min_l2or})")

    # ── Build figure layout ───────────────────────────────────────────────────
    has_model   = bool(transcripts)
    has_editing = bool(sites_by_condition)
    has_eclip   = bool(peaks_by_dataset)
    n_cov       = len(coverage_data)
    n_editing   = len(sites_by_condition)
    n_eclip_ds  = len(peaks_by_dataset)

    height_ratios = []
    panel_names   = []

    if has_model:
        n_tx = min(len(transcripts), args.max_transcripts)
        height_ratios.append(max(0.8, n_tx * 0.35))
        panel_names.append('model')

    if has_eclip:
        height_ratios.append(max(0.8, n_eclip_ds * 0.5))
        panel_names.append('eclip')

    if has_editing:
        height_ratios.append(max(1.0, n_editing * 0.6))
        panel_names.append('editing')

    for _ in coverage_data:
        height_ratios.append(1.0)
        panel_names.append('coverage')

    height_ratios.append(0.25)
    panel_names.append('xaxis')

    total_height = sum(height_ratios) * args.fig_height_per_track + 1.0
    fig = plt.figure(figsize=(args.fig_width, max(6, total_height)))
    gs = gridspec.GridSpec(
        len(height_ratios), 1,
        figure=fig, hspace=0.04,
        height_ratios=height_ratios
    )

    axes = [fig.add_subplot(gs[i]) for i in range(len(height_ratios))]

    for ax in axes[1:]:
        ax.sharex(axes[0])

    # ── Draw panels ───────────────────────────────────────────────────────────
    ax_idx = 0

    if has_model:
        _draw_gene_model(
            axes[ax_idx], transcripts,
            region_start, region_end, strand,
            max_tx=args.max_transcripts,
            transform=transform
        )
        axes[ax_idx].set_ylabel('Gene\nmodel', fontsize=7, rotation=0,
                                 labelpad=30, va='center')
        ax_idx += 1

    if has_eclip:
        _draw_eclip_peaks(
            axes[ax_idx], peaks_by_dataset,
            region_start, region_end, eclip_ds_colors,
            transform=transform
        )
        axes[ax_idx].set_ylabel('eCLIP\npeaks', fontsize=7, rotation=0,
                                 labelpad=30, va='center')
        ax_idx += 1

    if has_editing:
        _draw_editing_sites(
            axes[ax_idx], sites_by_condition,
            region_start, region_end, cond_colors,
            transform=transform
        )
        axes[ax_idx].set_ylabel('Editing\nsites', fontsize=7, rotation=0,
                                 labelpad=30, va='center')
        ax_idx += 1

    if args.ymax is None and not args.no_rpm:
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
                ymax=shared_ymax,
                transform=transform,
                pre_normalized=cd.get('pre_normalized', False)
            )
            ax.axhline(0, color='#ddd', linewidth=0.5)
        else:
            ax.text(0.5, 0.5, f"{cd['label']}: no data",
                    transform=ax.transAxes, ha='center', va='center',
                    color='#aaa', fontsize=8)
            ax.set_yticks([])
        ax_idx += 1

    # ── X-axis ticks ──────────────────────────────────────────────────────────
    ax_x = axes[ax_idx]
    xl0, xl1 = _xlim(region_start, region_end, transform)
    ax_x.set_xlim(xl0, xl1)
    ax_x.set_ylim(0, 1)
    ax_x.axis('off')

    if transform:
        ticks = transform.tick_positions(n=7)
        for t_d, t_label in ticks:
            ax_x.text(t_d, 0.8, t_label, fontsize=7,
                      ha='center', va='top', color='#555')
        # Indicate scale compression
        scale_pct = transform.total / region_span * 100
        ax_x.text(
            (xl0 + xl1) / 2, 0.1,
            f'{chrom}  |  introns compressed to {args.intron_scale*100:.0f}%  '
            f'({scale_pct:.0f}% of genomic span shown)',
            ha='center', va='bottom', fontsize=7, color='#999'
        )
    else:
        ticks = np.linspace(region_start, region_end, 6)
        for t in ticks:
            ax_x.text(t, 0.8, f'{int(t):,}', fontsize=7,
                      ha='center', va='top', color='#555')
        ax_x.text(
            (region_start + region_end) / 2, 0.1,
            f'{chrom}  ({region_span:,} bp)',
            ha='center', va='bottom', fontsize=8, color='#777'
        )

    # ── Shade compressed intron regions on all panels ─────────────────────────
    _shade_compressed(axes, transform)

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

    if has_editing:
        editing_legend = [
            mpatches.Patch(facecolor=cond_colors[c], label=c, alpha=0.85)
            for c in sites_by_condition
        ]
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
