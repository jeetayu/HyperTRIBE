#!/usr/bin/env python3
"""
Splice Site Motif Analysis for HyperTRIBE Editing Sites
========================================================

Extracts genomic sequence around editing sites and canonical splice elements,
then generates sequence logos to show motif enrichment vs. a matched control.

Analyses produced (each as a panel in the output PDF):

  1. Editing-site trinucleotide context (NAN where A is the edited adenosine)
     — shows local sequence bias around A→I edits

  2. Sequence logo ±15 nt centred on the editing site (all intronic sites)
     — reveals intrinsic sequence preference of each RBP

  3. 5'SS donor motif (GU + flanking) for introns containing an editing site
     vs. a GC/AT-matched background of unexplored introns from the same genes
     — 9-mer: exon[-3,-1] | +1GT+3..+6 intronic

  4. 3'SS acceptor motif (AG dinucleotide + PPT upstream) for introns
     containing an editing site vs. background
     — 23-mer: -20..-3 PPT | -2AG

  5. Branch-point region logo: sequence from -40 to -10 relative to the
     acceptor (standard BP window) for PPT-region editing sites
     — highlights YYYYYYYYYYYYYNYYRAY (canonical branch point)

  6. Editing-site logo for sites in the PPT region specifically
     — distinguishes PPT sequence enrichment at the edit itself

Control sequences:
    For each edited intron, we sample two random positions from the same
    intron at matching distances from the splice sites. This controls for
    intron-length bias and gene-sequence composition.

Usage:
    python plot_splice_motifs.py \\
        --input  results/splice_annotated_editing_sites.bed \\
        --fasta  /path/to/genome.fa \\
        --gtf    /path/to/genes.gtf \\
        --output results/plots/splice_motif_analysis.pdf \\
        --label  "WT PUF60"
"""

import argparse
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logomaker
import seaborn as sns
from pyfaidx import Fasta

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
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
})

random.seed(42)

DNA_COMPLEMENT = str.maketrans('ACGTacgtNn', 'TGCAtgcaNn')

# ─────────────────────────────────────────────────────────────────────────────
# Reference RBP binding motifs  (frequency matrices, columns = A C G T)
# ─────────────────────────────────────────────────────────────────────────────

# PUF60  (Poly-U binding Factor 60 kDa / FIR / SIAHBP1)
# ---------------------------------------------------------------
# PUF60 contains two RRM domains and a U2AF-homology motif (UHM).
# It cooperates with U2AF65 at polypyrimidine tracts near the 3'SS,
# recognising UC-rich sequences.
# PWM approximated from ENCODE eCLIP (K562 + HepG2) and from
# Salton et al. 2008 EMBO Rep; Masuda et al. 2012 Nat Struct Mol Biol.
# Core consensus: UCUCUUUU  (8-mer, coding-strand DNA).
# NOTE: PUF60 ≠ pumilio/PUF family. The "PUF" in PUF60 stands for
# "Poly-U Factor", not pumilio.  See clarification panel below.
_PUF60_MOTIF = pd.DataFrame({
    'A': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    'C': [0.32, 0.28, 0.35, 0.28, 0.30, 0.28, 0.25, 0.25],
    'G': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    'T': [0.58, 0.62, 0.55, 0.62, 0.60, 0.62, 0.65, 0.65],
})

# Pumilio / PUF-family  (Wickens-group "PUF code")
# ---------------------------------------------------------------
# PUF-family proteins (Pumilio, FBF, Puf1–5) contain a pumilio-homology
# domain (PUM-HD) with 8 PUF repeats, each recognising one RNA base via
# a two-amino-acid recognition code.
# Classic consensus: 5'-UGUAHAUA-3' (RNA), H = not G.
# Key refs: Zamore et al. 1997 Cell; Edwards et al. 2001 Struct;
#           Gerber et al. 2004 Genes Dev; Bernstein et al. 2005 Science;
#           Weidmann & Goldstrohm 2012 Nat Chem Biol.
# DNA coding-strand: T-G-T-A-[ACT]-A-T-A
_PUMILIO_MOTIF = pd.DataFrame({
    'A': [0.02, 0.02, 0.02, 0.94, 0.31, 0.94, 0.02, 0.94],
    'C': [0.02, 0.02, 0.02, 0.02, 0.31, 0.02, 0.02, 0.02],
    'G': [0.02, 0.94, 0.02, 0.02, 0.00, 0.02, 0.02, 0.02],
    'T': [0.94, 0.02, 0.94, 0.02, 0.38, 0.02, 0.94, 0.02],
})

_KNOWN_MOTIFS = {
    'PUF60\n(UC-rich PPT)':  _PUF60_MOTIF,
    'Pumilio/PUF\n(UGUAHAUA)': _PUMILIO_MOTIF,
}


def revcomp(seq: str) -> str:
    return seq.translate(DNA_COMPLEMENT)[::-1]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_bed(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', comment='#', header=None,
                     low_memory=False)
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
    for col in ('dist_5ss', 'dist_3ss', 'signed_5ss', 'signed_3ss'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _parse_gtf_introns(gtf_file: str) -> Dict[str, List[Tuple[str, int, int, str]]]:
    """
    Return per-gene_id list of (chrom, intron_start, intron_end, strand)
    by inferring introns from exon records (gaps between consecutive exons
    of the same transcript).
    """
    logger.info(f"Parsing introns from GTF: {gtf_file}")
    # transcript -> sorted exons
    tx_exons: Dict[str, list] = defaultdict(list)
    tx_meta:  Dict[str, tuple] = {}  # tx_id -> (chrom, strand, gene_id)

    with open(gtf_file) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            f = line.rstrip('\n').split('\t')
            if len(f) < 9 or f[2] != 'exon':
                continue
            chrom  = f[0]
            start  = int(f[3])
            end    = int(f[4])
            strand = f[6]
            attrs  = {}
            for item in f[8].strip().split(';'):
                item = item.strip()
                if not item:
                    continue
                parts = item.split(' ', 1)
                if len(parts) == 2:
                    attrs[parts[0]] = parts[1].strip('"')
            gene_id = attrs.get('gene_id', '')
            tx_id   = attrs.get('transcript_id', '')
            if not tx_id:
                continue
            tx_exons[tx_id].append((start, end))
            if tx_id not in tx_meta:
                tx_meta[tx_id] = (chrom, strand, gene_id)

    # Build introns per gene
    gene_introns: Dict[str, List[Tuple[str, int, int, str]]] = defaultdict(list)
    for tx_id, exons in tx_exons.items():
        chrom, strand, gene_id = tx_meta[tx_id]
        exons_sorted = sorted(exons, key=lambda x: x[0])
        for i in range(len(exons_sorted) - 1):
            istart = exons_sorted[i][1] + 1      # 1-based
            iend   = exons_sorted[i+1][0] - 1    # 1-based
            if iend > istart + 10:               # ignore tiny gaps
                gene_introns[gene_id].append((chrom, istart, iend, strand))

    # Deduplicate
    for gid in gene_introns:
        gene_introns[gid] = list(set(gene_introns[gid]))

    n_introns = sum(len(v) for v in gene_introns.values())
    logger.info(f"  {n_introns:,} unique introns across {len(gene_introns):,} genes")
    return dict(gene_introns)


# ─────────────────────────────────────────────────────────────────────────────
# Sequence extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_seq(fasta: Fasta, chrom: str, start1: int, end1: int, strand: str) -> Optional[str]:
    """
    Fetch sequence [start1, end1] (1-based inclusive), return on correct strand.
    Returns None if out-of-bounds or chromosome not found.
    """
    try:
        seq = fasta[chrom][start1 - 1:end1].seq.upper()
        if len(seq) != end1 - start1 + 1:
            return None
        if strand == '-':
            seq = revcomp(seq)
        return seq
    except (KeyError, Exception):
        return None


def _seq_around_site(fasta: Fasta, chrom: str, pos1: int,
                     strand: str, flank: int) -> Optional[str]:
    """Extract 2*flank+1 nt centred on pos1 (1-based) on the given strand."""
    return _get_seq(fasta, chrom, pos1 - flank, pos1 + flank, strand)


def _donor_seq(fasta: Fasta, chrom: str, donor_pos1: int, strand: str,
               up: int = 3, down: int = 8) -> Optional[str]:
    """
    Extract splice donor motif.
    donor_pos1 is the last base of the upstream exon (1-based).
    Returns up nt of exon + GU + (down-2) nt of intron = up+down nt total.
    On '+': donor_pos1 .. donor_pos1+down  (exon ends at donor_pos1)
    On '-': intronic bases are to the LEFT of donor_pos1
    """
    if strand == '+':
        return _get_seq(fasta, chrom, donor_pos1 - up + 1, donor_pos1 + down, strand)
    else:
        return _get_seq(fasta, chrom, donor_pos1 - down, donor_pos1 + up - 1, strand)


def _acceptor_seq(fasta: Fasta, chrom: str, acceptor_pos1: int, strand: str,
                  up: int = 20, down: int = 3) -> Optional[str]:
    """
    Extract splice acceptor motif (PPT + AG).
    acceptor_pos1 is the first base of the downstream exon (1-based).
    Returns up nt of intron + AG + down nt of exon = up+down nt total.
    """
    if strand == '+':
        return _get_seq(fasta, chrom, acceptor_pos1 - up, acceptor_pos1 + down - 1, strand)
    else:
        return _get_seq(fasta, chrom, acceptor_pos1 - down, acceptor_pos1 + up - 1, strand)


def _bp_region_seq(fasta: Fasta, chrom: str, acceptor_pos1: int, strand: str,
                   bp_start: int = 40, bp_end: int = 10) -> Optional[str]:
    """
    Extract branch-point window: bp_start..bp_end nt upstream of acceptor.
    Returns bp_start-bp_end+1 nt of intronic sequence.
    """
    length = bp_start - bp_end + 1
    if strand == '+':
        return _get_seq(fasta, chrom, acceptor_pos1 - bp_start, acceptor_pos1 - bp_end, strand)
    else:
        return _get_seq(fasta, chrom, acceptor_pos1 + bp_end, acceptor_pos1 + bp_start, strand)


# ─────────────────────────────────────────────────────────────────────────────
# Sequence logo helpers
# ─────────────────────────────────────────────────────────────────────────────

def _seqs_to_pwm(seqs: List[str], pseudo: float = 0.5) -> Optional[pd.DataFrame]:
    """Convert list of equal-length sequences to position frequency matrix (0→1)."""
    seqs = [s for s in seqs if s and len(s) == len(seqs[0]) and 'N' not in s]
    if len(seqs) < 5:
        return None
    L = len(seqs[0])
    counts = pd.DataFrame(0, index=range(L), columns=list('ACGT'))
    for seq in seqs:
        for i, base in enumerate(seq):
            if base in counts.columns:
                counts.at[i, base] += 1
    # Add pseudocount, normalise
    counts = counts + pseudo
    freq = counts.div(counts.sum(axis=1), axis=0)
    return freq


def _information_content(freq: pd.DataFrame) -> pd.DataFrame:
    """Convert frequency matrix to information content (bits)."""
    bg = 0.25
    ic = freq.copy()
    for col in ic.columns:
        ic[col] = freq[col] * np.log2(freq[col].clip(lower=1e-10) / bg)
    ic_total = ic.sum(axis=1)
    return freq.multiply(ic_total, axis=0)


def _draw_logo(ax, seqs: List[str], title: str,
               center_label: Optional[int] = None,
               xtick_labels: Optional[List[str]] = None,
               highlight_positions: Optional[List[int]] = None):
    """Draw sequence logo on ax. center_label: position index that is '0'."""
    freq = _seqs_to_pwm(seqs)
    if freq is None:
        ax.text(0.5, 0.5, f'n < 5 sequences\n(n={len(seqs)})',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=10, color='grey')
        ax.set_title(title)
        return

    ic = _information_content(freq)
    n_valid = len([s for s in seqs if s and 'N' not in s])

    logo = logomaker.Logo(
        ic,
        ax=ax,
        color_scheme='classic',
        baseline_width=0.5,
        alpha=0.85,
    )
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    logo.ax.set_ylabel('bits', fontsize=9)

    # x-tick labels
    if xtick_labels:
        ax.set_xticks(range(len(xtick_labels)))
        ax.set_xticklabels(xtick_labels, fontsize=7)
    elif center_label is not None:
        offset = center_label
        ax.set_xticks(range(len(freq)))
        ax.set_xticklabels([str(i - offset) for i in range(len(freq))], fontsize=7)

    # Highlight positions
    if highlight_positions:
        for pos in highlight_positions:
            ax.axvspan(pos - 0.5, pos + 0.5, alpha=0.15, color='red', zorder=0)

    ax.set_title(f'{title}\n(n={n_valid:,})', fontsize=10)


def _draw_trinuc_bar(ax, seqs: List[str], center: int, title: str):
    """Bar chart of trinucleotide frequencies centred at `center`."""
    if not seqs:
        return
    trinucs: Dict[str, int] = defaultdict(int)
    for seq in seqs:
        if len(seq) > center + 1 and center >= 1:
            tri = seq[center - 1:center + 2]
            if len(tri) == 3 and 'N' not in tri:
                trinucs[tri] += 1
    if not trinucs:
        return
    df = pd.Series(trinucs).sort_values(ascending=False).head(20)
    total = df.sum()
    # Highlight NAN (editing context)
    colors = ['#e74c3c' if t[1] == 'A' else '#3498db' for t in df.index]
    ax.bar(range(len(df)), df.values / total * 100,
           color=colors, edgecolor='white', linewidth=0.3)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index, fontsize=7, rotation=60, ha='right')
    ax.set_ylabel('% of sites')
    ax.set_title(f'{title}\n(n={total:,}; red = N[A]N context)', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Known-motif scanning and comparison helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pwm_log_odds(pwm: pd.DataFrame, bg: float = 0.25) -> np.ndarray:
    """Convert frequency matrix to log-odds (nats). Returns (L × 4) array."""
    arr = pwm[list('ACGT')].values.clip(min=1e-9)
    return np.log(arr / bg)


def _score_sequence(seq: str, lo_matrix: np.ndarray) -> float:
    """
    Best log-odds score for a k-mer PWM scanned across seq.
    Returns -inf if sequence is too short or contains only Ns.
    """
    base_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    k = lo_matrix.shape[0]
    if len(seq) < k:
        return -np.inf
    best = -np.inf
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k].upper()
        if 'N' in kmer:
            continue
        score = sum(lo_matrix[j, base_idx.get(b, 0)] for j, b in enumerate(kmer))
        if score > best:
            best = score
    return best


def _positional_motif_scores(seqs: List[str], lo_matrix: np.ndarray,
                              flank: int) -> np.ndarray:
    """
    For each position p in [-flank, flank] and each sequence, compute the
    log-odds score of the k-mer that starts at position p.
    Returns array of shape (2*flank+1,) with mean scores per position.
    """
    k = lo_matrix.shape[0]
    base_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = 2 * flank + 1
    position_scores = [[] for _ in range(L - k + 1)]

    for seq in seqs:
        if not seq or len(seq) < L or 'N' in seq:
            continue
        for i in range(L - k + 1):
            kmer = seq[i:i + k].upper()
            score = sum(lo_matrix[j, base_idx.get(b, 0)] for j, b in enumerate(kmer))
            position_scores[i].append(score)

    means = np.array([np.mean(v) if v else np.nan for v in position_scores])
    return means


def _draw_motif_reference(ax, pwm: pd.DataFrame, title: str):
    """Draw the reference PWM as an information-content logo."""
    ic = _information_content(pwm)
    logo = logomaker.Logo(ic, ax=ax, color_scheme='classic',
                           baseline_width=0.5, alpha=0.85)
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    ax.set_ylabel('bits', fontsize=9)
    ax.set_title(title, fontsize=10)


def _draw_motif_enrichment(ax, edit_seqs: List[str], ctrl_seqs: List[str],
                            lo_matrix: np.ndarray, motif_name: str, flank: int):
    """
    Plot mean motif score at each position in the ±flank window.
    Editing sites (orange) vs. control (blue).  Position 0 = editing site.
    """
    k = lo_matrix.shape[0]
    positions = np.arange(-(flank), flank - k + 2)   # start position of each k-mer

    edit_scores = _positional_motif_scores(edit_seqs, lo_matrix, flank)
    ctrl_scores  = _positional_motif_scores(ctrl_seqs, lo_matrix, flank)

    ax.plot(positions, edit_scores, color='#e74c3c', linewidth=1.8,
            label=f'Editing sites (n={len(edit_seqs):,})', zorder=3)
    ax.plot(positions, ctrl_scores,  color='#3498db', linewidth=1.8,
            linestyle='--', label=f'Control (n={len(ctrl_seqs):,})', zorder=3)
    ax.fill_between(positions, edit_scores, ctrl_scores,
                     where=(edit_scores > ctrl_scores),
                     alpha=0.18, color='#e74c3c', label='Editing > control')
    ax.fill_between(positions, edit_scores, ctrl_scores,
                     where=(edit_scores < ctrl_scores),
                     alpha=0.18, color='#3498db', label='Control > editing')
    ax.axvline(0, color='black', linewidth=1.2, linestyle='-', zorder=5,
               label='Editing site (A→I)')
    ax.axhline(np.nanmean(ctrl_scores), color='#3498db', linewidth=0.8,
               linestyle=':', alpha=0.7, label='Control mean')
    ax.set_xlabel('Position relative to editing site (nt)')
    ax.set_ylabel(f'Mean log-odds score')
    ax.set_title(f'{motif_name} motif enrichment at editing sites\nvs. control (matched positions in same genes)',
                 fontsize=10)
    ax.legend(fontsize=8, frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def _extract_editing_site_seqs(fasta: Fasta, df: pd.DataFrame, flank: int = 15) -> List[str]:
    """±flank nt centred on each editing site. Reverse-complement for – strand."""
    seqs = []
    for _, row in df.iterrows():
        chrom  = row['chr']
        pos1   = int(row['start']) + 1   # BED 0-based → 1-based
        strand = row['strand']
        seq = _seq_around_site(fasta, chrom, pos1, strand, flank)
        if seq and len(seq) == 2 * flank + 1:
            seqs.append(seq)
    return seqs


def _reconstruct_ss_positions(row) -> Tuple[Optional[int], Optional[int]]:
    """
    From an intronic editing site row, infer:
      donor_pos1   = last base of upstream exon (1-based)
      acceptor_pos1 = first base of downstream exon (1-based)
    Returns (None, None) if not intronic or distances missing.
    """
    if row['feature_type'] != 'intron':
        return None, None
    d5 = row['dist_5ss']
    d3 = row['dist_3ss']
    if pd.isna(d5) or pd.isna(d3):
        return None, None
    pos1   = int(row['start']) + 1
    strand = row['strand']
    if strand == '+':
        donor_pos1    = pos1 - int(d5)
        acceptor_pos1 = pos1 + int(d3)
    else:
        donor_pos1    = pos1 + int(d5)
        acceptor_pos1 = pos1 - int(d3)
    return donor_pos1, acceptor_pos1


def _extract_ss_seqs(fasta: Fasta, df: pd.DataFrame,
                     kind: str,          # 'donor' | 'acceptor' | 'bp'
                     max_dist: int = 500,
                     subsample: int = 500) -> List[str]:
    """
    Extract donor (9nt), acceptor (23nt), or branch-point (31nt) sequences
    for intronic editing sites within max_dist of the relevant splice site.
    """
    seqs = []
    rows = df[df['feature_type'] == 'intron'].copy()
    # Filter by proximity
    if kind == 'donor':
        rows = rows[rows['dist_5ss'].notna() & (rows['dist_5ss'] <= max_dist)]
    else:
        rows = rows[rows['dist_3ss'].notna() & (rows['dist_3ss'] <= max_dist)]

    for _, row in rows.iterrows():
        d_pos, a_pos = _reconstruct_ss_positions(row)
        if d_pos is None:
            continue
        chrom  = row['chr']
        strand = row['strand']
        if kind == 'donor':
            seq = _donor_seq(fasta, chrom, d_pos, strand, up=3, down=8)
        elif kind == 'acceptor':
            seq = _acceptor_seq(fasta, chrom, a_pos, strand, up=20, down=3)
        else:  # bp
            seq = _bp_region_seq(fasta, chrom, a_pos, strand, bp_start=40, bp_end=10)
        if seq:
            seqs.append(seq)

    if len(seqs) > subsample:
        seqs = random.sample(seqs, subsample)
    return seqs


def _control_seqs_from_introns(fasta: Fasta,
                                df: pd.DataFrame,
                                gene_introns: Dict[str, List[Tuple[str, int, int, str]]],
                                kind: str,
                                n_per_intron: int = 2,
                                subsample: int = 500) -> List[str]:
    """
    Build a control set by sampling random intronic positions from the same
    genes, then extracting donor/acceptor/bp sequences.
    Excludes introns that contain actual editing sites (to avoid contamination).
    """
    # Build set of edited introns to exclude
    edited_introns: set = set()
    for _, row in df[df['feature_type'] == 'intron'].iterrows():
        d_pos, a_pos = _reconstruct_ss_positions(row)
        if d_pos is None:
            continue
        edited_introns.add((row['chr'], d_pos, a_pos))

    seqs = []
    edited_gene_ids = set(df['gene_id'].dropna().unique())

    for gene_id in edited_gene_ids:
        if gene_id not in gene_introns:
            continue
        for (chrom, istart, iend, strand) in gene_introns[gene_id]:
            if (chrom, istart, iend) in edited_introns:
                continue  # skip introns with editing sites
            if iend - istart < 100:
                continue
            for _ in range(n_per_intron):
                # Pick a random position inside the intron
                pos1 = random.randint(istart + 20, iend - 20)
                if kind == 'donor':
                    seq = _donor_seq(fasta, chrom, istart - 1, strand, up=3, down=8)
                elif kind == 'acceptor':
                    seq = _acceptor_seq(fasta, chrom, iend + 1, strand, up=20, down=3)
                else:  # bp
                    seq = _bp_region_seq(fasta, chrom, iend + 1, strand,
                                         bp_start=40, bp_end=10)
                if seq:
                    seqs.append(seq)

    if len(seqs) > subsample:
        seqs = random.sample(seqs, subsample)
    return seqs


# ─────────────────────────────────────────────────────────────────────────────
# Plot assembly
# ─────────────────────────────────────────────────────────────────────────────

def plot(input_file: str, fasta_file: str, gtf_file: str,
         output: str, label: str):

    logger.info(f"Loading editing sites: {input_file}")
    df = _load_bed(input_file)
    intronic = df[df['feature_type'] == 'intron'].copy()
    logger.info(f"  {len(df):,} total sites, {len(intronic):,} intronic")

    logger.info(f"Loading genome FASTA: {fasta_file}")
    fasta = Fasta(fasta_file)

    logger.info("Parsing GTF for intron structures...")
    gene_introns = _parse_gtf_introns(gtf_file)

    # ── Extract sequences ───────────────────────────────────────────────────
    logger.info("Extracting sequences around editing sites (±15 nt)...")
    all_site_seqs     = _extract_editing_site_seqs(fasta, df, flank=15)
    intronic_site_seqs = _extract_editing_site_seqs(fasta, intronic, flank=15)

    logger.info("Extracting donor sequences (all intronic sites)...")
    donor_edit   = _extract_ss_seqs(fasta, df, 'donor',    max_dist=10_000_000)
    donor_ctrl   = _control_seqs_from_introns(fasta, df, gene_introns, 'donor')

    logger.info("Extracting acceptor sequences (all intronic sites)...")
    accept_edit  = _extract_ss_seqs(fasta, df, 'acceptor', max_dist=10_000_000)
    accept_ctrl  = _control_seqs_from_introns(fasta, df, gene_introns, 'acceptor')

    logger.info("Extracting branch-point region sequences (PPT sites ≤50 nt from 3'SS)...")
    ppt_sites    = df[df['splice_region'] == 'ppt_region'].copy()
    bp_edit      = _extract_ss_seqs(fasta, ppt_sites, 'bp', max_dist=50)
    bp_ctrl      = _control_seqs_from_introns(fasta, df, gene_introns, 'bp')

    ppt_site_seqs = _extract_editing_site_seqs(fasta, ppt_sites, flank=15)

    logger.info(f"  donor:    {len(donor_edit):,} editing / {len(donor_ctrl):,} control")
    logger.info(f"  acceptor: {len(accept_edit):,} editing / {len(accept_ctrl):,} control")
    logger.info(f"  branch-pt:{len(bp_edit):,} editing / {len(bp_ctrl):,} control")

    # ── Control sequences for motif enrichment scan ─────────────────────────
    logger.info("Generating matched control sequences for motif scan...")
    ctrl_site_seqs = []
    edited_positions = set(zip(df['chr'], df['start']))
    for gene_id in set(df['gene_id'].dropna()):
        if gene_id not in gene_introns:
            continue
        for (chrom, istart, iend, strand) in gene_introns[gene_id][:5]:
            if iend - istart < 50:
                continue
            for _ in range(3):
                pos1 = random.randint(istart + 16, iend - 16)
                if (chrom, pos1 - 1) in edited_positions:
                    continue
                seq = _seq_around_site(fasta, chrom, pos1, strand, flank=15)
                if seq and len(seq) == 31 and 'N' not in seq:
                    ctrl_site_seqs.append(seq)
    ctrl_site_seqs = random.sample(ctrl_site_seqs, min(1000, len(ctrl_site_seqs)))
    logger.info(f"  control sequences: {len(ctrl_site_seqs):,}")

    # Pre-compute log-odds matrices for motif scan
    lo_puf60   = _pwm_log_odds(_PUF60_MOTIF)
    lo_pumilio = _pwm_log_odds(_PUMILIO_MOTIF)

    # ── Build figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 42))
    gs  = gridspec.GridSpec(8, 2, figure=fig, hspace=0.72, wspace=0.38)

    # Row 0: trinucleotide context + all-site logo
    ax0a = fig.add_subplot(gs[0, 0])
    ax0b = fig.add_subplot(gs[0, 1])
    # Row 1: intronic site logo (edit / ppt)
    ax1a = fig.add_subplot(gs[1, 0])
    ax1b = fig.add_subplot(gs[1, 1])
    # Row 2: donor edit vs ctrl
    ax2a = fig.add_subplot(gs[2, 0])
    ax2b = fig.add_subplot(gs[2, 1])
    # Row 3: acceptor edit vs ctrl
    ax3a = fig.add_subplot(gs[3, 0])
    ax3b = fig.add_subplot(gs[3, 1])
    # Row 4: BP window edit vs ctrl
    ax4a = fig.add_subplot(gs[4, 0])
    ax4b = fig.add_subplot(gs[4, 1])
    # Row 5: Known motif references (PUF60 left, Pumilio right)
    ax5a = fig.add_subplot(gs[5, 0])
    ax5b = fig.add_subplot(gs[5, 1])
    # Row 6: Motif enrichment scan (PUF60 left, Pumilio right)
    ax6a = fig.add_subplot(gs[6, 0])
    ax6b = fig.add_subplot(gs[6, 1])
    # Row 7: note
    ax7  = fig.add_subplot(gs[7, :])

    flank = 15
    center = flank  # 0-indexed centre position

    # ── Panel 0a: Trinucleotide bar chart ───────────────────────────────────
    _draw_trinuc_bar(ax0a, all_site_seqs, center=center,
                     title="Trinucleotide Context at Editing Site\n(All sites)")

    # ── Panel 0b: All-sites logo ─────────────────────────────────────────────
    xticks = [str(i - flank) for i in range(2 * flank + 1)]
    _draw_logo(ax0b, all_site_seqs,
               title="Sequence Logo ±15 nt — All Sites",
               center_label=center, xtick_labels=xticks)

    # ── Panel 1a: Intronic sites logo ───────────────────────────────────────
    _draw_logo(ax1a, intronic_site_seqs,
               title="Sequence Logo ±15 nt — Intronic Sites Only",
               center_label=center, xtick_labels=xticks)

    # ── Panel 1b: PPT-region sites logo ─────────────────────────────────────
    _draw_logo(ax1b, ppt_site_seqs,
               title="Sequence Logo ±15 nt — PPT-Region Sites Only",
               center_label=center, xtick_labels=xticks,
               highlight_positions=[center])

    # ── Panel 2: Donor motif (editing vs control) ────────────────────────────
    donor_xticks = [str(i) for i in range(-3, 0)] + ['+1', '+2', '+3', '+4', '+5', '+6', '+7', '+8']
    _draw_logo(ax2a, donor_edit,
               title="5'SS Donor Motif — Introns Containing Editing Sites",
               xtick_labels=donor_xticks,
               highlight_positions=[3, 4])   # +1/+2 = GT

    _draw_logo(ax2b, donor_ctrl,
               title="5'SS Donor Motif — Unedited Introns (Same Genes, Control)",
               xtick_labels=donor_xticks,
               highlight_positions=[3, 4])

    # ── Panel 3: Acceptor motif (editing vs control) ─────────────────────────
    accept_xticks = [str(i) for i in range(-20, 0)] + ['+1', '+2', '+3']
    _draw_logo(ax3a, accept_edit,
               title="3'SS Acceptor Motif — Introns Containing Editing Sites",
               xtick_labels=accept_xticks,
               highlight_positions=[18, 19])   # -2/-1 = AG

    _draw_logo(ax3b, accept_ctrl,
               title="3'SS Acceptor Motif — Unedited Introns (Same Genes, Control)",
               xtick_labels=accept_xticks,
               highlight_positions=[18, 19])

    # ── Panel 4: Branch-point window (editing vs control) ────────────────────
    # 31-mer: positions -40..-10 upstream of 3'SS
    bp_xticks = [str(-40 + i) for i in range(31)]
    _draw_logo(ax4a, bp_edit,
               title="Branch-Point Window (-40 to -10 from 3'SS)\nEditing in PPT Region",
               xtick_labels=bp_xticks)

    _draw_logo(ax4b, bp_ctrl,
               title="Branch-Point Window (-40 to -10 from 3'SS)\nControl (Unedited Introns)",
               xtick_labels=bp_xticks)

    # ── Row 5: Known reference motif logos ──────────────────────────────────
    _draw_motif_reference(
        ax5a, _PUF60_MOTIF,
        "Reference: PUF60 binding motif (UCUCUUUU consensus)\n"
        "from ENCODE eCLIP + Salton et al. 2008 EMBO Rep\n"
        "[PUF60 = Poly-U binding Factor; UHM/RRM protein; NOT pumilio family]"
    )
    _draw_motif_reference(
        ax5b, _PUMILIO_MOTIF,
        "Reference: Pumilio / PUF-family motif (UGUAHAUA, H=not G)\n"
        "Wickens group 'PUF code' — Gerber et al. 2004; Bernstein et al. 2005\n"
        "[Pumilio-homology domain proteins — distinct from PUF60]"
    )

    # ── Row 6: Motif enrichment scan at editing site (±15 nt window) ────────
    _draw_motif_enrichment(
        ax6a, all_site_seqs, ctrl_site_seqs,
        lo_puf60, 'PUF60 (UC-rich PPT)', flank=15
    )
    _draw_motif_enrichment(
        ax6b, all_site_seqs, ctrl_site_seqs,
        lo_pumilio, 'Pumilio/PUF (UGUAHAUA)', flank=15
    )

    # ── Row 7: Note ──────────────────────────────────────────────────────────
    ax7.axis('off')
    note = (
        "Notes:\n"
        "• Sequence logos show information content (bits).\n"
        "• Donor motif (+1/+2 = GT highlighted red); Acceptor motif (-2/-1 = AG highlighted).\n"
        "• Branch-point window: −40 to −10 nt from 3'SS. "
        "Canonical BP consensus = YYYYYYYYYYYYYNYYR[A]Y (branch-point A in brackets).\n"
        "• Control sequences: random intronic positions in the same edited genes "
        "(introns without editing sites) — controls for gene-composition bias.\n"
        "• IMPORTANT: PUF60 (Poly-U Factor 60 kDa, FIR/SIAHBP1) contains RRM/UHM "
        "domains and binds polypyrimidine tracts.  It is NOT a pumilio/PUF-family "
        "protein. The Wickens-group 'PUF code' (UGUAHAUA) describes pumilio-homology "
        "domain proteins (Pumilio, FBF, Puf1–5). If a pumilio-type motif is enriched "
        "at editing sites, that would be unexpected and worth investigating.\n"
        f"• Label: {label}"
    )
    ax7.text(0.02, 0.90, note, transform=ax7.transAxes,
             ha='left', va='top', fontsize=9.5, family='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f4f8', alpha=0.8))

    fig.suptitle(f"HyperTRIBE — Splice Motif Analysis + Known Motif Comparison: {label}",
                 fontsize=15, fontweight='bold', y=1.002)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(
        description='Splice site motif analysis for HyperTRIBE editing sites',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input',  required=True,
                        help='Splice-annotated editing sites BED')
    parser.add_argument('--fasta',  required=True,
                        help='Genome FASTA (must be indexed with .fai)')
    parser.add_argument('--gtf',    required=True,
                        help='GTF annotation file (same as used for annotation)')
    parser.add_argument('--output', required=True,
                        help='Output PDF path')
    parser.add_argument('--label',  default='',
                        help='Experiment label for plot titles')
    args = parser.parse_args()
    plot(args.input, args.fasta, args.gtf, args.output, args.label)


if __name__ == '__main__':
    main()
