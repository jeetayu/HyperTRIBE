#!/usr/bin/env python3
"""
Parallel RNA Editing Site Caller for HyperTRIBE
================================================

This script identifies ADAR-mediated RNA editing sites by comparing control
and treatment BAM files. It uses multiprocessing for efficient analysis of
large human genomes.

Scores both A->G (plus-strand-equivalent) and T->C (minus-strand-equivalent)
positions per-locus based on the control sample's dominant base — see
_dominant_base_pair(). Before 2026-07-02 this only ever checked A->G, which
silently missed every editing site on a minus-strand-transcribed locus
(~half of all real sites in validation against an independent original
pipeline run; see project_rhadar_manuscript memory for the diagnosis).

Author: Optimized HyperTRIBE Pipeline v2.0
Date: 2026-01-28
"""

import argparse
import logging
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import pysam
import numpy as np
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('editing_caller.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class BaseCount:
    """Container for base counts at a position"""
    A: int = 0
    C: int = 0
    G: int = 0
    T: int = 0
    N: int = 0
    
    @property
    def total(self) -> int:
        """Total coverage at position"""
        return self.A + self.C + self.G + self.T
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary"""
        return {
            'A': self.A, 'C': self.C, 'G': self.G, 
            'T': self.T, 'N': self.N, 'total': self.total
        }


@dataclass
class EditingSite:
    """Container for an RNA editing site"""
    chromosome: str
    position: int
    strand: str
    gene: str
    control_count: BaseCount
    treatment_count: BaseCount
    edit_frequency: float
    fold_change: float
    p_value: float
    rep_idx: int = 0   # 0-based index of treatment BAM in --treatment list
    ref_base: str = 'A'   # which BaseCount field was actually scored as "unedited"
    edit_base: str = 'G'  # which BaseCount field was actually scored as "edited"
    # ('A','G') for plus-strand-equivalent sites, ('T','C') for minus-strand-
    # equivalent sites (see _dominant_base_pair()). FIXED 2026-07-29: to_bed_line()
    # used to hardcode .A/.G here regardless of which pair was actually scored,
    # so every minus-strand (T/C-scored) site silently wrote near-zero background
    # A/G counts into the control_A/control_G/treatment_A/treatment_G columns
    # instead of the real T/C counts actually used for edit_frequency/fold_change/
    # p_value in this same row -- filter_replicates.py then recomputed pooled
    # statistics from those wrong columns, discarding real signal (see
    # feedback_hypertribe_caller_ag_output_bug memory for the full incident/fix).

    def to_bed_line(self) -> str:
        """Convert to BED format line.

        control_A/control_G/treatment_A/treatment_G column names are kept as-is
        for backward compatibility with downstream scripts (filter_replicates.py,
        annotate_genes.py, etc.) -- but the VALUES are now always whichever base
        pair was actually scored (ref_base/edit_base), not literal .A/.G, so
        these columns are correct for both plus- and minus-strand sites.
        """
        return (
            f"{self.chromosome}\t"
            f"{self.position}\t"
            f"{self.position + 1}\t"
            f"{self.gene}\t"
            f"{self.edit_frequency:.2f}\t"
            f"{self.strand}\t"
            f"{self.control_count.total}\t"
            f"{getattr(self.control_count, self.ref_base)}\t"
            f"{getattr(self.control_count, self.edit_base)}\t"
            f"{self.treatment_count.total}\t"
            f"{getattr(self.treatment_count, self.ref_base)}\t"
            f"{getattr(self.treatment_count, self.edit_base)}\t"
            f"{self.fold_change:.2f}\t"
            f"{self.p_value:.2e}\t"
            f"{self.rep_idx}"
        )


def get_base_counts_at_position(
    bam_file: str,
    chromosome: str,
    position: int,
    min_base_quality: int = 20,
    min_mapping_quality: int = 10
) -> BaseCount:
    """
    Extract base counts at a specific genomic position.
    
    Args:
        bam_file: Path to BAM file
        chromosome: Chromosome name
        position: Genomic position (0-based)
        min_base_quality: Minimum base quality score
        min_mapping_quality: Minimum read mapping quality
        
    Returns:
        BaseCount object with counts for each base
    """
    counts = BaseCount()
    
    try:
        with pysam.AlignmentFile(bam_file, "rb") as bam:
            for pileupcolumn in bam.pileup(
                chromosome, position, position + 1,
                min_base_quality=min_base_quality,
                min_mapping_quality=min_mapping_quality,
                stepper='nofilter',
                truncate=True
            ):
                if pileupcolumn.pos != position:
                    continue
                    
                for pileupread in pileupcolumn.pileups:
                    if pileupread.is_del or pileupread.is_refskip:
                        continue
                        
                    base = pileupread.alignment.query_sequence[pileupread.query_position]
                    
                    if base == 'A':
                        counts.A += 1
                    elif base == 'C':
                        counts.C += 1
                    elif base == 'G':
                        counts.G += 1
                    elif base == 'T':
                        counts.T += 1
                    else:
                        counts.N += 1
                        
    except Exception as e:
        logger.error(f"Error reading position {chromosome}:{position} in {bam_file}: {e}")
        
    return counts


def _dominant_base_pair(
    bc: BaseCount, min_count: float
) -> Optional[Tuple[str, str]]:
    """
    Infer (reference_base, edited_base) at a position from the control's
    base counts, without requiring a FASTA reference or per-read strand tags.

    ADAR-mediated editing is A->I(G) on the sense strand of the transcribed
    RNA. On a minus-strand-transcribed locus this appears as T->C on the
    plus-strand-mapped genomic reference (the reverse complement of A->G).
    A caller that only ever checks A/G silently misses every site on a
    minus-strand transcript. Genomic positions are overwhelmingly single-base
    in real data, so whichever of A or T clears the coverage threshold in the
    control sample is a reliable proxy for the transcribed strand's
    reference base (matches the pre-existing 'bc.A >= min_coverage * 0.5'
    absolute-count convention, extended symmetrically to T).

    Returns None if neither A nor T reaches min_count (ambiguous / not a
    clean single-base position). If both do (rare: true heterozygous SNP or
    bidirectional transcription), the higher-count base wins.
    """
    a_ok = bc.A >= min_count
    t_ok = bc.T >= min_count
    if a_ok and t_ok:
        return ('A', 'G') if bc.A >= bc.T else ('T', 'C')
    if a_ok:
        return ('A', 'G')
    if t_ok:
        return ('T', 'C')
    return None


def calculate_editing_statistics(
    control_counts: BaseCount,
    treatment_counts: BaseCount,
    stat_test: str = 'fisher',
    ref_base: str = 'A',
    edit_base: str = 'G',
) -> Tuple[float, float, float]:
    """
    Calculate editing frequency, fold change, and statistical significance.

    ref_base/edit_base select which base pair to score (('A','G') for
    plus-strand-equivalent editing, ('T','C') for minus-strand-equivalent —
    see _dominant_base_pair()). Defaults preserve the original A/G-only
    behavior for any external caller that doesn't pass these explicitly.

    stat_test options:
      'fisher'        - one-sided Fisher's exact test on 2x2 contingency table
      'hypergeometric'- hypergeometric sampling model (treat reads drawn from
                        control-defined background; falls back to binomial when
                        treatment depth exceeds control depth)
      'none'          - no statistical test; p_value set to 0.0 so all sites
                        passing frequency/fold-change thresholds are retained
    """
    ctrl_ref = getattr(control_counts, ref_base)
    ctrl_edit = getattr(control_counts, edit_base)
    treat_ref = getattr(treatment_counts, ref_base)
    treat_edit = getattr(treatment_counts, edit_base)

    control_edit = (ctrl_edit / ctrl_ref) if ctrl_ref > 0 else 0
    treatment_edit = (treat_edit / treat_ref) if treat_ref > 0 else 0
    edit_freq = treatment_edit * 100
    fold_change = treatment_edit / control_edit if control_edit > 0 else float('inf')

    if stat_test == 'none':
        p_value = 0.0

    elif stat_test == 'hypergeometric':
        # Hypergeometric: P(X >= treat_edit) where X ~ Hypergeom(M, n, N)
        #   M = control total ref+edit reads (population size)
        #   n = control edit reads            (successes in population)
        #   N = treatment total ref+edit reads (draw size)
        #   k = treatment edit reads           (observed successes)
        # Falls back to binomial when N > M (treatment depth exceeds control).
        M = ctrl_ref + ctrl_edit
        n = ctrl_edit
        N = treat_ref + treat_edit
        k = treat_edit
        try:
            if k == 0:
                p_value = 1.0
            elif N <= M and M > 0:
                p_value = float(stats.hypergeom.sf(k - 1, M, n, N))
            else:
                bg_rate = n / M if M > 0 else 0.0
                p_value = float(stats.binom.sf(k - 1, N, bg_rate)) if bg_rate > 0 else 1.0
        except Exception:
            p_value = 1.0

    else:  # fisher (default)
        contingency_table = [
            [treat_edit, treat_ref],
            [ctrl_edit,  ctrl_ref],
        ]
        try:
            _, p_value = stats.fisher_exact(contingency_table, alternative='greater')
        except Exception:
            p_value = 1.0

    return edit_freq, fold_change, p_value


def _pileup_base_counts(
    bam_paths: List[str],
    chromosome: str,
    start: int,
    end: int,
    min_base_quality: int,
    min_mapping_quality: int,
) -> Dict[int, BaseCount]:
    """
    Single-pass pileup over all BAMs for a genomic chunk.
    Returns merged BaseCount per covered position.
    Opening each BAM once per chunk is far cheaper than random-access per position.
    """
    merged: Dict[int, BaseCount] = defaultdict(BaseCount)
    for bam_path in bam_paths:
        try:
            with pysam.AlignmentFile(bam_path, "rb") as bam:
                for col in bam.pileup(
                    chromosome, start, end,
                    min_base_quality=min_base_quality,
                    min_mapping_quality=min_mapping_quality,
                    stepper='nofilter',
                    truncate=True,
                ):
                    pos = col.reference_pos
                    bc = merged[pos]
                    for pr in col.pileups:
                        if pr.is_del or pr.is_refskip:
                            continue
                        base = pr.alignment.query_sequence[pr.query_position]
                        if base == 'A':
                            bc.A += 1
                        elif base == 'C':
                            bc.C += 1
                        elif base == 'G':
                            bc.G += 1
                        elif base == 'T':
                            bc.T += 1
                        else:
                            bc.N += 1
        except (ValueError, KeyError):
            # Chromosome not present in this BAM (e.g. chrY in female sample)
            pass
    return merged


def process_chromosome_chunk(args) -> List[EditingSite]:
    """
    Process all covered positions in a chromosome chunk.
    Uses a single pileup pass per BAM rather than per-position random access,
    so every A-covered position is examined (not a 100-bp subsample).
    """
    chromosome, start, end, control_bams, treatment_bams, params = args

    logger.info(f"Processing {chromosome}:{start}-{end}")

    min_bq = params['min_base_quality']
    min_mq = params['min_mapping_quality']
    min_cov = params['min_coverage']

    # ── Merge control counts across all control BAMs in one pass ─────────────
    ctrl_counts = _pileup_base_counts(control_bams, chromosome, start, end, min_bq, min_mq)

    # ── Pre-filter to positions with adequate control coverage of either the
    #    A/G (plus-strand-equivalent) or T/C (minus-strand-equivalent) base
    #    pair, and acceptable control background on whichever pair applies.
    #    See _dominant_base_pair() docstring for why both are needed. ───────
    candidate_positions: Dict[int, Tuple[BaseCount, str, str]] = {}
    for pos, bc in ctrl_counts.items():
        if bc.total < min_cov:
            continue
        base_pair = _dominant_base_pair(bc, min_cov * 0.5)
        if base_pair is None:
            continue
        ref_base, edit_base = base_pair
        ctrl_ref = getattr(bc, ref_base)
        ctrl_edit = getattr(bc, edit_base)
        if (ctrl_edit / ctrl_ref if ctrl_ref > 0 else 0) <= params['max_control_edit_freq']:
            candidate_positions[pos] = (bc, ref_base, edit_base)

    if not candidate_positions:
        return []

    # ── Pileup each treatment BAM once; only score candidate positions ────────
    editing_sites = []
    for rep_idx, treat_bam in enumerate(treatment_bams):
        treat_counts = _pileup_base_counts([treat_bam], chromosome, start, end, min_bq, min_mq)

        for pos, (ctrl_bc, ref_base, edit_base) in candidate_positions.items():
            treat_bc = treat_counts.get(pos, BaseCount())
            if treat_bc.total < min_cov:
                continue

            edit_freq, fold_change, p_value = calculate_editing_statistics(
                ctrl_bc, treat_bc, stat_test=params.get('stat_test', 'fisher'),
                ref_base=ref_base, edit_base=edit_base,
            )

            ctrl_ref = getattr(ctrl_bc, ref_base)
            ctrl_edit = getattr(ctrl_bc, edit_base)
            ctrl_edit_rate = ctrl_edit / ctrl_ref if ctrl_ref > 0 else 0
            if (edit_freq >= params['min_edit_freq'] * 100
                    and (params['edit_fold_change'] == 0 or fold_change >= params['edit_fold_change'])
                    and ctrl_edit_rate <= params['max_control_edit_freq']
                    and p_value < params['p_value_threshold']):
                editing_sites.append(EditingSite(
                    chromosome=chromosome,
                    position=pos,
                    strand='+' if ref_base == 'A' else '-',
                    gene='.',
                    control_count=ctrl_bc,
                    treatment_count=treat_bc,
                    edit_frequency=edit_freq,
                    fold_change=fold_change,
                    p_value=p_value,
                    rep_idx=rep_idx,
                    ref_base=ref_base,
                    edit_base=edit_base,
                ))

    logger.info(f"Found {len(editing_sites)} sites in {chromosome}:{start}-{end}")
    return editing_sites


def get_chromosome_chunks(
    bam_file: str,
    chromosomes: Optional[List[str]] = None,
    chunk_size: int = 10_000_000
) -> List[Tuple[str, int, int]]:
    """
    Divide chromosomes into chunks for parallel processing.
    
    Args:
        bam_file: Path to BAM file (to get chromosome lengths)
        chromosomes: List of chromosomes to process (None = all)
        chunk_size: Size of each chunk in base pairs
        
    Returns:
        List of (chromosome, start, end) tuples
    """
    chunks = []
    
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        for ref_name, ref_length in zip(bam.references, bam.lengths):
            # Filter chromosomes if specified
            if chromosomes and ref_name not in chromosomes:
                continue
                
            # Create chunks for this chromosome
            for start in range(0, ref_length, chunk_size):
                end = min(start + chunk_size, ref_length)
                chunks.append((ref_name, start, end))
    
    return chunks


def _apply_min_site_distance(
    sites: List[EditingSite], min_dist: int
) -> List[EditingSite]:
    """
    Remove sites closer than min_dist nt to an adjacent retained site.
    Sites must already be sorted by (chromosome, position).
    When two sites are within min_dist, the one with the higher p_value is dropped;
    ties are broken by keeping the site with the higher fold_change.
    """
    if not sites:
        return sites
    kept: List[EditingSite] = []
    last_chrom: Optional[str] = None
    last_pos: int = -min_dist - 1
    for site in sites:
        # Same position = different replicate calling the same site; always keep
        same_pos = (site.chromosome == last_chrom and site.position == last_pos)
        if same_pos or site.chromosome != last_chrom or (site.position - last_pos) >= min_dist:
            kept.append(site)
            last_chrom = site.chromosome
            last_pos = site.position
        else:
            # Replace last kept site if current site is more significant
            prev = kept[-1]
            if site.p_value < prev.p_value or (
                site.p_value == prev.p_value and site.fold_change > prev.fold_change
            ):
                kept[-1] = site
                last_pos = site.position
    return kept


def main():
    """Main function to run the editing site caller"""
    parser = argparse.ArgumentParser(
        description='Call RNA editing sites from HyperTRIBE data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input files
    parser.add_argument('--control', nargs='+', required=True,
                       help='Control BAM files (wildtype or gDNA)')
    parser.add_argument('--treatment', nargs='+', required=True,
                       help='Treatment BAM files (HyperTRIBE samples)')
    
    # Output
    parser.add_argument('--output', required=True,
                       help='Output BED file')
    
    # Processing parameters
    parser.add_argument('--threads', type=int, default=1,
                       help='Number of parallel threads (default: 1)')
    parser.add_argument('--chunk-size', type=int, default=10_000_000,
                       help='Chunk size for parallel processing (default: 10M bp)')
    parser.add_argument('--chromosomes', nargs='+',
                       help='Specific chromosomes to analyze')
    
    # Filtering parameters
    parser.add_argument('--min-coverage', type=int, default=20,
                       help='Minimum read coverage (default: 20)')
    parser.add_argument('--min-edit-freq', type=float, default=0.05,
                       help='Minimum editing frequency (default: 0.05)')
    parser.add_argument('--edit-fold-change', type=float, default=2.0,
                       help='Minimum fold change vs control (default: 2.0). '
                            'Set to 0 to disable.')
    parser.add_argument('--max-control-edit-freq', type=float, default=1.0,
                       help='Maximum editing frequency allowed in control (default: 1.0 = disabled). '
                            'Set e.g. 0.1 to require <10%% control editing (original TRIBE approach).')
    parser.add_argument('--p-value-threshold', type=float, default=0.05,
                       help='P-value threshold for significance (default: 0.05)')
    
    # Quality filters
    parser.add_argument('--min-base-quality', type=int, default=20,
                       help='Minimum base quality score (default: 20)')
    parser.add_argument('--min-mapping-quality', type=int, default=10,
                       help='Minimum read mapping quality (default: 10)')

    # Statistical test
    parser.add_argument('--stat-test', default='fisher',
                       choices=['fisher', 'hypergeometric', 'none'],
                       help='Statistical test for significance (default: fisher). '
                            '"none" retains all sites passing freq/fold-change thresholds.')

    # Proximity filter
    parser.add_argument('--min-site-distance', type=int, default=0,
                       help='Minimum distance (nt) between adjacent called sites. '
                            'When two sites are closer than this, the lower-confidence '
                            'site (higher p-value) is dropped. 0 = disabled (default: 0).')

    args = parser.parse_args()
    
    # Validate input files
    for bam_file in args.control + args.treatment:
        if not Path(bam_file).exists():
            logger.error(f"BAM file not found: {bam_file}")
            sys.exit(1)
        if not Path(f"{bam_file}.bai").exists():
            logger.error(f"BAM index not found: {bam_file}.bai")
            sys.exit(1)
    
    logger.info("="*60)
    logger.info("RNA Editing Site Caller for HyperTRIBE")
    logger.info("="*60)
    logger.info(f"Control samples: {len(args.control)}")
    logger.info(f"Treatment samples: {len(args.treatment)}")
    logger.info(f"Threads: {args.threads}")
    logger.info(f"Min coverage: {args.min_coverage}")
    logger.info(f"Min editing frequency: {args.min_edit_freq}")
    logger.info(f"Min fold change: {args.edit_fold_change} ({'disabled' if args.edit_fold_change == 0 else 'active'})")
    logger.info(f"Max control edit freq: {args.max_control_edit_freq} ({'disabled' if args.max_control_edit_freq >= 1.0 else 'active'})")
    
    # Get chromosome chunks for parallel processing
    logger.info("Dividing genome into chunks...")
    chunks = get_chromosome_chunks(
        args.control[0],
        chromosomes=args.chromosomes,
        chunk_size=args.chunk_size
    )
    logger.info(f"Created {len(chunks)} chunks for processing")
    
    logger.info(f"Stat test:        {args.stat_test}")
    logger.info(f"Min site distance: {args.min_site_distance} nt "
                f"({'disabled' if args.min_site_distance == 0 else 'active'})")

    # Prepare parameters for workers
    params = {
        'min_coverage': args.min_coverage,
        'min_edit_freq': args.min_edit_freq,
        'edit_fold_change': args.edit_fold_change,
        'max_control_edit_freq': args.max_control_edit_freq,
        'p_value_threshold': args.p_value_threshold,
        'min_base_quality': args.min_base_quality,
        'min_mapping_quality': args.min_mapping_quality,
        'stat_test': args.stat_test,
    }
    
    # Prepare arguments for each chunk
    chunk_args = [
        (chrom, start, end, args.control, args.treatment, params)
        for chrom, start, end in chunks
    ]
    
    # Process chunks in parallel
    logger.info("Processing chunks in parallel...")
    all_editing_sites = []
    
    if args.threads > 1:
        with mp.Pool(processes=args.threads) as pool:
            results = pool.map(process_chromosome_chunk, chunk_args)
            for chunk_sites in results:
                all_editing_sites.extend(chunk_sites)
    else:
        for chunk_arg in chunk_args:
            chunk_sites = process_chromosome_chunk(chunk_arg)
            all_editing_sites.extend(chunk_sites)
    
    logger.info(f"Total editing sites found (pre-distance filter): {len(all_editing_sites)}")

    # Apply minimum inter-site distance filter (global, after sorting)
    if args.min_site_distance > 0:
        all_editing_sites = _apply_min_site_distance(
            all_editing_sites, args.min_site_distance
        )
        logger.info(
            f"After {args.min_site_distance}-nt distance filter: {len(all_editing_sites)} sites"
        )

    # Write results
    logger.info(f"Writing results to {args.output}")
    with open(args.output, 'w') as out:
        # Write header
        header = (
            "#chromosome\tstart\tend\tgene\tedit_freq\tstrand\t"
            "control_cov\tcontrol_A\tcontrol_G\t"
            "treatment_cov\ttreatment_A\ttreatment_G\t"
            "fold_change\tp_value\trep_idx\n"
        )
        out.write(header)
        
        # Sort sites by chromosome and position
        all_editing_sites.sort(key=lambda x: (x.chromosome, x.position))
        
        # Write sites
        for site in all_editing_sites:
            out.write(site.to_bed_line() + '\n')
    
    logger.info("="*60)
    logger.info("Analysis complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
