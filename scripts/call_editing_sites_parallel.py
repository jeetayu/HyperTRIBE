#!/usr/bin/env python3
"""
Parallel RNA Editing Site Caller for HyperTRIBE
================================================

This script identifies A-to-G RNA editing sites by comparing control and treatment
BAM files. It uses multiprocessing for efficient analysis of large human genomes.

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
    
    def to_bed_line(self) -> str:
        """Convert to BED format line"""
        return (
            f"{self.chromosome}\t"
            f"{self.position}\t"
            f"{self.position + 1}\t"
            f"{self.gene}\t"
            f"{self.edit_frequency:.2f}\t"
            f"{self.strand}\t"
            f"{self.control_count.total}\t"
            f"{self.control_count.A}\t"
            f"{self.control_count.G}\t"
            f"{self.treatment_count.total}\t"
            f"{self.treatment_count.A}\t"
            f"{self.treatment_count.G}\t"
            f"{self.fold_change:.2f}\t"
            f"{self.p_value:.2e}"
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


def calculate_editing_statistics(
    control_counts: BaseCount,
    treatment_counts: BaseCount
) -> Tuple[float, float, float]:
    """
    Calculate editing frequency, fold change, and statistical significance.
    
    Args:
        control_counts: Base counts from control sample
        treatment_counts: Base counts from treatment sample
        
    Returns:
        Tuple of (editing_frequency, fold_change, p_value)
    """
    # Calculate editing frequencies
    control_edit = (control_counts.G / control_counts.A) if control_counts.A > 0 else 0
    treatment_edit = (treatment_counts.G / treatment_counts.A) if treatment_counts.A > 0 else 0
    
    # Editing frequency as percentage
    edit_freq = treatment_edit * 100
    
    # Fold change
    fold_change = treatment_edit / control_edit if control_edit > 0 else float('inf')
    
    # Fisher's exact test for significance
    # Contingency table: [[treatment_G, treatment_A], [control_G, control_A]]
    contingency_table = [
        [treatment_counts.G, treatment_counts.A],
        [control_counts.G, control_counts.A]
    ]
    
    try:
        _, p_value = stats.fisher_exact(contingency_table, alternative='greater')
    except:
        p_value = 1.0
    
    return edit_freq, fold_change, p_value


def process_chromosome_chunk(args) -> List[EditingSite]:
    """
    Process a chunk of positions on a chromosome.
    
    This function is designed to be called by multiprocessing.Pool.
    
    Args:
        args: Tuple containing (chromosome, start, end, control_bams, 
              treatment_bams, params)
              
    Returns:
        List of EditingSite objects
    """
    chromosome, start, end, control_bams, treatment_bams, params = args
    
    logger.info(f"Processing {chromosome}:{start}-{end}")
    
    editing_sites = []
    
    # Sample positions (full iteration would check every position)
    # For efficiency, we only check positions with coverage
    # This is a simplified version - full implementation would use pileup
    
    for position in range(start, end, 100):  # Sample every 100bp for demo
        # Get base counts from all control samples
        control_counts_all = []
        for bam in control_bams:
            counts = get_base_counts_at_position(
                bam, chromosome, position,
                params['min_base_quality'],
                params['min_mapping_quality']
            )
            if counts.total > 0:
                control_counts_all.append(counts)
        
        if not control_counts_all:
            continue
        
        # Merge control counts
        merged_control = BaseCount()
        for counts in control_counts_all:
            merged_control.A += counts.A
            merged_control.C += counts.C
            merged_control.G += counts.G
            merged_control.T += counts.T
            merged_control.N += counts.N
        
        # Skip if insufficient control coverage
        if merged_control.total < params['min_coverage']:
            continue
            
        # Only consider positions with A in control
        if merged_control.A < params['min_coverage'] * 0.5:
            continue
        
        # Process each treatment sample
        for treatment_bam in treatment_bams:
            treatment_counts = get_base_counts_at_position(
                treatment_bam, chromosome, position,
                params['min_base_quality'],
                params['min_mapping_quality']
            )
            
            if treatment_counts.total < params['min_coverage']:
                continue
            
            # Calculate statistics
            edit_freq, fold_change, p_value = calculate_editing_statistics(
                merged_control, treatment_counts
            )
            
            # Apply filters
            if (edit_freq >= params['min_edit_freq'] * 100 and
                fold_change >= params['edit_fold_change'] and
                p_value < params['p_value_threshold']):
                
                site = EditingSite(
                    chromosome=chromosome,
                    position=position,
                    strand='.',
                    gene='.',
                    control_count=merged_control,
                    treatment_count=treatment_counts,
                    edit_frequency=edit_freq,
                    fold_change=fold_change,
                    p_value=p_value
                )
                editing_sites.append(site)
    
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
                       help='Minimum fold change vs control (default: 2.0)')
    parser.add_argument('--p-value-threshold', type=float, default=0.05,
                       help='P-value threshold for significance (default: 0.05)')
    
    # Quality filters
    parser.add_argument('--min-base-quality', type=int, default=20,
                       help='Minimum base quality score (default: 20)')
    parser.add_argument('--min-mapping-quality', type=int, default=10,
                       help='Minimum read mapping quality (default: 10)')
    
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
    logger.info(f"Min fold change: {args.edit_fold_change}")
    
    # Get chromosome chunks for parallel processing
    logger.info("Dividing genome into chunks...")
    chunks = get_chromosome_chunks(
        args.control[0],
        chromosomes=args.chromosomes,
        chunk_size=args.chunk_size
    )
    logger.info(f"Created {len(chunks)} chunks for processing")
    
    # Prepare parameters for workers
    params = {
        'min_coverage': args.min_coverage,
        'min_edit_freq': args.min_edit_freq,
        'edit_fold_change': args.edit_fold_change,
        'p_value_threshold': args.p_value_threshold,
        'min_base_quality': args.min_base_quality,
        'min_mapping_quality': args.min_mapping_quality
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
    
    logger.info(f"Total editing sites found: {len(all_editing_sites)}")
    
    # Write results
    logger.info(f"Writing results to {args.output}")
    with open(args.output, 'w') as out:
        # Write header
        header = (
            "#chromosome\tstart\tend\tgene\tedit_freq\tstrand\t"
            "control_cov\tcontrol_A\tcontrol_G\t"
            "treatment_cov\ttreatment_A\ttreatment_G\t"
            "fold_change\tp_value\n"
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
