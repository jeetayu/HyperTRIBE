#!/usr/bin/env python3
"""
Mutation Spectrum Analysis for TRIBE experiments.

Computes the frequency of all 12 base conversion types (A→C, A→G, A→T,
C→A, C→G, C→T, G→A, G→C, G→T, T→A, T→C, T→G) in treatment vs. control
BAMs, using the same pileup infrastructure as call_editing_sites_parallel.py.

The majority control base at each position is treated as the "reference".
At A-dominant positions this is directly comparable to the TRIBE A→G caller:
A→G should be dramatically enriched while A→C and A→T should sit at
background (sequencing error / PCR error) rates.

Usage:
    python mutation_spectrum.py \\
        --control   ctrl_pooled.nodup.bam \\
        --treatment rep1.nodup.bam rep2.nodup.bam \\
        --output    mutation_spectrum.tsv \\
        --threads   16 \\
        [--min-coverage 20] \\
        [--chromosomes chr1 chr2 ...]  # default: all chr1-22,X,Y

Output files:
    {output}                  — per-conversion-type summary (12 rows × metrics)
    {output}.positions.tsv.gz — per-position records (large; for QC)
"""

import argparse
import gzip
import logging
import multiprocessing as mp
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pysam
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

BASES = ("A", "C", "G", "T")
CONVERSIONS = [(ref, alt) for ref in BASES for alt in BASES if ref != alt]  # 12 pairs
HUMAN_CHROMS = [f"chr{i}" for i in list(range(1, 23)) + ["X", "Y"]]

# ── Data containers ────────────────────────────────────────────────────────────

@dataclass
class BaseCounts:
    A: int = 0
    C: int = 0
    G: int = 0
    T: int = 0

    @property
    def total(self) -> int:
        return self.A + self.C + self.G + self.T

    def majority_base(self) -> Optional[str]:
        counts = {"A": self.A, "C": self.C, "G": self.G, "T": self.T}
        if self.total == 0:
            return None
        mb = max(counts, key=counts.get)
        # Require majority: at least 50% of reads are the reference base
        if counts[mb] / self.total >= 0.5:
            return mb
        return None

    def get(self, base: str) -> int:
        return getattr(self, base, 0)


@dataclass
class ConversionAccumulator:
    """Accumulates counts for one (ref_base, alt_base) conversion type."""
    ref_base: str
    alt_base: str
    n_positions: int = 0
    total_ref_reads: int = 0
    total_alt_reads: int = 0
    freq_sum: float = 0.0        # sum of per-position alt frequencies (for mean)
    freq_sum_sq: float = 0.0     # for variance

    def update(self, ref_reads: int, alt_reads: int):
        total = ref_reads + alt_reads
        if total == 0:
            return
        freq = alt_reads / total
        self.n_positions += 1
        self.total_ref_reads += ref_reads
        self.total_alt_reads += alt_reads
        self.freq_sum += freq
        self.freq_sum_sq += freq * freq

    @property
    def global_freq(self) -> float:
        denom = self.total_ref_reads + self.total_alt_reads
        return self.total_alt_reads / denom if denom > 0 else 0.0

    @property
    def mean_freq(self) -> float:
        return self.freq_sum / self.n_positions if self.n_positions > 0 else 0.0

    @property
    def std_freq(self) -> float:
        if self.n_positions < 2:
            return 0.0
        var = (self.freq_sum_sq - self.freq_sum**2 / self.n_positions) / (self.n_positions - 1)
        return float(np.sqrt(max(var, 0.0)))

    def to_dict(self) -> dict:
        return {
            "conversion": f"{self.ref_base}>{self.alt_base}",
            "ref_base": self.ref_base,
            "alt_base": self.alt_base,
            "n_positions": self.n_positions,
            "total_ref_reads": self.total_ref_reads,
            "total_alt_reads": self.total_alt_reads,
            "global_freq": self.global_freq,
            "mean_pos_freq": self.mean_freq,
            "std_pos_freq": self.std_freq,
        }


# ── Pileup helpers (shared with call_editing_sites_parallel.py approach) ──────

def _pileup_chunk(bam_paths: List[str], chrom: str, start: int, end: int,
                  min_bq: int = 20, min_mq: int = 10) -> Dict[int, BaseCounts]:
    merged: Dict[int, BaseCounts] = defaultdict(BaseCounts)
    for bam_path in bam_paths:
        try:
            with pysam.AlignmentFile(bam_path, "rb") as bam:
                for col in bam.pileup(chrom, start, end,
                                      min_base_quality=min_bq,
                                      min_mapping_quality=min_mq,
                                      stepper="nofilter",
                                      truncate=True):
                    pos = col.reference_pos
                    bc = merged[pos]
                    for pr in col.pileups:
                        if pr.is_del or pr.is_refskip:
                            continue
                        base = pr.alignment.query_sequence[pr.query_position]
                        if base == "A":
                            bc.A += 1
                        elif base == "C":
                            bc.C += 1
                        elif base == "G":
                            bc.G += 1
                        elif base == "T":
                            bc.T += 1
        except (ValueError, KeyError):
            pass
    return merged


def _process_chunk(args) -> Tuple[Dict, List]:
    """
    Worker: pileup one genomic chunk, accumulate conversion counts.
    Returns (accum_dict, position_records).
    accum_dict: {(ref, alt): [n_pos, total_ref, total_alt, freq_sum, freq_sum_sq]}
    position_records: list of (chrom, pos, ref_base, ctrl_total, treat_A, treat_C,
                               treat_G, treat_T, treat_total) for A-dominant positions
    """
    chrom, start, end, ctrl_bams, treat_bams, params = args
    min_cov = params["min_coverage"]
    min_bq  = params["min_bq"]
    min_mq  = params["min_mq"]

    # Pileup control (determines reference base)
    ctrl = _pileup_chunk(ctrl_bams, chrom, start, end, min_bq, min_mq)

    # Only keep positions with adequate control coverage and a clear majority base
    candidates = {}
    for pos, bc in ctrl.items():
        if bc.total < min_cov:
            continue
        mb = bc.majority_base()
        if mb is None:
            continue
        candidates[pos] = (mb, bc)

    if not candidates:
        return {}, []

    # Pileup pooled treatment
    treat = _pileup_chunk(treat_bams, chrom, start, end, min_bq, min_mq)

    # Accumulate per (ref, alt) pair
    # accum: {(ref, alt): [n_pos, total_ref, total_alt, freq_sum, freq_sum_sq]}
    accum = defaultdict(lambda: [0, 0, 0, 0.0, 0.0])
    pos_records = []  # only for A-dominant positions

    for pos, (ref_base, ctrl_bc) in candidates.items():
        tbc = treat.get(pos, BaseCounts())
        if tbc.total < min_cov:
            continue

        ref_reads = tbc.get(ref_base)
        total = tbc.total

        for alt_base in BASES:
            if alt_base == ref_base:
                continue
            alt_reads = tbc.get(alt_base)
            if ref_reads + alt_reads == 0:
                continue
            freq = alt_reads / (ref_reads + alt_reads)
            key = (ref_base, alt_base)
            accum[key][0] += 1                 # n_positions
            accum[key][1] += ref_reads          # total_ref
            accum[key][2] += alt_reads          # total_alt
            accum[key][3] += freq               # freq_sum
            accum[key][4] += freq * freq        # freq_sum_sq

        # Save position record for A-dominant positions (for per-site output)
        if ref_base == "A":
            pos_records.append((
                chrom, pos, ref_base,
                ctrl_bc.total,
                tbc.A, tbc.C, tbc.G, tbc.T, total,
            ))

    return dict(accum), pos_records


# ── Main ──────────────────────────────────────────────────────────────────────

def get_chrom_lengths(bam_path: str, chroms: List[str]) -> List[Tuple[str, int]]:
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        lengths = {sq["SN"]: sq["LN"] for sq in bam.header["SQ"]}
    return [(c, lengths[c]) for c in chroms if c in lengths]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control",     nargs="+", required=True)
    ap.add_argument("--treatment",   nargs="+", required=True)
    ap.add_argument("--output",      required=True)
    ap.add_argument("--threads",     type=int, default=16)
    ap.add_argument("--min-coverage",type=int, default=20)
    ap.add_argument("--chunk-size",  type=int, default=10_000_000)
    ap.add_argument("--min-bq",      type=int, default=20)
    ap.add_argument("--min-mq",      type=int, default=10)
    ap.add_argument("--chromosomes", nargs="*", default=None,
                    help="Chromosomes to scan (default: chr1-22,X,Y)")
    ap.add_argument("--save-positions", action="store_true",
                    help="Save per-position records for A-dominant sites (large file)")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chroms = args.chromosomes or HUMAN_CHROMS
    chrom_lengths = get_chrom_lengths(args.control[0], chroms)
    logger.info(f"Scanning {len(chrom_lengths)} chromosomes, "
                f"{args.threads} threads, min_coverage={args.min_coverage}")

    params = {
        "min_coverage": args.min_coverage,
        "min_bq": args.min_bq,
        "min_mq": args.min_mq,
    }

    # Build chunks
    chunks = []
    for chrom, length in chrom_lengths:
        for start in range(0, length, args.chunk_size):
            end = min(start + args.chunk_size, length)
            chunks.append((chrom, start, end, args.control, args.treatment, params))

    logger.info(f"Processing {len(chunks)} chunks across {len(chrom_lengths)} chromosomes...")

    # Parallel processing
    global_accum = defaultdict(lambda: [0, 0, 0, 0.0, 0.0])
    all_pos_records = []

    with mp.Pool(processes=args.threads) as pool:
        for i, (chunk_accum, pos_recs) in enumerate(
            pool.imap_unordered(_process_chunk, chunks, chunksize=1)
        ):
            for key, vals in chunk_accum.items():
                ga = global_accum[key]
                for j in range(5):
                    ga[j] += vals[j]
            if args.save_positions:
                all_pos_records.extend(pos_recs)
            if (i + 1) % 50 == 0:
                logger.info(f"  {i+1}/{len(chunks)} chunks done")

    logger.info("Aggregating results...")

    # Build summary table
    rows = []
    for ref_base in BASES:
        for alt_base in BASES:
            if ref_base == alt_base:
                continue
            key = (ref_base, alt_base)
            vals = global_accum.get(key, [0, 0, 0, 0.0, 0.0])
            n_pos, total_ref, total_alt, freq_sum, freq_sum_sq = vals
            global_freq = total_alt / (total_ref + total_alt) if (total_ref + total_alt) > 0 else 0.0
            mean_freq   = freq_sum / n_pos if n_pos > 0 else 0.0
            std_freq    = 0.0
            if n_pos > 1:
                var = (freq_sum_sq - freq_sum**2 / n_pos) / (n_pos - 1)
                std_freq = float(np.sqrt(max(var, 0.0)))

            rows.append({
                "conversion":      f"{ref_base}>{alt_base}",
                "ref_base":        ref_base,
                "alt_base":        alt_base,
                "n_positions":     n_pos,
                "total_ref_reads": total_ref,
                "total_alt_reads": total_alt,
                "global_freq":     global_freq,
                "mean_pos_freq":   mean_freq,
                "std_pos_freq":    std_freq,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, sep="\t", index=False, float_format="%.8f")
    logger.info(f"Wrote {len(df)} conversion types → {out_path}")

    # Print summary
    print("\n=== Mutation Spectrum Summary ===")
    print(f"{'Conversion':>12}  {'n_positions':>12}  {'global_freq':>12}  {'mean_pos_freq':>14}")
    print("-" * 56)
    for _, row in df.sort_values("global_freq", ascending=False).iterrows():
        marker = " ◀ ADAR" if row["conversion"] == "A>G" else ""
        print(f"  {row['conversion']:>10}  {row['n_positions']:>12,}  "
              f"{row['global_freq']:>12.6f}  {row['mean_pos_freq']:>14.6f}{marker}")

    # Save per-position A-dominant records if requested
    if args.save_positions and all_pos_records:
        pos_path = str(out_path) + ".positions.tsv.gz"
        pos_df = pd.DataFrame(all_pos_records,
                              columns=["chr", "pos", "ref_base", "ctrl_cov",
                                       "treat_A", "treat_C", "treat_G", "treat_T",
                                       "treat_total"])
        pos_df.to_csv(pos_path, sep="\t", index=False, compression="gzip")
        logger.info(f"Wrote {len(pos_df):,} A-dominant position records → {pos_path}")


if __name__ == "__main__":
    main()
