#!/usr/bin/env python3
"""
Annotate Editing Sites with Splice Site Proximity
==================================================

For each editing site within a gene, classifies its position relative to
exon/intron structure and annotates proximity to splice site elements:

  feature_type : CDS_exon | UTR | intron | intergenic
  dist_5ss     : nt from the 5' splice site (donor); NA if not intronic
  dist_3ss     : nt from the 3' splice site (acceptor); NA if not intronic
  splice_region:
      5ss_proximal  — within 8 nt of donor (GU consensus region)
      3ss_proximal  — within 3 nt of acceptor (AG dinucleotide)
      ppt_region    — 4–50 nt upstream of acceptor (polypyrimidine tract)
      deep_intronic — > 50 nt from both splice sites
      CDS_exon / UTR / intergenic — not intronic

Distances are strand-aware:
  + strand: 5'SS is at the right edge of the upstream exon;
             3'SS is at the left edge of the downstream exon.
  - strand: 5'SS is at the left edge of the downstream (higher-coord) exon;
             3'SS is at the right edge of the upstream (lower-coord) exon.

Usage:
    python annotate_splice_sites.py \\
        --editing-sites results/annotated_editing_sites.bed \\
        --gtf           /path/to/genes.gtf \\
        --output        results/splice_annotated_editing_sites.bed
"""

import argparse
import logging
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Splice site proximity thresholds (nucleotides from boundary)
DONOR_PROXIMAL_NT    = 8   # GU + 6 nt of 5'SS consensus
ACCEPTOR_PROXIMAL_NT = 3   # AG + 1 intronic nt
PPT_MAX_NT           = 50  # polypyrimidine tract upper bound from 3'SS
PPT_MIN_NT           = 4   # PPT starts just after the 3'SS proximal zone


def parse_attributes(attr_string: str) -> Dict[str, str]:
    attrs = {}
    for item in attr_string.strip().split(';'):
        item = item.strip()
        if not item:
            continue
        parts = item.split(' ', 1)
        if len(parts) == 2:
            attrs[parts[0]] = parts[1].strip('"')
    return attrs


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping/adjacent 1-based inclusive intervals."""
    if not intervals:
        return []
    merged = [sorted(intervals)[0]]
    for start, end in sorted(intervals)[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


class SpliceSiteAnnotator:
    """
    Builds per-gene exon and UTR databases from a GTF, then classifies
    editing site positions relative to the splicing machinery.
    """

    def __init__(self, gtf_file: str):
        self.gtf_file = gtf_file
        # (chrom, gene_id) -> sorted merged exon intervals [(start, end), ...] 1-based
        self._exons:  Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
        # (chrom, gene_id) -> sorted merged UTR intervals
        self._utrs:   Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
        # (chrom, gene_id) -> strand
        self._strand: Dict[Tuple[str, str], str] = {}

    # ------------------------------------------------------------------ #
    # GTF parsing                                                          #
    # ------------------------------------------------------------------ #

    def parse(self):
        logger.info(f"Parsing GTF: {self.gtf_file}")

        raw_exons: Dict[Tuple[str, str], List[Tuple[int, int]]] = defaultdict(list)
        raw_utrs:  Dict[Tuple[str, str], List[Tuple[int, int]]] = defaultdict(list)

        with open(self.gtf_file) as fh:
            for line in fh:
                if line.startswith('#'):
                    continue
                fields = line.rstrip('\n').split('\t')
                if len(fields) < 9:
                    continue

                feature = fields[2]
                if feature not in ('exon', 'UTR'):
                    continue

                chrom  = fields[0]
                start  = int(fields[3])   # 1-based, inclusive
                end    = int(fields[4])
                strand = fields[6]
                attrs  = parse_attributes(fields[8])

                gene_id = attrs.get('gene_id', '')
                if not gene_id:
                    continue

                key = (chrom, gene_id)
                self._strand[key] = strand

                if feature == 'exon':
                    raw_exons[key].append((start, end))
                else:  # UTR
                    raw_utrs[key].append((start, end))

        # Merge overlapping intervals per gene
        for key, ivs in raw_exons.items():
            self._exons[key] = _merge_intervals(ivs)
        for key, ivs in raw_utrs.items():
            self._utrs[key] = _merge_intervals(ivs)

        n_genes = len(self._exons)
        logger.info(f"Loaded exon structures for {n_genes:,} gene loci")

    # ------------------------------------------------------------------ #
    # Site classification                                                   #
    # ------------------------------------------------------------------ #

    def classify(
        self,
        chrom: str,
        pos_0based: int,
        gene_id: str,
        strand: str,
    ) -> Tuple[str, Optional[int], Optional[int], str]:
        """
        Classify one editing site.

        Parameters
        ----------
        chrom      : chromosome (must match GTF)
        pos_0based : 0-based BED start coordinate
        gene_id    : gene_id from the annotation BED (column 16)
        strand     : strand from the annotation BED (column 6)

        Returns
        -------
        (feature_type, dist_5ss, dist_3ss, splice_region)
        feature_type : 'CDS_exon' | 'UTR' | 'intron' | 'intergenic'
        dist_5ss     : int or None
        dist_3ss     : int or None
        splice_region: string label
        """
        if gene_id in ('.', ''):
            return 'intergenic', None, None, 'intergenic'

        pos1 = pos_0based + 1   # convert to 1-based

        key = (chrom, gene_id)
        exons = self._exons.get(key)
        if not exons:
            return 'intergenic', None, None, 'intergenic'

        # ── Is the site in a UTR? ───────────────────────────────────────
        utrs = self._utrs.get(key, [])
        in_utr = any(s <= pos1 <= e for s, e in utrs)

        # ── Is the site in any exon? ────────────────────────────────────
        in_exon = any(s <= pos1 <= e for s, e in exons)

        if in_exon:
            ftype = 'UTR' if in_utr else 'CDS_exon'
            return ftype, None, None, ftype

        # ── Intronic: find flanking exons in genomic coordinates ────────
        left_exon  = None   # exon with max end  < pos1
        right_exon = None   # exon with min start > pos1

        for s, e in exons:
            if e < pos1:
                left_exon = (s, e)
            elif s > pos1 and right_exon is None:
                right_exon = (s, e)
                break

        if left_exon is None or right_exon is None:
            # Outside the exon span (e.g. far upstream/downstream UTR extension)
            return 'intergenic', None, None, 'intergenic'

        # ── Strand-aware splice site distances ─────────────────────────
        # dist_to_left  = how far the site is from the right edge of the left exon
        # dist_to_right = how far the site is from the left edge of the right exon
        dist_to_left  = pos1 - left_exon[1]    # > 0 when intronic
        dist_to_right = right_exon[0] - pos1   # > 0 when intronic

        if strand == '+':
            # 5'SS donor  is at the right boundary of the left exon
            # 3'SS acceptor is at the left boundary of the right exon
            dist_5ss = dist_to_left
            dist_3ss = dist_to_right
        else:
            # On – strand the pre-mRNA runs right→left in genomic coords.
            # 5'SS donor  is at the left boundary of the right exon
            # 3'SS acceptor is at the right boundary of the left exon
            dist_5ss = dist_to_right
            dist_3ss = dist_to_left

        # ── Classify splice region ─────────────────────────────────────
        if dist_5ss <= DONOR_PROXIMAL_NT:
            region = '5ss_proximal'
        elif dist_3ss <= ACCEPTOR_PROXIMAL_NT:
            region = '3ss_proximal'
        elif dist_3ss <= PPT_MAX_NT:
            region = 'ppt_region'
        else:
            region = 'deep_intronic'

        return 'intron', dist_5ss, dist_3ss, region


# ------------------------------------------------------------------ #
# Main annotation loop                                                  #
# ------------------------------------------------------------------ #

def annotate(editing_sites: str, gtf: str, output: str):
    annotator = SpliceSiteAnnotator(gtf)
    annotator.parse()

    counts: Dict[str, int] = defaultdict(int)
    total = 0

    with open(editing_sites) as inf, open(output, 'w') as outf:
        for line in inf:
            if line.startswith('#'):
                header = line.rstrip('\n') + '\tfeature_type\tdist_5ss\tdist_3ss\tsplice_region\n'
                outf.write(header)
                continue

            fields = line.rstrip('\n').split('\t')
            chrom   = fields[0]
            pos     = int(fields[1])          # 0-based BED
            strand  = fields[5]  if len(fields) > 5  else '.'
            gene_id = fields[15] if len(fields) > 15 else '.'

            ftype, d5, d3, region = annotator.classify(chrom, pos, gene_id, strand)

            d5_str = str(d5) if d5 is not None else 'NA'
            d3_str = str(d3) if d3 is not None else 'NA'

            outf.write('\t'.join(fields) + f'\t{ftype}\t{d5_str}\t{d3_str}\t{region}\n')
            counts[region] += 1
            total += 1

    logger.info(f"Annotated {total:,} editing sites:")
    for region, n in sorted(counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {region:20s}: {n:5d}  ({100*n/total:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Annotate editing sites with splice site proximity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--editing-sites', required=True,
                        help='Annotated editing sites BED (output of annotate_genes.py)')
    parser.add_argument('--gtf',           required=True,
                        help='GTF annotation file (GENCODE recommended)')
    parser.add_argument('--output',        required=True,
                        help='Output BED with splice site columns appended')
    args = parser.parse_args()

    logger.info('=' * 60)
    logger.info('Splice Site Proximity Annotation')
    logger.info('=' * 60)
    annotate(args.editing_sites, args.gtf, args.output)
    logger.info('=' * 60)
    logger.info('Done.')
    logger.info('=' * 60)


if __name__ == '__main__':
    main()
