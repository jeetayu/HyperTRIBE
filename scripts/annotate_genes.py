#!/usr/bin/env python3
"""
Annotate Editing Sites with Gene Information
=============================================

Adds gene annotation from GTF file to editing sites.
INCLUDES FIX for gene annotation bug (prioritizes protein-coding over ncRNA).

Author: Optimized HyperTRIBE Pipeline v2.0
"""

import argparse
import sys
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Biotype priority for proper gene selection
# Higher number = higher priority
BIOTYPE_PRIORITY = {
    'protein_coding': 100,
    'lncRNA': 50,
    'lincRNA': 50,          # GENCODE uses lincRNA
    'miRNA': 45,
    'snRNA': 40,
    'snoRNA': 40,
    'scRNA': 35,
    'rRNA': 30,
    'misc_RNA': 20,
    'processed_transcript': 15,
    'sense_intronic': 12,
    'antisense': 12,
    'ribozyme': 10,
    'sRNA': 10,
    'scaRNA': 10,
    'ncRNA': 10,
    'TEC': 8,
    'pseudogene': 5,
    'processed_pseudogene': 4,
    'unprocessed_pseudogene': 3,
    'transcribed_unprocessed_pseudogene': 3,
    'transcribed_processed_pseudogene': 3,
    'Metazoa_SRP': 1,
    'Mt_tRNA': 1,
    'Mt_rRNA': 1,
}


class GTFParser:
    """Parse GTF annotation file with proper biotype prioritization"""
    
    def __init__(self, gtf_file: str):
        self.gtf_file = gtf_file
        self.genes = defaultdict(list)  # chromosome -> list of gene intervals
        
    @staticmethod
    def _infer_gene_type(attributes: Dict[str, str]) -> str:
        """
        Extract gene_type from GTF attributes.
        GENCODE/Ensembl GTFs carry gene_type or gene_biotype directly.
        UCSC iGenomes GTFs lack this field — infer from transcript_id prefix:
          NM_/XM_ = protein_coding,  NR_/XR_ = ncRNA.
        """
        gene_type = attributes.get('gene_type', attributes.get('gene_biotype', ''))
        if gene_type:
            return gene_type
        transcript_id = attributes.get('transcript_id', '')
        if transcript_id.startswith(('NM_', 'XM_')):
            return 'protein_coding'
        if transcript_id.startswith(('NR_', 'XR_')):
            return 'ncRNA'
        return '.'

    def parse(self):
        """
        Parse GTF file and build gene-level intervals.

        Handles two GTF flavours:
        - GENCODE/Ensembl: explicit 'gene' feature records with gene_type.
        - UCSC iGenomes:   only 'exon' records; gene intervals are reconstructed
                           by merging all exons per gene_id (min start, max end).
        """
        logger.info(f"Parsing GTF file: {self.gtf_file}")

        # key: (chrom, gene_id) -> gene_info dict (extended as we see more exons)
        gene_map: Dict[Tuple[str, str], dict] = {}

        with open(self.gtf_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue

                fields = line.strip().split('\t')
                if len(fields) < 9:
                    continue

                chrom  = fields[0]
                start  = int(fields[3])
                end    = int(fields[4])
                strand = fields[6]

                attributes = self.parse_attributes(fields[8])
                gene_id   = attributes.get('gene_id', '')
                gene_name = attributes.get('gene_name', gene_id)
                if not gene_id:
                    continue

                key = (chrom, gene_id)
                if key not in gene_map:
                    gene_map[key] = {
                        'start':     start,
                        'end':       end,
                        'strand':    strand,
                        'gene_name': gene_name,
                        'gene_id':   gene_id,
                        'gene_type': self._infer_gene_type(attributes),
                    }
                else:
                    # Extend the gene span to cover all its features (exons, CDS, etc.)
                    gene_map[key]['start'] = min(gene_map[key]['start'], start)
                    gene_map[key]['end']   = max(gene_map[key]['end'],   end)

        # Load into per-chromosome lists
        for (chrom, _), gene_info in gene_map.items():
            self.genes[chrom].append(gene_info)

        gene_count = sum(len(v) for v in self.genes.values())
        logger.info(f"Parsed {gene_count} genes")

        # Sort genes by start position for each chromosome
        for chrom in self.genes:
            self.genes[chrom].sort(key=lambda x: x['start'])
    
    def parse_attributes(self, attr_string: str) -> Dict[str, str]:
        """Parse GTF attributes field"""
        attributes = {}
        for item in attr_string.strip().split(';'):
            item = item.strip()
            if not item:
                continue
            
            parts = item.split(' ', 1)
            if len(parts) == 2:
                key = parts[0]
                value = parts[1].strip('"')
                attributes[key] = value
        
        return attributes
    
    def find_overlapping_genes(self, chrom: str, position: int) -> List[dict]:
        """
        Find genes that overlap a given position.
        Returns list sorted by priority (best gene first).
        """
        if chrom not in self.genes:
            return []
        
        overlapping = []
        for gene in self.genes[chrom]:
            if gene['start'] <= position <= gene['end']:
                overlapping.append(gene)
            elif gene['start'] > position:
                # Genes are sorted by start, so we can stop
                break
        
        if not overlapping:
            return []
        
        # CRITICAL FIX: Sort by biotype priority
        # This prevents generic ncRNA classes from dominating
        overlapping.sort(key=lambda g: (
            BIOTYPE_PRIORITY.get(g['gene_type'], 0),  # Highest priority biotype
            -(g['end'] - g['start']),  # Smallest gene (more specific)
            g['gene_name']  # Alphabetically (deterministic)
        ), reverse=True)
        
        return overlapping


def annotate_editing_sites(
    editing_sites_file: str,
    gtf_file: str,
    output_file: str
):
    """
    Annotate editing sites with gene information.
    
    Args:
        editing_sites_file: Input BED file with editing sites
        gtf_file: GTF annotation file
        output_file: Output annotated BED file
    """
    # Parse GTF
    gtf_parser = GTFParser(gtf_file)
    gtf_parser.parse()
    
    # Process editing sites
    logger.info(f"Annotating editing sites from {editing_sites_file}")
    
    annotated_count = 0
    unannotated_count = 0
    biotype_counts = defaultdict(int)
    multi_gene_count = 0
    
    with open(editing_sites_file, 'r') as inf, open(output_file, 'w') as outf:
        for line in inf:
            if line.startswith('#'):
                # Add extra columns to header
                header = line.strip() + '\tgene_id\tgene_type\n'
                outf.write(header)
                continue
            
            fields = line.strip().split('\t')
            chrom = fields[0]
            position = int(fields[1])
            
            # Find overlapping genes (sorted by priority)
            genes = gtf_parser.find_overlapping_genes(chrom, position)
            
            if genes:
                # Take best gene (first after priority sorting)
                # Protein-coding > lncRNA > ncRNA > Metazoa_SRP
                gene = genes[0]
                
                if len(genes) > 1:
                    multi_gene_count += 1
                    # Log first few cases for verification
                    if multi_gene_count <= 5:
                        gene_list = ', '.join([f"{g['gene_name']}({g['gene_type']})" 
                                              for g in genes[:3]])
                        logger.info(f"  Multiple genes at {chrom}:{position}, "
                                  f"chose {gene['gene_name']} ({gene['gene_type']}) "
                                  f"over {gene_list}")
                
                # Update gene name if not already set
                if fields[3] == '.':
                    fields[3] = gene['gene_name']
                
                # Update strand if not already set
                if fields[5] == '.':
                    fields[5] = gene['strand']
                
                # Add gene_id and gene_type
                annotated_line = '\t'.join(fields) + f"\t{gene['gene_id']}\t{gene['gene_type']}\n"
                annotated_count += 1
                biotype_counts[gene['gene_type']] += 1
            else:
                # No overlapping gene
                annotated_line = '\t'.join(fields) + "\t.\t.\n"
                unannotated_count += 1
            
            outf.write(annotated_line)
    
    logger.info(f"Annotated {annotated_count} sites with gene information")
    logger.info(f"{unannotated_count} sites in intergenic regions")
    if multi_gene_count > 0:
        logger.info(f"{multi_gene_count} sites had multiple overlapping genes "
                   f"(chose highest priority biotype)")
    
    logger.info(f"\nGene biotypes assigned:")
    for biotype, count in sorted(biotype_counts.items(), key=lambda x: -x[1])[:10]:
        pct = 100 * count / annotated_count if annotated_count > 0 else 0
        logger.info(f"  {biotype}: {count} ({pct:.1f}%)")
    
    logger.info(f"\nWrote annotated sites to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Annotate editing sites with gene information from GTF'
    )
    
    parser.add_argument('--editing-sites', required=True,
                       help='Input BED file with editing sites')
    parser.add_argument('--gtf', required=True,
                       help='GTF annotation file')
    parser.add_argument('--output', required=True,
                       help='Output annotated BED file')
    parser.add_argument('--threads', type=int, default=1,
                       help='Number of threads (not currently used)')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Annotating Editing Sites with Gene Information")
    logger.info("="*60)
    
    annotate_editing_sites(
        args.editing_sites,
        args.gtf,
        args.output
    )
    
    logger.info("="*60)
    logger.info("Annotation complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
