# Optimized HyperTRIBE Snakemake Workflow
# ==========================================
# Author: Optimized HyperTRIBE v2.0
# For human genome analysis (hg38)

import os
from pathlib import Path

# ============================================================================
# Pipeline paths (resolved relative to the Snakefile, not the working dir)
# This lets users run: snakemake --snakefile /path/to/Snakefile
# ============================================================================

PIPELINE_DIR  = workflow.basedir
SCRIPTS_DIR   = os.path.join(PIPELINE_DIR, "scripts")
NOTEBOOKS_DIR = os.path.join(PIPELINE_DIR, "notebooks")

# ============================================================================
# Configuration
# ============================================================================

configfile: "config.yaml"

# Extract configuration
SAMPLES = config["samples"]
CONTROL_SAMPLES = config["control_samples"]
TREATMENT_SAMPLES = config["treatment_samples"]
GENOME = config["genome"]
STAR_INDEX = config["star_index"]
ANNOTATION = config["annotation"]

# All samples combined
ALL_SAMPLES = CONTROL_SAMPLES + TREATMENT_SAMPLES

# ============================================================================
# Rule: all - Define final outputs
# ============================================================================

rule all:
    input:
        # QC reports
        expand("qc/{sample}_fastp.html", sample=ALL_SAMPLES),
        # Alignments
        expand("aligned/{sample}.Aligned.sortedByCoord.out.bam.bai", sample=ALL_SAMPLES),
        # Editing sites
        "results/raw_editing_sites.bed",
        "results/filtered_editing_sites.bed",
        "results/annotated_editing_sites.bed",
        # Gene lists
        "results/target_genes.txt",
        "results/target_genes_by_editcount.txt",
        # Summary reports
        "results/analysis_report.html",
        "results/alignment_stats.txt",
        # Visualization
        "results/plots/editing_frequency_distribution.pdf",
        "results/plots/chromosome_distribution.pdf",
        "results/plots/gene_biotype_distribution.pdf"

# ============================================================================
# Quality Control and Trimming
# ============================================================================

rule fastp_trim:
    """
    Trim adapters and low-quality bases using fastp.
    
    fastp is faster than Trimmomatic and provides comprehensive QC.
    Removes 6bp from 5' end to avoid random hexamer priming artifacts.
    """
    input:
        r1 = "data/fastq/{sample}_R1.fastq.gz",
        r2 = "data/fastq/{sample}_R2.fastq.gz"
    output:
        r1 = "trimmed/{sample}_R1_trimmed.fastq.gz",
        r2 = "trimmed/{sample}_R2_trimmed.fastq.gz",
        html = "qc/{sample}_fastp.html",
        json = "qc/{sample}_fastp.json"
    params:
        min_length = config.get("min_read_length", 50),
        min_quality = config.get("min_base_quality", 20),
        trim_front = 6  # Remove 6bp from 5' end
    threads: 4
    log: "logs/{sample}_fastp.log"
    shell:
        """
        fastp \
            -i {input.r1} -I {input.r2} \
            -o {output.r1} -O {output.r2} \
            --thread {threads} \
            --html {output.html} \
            --json {output.json} \
            --qualified_quality_phred {params.min_quality} \
            --length_required {params.min_length} \
            --trim_front1 {params.trim_front} \
            --trim_front2 {params.trim_front} \
            --detect_adapter_for_pe \
            2> {log}
        """

# ============================================================================
# Alignment to Human Genome
# ============================================================================

rule star_align:
    """
    Align reads to human transcriptome using STAR.
    
    Uses 2-pass mode for better splice junction detection.
    Optimized parameters for human RNA-seq.
    """
    input:
        r1 = "trimmed/{sample}_R1_trimmed.fastq.gz",
        r2 = "trimmed/{sample}_R2_trimmed.fastq.gz"
    output:
        bam = "aligned/{sample}.Aligned.sortedByCoord.out.bam",
        log_final = "aligned/{sample}.Log.final.out",
        log_progress = "aligned/{sample}.Log.progress.out",
        log_out = "aligned/{sample}.Log.out",
        sj = "aligned/{sample}.SJ.out.tab"
    params:
        index = STAR_INDEX,
        prefix = "aligned/{sample}.",
        max_multimap = config.get("max_multimapping", 10),
        max_mismatch = config.get("max_mismatches", 3),
        max_intron = config.get("max_intron_length", 500000)
    threads: config.get("star_threads", 8)
    log: "logs/{sample}_star.log"
    shell:
        """
        STAR \
            --runThreadN {threads} \
            --genomeDir {params.index} \
            --readFilesIn {input.r1} {input.r2} \
            --readFilesCommand zcat \
            --outFileNamePrefix {params.prefix} \
            --outSAMtype BAM SortedByCoordinate \
            --outSAMattributes All \
            --outFilterMultimapNmax {params.max_multimap} \
            --outFilterMismatchNmax {params.max_mismatch} \
            --alignIntronMax {params.max_intron} \
            --limitBAMsortRAM 32000000000 \
            --outSAMstrandField intronMotif \
            --twopassMode Basic \
            2> {log}
        """

rule index_bam:
    """Index BAM files for rapid random access"""
    input:
        "aligned/{sample}.Aligned.sortedByCoord.out.bam"
    output:
        "aligned/{sample}.Aligned.sortedByCoord.out.bam.bai"
    threads: 1
    log: "logs/{sample}_index.log"
    shell:
        "samtools index {input} 2> {log}"

# ============================================================================
# Editing Site Calling
# ============================================================================

rule call_editing_sites:
    """
    Call RNA editing sites using optimized parallel algorithm.
    
    This replaces the MySQL-based approach with direct BAM processing.
    Dramatically faster and more memory efficient.
    """
    input:
        control = expand("aligned/{sample}.Aligned.sortedByCoord.out.bam", 
                        sample=CONTROL_SAMPLES),
        control_idx = expand("aligned/{sample}.Aligned.sortedByCoord.out.bam.bai",
                            sample=CONTROL_SAMPLES),
        treatment = expand("aligned/{sample}.Aligned.sortedByCoord.out.bam",
                          sample=TREATMENT_SAMPLES),
        treatment_idx = expand("aligned/{sample}.Aligned.sortedByCoord.out.bam.bai",
                              sample=TREATMENT_SAMPLES)
    output:
        "results/raw_editing_sites.bed"
    params:
        min_coverage = config.get("min_coverage", 20),
        min_edit_freq = config.get("min_edit_freq", 0.05),
        edit_fold_change = config.get("edit_fold_change", 2.0),
        p_threshold = config.get("p_value_threshold", 0.05),
        chunk_size = config.get("chunk_size", 10000000)
    threads: config.get("calling_threads", 16)
    log: "logs/call_editing_sites.log"
    shell:
        """
        python {SCRIPTS_DIR}/call_editing_sites_parallel.py \
            --control {input.control} \
            --treatment {input.treatment} \
            --output {output} \
            --threads {threads} \
            --chunk-size {params.chunk_size} \
            --min-coverage {params.min_coverage} \
            --min-edit-freq {params.min_edit_freq} \
            --edit-fold-change {params.edit_fold_change} \
            --p-value-threshold {params.p_threshold} \
            2> {log}
        """

# ============================================================================
# Filtering and Annotation
# ============================================================================

rule filter_replicates:
    """
    Filter editing sites for reproducibility across replicates.
    
    Sites must appear in minimum number of replicates to be retained.
    """
    input:
        "results/raw_editing_sites.bed"
    output:
        "results/filtered_editing_sites.bed"
    params:
        min_replicates = config.get("min_replicates", 2)
    log: "logs/filter_replicates.log"
    shell:
        """
        python {SCRIPTS_DIR}/filter_replicates.py \
            --input {input} \
            --min-replicates {params.min_replicates} \
            --output {output} \
            2> {log}
        """

rule annotate_genes:
    """
    Annotate editing sites with gene information from GTF.
    
    Adds gene name, gene ID, transcript, exon number, biotype, etc.
    """
    input:
        editing_sites = "results/filtered_editing_sites.bed",
        gtf = ANNOTATION
    output:
        "results/annotated_editing_sites.bed"
    threads: 4
    log: "logs/annotate_genes.log"
    shell:
        """
        python {SCRIPTS_DIR}/annotate_genes.py \
            --editing-sites {input.editing_sites} \
            --gtf {input.gtf} \
            --output {output} \
            --threads {threads} \
            2> {log}
        """

# ============================================================================
# Extract Target Genes
# ============================================================================

rule extract_target_genes:
    """Extract unique list of target genes"""
    input:
        "results/annotated_editing_sites.bed"
    output:
        "results/target_genes.txt"
    shell:
        """
        # Extract gene names (column 4), remove duplicates, sort
        awk '!/^#/ {{print $4}}' {input} | \
            sort -u > {output}
        """

rule rank_genes_by_editing:
    """
    Rank genes by number of editing sites.
    
    Identifies the most heavily edited genes.
    """
    input:
        "results/annotated_editing_sites.bed"
    output:
        "results/target_genes_by_editcount.txt"
    shell:
        """
        # Count editing sites per gene, sort by count
        awk '!/^#/ {{print $4}}' {input} | \
            sort | uniq -c | sort -rn | \
            awk '{{print $2"\\t"$1}}' > {output}
        """

# ============================================================================
# Quality Control and Statistics
# ============================================================================

rule compile_alignment_stats:
    """Compile alignment statistics from all samples"""
    input:
        expand("aligned/{sample}.Log.final.out", sample=ALL_SAMPLES)
    output:
        "results/alignment_stats.txt"
    shell:
        """
        echo "Sample\tTotal_Reads\tUniquely_Mapped\tUniquely_Mapped_Pct\tMulti_Mapped\tUnmapped" > {output}
        for log in {input}; do
            sample=$(basename $log .Log.final.out)
            total=$(grep "Number of input reads" $log | awk '{{print $NF}}')
            unique=$(grep "Uniquely mapped reads number" $log | awk '{{print $NF}}')
            unique_pct=$(grep "Uniquely mapped reads %" $log | awk '{{print $NF}}')
            multi=$(grep "Number of reads mapped to multiple loci" $log | awk '{{print $NF}}')
            unmapped=$(grep "Number of reads unmapped: too short" $log | awk '{{print $NF}}')
            echo -e "$sample\t$total\t$unique\t$unique_pct\t$multi\t$unmapped"
        done >> {output}
        """

# ============================================================================
# Visualization
# ============================================================================

rule plot_editing_frequency_distribution:
    """Plot distribution of editing frequencies"""
    input:
        "results/annotated_editing_sites.bed"
    output:
        "results/plots/editing_frequency_distribution.pdf"
    log: "logs/plot_edit_freq.log"
    shell:
        """
        python {SCRIPTS_DIR}/plot_editing_distribution.py \
            --input {input} \
            --output {output} \
            2> {log}
        """

rule plot_chromosome_distribution:
    """Plot distribution of editing sites across chromosomes"""
    input:
        "results/annotated_editing_sites.bed"
    output:
        "results/plots/chromosome_distribution.pdf"
    log: "logs/plot_chr_dist.log"
    shell:
        """
        python {SCRIPTS_DIR}/plot_chromosome_distribution.py \
            --input {input} \
            --output {output} \
            2> {log}
        """

rule plot_gene_biotype_distribution:
    """Plot distribution of editing sites by gene biotype"""
    input:
        "results/annotated_editing_sites.bed"
    output:
        "results/plots/gene_biotype_distribution.pdf"
    log: "logs/plot_biotype.log"
    shell:
        """
        python {SCRIPTS_DIR}/plot_biotype_distribution.py \
            --input {input} \
            --output {output} \
            2> {log}
        """

# ============================================================================
# Generate Final Report
# ============================================================================

rule generate_report:
    """
    Render the Quarto HTML report with all pipeline results.

    Quarto executes the notebook in the analysis working directory so that
    relative paths (results/, qc/, config.yaml) resolve correctly.
    The output is a self-contained HTML file with embedded figures.
    """
    input:
        editing_sites  = "results/annotated_editing_sites.bed",
        alignment_stats = "results/alignment_stats.txt",
        fastp_jsons    = expand("qc/{sample}_fastp.json", sample=ALL_SAMPLES),
        gene_list      = "results/target_genes.txt",
        gene_ranks     = "results/target_genes_by_editcount.txt",
        plots = [
            "results/plots/editing_frequency_distribution.pdf",
            "results/plots/chromosome_distribution.pdf",
            "results/plots/gene_biotype_distribution.pdf",
        ],
        notebook = os.path.join(NOTEBOOKS_DIR, "hypertribe_report.qmd"),
    output:
        "results/analysis_report.html"
    params:
        outdir   = lambda wc, output: str(Path(output[0]).parent.resolve()),
        execdir  = lambda wc: str(Path.cwd()),
        notebook = os.path.join(NOTEBOOKS_DIR, "hypertribe_report.qmd"),
    log: "logs/generate_report.log"
    shell:
        """
        quarto render {params.notebook} \
            --execute-dir {params.execdir} \
            --output-dir  {params.outdir} \
            --output      analysis_report.html \
            2> {log}
        """

# ============================================================================
# Optional Rules
# ============================================================================

rule generate_bigwig:
    """
    Generate BigWig files for genome browser visualization.
    
    Optional - only runs if enabled in config.
    """
    input:
        "aligned/{sample}.Aligned.sortedByCoord.out.bam"
    output:
        "bigwig/{sample}.bw"
    threads: 4
    log: "logs/{sample}_bigwig.log"
    shell:
        """
        bamCoverage \
            -b {input} \
            -o {output} \
            --binSize 10 \
            --normalizeUsing CPM \
            --numberOfProcessors {threads} \
            2> {log}
        """

# ============================================================================
# Cleanup Rules
# ============================================================================

rule clean_temp:
    """Remove temporary files to save disk space"""
    shell:
        """
        rm -f aligned/*.tab
        rm -f aligned/*.progress.out
        rm -f aligned/*.out
        """

# ============================================================================
# End of Snakefile
# ============================================================================
