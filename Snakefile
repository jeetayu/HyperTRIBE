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

# Singularity containers — all bioinformatics tools run via Singularity
CUTADAPT_SIF  = "/data1/abdelwao/shared/containers/cutadapt_latest.sif"
STAR_SIF      = "/data1/abdelwao/shared/containers/star_2.7.10a_alpha_220506.sif"
SAMTOOLS_SIF  = "/data1/abdelwao/shared/containers/samtools_latest.sif"
BEDTOOLS_SIF  = "/data1/abdelwao/shared/containers/bedtools_v2.27.1dfsg-4-deb_cv1.sif"
DEEPTOOLS_SIF = "/data1/abdelwao/shared/containers/deeptools_latest.sif"
PICARD_SIF    = "/data1/abdelwao/shared/containers/picard_3.4.0.sif"
MULTIQC_SIF   = "/data1/abdelwao/shared/containers/multiqc_latest.sif"
SINGULARITY_BIND = "/data1/abdelwao:/data1/abdelwao"

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

# If control_dir is set in config, controls are already trimmed/aligned/deduped
# in that shared directory. Only treatment samples need local processing.
# Defaults to "aligned" (local) for backward compatibility.
CONTROL_DIR = config.get("control_dir", "aligned")
LOCAL_SAMPLES = TREATMENT_SAMPLES if CONTROL_DIR != "aligned" else ALL_SAMPLES

# ============================================================================
# Rule: all - Define final outputs
# ============================================================================

rule all:
    input:
        # Per-sample cutadapt QC logs (parsed by MultiQC)
        expand("qc/{sample}_cutadapt.txt", sample=LOCAL_SAMPLES),
        # Deduplicated + indexed alignments (input to editing caller)
        expand("aligned/{sample}.nodup.bam.bai", sample=LOCAL_SAMPLES),
        # MultiQC report (cutadapt + STAR + Picard all in one)
        "qc/multiqc_report.html",
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

rule cutadapt_trim:
    """
    Trim adapters and low-quality bases using cutadapt via Singularity.

    Matches the parameters used in validated HyperTRIBE scripts:
      -q 20        : trim low-quality bases from 3' end
      -u 6         : remove 6bp from 5' end of R1 (random hexamer artifact)
      -U 6         : remove 6bp from 5' end of R2
      --trim-n     : remove flanking N bases
      --minimum-length 25 : discard reads shorter than 25 bp after trimming
    MultiQC recognises cutadapt's stdout report format directly.
    """
    input:
        r1 = "data/fastq/{sample}_R1.fastq.gz",
        r2 = "data/fastq/{sample}_R2.fastq.gz"
    output:
        r1  = "trimmed/{sample}_R1_trimmed.fastq.gz",
        r2  = "trimmed/{sample}_R2_trimmed.fastq.gz",
        log = "qc/{sample}_cutadapt.txt"
    params:
        min_length  = config.get("min_read_length", 25),
        min_quality = config.get("min_base_quality", 20),
        sif  = CUTADAPT_SIF,
        bind = SINGULARITY_BIND
    threads: 4
    log: "logs/{sample}_cutadapt.log"
    shell:
        """
        singularity exec --bind {params.bind} {params.sif} \
            cutadapt \
            -q {params.min_quality} \
            -u 6 -U 6 \
            --trim-n \
            --minimum-length {params.min_length} \
            -j {threads} \
            -o {output.r1} -p {output.r2} \
            {input.r1} {input.r2} \
            > {output.log} 2> {log}
        """

# ============================================================================
# Alignment to Human Genome
# ============================================================================

rule star_align:
    """
    Align reads to human genome using STAR with HyperTRIBE-specific parameters.

    Key HyperTRIBE requirements:
      - Only uniquely mapping reads (outFilterMultimapNmax 1) to avoid
        multi-mappers inflating A→G counts at repetitive loci
      - Fraction-based mismatch filter (outFilterMismatchNoverLmax 0.06)
        rather than absolute count, which is better for variable read lengths
      - Strict splice-junction filters to reduce spurious editing calls
        at non-canonical junctions
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
        index      = STAR_INDEX,
        prefix     = "aligned/{sample}.",
        max_intron = config.get("max_intron_length", 500000),
        sif        = STAR_SIF,
        bind       = SINGULARITY_BIND
    threads: config.get("star_threads", 8)
    log: "logs/{sample}_star.log"
    shell:
        """
        singularity exec --bind {params.bind} {params.sif} \
        STAR \
            --runThreadN {threads} \
            --genomeDir {params.index} \
            --readFilesIn {input.r1} {input.r2} \
            --readFilesCommand zcat \
            --outFileNamePrefix {params.prefix} \
            --outSAMtype BAM SortedByCoordinate \
            --outSAMattributes All \
            --twopassMode Basic \
            --quantMode GeneCounts \
            --outFilterMultimapNmax 1 \
            --outFilterMismatchNoverLmax 0.06 \
            --outFilterScoreMinOverLread 0.3 \
            --outFilterMatchNminOverLread 0.3 \
            --outFilterMatchNmin 15 \
            --outSJfilterReads Unique \
            --outSAMstrandField intronMotif \
            --outFilterIntronMotifs RemoveNoncanonical \
            --alignMatesGapMax 25000 \
            --alignIntronMax {params.max_intron} \
            --limitBAMsortRAM 48000000000 \
            2> {log}
        """

rule index_bam:
    """Index initial BAM for Picard input"""
    input:
        "aligned/{sample}.Aligned.sortedByCoord.out.bam"
    output:
        "aligned/{sample}.Aligned.sortedByCoord.out.bam.bai"
    params:
        sif  = SAMTOOLS_SIF,
        bind = SINGULARITY_BIND
    threads: 1
    log: "logs/{sample}_index.log"
    shell:
        "singularity exec --bind {params.bind} {params.sif} samtools index {input} 2> {log}"

rule add_read_groups:
    """
    Add @RG header and per-read RG:Z tags required by Picard MarkDuplicates 3.x.

    STAR does not emit read-group tags by default. Picard 3.x throws a
    NullPointerException if any read lacks an RG tag. This lightweight step
    stamps every read with a minimal read group (ID + SM) using samtools
    addreplacerg, which runs in O(n) time without touching alignment data.
    """
    input:
        bam = "aligned/{sample}.Aligned.sortedByCoord.out.bam",
        bai = "aligned/{sample}.Aligned.sortedByCoord.out.bam.bai"
    output:
        bam = temp("aligned/{sample}.rg.bam")
    params:
        sif  = SAMTOOLS_SIF,
        bind = SINGULARITY_BIND
    threads: 1
    log: "logs/{sample}_addRG.log"
    shell:
        """
        singularity exec --bind {params.bind} {params.sif} \
            samtools addreplacerg \
            -r $'ID:{wildcards.sample}\\tSM:{wildcards.sample}\\tPL:ILLUMINA\\tLB:{wildcards.sample}\\tPU:{wildcards.sample}' \
            -m overwrite_all \
            -o {output.bam} \
            {input.bam} \
            2> {log}
        """

rule picard_dedup:
    """
    Remove PCR duplicates with Picard MarkDuplicates.

    Critical for HyperTRIBE: PCR duplicates amplify editing signals from
    single molecules, creating false-positive high-frequency edit sites.
    Removing them ensures each A→G call represents an independent RNA molecule.
    Uses the pre-built Singularity container (Picard not in conda env).
    """
    input:
        bam = "aligned/{sample}.rg.bam"
    output:
        bam     = "aligned/{sample}.nodup.bam",
        metrics = "aligned/{sample}.dup_metrics.txt"
    params:
        picard_sif = PICARD_SIF,
        bind       = SINGULARITY_BIND,
        tmp        = "tmp"
    log: "logs/{sample}_dedup.log"
    shell:
        """
        mkdir -p {params.tmp}
        singularity exec --bind {params.bind} {params.picard_sif} \
            java -jar /usr/picard/picard.jar MarkDuplicates \
            INPUT={input.bam} \
            OUTPUT={output.bam} \
            METRICS_FILE={output.metrics} \
            VALIDATION_STRINGENCY=LENIENT \
            REMOVE_DUPLICATES=true \
            TMP_DIR={params.tmp} \
            ASSUME_SORTED=true \
            2> {log}
        """

rule index_dedup_bam:
    """Index deduplicated BAM — this is the final BAM used for all downstream steps"""
    input:
        "aligned/{sample}.nodup.bam"
    output:
        "aligned/{sample}.nodup.bam.bai"
    params:
        sif  = SAMTOOLS_SIF,
        bind = SINGULARITY_BIND
    threads: 1
    log: "logs/{sample}_index_dedup.log"
    shell:
        "singularity exec --bind {params.bind} {params.sif} samtools index {input} 2> {log}"

# ============================================================================
# MultiQC — aggregate QC across all samples
# ============================================================================

rule multiqc:
    """
    Aggregate cutadapt trimming stats, STAR alignment stats, and Picard duplicate
    metrics into a single MultiQC HTML report.

    Inputs are all collected before this rule fires so the report always covers
    every sample. Uses the pre-built Singularity container.

    Key sections in the report:
      cutadapt → read quality, adapter content, 6-bp 5'-trim effect
      STAR     → uniquely mapped %, multimapper %, unmapped reads
      Picard   → duplication rate per sample (critical QC for HyperTRIBE)
    """
    input:
        cutadapt_logs = expand("qc/{sample}_cutadapt.txt", sample=LOCAL_SAMPLES),
        star_logs     = expand("aligned/{sample}.Log.final.out", sample=LOCAL_SAMPLES),
        dup_metrics   = expand("aligned/{sample}.dup_metrics.txt", sample=LOCAL_SAMPLES)
    output:
        html = "qc/multiqc_report.html",
        data = directory("qc/multiqc_report_data")
    params:
        multiqc_sif = MULTIQC_SIF,
        bind        = SINGULARITY_BIND
    log: "logs/multiqc.log"
    shell:
        """
        singularity exec --bind {params.bind} {params.multiqc_sif} \
            multiqc \
            qc/ \
            aligned/ \
            --outdir qc/ \
            --filename multiqc_report.html \
            --force \
            --title "HyperTRIBE QC Report" \
            2> {log}
        """

# ============================================================================
# Editing Site Calling
# ============================================================================

rule call_editing_sites:
    """
    Call RNA editing sites using optimized parallel algorithm.

    This replaces the MySQL-based approach with direct BAM processing.
    Inputs are deduplicated BAMs from picard_dedup.
    """
    input:
        control = expand(CONTROL_DIR + "/{sample}.nodup.bam",
                        sample=CONTROL_SAMPLES),
        control_idx = expand(CONTROL_DIR + "/{sample}.nodup.bam.bai",
                            sample=CONTROL_SAMPLES),
        treatment = expand("aligned/{sample}.nodup.bam",
                          sample=TREATMENT_SAMPLES),
        treatment_idx = expand("aligned/{sample}.nodup.bam.bai",
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
        expand("aligned/{sample}.Log.final.out", sample=LOCAL_SAMPLES)
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
        cutadapt_logs  = expand("qc/{sample}_cutadapt.txt", sample=LOCAL_SAMPLES),
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
    """Generate BigWig files for genome browser visualization (uses deduped BAM)."""
    input:
        bam = "aligned/{sample}.nodup.bam",
        bai = "aligned/{sample}.nodup.bam.bai"
    output:
        "bigwig/{sample}.bw"
    params:
        sif  = DEEPTOOLS_SIF,
        bind = SINGULARITY_BIND
    threads: 4
    log: "logs/{sample}_bigwig.log"
    shell:
        """
        singularity exec --bind {params.bind} {params.sif} \
            bamCoverage \
            -b {input.bam} \
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
