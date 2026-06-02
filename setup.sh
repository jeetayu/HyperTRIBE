#!/usr/bin/env bash
# =============================================================================
# HyperTRIBE Pipeline v2.0 — Interactive Setup Wizard
# =============================================================================
# Prompts for SLURM cluster settings and reference file paths, then writes:
#   config.yaml          — analysis parameters and sample info template
#   slurm/cluster.yaml   — per-rule SLURM resource configuration
#   submit_pipeline.sh   — SLURM controller job script
#
# Usage:
#   bash setup.sh              # interactive
#   bash setup.sh --defaults   # write defaults without prompting (CI/test)
# =============================================================================

set -euo pipefail

# ── Terminal colours ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

# ── Helpers ───────────────────────────────────────────────────────────────────

banner() {
    echo -e "\n${BLUE}${BOLD}══════════════════════════════════════════════${NC}"
    echo -e "${BLUE}${BOLD}  $1${NC}"
    echo -e "${BLUE}${BOLD}══════════════════════════════════════════════${NC}\n"
}

section() {
    echo -e "\n${CYAN}── $1 ──${NC}"
}

ok()   { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC}  $1"; }
err()  { echo -e "${RED}✗${NC}  $1"; }
info() { echo -e "  ${BLUE}→${NC} $1"; }

# Prompt with optional default. Sets variable $REPLY_VAL.
ask() {
    local prompt="$1"
    local default="${2:-}"
    if [[ -n "$default" ]]; then
        echo -ne "  ${prompt} ${YELLOW}[${default}]${NC}: "
    else
        echo -ne "  ${prompt}: "
    fi
    read -r user_input
    REPLY_VAL="${user_input:-$default}"
}

# Prompt for a filesystem path; warns if it doesn't exist (non-fatal).
ask_path() {
    local prompt="$1"
    local default="${2:-}"
    ask "$prompt" "$default"
    if [[ -n "$REPLY_VAL" && "$REPLY_VAL" != "SKIP" && ! -e "$REPLY_VAL" ]]; then
        warn "Path not found yet: $REPLY_VAL"
        info "(The path will be written to config — ensure it exists on the HPC)"
    fi
}

# y/n question; returns 0 for yes, 1 for no.
confirm() {
    local prompt="$1"
    local default="${2:-y}"
    ask "$prompt (y/n)" "$default"
    [[ "$REPLY_VAL" =~ ^[Yy] ]]
}

# ── Default mode ──────────────────────────────────────────────────────────────
DEFAULTS_MODE=false
[[ "${1:-}" == "--defaults" ]] && DEFAULTS_MODE=true

if $DEFAULTS_MODE; then
    echo "Running in --defaults mode (non-interactive)"
fi

# =============================================================================
# WELCOME
# =============================================================================

clear
banner "HyperTRIBE Pipeline v2.0 — Setup Wizard"
echo "This wizard will configure the pipeline for your HPC environment."
echo "It writes three files:"
echo "  • config.yaml          — analysis parameters and sample names"
echo "  • slurm/cluster.yaml   — SLURM resource limits per pipeline rule"
echo "  • submit_pipeline.sh   — controller job to submit to SLURM"
echo ""
echo "You can re-run this wizard at any time to update your configuration."
echo ""

if ! $DEFAULTS_MODE; then
    read -rp "Press Enter to begin, or Ctrl-C to cancel..."
fi

# =============================================================================
# STEP 1: PROJECT INFORMATION
# =============================================================================

banner "Step 1: Project Information"

ask "Project name (no spaces)" "HyperTRIBE_Analysis"
PROJECT_NAME="$REPLY_VAL"

ask "Your name (for SLURM email subject)"  "Researcher"
USER_NAME="$REPLY_VAL"

ask "Notification email" "jbiswas@gmail.com"
USER_EMAIL="$REPLY_VAL"

ask "Organism (Homo sapiens / Mus musculus)" "Homo sapiens"
ORGANISM="$REPLY_VAL"

ask "Genome build (hg38 / mm10)" "hg38"
GENOME_BUILD="$REPLY_VAL"

ok "Project: ${PROJECT_NAME} | Genome: ${GENOME_BUILD}"

# =============================================================================
# STEP 2: SLURM CLUSTER SETTINGS
# =============================================================================

banner "Step 2: SLURM Cluster Settings"
echo "These settings control how jobs are submitted to your cluster."
echo ""
info "To find your account/partition names, run: sacctmgr show user \$USER"
echo ""

ask "SLURM account name" ""
SLURM_ACCOUNT="$REPLY_VAL"
[[ -z "$SLURM_ACCOUNT" ]] && warn "No account set — some clusters require this"

ask "Default partition (standard jobs)" "componc_cpu"
SLURM_PARTITION="$REPLY_VAL"

ask "High-memory partition (edit-calling step, ≥128 GB)" "${SLURM_PARTITION}"
SLURM_BIGMEM_PARTITION="$REPLY_VAL"

ask "Max wall time for STAR alignment (HH:MM:SS)" "08:00:00"
STAR_TIME="$REPLY_VAL"

ask "Max wall time for edit calling (HH:MM:SS)" "48:00:00"
CALLING_TIME="$REPLY_VAL"

ask "Max concurrent SLURM jobs" "50"
MAX_JOBS="$REPLY_VAL"

ok "SLURM: account=${SLURM_ACCOUNT} partition=${SLURM_PARTITION}"

# =============================================================================
# STEP 3: REFERENCE FILES
# =============================================================================

banner "Step 3: Reference Files (HPC paths)"
echo "Enter the full paths to reference files on the HPC."
echo "Type SKIP to leave a placeholder (edit config.yaml manually later)."
echo ""
info "GENCODE annotation: /data/genomes/hg38/gencode.v38.annotation.gtf"
info "STAR index example: /data/genomes/hg38/STAR_index/"
info "Genome FASTA:       /data/genomes/hg38/GRCh38.primary_assembly.genome.fa"
echo ""

ask_path "Genome FASTA file (.fa / .fasta)" "/path/to/hg38.fa"
[[ "$REPLY_VAL" == "SKIP" ]] && REPLY_VAL="/path/to/hg38.fa"
GENOME_FASTA="$REPLY_VAL"

ask_path "STAR genome index directory" "/path/to/star_index/hg38"
[[ "$REPLY_VAL" == "SKIP" ]] && REPLY_VAL="/path/to/star_index/hg38"
STAR_INDEX="$REPLY_VAL"

ask_path "Gene annotation GTF (GENCODE recommended)" "/path/to/gencode.v38.annotation.gtf"
[[ "$REPLY_VAL" == "SKIP" ]] && REPLY_VAL="/path/to/gencode.v38.annotation.gtf"
ANNOTATION_GTF="$REPLY_VAL"

echo ""
info "REDIportal is used to filter out background endogenous ADAR editing sites."
info "Download from: http://srv00.recas.ba.infn.it/atlas/index.html"
ask_path "REDIportal database (TABLE1_hg38.txt) — press Enter to skip" "SKIP"
[[ "$REPLY_VAL" == "SKIP" || -z "$REPLY_VAL" ]] && REPLY_VAL=""
REDIPORTAL="$REPLY_VAL"

ok "Reference files configured"

# =============================================================================
# STEP 4: ANALYSIS PARAMETERS
# =============================================================================

banner "Step 4: Analysis Parameters"
echo "Default values are shown in brackets. Press Enter to accept."
echo ""
info "Recommended for well-covered human RNA-seq (50M+ reads):"
info "  coverage ≥20, edit_freq ≥5%, fold_change ≥2×, 2/3 replicates"
echo ""

ask "Minimum read coverage at editing site"              "20";  MIN_COVERAGE="$REPLY_VAL"
ask "Minimum editing frequency (0–1 scale, 0.05 = 5%)"  "0.05"; MIN_EDIT_FREQ="$REPLY_VAL"
ask "Minimum fold change vs control"                     "2.0";  EDIT_FOLD_CHANGE="$REPLY_VAL"
ask "Minimum number of treatment replicates"             "2";    MIN_REPLICATES="$REPLY_VAL"
ask "Minimum base quality score (Phred)"                 "20";   MIN_BASE_QUALITY="$REPLY_VAL"
ask "CPU threads for STAR alignment"                     "8";    STAR_THREADS="$REPLY_VAL"
ask "CPU threads for parallel edit calling"              "16";   CALLING_THREADS="$REPLY_VAL"

ok "Parameters configured"

# =============================================================================
# STEP 5: SAMPLE INFORMATION
# =============================================================================

banner "Step 5: Sample Information"
echo "Enter sample names (without .fastq.gz extension)."
echo "FASTQ files must be named:  data/fastq/{sample}_R1.fastq.gz"
echo "                            data/fastq/{sample}_R2.fastq.gz"
echo ""
warn "You can also edit sample names directly in config.yaml after setup."
echo ""

ask "Number of control samples (wildtype / gDNA)" "2"
N_CONTROL=$REPLY_VAL

CONTROL_SAMPLES=()
for (( i=1; i<=N_CONTROL; i++ )); do
    ask "  Control sample $i name" "control_rep${i}"
    CONTROL_SAMPLES+=("$REPLY_VAL")
done

ask "Number of treatment samples (HyperTRIBE)" "3"
N_TREATMENT=$REPLY_VAL

TREATMENT_SAMPLES=()
for (( i=1; i<=N_TREATMENT; i++ )); do
    ask "  Treatment sample $i name" "treatment_rep${i}"
    TREATMENT_SAMPLES+=("$REPLY_VAL")
done

ok "Samples: ${#CONTROL_SAMPLES[@]} control, ${#TREATMENT_SAMPLES[@]} treatment"

# =============================================================================
# WRITE FILES
# =============================================================================

banner "Writing Configuration Files"

# Build YAML lists for samples
format_yaml_list() {
    local arr=("$@")
    local result=""
    for item in "${arr[@]}"; do
        result+="  - \"${item}\"\n"
    done
    echo -e "$result"
}

ALL_SAMPLES=("${CONTROL_SAMPLES[@]}" "${TREATMENT_SAMPLES[@]}")

CONTROL_YAML=$(format_yaml_list "${CONTROL_SAMPLES[@]}")
TREATMENT_YAML=$(format_yaml_list "${TREATMENT_SAMPLES[@]}")
ALL_YAML=$(format_yaml_list "${ALL_SAMPLES[@]}")
REDIPORTAL_YAML="${REDIPORTAL:-/path/to/rediportal/TABLE1_hg38.txt}"

# ── config.yaml ───────────────────────────────────────────────────────────────
cat > config.yaml << YAML
# HyperTRIBE Analysis Configuration
# Generated by setup.sh on $(date)
# ============================================================

project_name:  "${PROJECT_NAME}"
organism:      "${ORGANISM}"
genome_build:  "${GENOME_BUILD}"

# ── Sample Information ───────────────────────────────────────────────────────
# FASTQ files must be named: data/fastq/{sample}_R1.fastq.gz
#                            data/fastq/{sample}_R2.fastq.gz

samples:
$(echo -e "$ALL_YAML")
control_samples:
$(echo -e "$CONTROL_YAML")
treatment_samples:
$(echo -e "$TREATMENT_YAML")

# ── Reference Files ──────────────────────────────────────────────────────────
# Update these to match paths on your HPC

genome:        "${GENOME_FASTA}"
star_index:    "${STAR_INDEX}"
annotation:    "${ANNOTATION_GTF}"
rediportal_db: "${REDIPORTAL_YAML}"

# ── QC Parameters ────────────────────────────────────────────────────────────
min_read_length:     50
min_base_quality:    ${MIN_BASE_QUALITY}
adapter_trimming:    true
min_mapping_quality: 10

# ── STAR Alignment ────────────────────────────────────────────────────────────
star_threads:      ${STAR_THREADS}
max_multimapping:  10
max_mismatches:    3
max_intron_length: 500000

# ── Editing Site Calling ──────────────────────────────────────────────────────
min_coverage:       ${MIN_COVERAGE}
min_edit_freq:      ${MIN_EDIT_FREQ}
edit_fold_change:   ${EDIT_FOLD_CHANGE}
p_value_threshold:  0.05
min_replicates:     ${MIN_REPLICATES}

# ── Performance ───────────────────────────────────────────────────────────────
calling_threads: ${CALLING_THREADS}
chunk_size:      10000000

# ── Output Options ────────────────────────────────────────────────────────────
generate_bigwig:          false
keep_intermediate:        false
compress_output:          true
remove_homopolymers:      true
homopolymer_length:       4
filter_known_editing:     ${REDIPORTAL:+true}${REDIPORTAL:-false}
strand_specific:          false

# ── Annotation Filters ────────────────────────────────────────────────────────
include_biotypes:
  - "protein_coding"
  - "lncRNA"
  - "miRNA"
  - "snRNA"
  - "snoRNA"

include_regions:
  - "exon"
  - "intron"
  - "5UTR"
  - "3UTR"

# ── Reporting ─────────────────────────────────────────────────────────────────
max_genes_plot:  50
top_n_genes:     20
YAML

ok "Written: config.yaml"

# ── slurm/cluster.yaml ────────────────────────────────────────────────────────
mkdir -p slurm

cat > slurm/cluster.yaml << YAML
# HyperTRIBE SLURM Resource Configuration
# Generated by setup.sh on $(date)

__default__:
  account:   "${SLURM_ACCOUNT}"
  partition: "${SLURM_PARTITION}"
  time:      "24:00:00"
  mem:       "32G"

fastp_trim:
  time:      "04:00:00"
  mem:       "16G"

star_align:
  time:      "${STAR_TIME}"
  mem:       "64G"

index_bam:
  time:      "01:00:00"
  mem:       "4G"

call_editing_sites:
  partition: "${SLURM_BIGMEM_PARTITION}"
  time:      "${CALLING_TIME}"
  mem:       "128G"

filter_replicates:
  time:      "02:00:00"
  mem:       "16G"

annotate_genes:
  time:      "02:00:00"
  mem:       "16G"

extract_target_genes:
  time:      "00:10:00"
  mem:       "2G"

rank_genes_by_editing:
  time:      "00:10:00"
  mem:       "2G"

compile_alignment_stats:
  time:      "00:10:00"
  mem:       "2G"

plot_editing_frequency_distribution:
  time:      "00:30:00"
  mem:       "8G"

plot_chromosome_distribution:
  time:      "00:30:00"
  mem:       "8G"

plot_gene_biotype_distribution:
  time:      "00:30:00"
  mem:       "8G"

generate_report:
  time:      "01:00:00"
  mem:       "16G"
YAML

ok "Written: slurm/cluster.yaml"

# ── submit_pipeline.sh ────────────────────────────────────────────────────────
ACCOUNT_FLAG=""
[[ -n "$SLURM_ACCOUNT" ]] && ACCOUNT_FLAG="#SBATCH --account=${SLURM_ACCOUNT}"

cat > submit_pipeline.sh << BASH
#!/usr/bin/env bash
# HyperTRIBE Pipeline — SLURM Controller Job
# Generated by setup.sh on $(date)
# Submit with: sbatch submit_pipeline.sh

#SBATCH --job-name=ht_${PROJECT_NAME}
${ACCOUNT_FLAG}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --time=96:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/snakemake_controller_%j.out
#SBATCH --error=logs/slurm/snakemake_controller_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${USER_EMAIL}

set -euo pipefail

echo "========================================"
echo "HyperTRIBE Pipeline — ${PROJECT_NAME}"
echo "Started: \$(date)"
echo "Host:    \$(hostname)"
echo "========================================"

# Activate conda environment
source ~/.bashrc
conda activate hypertribe

mkdir -p logs/slurm results/plots qc trimmed aligned

# Dry run preview
echo "=== Planned jobs (dry run) ==="
snakemake --profile slurm --configfile config.yaml --jobs ${MAX_JOBS} --dry-run --quiet 2>&1 | head -40

echo ""
echo "=== Submitting pipeline ==="
snakemake \\
    --profile slurm \\
    --configfile config.yaml \\
    --jobs ${MAX_JOBS} \\
    --rerun-incomplete \\
    --keep-going \\
    --printshellcmds \\
    --reason

EXIT_CODE=\$?
echo "========================================"
echo "Finished: \$(date) | Exit: \${EXIT_CODE}"
echo "========================================"
[[ \$EXIT_CODE -eq 0 ]] && echo "SUCCESS — open results/analysis_report.html" \
                         || echo "FAILED — check logs/slurm/ for details"
exit \$EXIT_CODE
BASH

chmod +x submit_pipeline.sh
ok "Written: submit_pipeline.sh"

# =============================================================================
# DONE
# =============================================================================

banner "Setup Complete!"

echo "Files written:"
echo "  ✓ config.yaml"
echo "  ✓ slurm/cluster.yaml"
echo "  ✓ submit_pipeline.sh"
echo ""
echo -e "${BOLD}Next steps:${NC}"
echo ""
echo "  1. Review and edit config.yaml:"
echo "     • Confirm sample names match your FASTQ filenames"
echo "     • Verify reference file paths are accessible on the HPC"
echo ""
echo "  2. Install the conda environment (first time only):"
echo "     ${CYAN}conda env create -f environment.yml${NC}"
echo ""
echo "  3. Place FASTQ files:"
echo "     ${CYAN}mkdir -p data/fastq${NC}"
echo "     Files must be named:  data/fastq/{sample}_R1.fastq.gz"
echo ""
echo "  4. Dry run to verify the pipeline:"
echo "     ${CYAN}snakemake --profile slurm --configfile config.yaml -n${NC}"
echo ""
echo "  5. Submit the pipeline:"
echo "     ${CYAN}sbatch submit_pipeline.sh${NC}"
echo ""
echo "  6. Monitor jobs:"
echo "     ${CYAN}squeue -u \$USER${NC}"
echo "     ${CYAN}tail -f logs/slurm/snakemake_controller_*.out${NC}"
echo ""
echo -e "${GREEN}${BOLD}Good luck with your analysis!${NC}"
echo ""
