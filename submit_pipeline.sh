#!/usr/bin/env bash
# =============================================================================
# HyperTRIBE Pipeline — SLURM Controller Job
# =============================================================================
# This script runs the Snakemake controller as a SLURM job. Snakemake then
# submits each pipeline rule as its own individual SLURM job (defined in
# slurm/config.yaml and slurm/cluster.yaml).
#
# Usage:
#   1. Edit the #SBATCH directives below to match your cluster (account, partition)
#   2. Edit CONDA_ENV and CONFIG below if needed
#   3. Submit: sbatch submit_pipeline.sh
#   4. Monitor: squeue -u $USER
#              tail -f logs/slurm/snakemake_controller_<JOBID>.out
#
# Dry-run before submitting:
#   conda activate hypertribe
#   snakemake --profile slurm --configfile config.yaml -n
# =============================================================================

#SBATCH --job-name=ht_controller
#SBATCH --account=YOUR_SLURM_ACCOUNT          # <-- replace
#SBATCH --partition=YOUR_PARTITION            # <-- replace (use a short/interactive queue)
#SBATCH --time=96:00:00                       # controller can run up to 4 days
#SBATCH --mem=4G                              # controller only needs minimal RAM
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/snakemake_controller_%j.out
#SBATCH --error=logs/slurm/snakemake_controller_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jbiswas@gmail.com         # <-- your email

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV="hypertribe"
CONFIG="config.yaml"
PROFILE="slurm"
MAX_JOBS=50

# ── Environment setup ─────────────────────────────────────────────────────────
echo "========================================"
echo "HyperTRIBE Pipeline"
echo "Started: $(date)"
echo "Host:    $(hostname)"
echo "Dir:     $(pwd)"
echo "========================================"

# Load conda — adjust this for your cluster's module system
# Option A: if conda is in PATH via .bashrc
source ~/.bashrc
# Option B: if your cluster uses modules
# module load anaconda3/2023.09
# Option C: direct conda init
# source /path/to/miniconda3/etc/profile.d/conda.sh

conda activate "${CONDA_ENV}"
echo "Conda env: ${CONDA_ENV} ($(python --version))"
echo "Snakemake:  $(snakemake --version)"

# ── Create required directories ───────────────────────────────────────────────
mkdir -p logs/slurm results/plots qc trimmed aligned

# ── Dry run first (prints planned jobs) ──────────────────────────────────────
echo ""
echo "=== Dry run ==="
snakemake \
    --profile "${PROFILE}" \
    --configfile "${CONFIG}" \
    --jobs "${MAX_JOBS}" \
    --dry-run \
    --quiet 2>&1 | head -60

echo ""
echo "=== Submitting pipeline ==="

# ── Main pipeline run ─────────────────────────────────────────────────────────
snakemake \
    --profile "${PROFILE}" \
    --configfile "${CONFIG}" \
    --jobs "${MAX_JOBS}" \
    --rerun-incomplete \
    --keep-going \
    --printshellcmds \
    --reason

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "========================================"

if [ "${EXIT_CODE}" -eq 0 ]; then
    echo "SUCCESS — results in results/"
    echo "Open results/analysis_report.html to view the summary"
else
    echo "PIPELINE FAILED — check logs/slurm/ for error details"
    echo "Re-run the same sbatch command to resume from the last checkpoint"
fi

exit "${EXIT_CODE}"
