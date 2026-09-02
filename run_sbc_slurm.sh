#!/bin/bash
#SBATCH --job-name=nre-sbc
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-39                 # must match sbc_draw_truths.py's --n-truths - 1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                    # UNVERIFIED -- MCMC + a single mock-environment build,
                                      # should be far lighter than the coeval-box jobs, but not
                                      # timed yet. Adjust after the first task or two complete.
#SBATCH --time=02:00:00              # UNVERIFIED -- same, adjust after seeing real runtimes.
#SBATCH --partition=astro2_short     # verify this partition name/limit exists on your cluster
##SBATCH --account=your_account

# =============================================================================
# SBC (simulation-based calibration) for the balanced full-environment NRE
# model -- item 4 of the NRE to-do list. Each array task handles ONE test
# truth (drawn ahead of time by sbc_draw_truths.py from the UVLF-only
# posterior, not a uniform prior draw): builds a fresh mock observation,
# runs inference, and records the per-parameter rank statistic. Run
# run_sbc_aggregate.py once every task has finished.
#
# Prerequisite (run once, NOT part of this array):
#   python sbc_draw_truths.py \
#       --uvlf-posterior /groups/astro/ivannik/projects/Neighbors/UVLF_only_true/posterior_samples_N0_uvlfonly.npy \
#       --n-truths 40 --seed 42 \
#       --output /groups/astro/ivannik/projects/Neighbors/sbc/sbc_truths.npy
# =============================================================================

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate galaxy-neighbors        # UNVERIFIED -- confirm this is where py21cmfast/torch/emcee live

mkdir -p logs

NEIGHBORS=/groups/astro/ivannik/projects/Neighbors
TRUTHS_FILE=$NEIGHBORS/sbc/sbc_truths.npy
MODEL_DIR=$NEIGHBORS/nre_model_balanced_only_ang_only_ang   # same model used for the highstoch-ref check
OUTPUT_DIR=$NEIGHBORS/sbc

echo "======================================================"
echo "Job:      $SLURM_JOB_NAME  (array task $SLURM_ARRAY_TASK_ID / job $SLURM_ARRAY_JOB_ID)"
echo "Truth idx: $SLURM_ARRAY_TASK_ID"
echo "Started:  $(date)"
echo "======================================================"

python run_sbc_one_truth.py \
    --truths-file "$TRUTHS_FILE" \
    --truth-index "$SLURM_ARRAY_TASK_ID" \
    --model-dir "$MODEL_DIR" \
    --n-obs 50 --n-thin 20 \
    --output-dir "$OUTPUT_DIR"

echo "======================================================"
echo "Finished: $(date)"
echo "======================================================"