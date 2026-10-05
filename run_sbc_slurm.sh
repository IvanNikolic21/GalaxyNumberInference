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
# SBC (simulation-based calibration) for the balanced d1-only multi-box NRE
# model (nre_model_d1_balanced_multibox4_only_ang) -- item 4 of the NRE
# to-do list. Each array task handles ONE test truth (drawn ahead of time
# by sbc_draw_truths.py from the UVLF-only posterior, not a uniform prior
# draw): builds a fresh mock observation, runs inference, and records the
# per-parameter rank statistic. Run run_sbc_aggregate.py once every task
# has finished.
#
# --infer-script points run_sbc_one_truth.py at infer_nre_d1.py (the d1/
# D1NRENetwork-compatible inference script) instead of its own default,
# infer_nre.py (full-environment NRENetwork architecture -- loading a
# D1NRENetwork checkpoint into that raises a state_dict key mismatch).
#
# Prerequisite (run once, NOT part of this array):
#   python sbc_draw_truths.py \
#       --uvlf-posterior /groups/astro/ivannik/projects/Neighbors/UVLF_only_true_multibox4/posterior_samples_d1_N0.npy \
#       --n-truths 40 --seed 42 \
#       --output /groups/astro/ivannik/projects/Neighbors/sbc/sbc_truths_multibox4.npy
# =============================================================================

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate UVLF_clust

mkdir -p logs

NEIGHBORS=/groups/astro/ivannik/projects/Neighbors
TRUTHS_FILE=$NEIGHBORS/sbc/sbc_truths_multibox4.npy
MODEL_DIR=$NEIGHBORS/nre_model_d1_balanced_multibox4_only_ang
OUTPUT_DIR=$NEIGHBORS/sbc_multibox4

echo "======================================================"
echo "Job:      $SLURM_JOB_NAME  (array task $SLURM_ARRAY_TASK_ID / job $SLURM_ARRAY_JOB_ID)"
echo "Truth idx: $SLURM_ARRAY_TASK_ID"
echo "Started:  $(date)"
echo "======================================================"

python run_sbc_one_truth.py \
    --truths-file "$TRUTHS_FILE" \
    --truth-index "$SLURM_ARRAY_TASK_ID" \
    --model-dir "$MODEL_DIR" \
    --infer-script infer_nre_d1.py \
    --n-obs 50 --n-thin 20 \
    --output-dir "$OUTPUT_DIR"

echo "======================================================"
echo "Finished: $(date)"
echo "======================================================"