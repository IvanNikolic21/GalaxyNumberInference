#!/bin/bash
#SBATCH --job-name=nre-sbc-full
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-39                 # must match sbc_draw_truths.py's --n-truths - 1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                    # UNVERIFIED -- same caveat as run_sbc_slurm.sh.
#SBATCH --time=04:00:00              # UNVERIFIED, but bumped up from run_sbc_slurm.sh's 2h: the
                                      # full-environment model's training ran ~3x slower per
                                      # epoch than the d1 model (~2.6 min vs ~46s), so its
                                      # per-task MCMC inference is likely slower too. Adjust
                                      # after seeing the first task or two's real runtime.
#SBATCH --partition=astro2_short     # verify this partition name/limit exists on your cluster
##SBATCH --account=your_account

# =============================================================================
# SBC (simulation-based calibration) for the balanced FULL-ENVIRONMENT
# multi-box NRE model (nre_model_full_balanced_multibox4_only_ang) -- the
# full-environment counterpart to run_sbc_slurm.sh's d1-only check. Same
# test truths are reused here (sbc_truths_multibox4.npy) rather than
# redrawing: the UVLF-only posterior those truths were drawn from is purely
# the analytic UVLF likelihood x prior, independent of which NRE model's
# weights get loaded (--uvlf-only forces the environment/NRE term to zero
# either way), so it's valid -- and preferable, for a clean apples-to-apples
# comparison -- to test both models against the identical 40 truths.
#
# No --infer-script override needed: run_sbc_one_truth.py's default
# (infer_nre.py) is already the full-environment/NRENetwork-compatible
# script -- that default is only wrong for the d1 model, which is why
# run_sbc_slurm.sh overrides it to infer_nre_d1.py.
#
# Prerequisite: already satisfied by the d1 run's prerequisite step (same
# truths file, see run_sbc_slurm.sh's header for that command) -- nothing
# new to run before this array.
# =============================================================================

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate UVLF_clust

mkdir -p logs

NEIGHBORS=/groups/astro/ivannik/projects/Neighbors
TRUTHS_FILE=$NEIGHBORS/sbc/sbc_truths_multibox4.npy
MODEL_DIR=$NEIGHBORS/nre_model_full_balanced_multibox4_only_ang
OUTPUT_DIR=$NEIGHBORS/sbc_full_multibox4

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
