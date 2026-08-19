#!/bin/bash
#SBATCH --job-name=nre-overnight
#SBATCH --output=logs/%x_%j.out      # stdout  → logs/nre-overnight_<jobid>.out
#SBATCH --error=logs/%x_%j.err       # stderr  → logs/nre-overnight_<jobid>.err
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=10:00:00              # ~8h budget + buffer -- see partition note below
#SBATCH --partition=astro2_long      # switched from _short: this run needs up to ~10h,
                                      # longer than a "_short" partition will typically allow.
                                      # Verify this partition name/limit exists on your cluster
                                      # before submitting -- swap back to astro2_short only if
                                      # you trim --time well under its cap.
##SBATCH --account=your_account

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate galaxy-neighbors

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export LD_LIBRARY_PATH=/groups/astro/ivannik/miniconda3/envs/galaxy-neighbors/lib:$LD_LIBRARY_PATH
mkdir -p logs

echo "======================================================"
echo "Job:      $SLURM_JOB_NAME  ($SLURM_JOB_ID)"
echo "Node:     $SLURMD_NODENAME"
echo "CPUs:     $SLURM_CPUS_PER_TASK"
echo "Started:  $(date)"
echo "======================================================"

# ---------------------------------------------------------------------------
# Paths -- edit before submitting if any of these differ
# ---------------------------------------------------------------------------
NEIGHBORS=/groups/astro/ivannik/projects/Neighbors
OBS_DIR=$NEIGHBORS/nre_database_new
OBS_DIR_V2=$NEIGHBORS/nre_database_new_v2
MODEL_FULL_CAPPED=$NEIGHBORS/nre_model_capped_only_ang
MODEL_D1_CAPPED=$NEIGHBORS/nre_model_d1_capped_only_ang

# EDIT THIS: point at the richest available training database for the
# reweighted retrain. Weighting has nothing to correct if it's fed a database
# already capped at build time -- the ORIGINAL uncapped database (the one
# whose file sizes ranged 858B-266KB, built ~Apr 21) is the right target if
# you still have/remember its path. Falling back to the capped one below is
# safe (the script will still run) but gives --weight-by-catalog-count less
# room to matter, since build-time capping already flattened it once.
PRIOR_DB=$NEIGHBORS/nre_database_prior_capped   # <-- replace with the original uncapped path if you have it

echo "Using PRIOR_DB=$PRIOR_DB"
echo "======================================================"

# =============================================================================
# PHASE 1 -- fast diagnostics (~30-45 min). Failures here do NOT block Phase 2.
# =============================================================================

echo ">>> [1a] Reproducibility check, step 1: rerun same obs-file/model"
python infer_nre.py --obs-file $OBS_DIR/nre_Madd0p30_sa-0p34_sb2p40.npz --n-obs 10 \
    --model-dir $MODEL_FULL_CAPPED --output-dir $NEIGHBORS/coverage_check_v2/repro_step1 \
    || echo "  [1a] FAILED, continuing."

echo ">>> [1b] Reproducibility check, step 2: rebuild obs-file with random subsampling, rerun"
python build_nre_database.py --param-file coverage_params.dat --output-dir $OBS_DIR_V2 \
    || echo "  [1b-build] FAILED, continuing."
python infer_nre.py --obs-file $OBS_DIR_V2/nre_Madd0p30_sa-0p34_sb2p40.npz --n-obs 10 \
    --model-dir $MODEL_FULL_CAPPED --output-dir $NEIGHBORS/coverage_check_v2/repro_step2 \
    || echo "  [1b-infer] FAILED, continuing."

echo ">>> [1c] Fill in missing N=1 runs for plot_posterior_gain.py (full-MLP, capped model)"
python infer_nre.py --obs-file $OBS_DIR/nre_Madd0p30_sa-0p34_sb0p60.npz --n-obs 1 \
    --model-dir $MODEL_FULL_CAPPED --output-dir $NEIGHBORS/UVLF_only \
    || echo "  [1c] FAILED, continuing."
python infer_nre.py --obs-file $OBS_DIR/nre_Madd0p30_sa-0p34_sb0p60.npz --n-obs 1 \
    --model-dir $MODEL_FULL_CAPPED --output-dir $NEIGHBORS/UVLF_only --use-uvlf \
    || echo "  [1c-uvlf] FAILED, continuing."

echo ">>> [1d] Regenerate presentation comparison figures"
python plot_posterior_gain.py || echo "  [1d] FAILED, continuing."

echo "======================================================"
echo "Phase 1 done: $(date)"
echo "======================================================"

# =============================================================================
# PHASE 2 -- the main overnight investment: reweighted-loss retrain (~4-6h),
# then re-run the coverage sweep against the new models.
# =============================================================================

echo ">>> [2a] Retrain full-MLP with --weight-by-catalog-count --max-per-catalog 0"
python train_nre.py --prior-only --prior-database-dir $PRIOR_DB \
    --only-angular --epochs 200 \
    --max-per-catalog 0 --weight-by-catalog-count \
    --output-dir $NEIGHBORS/nre_model_reweighted

echo ">>> [2b] Retrain d1-only with --weight-by-catalog-count --max-per-catalog 0"
python train_nre_d1.py --prior-only --prior-database-dir $PRIOR_DB \
    --only-angular --epochs 200 \
    --max-per-catalog 0 --weight-by-catalog-count \
    --output-dir $NEIGHBORS/nre_model_d1_reweighted

echo ">>> [2c] Coverage sweep against the reweighted models"
python run_coverage_sweep.py --param-file coverage_params.dat --obs-dir $OBS_DIR \
    --model-dir-full $NEIGHBORS/nre_model_reweighted_only_ang \
    --model-dir-d1   $NEIGHBORS/nre_model_d1_reweighted_only_ang \
    --output-dir $NEIGHBORS/coverage_check_v3 --n-obs 10

echo "======================================================"
echo "Finished: $(date)"
echo "======================================================"