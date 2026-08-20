#!/bin/bash
#SBATCH --job-name=nre-resume
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=06:00:00              # only the full-MLP retrain + Phase 1 left -- d1 already done
#SBATCH --partition=astro2_long      # verify this partition name/limit exists on your cluster
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
echo "Started:  $(date)"
echo "======================================================"

NEIGHBORS=/groups/astro/ivannik/projects/Neighbors
OBS_DIR=$NEIGHBORS/nre_database_new
OBS_DIR_V2=$NEIGHBORS/nre_database_new_v2
PRIOR_DB=$NEIGHBORS/nre_database_prior_capped

# Corrected per last night's actual training-completion log ("Model saved to:
# .../nre_model_prioronly_capped_only_ang"). nre_model_capped_only_ang (no
# "prioronly") is NOT a model dir -- it's a past infer_nre.py --output-dir,
# which is why every [1a]/[1b]/[1c] call failed instantly last night.
MODEL_FULL_CAPPED=$NEIGHBORS/nre_model_prioronly_capped_only_ang
MODEL_D1_REWEIGHTED=$NEIGHBORS/nre_model_d1_reweighted_only_ang   # already trained successfully last night

# Fail fast and loudly if this guess is STILL wrong, rather than silently
# burning the whole job on broken inference calls again.
echo "Checking MODEL_FULL_CAPPED=$MODEL_FULL_CAPPED ..."
if [ ! -f "$MODEL_FULL_CAPPED/model_config.npz" ]; then
    echo "!!! model_config.npz NOT FOUND in $MODEL_FULL_CAPPED -- STOPPING."
    echo "!!! Find the real trained-model directory (ls for model_config.npz + nre_best.pt)"
    echo "!!! and fix MODEL_FULL_CAPPED above before resubmitting."
    exit 1
fi
echo "  OK, found model_config.npz."
echo "======================================================"

# =============================================================================
# PHASE 1 -- fast diagnostics, now pointed at the correct model dir
# =============================================================================

echo ">>> [1a] Reproducibility check, step 1"
python infer_nre.py --obs-file $OBS_DIR/nre_Madd0p30_sa-0p34_sb2p40.npz --n-obs 10 \
    --model-dir $MODEL_FULL_CAPPED --output-dir $NEIGHBORS/coverage_check_v2/repro_step1 \
    || echo "  [1a] FAILED, continuing."

echo ">>> [1b] Reproducibility check, step 2 (random-subsample obs-file rebuild)"
if [ ! -f "$OBS_DIR_V2/nre_Madd0p30_sa-0p34_sb2p40.npz" ]; then
    python build_nre_database.py --param-file coverage_params.dat --output-dir $OBS_DIR_V2 \
        || echo "  [1b-build] FAILED, continuing."
fi
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

echo "======================================================"
echo "Phase 1 done: $(date)"
echo "======================================================"

# =============================================================================
# PHASE 2 -- full-MLP reweighted retrain ONLY (d1 already succeeded last night).
# Bug fixed (augmentation reshape) + --summary-mode added to match the
# baseline architecture (537,601 params, input_dim=24) for a clean comparison.
# =============================================================================

echo ">>> [2a] Retrain full-MLP: --summary-mode --weight-by-catalog-count --max-per-catalog 0"
python train_nre.py --prior-only --prior-database-dir $PRIOR_DB \
    --only-angular --summary-mode --epochs 200 \
    --max-per-catalog 0 --weight-by-catalog-count \
    --output-dir $NEIGHBORS/nre_model_reweighted

echo ">>> [2b] Coverage sweep: reweighted full-MLP + already-trained reweighted d1"
python run_coverage_sweep.py --param-file coverage_params.dat --obs-dir $OBS_DIR \
    --model-dir-full $NEIGHBORS/nre_model_reweighted_only_ang \
    --model-dir-d1   $MODEL_D1_REWEIGHTED \
    --output-dir $NEIGHBORS/coverage_check_v3 --n-obs 10

# =============================================================================
# Regenerate the presentation comparison figures now that full-MLP N=1 exists
# =============================================================================
echo ">>> [3] Regenerate posterior-gain comparison figures"
python plot_posterior_gain.py || echo "  [3] FAILED, continuing."

echo "======================================================"
echo "Finished: $(date)"
echo "======================================================"