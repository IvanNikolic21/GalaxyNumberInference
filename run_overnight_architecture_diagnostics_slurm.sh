#!/bin/bash
#SBATCH --job-name=nre-arch-diag
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=10:00:00              # 3 balanced-NRE retrains (~1.5h each) + inference + diagnostics
#SBATCH --partition=astro2_long      # verify this partition name/limit exists on your cluster
##SBATCH --account=your_account

# =============================================================================
# Three diagnostics for the unexplained "bimodal" Muv_add structure seen in
# repro_balanced_sb2p4_N50 (2026-08-25). Run in order of cost, cheapest first,
# so a failure partway through still leaves the most informative results:
#
#   PHASE 1: MCMC mixing check (cheap, ~minutes) -- is it real, or is the
#            sampler stuck? Uses infer_nre.py --save-chain (new) +
#            analyze_mcmc_mixing.py (new).
#   PHASE 2: Network-independent physics check (cheap, no GPU, ~tens of
#            minutes depending on cluster CPU) -- are the two theta points
#            actually physically degenerate? Uses check_theta_degeneracy.py
#            (new), no network/training involved at all.
#   PHASE 3: Ensemble reproducibility check (expensive, ~4-5h) -- do 3
#            independently-seeded retrains of the SAME architecture show the
#            SAME bimodal structure (real/systematic) or different,
#            idiosyncratic bumps (training noise)?
#
# ASSUMPTION FLAGGED: Phase 3's training command reconstructs the balanced-NRE
# run's flags from what we discussed in chat (matching the existing reweighted
# run in run_overnight_resume_slurm.sh, --balanced --balance-lambda 100 added,
# --epochs 100 per the actual completed log) -- PRIOR_DB below is a best
# guess, not independently re-verified against whatever exact command
# produced nre_model_balanced_only_ang_only_ang. Double check before
# submitting if you're not confident it matches.
# =============================================================================

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
OBS_DIR_V2=$NEIGHBORS/nre_database_new_v2
PRIOR_DB=$NEIGHBORS/nre_database_prior_capped     # ASSUMPTION -- see header note above
MODEL_BALANCED=$NEIGHBORS/nre_model_balanced_only_ang_only_ang
SB24_OBS=$OBS_DIR_V2/nre_Madd0p30_sa-0p34_sb2p40.npz

echo "Checking MODEL_BALANCED=$MODEL_BALANCED ..."
if [ ! -f "$MODEL_BALANCED/model_config.npz" ]; then
    echo "!!! model_config.npz NOT FOUND -- STOPPING. Fix MODEL_BALANCED above before resubmitting."
    exit 1
fi
if [ ! -f "$SB24_OBS" ]; then
    echo "!!! sigma_b=2.4 obs-file NOT FOUND at $SB24_OBS -- STOPPING."
    exit 1
fi
echo "  OK."
echo "======================================================"

# =============================================================================
# PHASE 1 -- MCMC mixing check
# =============================================================================
echo ">>> [1] Rerun sigma_b=2.4, N=50, balanced model, with --save-chain"
python infer_nre.py --obs-file $SB24_OBS --n-obs 50 \
    --model-dir $MODEL_BALANCED --save-chain \
    --output-dir $NEIGHBORS/coverage_check_v3/repro_balanced_sb2p4_N50_chain \
    || echo "  [1] infer FAILED, continuing."

echo ">>> [1b] Analyze mixing"
python analyze_mcmc_mixing.py \
    --chain-dir $NEIGHBORS/coverage_check_v3/repro_balanced_sb2p4_N50_chain \
    --n-obs 50 --param-index 0 \
    || echo "  [1b] analysis FAILED, continuing."

echo "======================================================"
echo "Phase 1 done: $(date)"
echo "======================================================"

# =============================================================================
# PHASE 2 -- network-independent physics/degeneracy check (no GPU needed)
# =============================================================================
echo ">>> [2] Theta degeneracy check (mode A vs. boundary pile-up mode B)"
python check_theta_degeneracy.py --n-realizations 10 --max-bright-per-realization 3000 \
    || echo "  [2] FAILED, continuing."

echo "======================================================"
echo "Phase 2 done: $(date)"
echo "======================================================"

# =============================================================================
# PHASE 3 -- ensemble reproducibility: 3 more balanced-NRE retrains, different seeds
# =============================================================================
for SEED in 43 44 45; do
    MODEL_OUT=$NEIGHBORS/nre_model_balanced_seed${SEED}
    echo ">>> [3-seed${SEED}] Training balanced NRE, seed=${SEED}"
    python train_nre.py --prior-only --prior-database-dir $PRIOR_DB \
        --only-angular --summary-mode --epochs 100 \
        --max-per-catalog 0 --weight-by-catalog-count \
        --balanced --balance-lambda 100.0 --seed ${SEED} \
        --output-dir $MODEL_OUT \
        || { echo "  [3-seed${SEED}] training FAILED, skipping its inference."; continue; }

    echo ">>> [3-seed${SEED}] Inference, sigma_b=2.4, N=50"
    python infer_nre.py --obs-file $SB24_OBS --n-obs 50 \
        --model-dir ${MODEL_OUT}_only_ang \
        --output-dir $NEIGHBORS/coverage_check_v3/repro_balanced_sb2p4_N50_seed${SEED} \
        || echo "  [3-seed${SEED}] inference FAILED, continuing."
done

echo "======================================================"
echo "Finished: $(date)"
echo "======================================================"
