#!/bin/bash
#SBATCH --job-name=halo-mass-overnight
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                    # UNVERIFIED -- bump if the dataset-loading step (reads
                                      # every .npz across all 4 boxes into RAM) OOMs.
#SBATCH --time=10:00:00              # UNVERIFIED -- generous overnight budget; a single GPU
                                      # epoch on this dataset was ~tens of seconds in the prior
                                      # run, so 300 epochs + eval should fit well under this,
                                      # but leaving margin since hidden-dims is much bigger here.
#SBATCH --partition=astro_gpu        # confirmed real partition name (2026-10-12)
#SBATCH --gres=gpu:1                 # VERIFY: your cluster's gres syntax for requesting 1 GPU
##SBATCH --account=your_account

# =============================================================================
# Overnight halo-mass model retrain -- v7.
#
# v5 introduced beta-NLL (Seitzer et al. 2022, https://arxiv.org/abs/2203.09168)
# to fix the sigma-collapse that plagued v2/v4 under plain NLL -- this WORKED
# (mean predicted sigma 20.09 -> 0.369, 69.2% empirical coverage vs a 68%
# target) but ran the full 300 epochs with no early stopping and badly
# overfit past epoch ~100 (val loss 0.33 -> 11.06 by epoch 300).
#
# v6 added --early-stop-patience 30 to fix THAT -- but immediately hit a
# sharper bug: it stopped at epoch 41 with "best val" still 3.9191, the
# exact plain-NLL collapse floor. beta-NLL training was actually working
# fine underneath (train loss kept dropping normally), but escaping that
# early high-sigma regime causes a TEMPORARY SPIKE in val_loss (deliberately
# measured as plain NLL, beta=0, for comparability across --beta-nll
# settings) as mu starts moving before sigma re-calibrates -- v5 shows the
# same spike at epoch 50 (val=25.88) before recovering to val=0.53 by epoch
# 100. Patience alone can't tell "still stuck" apart from "mid-recovery
# from a known transient," so it fired right as real learning started and
# saved the single worst checkpoint in the run.
#
# v7's fix: --early-stop-warmup 70 -- the patience counter doesn't start
# ticking until after this many epochs (best_val/checkpointing still run
# every epoch regardless, so there's always a saved model). Verified by
# simulation against v5's actual trajectory: this correctly finds the real
# best epoch (~100) and stops at ~130, surviving the spike instead of being
# fooled by it.
#
# NOTE: the underlying mu shrinkage-to-the-mean pattern (the S-shape in
# eval_vs_truth.pdf) is NOT expected to change much here -- neither LDS,
# beta-NLL, nor early stopping target mu's point-estimate accuracy directly.
# That's the architecture question (Deep Sets / Set Transformer) still on
# the table for next.
#
# Re-evaluates against eval_vs_truth.pdf's own script at the end so the
# before/after comparison is a straight diff across all seven PDFs (v1
# unweighted, v2 collapsed, v3 clipped-raw-reweight, v4 LDS-still-collapsed,
# v5 LDS+beta-NLL-overfit, v6 early-stop-fooled-by-spike, v7 warmup-fixed).
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
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo "======================================================"

NEIGHBORS=/groups/astro/ivannik/projects/Neighbors
DB1=$NEIGHBORS/nre_database_prior_capped_seed1955
DB2=$NEIGHBORS/nre_database_prior_capped_seed2027
DB3=$NEIGHBORS/nre_database_prior_capped_seed3142
DB4=$NEIGHBORS/nre_database_prior_capped_seed4242
OUT=$NEIGHBORS/halo_mass_model_multibox4_v7

echo ">>> Training halo-mass model v7 (LDS reweighting + beta-NLL + warmup-protected early stopping)"
python train_halo_mass.py \
    --database-dir $DB1 $DB2 $DB3 $DB4 \
    --only-angular --epochs 300 --max-per-catalog 0 --weight-by-catalog-count \
    --reweight-by-mass --reweight-alpha 0.5 --reweight-bins 30 --lds-kernel-sigma 2.0 \
    --beta-nll 0.5 \
    --max-weight-ratio 10.0 --grad-clip 5.0 \
    --early-stop-patience 30 --early-stop-warmup 70 \
    --diag-every 10 --diag-bins 8 \
    --hidden-dims 512 512 512 512 512 512 \
    --batch-size 1024 \
    --output-dir $OUT

echo "======================================================"
echo ">>> Re-running eval_vs_truth for direct before/after comparison"
python evaluate_halo_mass_model.py \
    --model-dir $OUT \
    --database-dir $DB1 $DB2 $DB3 $DB4 \
    --n-examples 500 \
    --output $OUT/eval_vs_truth_v7.pdf

echo "======================================================"
echo "Finished: $(date)"
echo "======================================================"
