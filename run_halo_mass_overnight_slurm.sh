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
# Overnight halo-mass model retrain -- v6.
#
# v2-v4 chased LOSS REWEIGHTING instability; v4's gentler, Label-Distribution-
# Smoothed weights STILL collapsed (sigma pegged at its ceiling by epoch ~10),
# which ruled reweighting out as the actual mechanism. The real cause: plain
# NLL's gradient on mu is proportional to 1/sigma^2, so sigma drifting up
# starves mu's gradient, which leaves inflating sigma further as the only way
# to reduce loss -- a runaway feedback loop, independent of reweighting.
#
# v5's fix: beta-NLL (Seitzer et al. 2022, https://arxiv.org/abs/2203.09168
# -- see train_halo_mass.py's module docstring and gaussian_nll()). This
# WORKED -- v5's mean predicted sigma went from 20.09 (collapsed) to 0.369,
# with 69.2% empirical coverage (target 68%, i.e. genuinely well-calibrated)
# -- but v5 also ran the full 300 epochs with no early stopping, and its own
# val loss shows why that's wasteful: best val (0.3346) was reached around
# epoch ~100, then climbed to 11.06 by epoch 300 (pure overfitting the rest
# of the way). The final per-bin diagnostic table printed in that log was
# from the overfit epoch-300 model, not from halo_mass_best.pt -- misleading
# if read at face value.
#
# v6 adds --early-stop-patience 30: stops once val loss hasn't improved for
# 30 consecutive epochs, instead of always running to --epochs. Everything
# else (LDS reweighting, beta-NLL, bigger net) stays the same as v5 -- this
# run is purely about not wasting GPU time past the point of diminishing
# (negative) returns, and about keeping the final per-bin log table
# trustworthy (it should now be close to halo_mass_best.pt's own numbers,
# since training stops shortly after the best epoch rather than hundreds of
# epochs later).
#
# NOTE: the underlying mu shrinkage-to-the-mean pattern (the S-shape in
# eval_vs_truth.pdf) is NOT expected to change much here -- neither LDS,
# beta-NLL, nor early stopping target mu's point-estimate accuracy directly.
# That's the architecture question (Deep Sets / Set Transformer) still on
# the table for next.
#
# Re-evaluates against eval_vs_truth.pdf's own script at the end so the
# before/after comparison is a straight diff across all six PDFs (v1
# unweighted, v2 collapsed, v3 clipped-raw-reweight, v4 LDS-still-collapsed,
# v5 LDS+beta-NLL-overfit, v6 LDS+beta-NLL+early-stopping).
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
OUT=$NEIGHBORS/halo_mass_model_multibox4_v6

echo ">>> Training halo-mass model v6 (LDS reweighting + beta-NLL + early stopping + bigger net)"
python train_halo_mass.py \
    --database-dir $DB1 $DB2 $DB3 $DB4 \
    --only-angular --epochs 300 --max-per-catalog 0 --weight-by-catalog-count \
    --reweight-by-mass --reweight-alpha 0.5 --reweight-bins 30 --lds-kernel-sigma 2.0 \
    --beta-nll 0.5 \
    --max-weight-ratio 10.0 --grad-clip 5.0 \
    --early-stop-patience 30 \
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
    --output $OUT/eval_vs_truth_v6.pdf

echo "======================================================"
echo "Finished: $(date)"
echo "======================================================"
