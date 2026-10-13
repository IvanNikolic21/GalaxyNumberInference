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
# Overnight halo-mass model retrain -- v8.
#
# v5-v7 were all about the LOSS/TRAINING LOOP: beta-NLL fixed the sigma
# collapse (Seitzer et al. 2022, https://arxiv.org/abs/2203.09168), then
# early stopping (with a warmup to survive the post-collapse val-loss spike)
# fixed the overfitting waste. v7 is the first genuinely trustworthy run --
# mean predicted sigma 0.379, 73.0% coverage (target 68%), stopped cleanly
# at epoch 100 (best val 0.3328). But its RMS error (0.361 dex) and the
# per-bin bias table are essentially IDENTICAL to v1's, all the way back to
# before any of these fixes: the low-mass tail still over-predicts
# (bias=+0.337), the high-mass tail still under-predicts (bias=-0.218) --
# the same shrinkage-to-the-mean S-shape every version has shown. Expected:
# none of v2-v7's fixes ever touched HaloMassNetwork's actual environment
# ENCODING, only the loss function and training loop around it.
#
# v8 is the first architecture change: --architecture deepset (Zaheer et
# al. 2017, https://arxiv.org/abs/1703.06114, module docstring in
# train_halo_mass.py -- new HaloMassSetNetwork class). HaloMassNetwork
# flattens the distance-sorted, zero-padded neighbor list into one vector
# and feeds it to a plain MLP, which has to rediscover permutation-
# invariance from scratch; HaloMassSetNetwork applies a small shared MLP to
# EACH neighbor row independently (--phi-dims), masks out padding rows, and
# sum-pools (--pooling) into one fixed-size embedding before the same
# ResidualBlock stack as before. Permutation-invariant by construction --
# this is the first change actually aimed at the persistent S-shape rather
# than at training stability. Everything else (LDS, beta-NLL, warmup-
# protected early stopping) is kept identical to v7 for a clean comparison.
#
# Re-evaluates against eval_vs_truth.pdf's own script at the end so the
# before/after comparison is a straight diff across all eight PDFs (v1
# unweighted, v2 collapsed, v3 clipped-raw-reweight, v4 LDS-still-collapsed,
# v5 LDS+beta-NLL-overfit, v6 early-stop-fooled-by-spike, v7 warmup-fixed-
# MLP, v8 warmup-fixed-DeepSets).
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
OUT=$NEIGHBORS/halo_mass_model_multibox4_v8

echo ">>> Training halo-mass model v8 (Deep Sets architecture + LDS + beta-NLL + early stopping)"
python train_halo_mass.py \
    --database-dir $DB1 $DB2 $DB3 $DB4 \
    --only-angular --epochs 300 --max-per-catalog 0 --weight-by-catalog-count \
    --reweight-by-mass --reweight-alpha 0.5 --reweight-bins 30 --lds-kernel-sigma 2.0 \
    --beta-nll 0.5 \
    --max-weight-ratio 10.0 --grad-clip 5.0 \
    --early-stop-patience 30 --early-stop-warmup 70 \
    --diag-every 10 --diag-bins 8 \
    --architecture deepset --phi-dims 64 64 --pooling sum \
    --hidden-dims 512 512 512 512 512 512 \
    --batch-size 1024 \
    --output-dir $OUT

echo "======================================================"
echo ">>> Re-running eval_vs_truth for direct before/after comparison"
python evaluate_halo_mass_model.py \
    --model-dir $OUT \
    --database-dir $DB1 $DB2 $DB3 $DB4 \
    --n-examples 500 \
    --output $OUT/eval_vs_truth_v8.pdf

echo "======================================================"
echo "Finished: $(date)"
echo "======================================================"
