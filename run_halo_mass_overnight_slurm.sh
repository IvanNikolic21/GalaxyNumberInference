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
# Overnight halo-mass model retrain -- v4.
#
# v2 added --reweight-by-mass (raw inverse-bin-count loss reweighting) +
# bigger net, but collapsed (sigma pegged at its ceiling for ~every example,
# loss frozen bit-for-bit by epoch ~10) because a few examples in sparse
# bins got >100x the weight of others and dominated every batch's gradient.
# v3 patched over that with --max-weight-ratio (clip the combined weight to
# [median/10, median*10]) + --grad-clip -- this DID stop the collapse, but
# v3's own per-bin diagnostics (and eval_vs_truth_v3.pdf) showed the
# shrinkage-to-the-mean bias pattern was essentially unchanged from v1: the
# aggressive clipping needed to prevent collapse also neutered most of the
# reweighting's intended effect on exactly the sparse tail bins that needed it.
#
# v4 fixes the ROOT cause instead of clipping around it: Label Distribution
# Smoothing (Yang et al. 2021, "Delving into Deep Imbalanced Regression",
# https://arxiv.org/abs/2102.09554 -- see train_halo_mass.py's module
# docstring). Raw per-bin counts are a NOISY density estimate for a sparsely
# populated bin, which is what produced the >100x weight ratios in the first
# place; LDS kernel-smooths the count histogram across neighboring bins
# BEFORE inverting it (new --lds-kernel-sigma, in bin units), giving a much
# more stable weight distribution without needing aggressive clipping
# downstream. --max-weight-ratio/--grad-clip are left on as a safety net --
# check the training log's "N/len clipped" line to see how much they're
# actually still binding now that the input to them is smoother.
#
# Re-evaluates against eval_vs_truth.pdf's own script at the end so the
# before/after comparison is a straight diff across all four PDFs
# (v1 unweighted, v2 collapsed, v3 clipped-raw-reweight, v4 LDS).
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
OUT=$NEIGHBORS/halo_mass_model_multibox4_v4

echo ">>> Training halo-mass model v4 (LDS reweighting + bigger net + longer run)"
python train_halo_mass.py \
    --database-dir $DB1 $DB2 $DB3 $DB4 \
    --only-angular --epochs 300 --max-per-catalog 0 --weight-by-catalog-count \
    --reweight-by-mass --reweight-alpha 0.5 --reweight-bins 30 --lds-kernel-sigma 2.0 \
    --max-weight-ratio 10.0 --grad-clip 5.0 \
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
    --output $OUT/eval_vs_truth_v4.pdf

echo "======================================================"
echo "Finished: $(date)"
echo "======================================================"
