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
# Overnight halo-mass model retrain -- v5.
#
# v2/v3 were about LOSS REWEIGHTING (raw inverse-bin-count, then clipped,
# then (v4) Label-Distribution-Smoothed -- see train_halo_mass.py's module
# docstring, Yang et al. 2021). v4 made the reweighting itself demonstrably
# gentler (smoothed density range [377, 12198] vs v3's raw [0, 20404], and
# only 0.45% of examples needed clipping) -- and it STILL collapsed, with
# the exact same signature as v2 (loss frozen at log_sigma pegged near its
# ceiling by epoch ~10). Collapsing under two very different reweighting
# strengths rules OUT reweighting as the mechanism.
#
# The real cause: a known pathology of jointly training a Gaussian mean and
# variance from one NLL loss (independent of class/mass imbalance). Plain
# NLL's gradient on mu is proportional to 1/sigma^2, so sigma drifting up
# even slightly starves mu's gradient; mu stops improving; the only way left
# to reduce loss is inflating sigma further -- a runaway feedback loop.
# Verified numerically: at the v2/v4 collapse sigma (~20), mu's gradient
# under plain NLL is ~400x weaker than at sigma=1.
#
# v5's fix: beta-NLL (Seitzer, Tavakoli, Kirchhof & Archambeau 2022,
# "On the Pitfalls of Heteroscedastic Uncertainty Estimation with
# Probabilistic Neural Networks," https://arxiv.org/abs/2203.09168 -- see
# train_halo_mass.py's module docstring and gaussian_nll()). New
# --beta-nll 0.5 scales each example's NLL by sigma.detach()**(2*beta)
# during training (val_loss stays plain NLL for comparability across runs)
# -- at beta=0.5 that same 400x gradient-starvation ratio drops to ~20x,
# breaking the feedback loop while still training a genuinely
# heteroscedastic sigma. LDS reweighting (v4) is kept ON alongside this --
# they fix different problems and don't conflict.
#
# Re-evaluates against eval_vs_truth.pdf's own script at the end so the
# before/after comparison is a straight diff across all five PDFs
# (v1 unweighted, v2 collapsed, v3 clipped-raw-reweight, v4 LDS-still-
# collapsed, v5 LDS+beta-NLL).
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
OUT=$NEIGHBORS/halo_mass_model_multibox4_v5

echo ">>> Training halo-mass model v5 (LDS reweighting + beta-NLL + bigger net + longer run)"
python train_halo_mass.py \
    --database-dir $DB1 $DB2 $DB3 $DB4 \
    --only-angular --epochs 300 --max-per-catalog 0 --weight-by-catalog-count \
    --reweight-by-mass --reweight-alpha 0.5 --reweight-bins 30 --lds-kernel-sigma 2.0 \
    --beta-nll 0.5 \
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
    --output $OUT/eval_vs_truth_v5.pdf

echo "======================================================"
echo "Finished: $(date)"
echo "======================================================"
