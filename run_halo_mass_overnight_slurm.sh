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
#SBATCH --partition=astro2_gpu       # VERIFY: replace with your cluster's actual GPU partition
                                      # name before submitting -- this is a guess based on the
                                      # astro2_short/astro2_long naming seen in the other scripts.
#SBATCH --gres=gpu:1                 # VERIFY: your cluster's gres syntax for requesting 1 GPU
##SBATCH --account=your_account

# =============================================================================
# Overnight halo-mass model retrain -- v3.
#
# v2 (halo_mass_model_multibox4_v2, already run) added:
#   1. --reweight-by-mass: inverse-density loss reweighting by true log(Mh)
#      bin, to stop the dense middle band from dominating gradient updates
#      and starving the sparse low-/high-mass tails.
#   2. --diag-every 10: per-mass-bin val RMS/bias/68%-coverage logged
#      periodically.
#   3. Bigger network (6x512 vs the original 4x256) and more epochs (300 vs
#      100).
# ...but v2's own log showed train/val loss freeze bit-for-bit by epoch ~10,
# with log_sigma pegged at its ceiling for ~every example (mean predicted
# sigma = 20.09 dex, 100% "coverage" in eval_vs_truth_v2.pdf). Root cause:
# --reweight-by-mass x --weight-by-catalog-count gave some examples >100x
# the weight of others, a few of which dominated every batch's gradient and
# made inflating sigma the network's cheapest way to shrink their loss
# contribution. v3 adds the fix (see train_halo_mass.py's HaloMassDataset
# weight-clipping block and train_epoch's grad_clip):
#   4. --max-weight-ratio 10: clips the combined per-example weight to
#      [median/10, median*10] before training ever sees it.
#   5. --grad-clip 5.0 (train_halo_mass.py's new default, passed explicitly
#      here too for clarity): caps the gradient norm every step, as a second
#      line of defense against the same instability.
#
# Re-evaluates against eval_vs_truth.pdf's own script at the end so the
# before/after comparison is a straight diff of the three PDFs (v1, v2, v3).
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
OUT=$NEIGHBORS/halo_mass_model_multibox4_v3

echo ">>> Training halo-mass model v3 (reweighted + clipped + bigger net + longer run)"
python train_halo_mass.py \
    --database-dir $DB1 $DB2 $DB3 $DB4 \
    --only-angular --epochs 300 --max-per-catalog 0 --weight-by-catalog-count \
    --reweight-by-mass --reweight-alpha 0.5 --reweight-bins 30 \
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
    --output $OUT/eval_vs_truth_v3.pdf

echo "======================================================"
echo "Finished: $(date)"
echo "======================================================"
