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
# Overnight halo-mass model retrain, folding in all three improvements
# discussed after reviewing eval_vs_truth.pdf's S-shaped shrinkage-to-the-mean:
#
#   1. --reweight-by-mass: inverse-density loss reweighting by true log(Mh)
#      bin, to stop the dense middle band from dominating gradient updates
#      and starving the sparse low-/high-mass tails.
#   2. --diag-every 10: per-mass-bin val RMS/bias/68%-coverage logged
#      periodically, so tail convergence can be checked directly in the log
#      instead of inferring it from the (middle-band-dominated) mean val loss.
#   3. Bigger network (6x512 vs the original 4x256) and more epochs (300 vs
#      100), since GPU is available and the prior run showed no sign of the
#      shrinkage being a capacity ceiling vs. just an undertrained-tail issue
#      -- this is a cheap thing to try alongside (1)/(2), not the primary fix.
#
# Re-evaluates against eval_vs_truth.pdf's own script at the end so the
# before/after comparison is a straight diff of the two PDFs.
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
OUT=$NEIGHBORS/halo_mass_model_multibox4_v2

echo ">>> Training halo-mass model v2 (reweighted + bigger net + longer run)"
python train_halo_mass.py \
    --database-dir $DB1 $DB2 $DB3 $DB4 \
    --only-angular --epochs 300 --max-per-catalog 0 --weight-by-catalog-count \
    --reweight-by-mass --reweight-alpha 0.5 --reweight-bins 30 \
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
    --output $OUT/eval_vs_truth_v2.pdf

echo "======================================================"
echo "Finished: $(date)"
echo "======================================================"
