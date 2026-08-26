#!/bin/bash
#SBATCH --job-name=coeval-z105
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G                   # UNVERIFIED -- check against your original script's actual
                                      # memory usage (HALO_CATALOG_MEM_FACTOR=5.0 suggests this run
                                      # can be memory-hungry) and adjust before submitting.
#SBATCH --time=12:00:00              # UNVERIFIED -- same, check against actual past runtime.
#SBATCH --partition=astro2_long      # verify this partition name/limit exists on your cluster
##SBATCH --account=your_account

# =============================================================================
# Launches 4 independent 21cmFAST coeval boxes at z=10.5 as a SLURM job array
# -- all 4 seeds run in parallel (as cluster resources allow), each as its own
# array task with its own log, rather than looping through them one at a time
# in a single job. This is the "generate genuinely independent boxes" step
# motivated by the NRE calibration investigation: the whole training/inference
# pipeline currently shares ONE hardcoded halo catalog for the entire prior
# grid, and giving training real box-to-box (cosmic-variance) diversity is
# the leading hypothesis for the training instability found there.
#
# Seed 1955 is Ivan's original script's seed (kept first for continuity with
# anything already cached under it); 2026, 3141, 4242 are new, otherwise
# arbitrary -- swap freely, they just need to be distinct.
# =============================================================================

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate galaxy-neighbors        # UNVERIFIED -- confirm this is where py21cmfast actually lives

mkdir -p logs

SEEDS=(1955 2026 3141 4242)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "======================================================"
echo "Job:      $SLURM_JOB_NAME  (array task $SLURM_ARRAY_TASK_ID / job $SLURM_ARRAY_JOB_ID)"
echo "Seed:     $SEED"
echo "Node:     $SLURMD_NODENAME"
echo "Started:  $(date)"
echo "======================================================"

python generate_coeval_box.py --seed "$SEED" --n-threads "$SLURM_CPUS_PER_TASK"

echo "======================================================"
echo "Finished: $(date)"
echo "======================================================"