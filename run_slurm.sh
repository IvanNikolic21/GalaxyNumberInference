#!/bin/bash
#SBATCH --job-name=galaxy-d1s
#SBATCH --output=logs/%x_%j.out      # stdout  → logs/galaxy-d1s_<jobid>.out
#SBATCH --error=logs/%x_%j.err       # stderr  → logs/galaxy-d1s_<jobid>.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            # adjust to your node; used by numpy/scipy
#SBATCH --mem=32G                    # adjust based on your catalog sizes
#SBATCH --time=10:00:00              # hh:mm:ss — be generous the first time
#SBATCH --partition=astro2_short            # replace with your cluster's partition name
##SBATCH --account=your_account      # uncomment if your cluster requires this

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# Make conda available (adjust path if your conda lives elsewhere)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate galaxy-neighbors

# Tell numpy/scipy/h5py how many threads they're allowed to use.
# Without this they often spin up too many threads and fight each other.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------
mkdir -p logs

echo "======================================================"
echo "Job:      $SLURM_JOB_NAME  ($SLURM_JOB_ID)"
echo "Node:     $SLURMD_NODENAME"
echo "CPUs:     $SLURM_CPUS_PER_TASK"
echo "Started:  $(date)"
echo "======================================================"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

# First run: compute everything and cache
python run_analysis.py --muv-index 0 --force-recompute

# Uncomment to force a full recompute (e.g. after changing magnitude limits):
# python run_analysis.py --muv-index 0 --force-recompute

# Uncomment to skip plots and just build the cache (fast sanity check):
# python run_analysis.py --muv-index 0 --no-plots

echo "======================================================"
echo "Finished: $(date)"
echo "======================================================"
