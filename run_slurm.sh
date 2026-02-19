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

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate galaxy-neighbors

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p logs

echo "======================================================"
echo "Job:      $SLURM_JOB_NAME  ($SLURM_JOB_ID)"
echo "Node:     $SLURMD_NODENAME"
echo "CPUs:     $SLURM_CPUS_PER_TASK"
echo "Started:  $(date)"
echo "======================================================"

# ---------------------------------------------------------------------------
# Run  —  pick one redshift per submission
# ---------------------------------------------------------------------------
# Default realization counts reflect catalog sizes:
#   z=8.0  ->  1 realization
#   z=10.5 ->  1 realization
#   z=12.0 -> 50 realizations
#   z=14.0 -> 200 realizations

# z=10.5 (default for testing)
python run_analysis.py --redshift 10.5 --muv-realizations 20 --force-recompute

# z=8
 python run_analysis.py --redshift 8.0 --muv-realizations 5 --force-recompute

# z=12
 python run_analysis.py --redshift 12.0 --muv-realizations 50 --force-recompute

# z=14
 python run_analysis.py --redshift 14.0 --muv-realizations 200 --force-recompute

# Force recompute after changing magnitude grids:
# python run_analysis.py --redshift 10.5 --muv-index 0 --force-recompute

# Build cache only, no plots:
# python run_analysis.py --redshift 14.0 --muv-realizations 200 --no-plots

echo "======================================================"
echo "Finished: $(date)"
echo "======================================================"
