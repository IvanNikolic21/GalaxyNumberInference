#!/bin/bash
#SBATCH --job-name=info-floor
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                    # UNVERIFIED -- features are small; this covers reading the
                                      # largest .npz files of the deep databases.
#SBATCH --time=12:00:00              # UNVERIFIED -- two passes (projected, 3D) of 6 feature sets
                                      # x 4 box-folds; the 14-feature k-NN queries are the slow part.
#SBATCH --partition=astro2_long      # verify this partition name/limit exists on your cluster
##SBATCH --account=your_account

# =============================================================================
# Model-free information floor for the bright galaxy's halo mass
# (diagnose_information_floor.py), 2026-10-22. Runs it twice on one set of
# databases: projected distances (imaging) and 3D distances (--use-dz, an
# upper bound on perfect line-of-sight information). Compare the printed
# bounds with the v11 network's 0.243 dex.
#
# Usage:
#   sbatch run_information_floor_slurm.sh            # the existing -18.5 databases
#   sbatch run_information_floor_slurm.sh -17.5      # deep databases from run_deep_database_slurm.sh
#   sbatch --dependency=afterok:<deep-db job id> run_information_floor_slurm.sh -17.5
#
# For a deep run, only the boxes whose database directory exists are used
# (at least 2 needed: the script cross-validates by box). The -18.5 run uses
# the same UVLF cut, so its bright galaxies are the same ones -> the floors
# at different depths are directly comparable.
# =============================================================================

FAINT_LIMIT=${1:--18.5}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate UVLF_clust

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p logs

NEIGHBORS=/groups/astro/ivannik/projects/Neighbors
SEEDS=(1955 2027 3142 4242)

if [ "$FAINT_LIMIT" = "-18.5" ]; then
    PREFIX=$NEIGHBORS/nre_database_prior_capped_seed
    ENV_CAP=200
    TAG=faint18p5
else
    DEPTH_TAG=$(echo "${FAINT_LIMIT#-}" | tr '.' 'p')
    PREFIX=$NEIGHBORS/nre_database_uvlf5_faint${DEPTH_TAG}_seed
    ENV_CAP=1000                     # run_deep_database_slurm.sh's --max-env-neighbors
    TAG=faint${DEPTH_TAG}
fi

DBS=()
for s in "${SEEDS[@]}"; do
    [ -d "${PREFIX}${s}" ] && DBS+=("${PREFIX}${s}")
done
if [ ${#DBS[@]} -lt 2 ]; then
    echo "Found ${#DBS[@]} database dir(s) matching ${PREFIX}<seed>; need >= 2 (CV by box). Aborting."
    exit 1
fi

OUT=$NEIGHBORS/information_floor
mkdir -p "$OUT"

echo "======================================================"
echo "Job:         $SLURM_JOB_NAME  ($SLURM_JOB_ID)"
echo "Faint limit: $FAINT_LIMIT   (env cap $ENV_CAP)"
echo "Databases:   ${DBS[*]}"
echo "Started:     $(date)"
echo "======================================================"

echo ">>> Projected distances (imaging)"
python diagnose_information_floor.py \
    --database-dir "${DBS[@]}" \
    --faint-limit "$FAINT_LIMIT" --env-cap $ENV_CAP \
    --output "$OUT/floor_${TAG}_projected.npz"

echo "======================================================"
echo ">>> 3D distances (--use-dz: perfect line-of-sight upper bound)"
python diagnose_information_floor.py \
    --database-dir "${DBS[@]}" \
    --faint-limit "$FAINT_LIMIT" --env-cap $ENV_CAP --use-dz \
    --output "$OUT/floor_${TAG}_3d.npz"

echo "======================================================"
echo "Finished: $(date)"
echo "======================================================"
