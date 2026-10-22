#!/bin/bash
#SBATCH --job-name=deep-db
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-3                  # one task per box; at least 2 boxes are needed for the
                                      # floor script (it cross-validates by box), e.g. --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                    # UNVERIFIED -- the halo catalog + cKDTree dominate, as in
                                      # run_multibox_database_slurm.sh; deeper limit adds only
                                      # per-environment arrays.
#SBATCH --time=48:00:00              # UNVERIFIED -- ~16% of theta rows, but more neighbors per
                                      # environment. Time the first catalogs in the .err log.
                                      # build_nre_database.py skips finished files, so a job that
                                      # hits the limit can simply be resubmitted to continue.
#SBATCH --partition=astro2_long      # verify this partition name/limit exists on your cluster
##SBATCH --account=your_account

# =============================================================================
# Depth-forecast NRE/halo-mass databases (2026-10-22): how much better can
# the bright galaxy's halo mass be inferred if the survey reaches fainter
# neighbors? Re-runs ONLY Stage 2 (build_nre_database.py) of
# run_multibox_database_slurm.sh -- the Stage-1 M_UV catalogs already exist
# and contain every halo, faint or not.
#
#   --faint-limit         survey depth (first argument to this script)
#   --max-env-neighbors   raised to 1000, so the 200 cap doesn't hide the
#                         extra neighbors a deeper limit brings in
#   --max-uvlf-sigma 5    only UVLF-consistent theta (~16% of prior.dat);
#                         rows keep their original index, so with the same
#                         --param-file/--seed/--max-environments-per-catalog
#                         as the original build the SAME bright galaxies are
#                         selected -> paired comparison with the -18.5 database
#
# Output stores bright_logmh and bright_muv natively (no patching needed).
# CHECK the .err log for "faint-neighbor sample is INCOMPLETE" warnings: they
# mean halos at the catalog's mass-resolution floor already reach this depth,
# i.e. the depth is beyond what the box can represent for that theta.
#
# Usage:
#   sbatch run_deep_database_slurm.sh -17.5
#   sbatch --array=0-1 run_deep_database_slurm.sh -16.5     # two boxes only
# Then: sbatch run_information_floor_slurm.sh -17.5
# =============================================================================

FAINT_LIMIT=${1:?usage: sbatch run_deep_database_slurm.sh <faint-limit, e.g. -17.5>}
DEPTH_TAG=$(echo "${FAINT_LIMIT#-}" | tr '.' 'p')     # -17.5 -> 17p5
MAX_ENV_NEIGHBORS=1000

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate UVLF_clust

mkdir -p logs

REPO=/groups/astro/ivannik/programs/NumberInference/GalaxyNumberInference
PRIOR_DAT=$REPO/prior.dat

NEIGHBORS=/groups/astro/ivannik/projects/Neighbors
CATALOGS_BASE=/lustre/astro/ivannik/catalogs_grid_prior
BOX_CACHE_BASE=/lustre/astro/ivannik/21cmFAST_cache/a4c5e3a912f09f0efa4f82b5a91a56e0
BOX_HASH2=ffa852ccaa39d8f82951cc98ff798ab4

SEEDS=(1955 2027 3142 4242)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
HALO_CATALOG=$BOX_CACHE_BASE/$SEED/$BOX_HASH2/10.5000/HaloCatalog.h5
CATALOG_DIR=${CATALOGS_BASE}_seed${SEED}
DB_OUTPUT_DIR=$NEIGHBORS/nre_database_uvlf5_faint${DEPTH_TAG}_seed${SEED}

echo "======================================================"
echo "Job:         $SLURM_JOB_NAME  (array task $SLURM_ARRAY_TASK_ID / job $SLURM_ARRAY_JOB_ID)"
echo "Seed:        $SEED"
echo "Faint limit: $FAINT_LIMIT   (max stored neighbors $MAX_ENV_NEIGHBORS)"
echo "Output:      $DB_OUTPUT_DIR"
echo "Started:     $(date)"
echo "======================================================"

# Fail early on a wrong path rather than after hours (cf. the 2026-10-16 patch run).
for f in "$HALO_CATALOG" "$CATALOG_DIR" "$PRIOR_DAT"; do
    [ -e "$f" ] || { echo "Missing: $f -- aborting."; exit 1; }
done

# --seed / --max-environments-per-catalog are left at the defaults (42 / 100),
# the values the original multibox build used -- required for the paired
# bright-galaxy selection described above.
python build_nre_database.py \
    --param-file "$PRIOR_DAT" \
    --halo-catalog-path "$HALO_CATALOG" \
    --catalog-dir "$CATALOG_DIR" \
    --output-dir "$DB_OUTPUT_DIR" \
    --faint-limit "$FAINT_LIMIT" \
    --max-env-neighbors $MAX_ENV_NEIGHBORS \
    --max-uvlf-sigma 5 \
    --n-workers "$SLURM_CPUS_PER_TASK"

echo "======================================================"
echo "Files written: $(ls "$DB_OUTPUT_DIR" | wc -l)"
echo "Finished: $(date)"
echo "======================================================"
