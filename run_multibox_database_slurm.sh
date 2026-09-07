#!/bin/bash
#SBATCH --job-name=nre-multibox-db
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                    # UNVERIFIED -- no timing data for this yet, check after
                                      # the first task or two before trusting this for all 4.
#SBATCH --time=08:00:00              # UNVERIFIED -- same.
#SBATCH --partition=astro2_long      # verify this partition name/limit exists on your cluster
##SBATCH --account=your_account

# =============================================================================
# 2026-09-07 multi-box NRE database rebuild -- see [[nre-training-imbalance]]
# memory for the full diagnosis this addresses (Phase 3: genuine seed-to-seed
# training instability, traced to every theta in the training grid sharing
# ONE hardcoded density-field realization; confirmed by the 2026-09-07 SBC
# sweep finding broad miscalibration on all 3 params). This runs the 2-stage
# database pipeline once per independent coeval box (4 boxes, z=10.5,
# generated earlier), each against the SAME prior.dat theta grid (deliberately
# NOT expanded this round -- see chat discussion 2026-09-07: isolating the
# box-diversity variable for a clean before/after SBC comparison; a larger
# grid is a reasonable follow-up once this is validated, not bundled in).
#
# Stage 1 (generate_catalog_database.py): ~4000 cheap MUV catalogs per box
#   (sample_muv draws on that box's existing halo masses -- no new N-body/
#   reionization physics, the expensive part is already done).
# Stage 2 (build_nre_database.py): the actual neighbor search -> NRE training
#   environments, against that same box's halo catalog + freshly-built
#   per-theta MUV catalogs from stage 1.
#
# Output per box (seed-suffixed, so all 4 combine cleanly via train_nre.py's
# --prior-database-dir <dir1> <dir2> <dir3> <dir4>, nargs='+' as of this
# session):
#   catalogs_grid_prior_seed<seed>/          (stage 1)
#   nre_database_prior_capped_seed<seed>/    (stage 2, final training data)
# =============================================================================

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate UVLF_clust

mkdir -p logs

REPO=/groups/astro/ivannik/programs/NumberInference/GalaxyNumberInference
PRIOR_DAT=$REPO/prior.dat

NEIGHBORS=/groups/astro/ivannik/projects/Neighbors
CATALOGS_BASE=/lustre/astro/ivannik/catalogs_grid_prior

# Coeval-box cache -- confirmed 2026-09-07: top-level hash differs from the
# original single-box catalog (a4c5e3a9... vs d12b21e8...), second-level
# hash matches. Worth double-checking generate_coeval_box.py's
# InputParameters against whatever produced the original catalog before
# fully trusting this is apples-to-apples (box size/resolution), but not
# blocking this run.
BOX_CACHE_BASE=/lustre/astro/ivannik/21cmFAST_cache/a4c5e3a912f09f0efa4f82b5a91a56e0
BOX_HASH2=ffa852ccaa39d8f82951cc98ff798ab4

SEEDS=(1955 2027 3142 4242)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
HALO_CATALOG=$BOX_CACHE_BASE/$SEED/$BOX_HASH2/10.5000/HaloCatalog.h5

CATALOG_DIR=${CATALOGS_BASE}_seed${SEED}
DB_OUTPUT_DIR=$NEIGHBORS/nre_database_prior_capped_seed${SEED}

echo "======================================================"
echo "Job:      $SLURM_JOB_NAME  (array task $SLURM_ARRAY_TASK_ID / job $SLURM_ARRAY_JOB_ID)"
echo "Seed:     $SEED"
echo "Halo cat: $HALO_CATALOG"
echo "Started:  $(date)"
echo "======================================================"

echo "--- Stage 1: generate_catalog_database.py ---"
python generate_catalog_database.py \
    --param-file "$PRIOR_DAT" \
    --halo-catalog-path "$HALO_CATALOG" \
    --output-dir "$CATALOG_DIR" \
    --n-workers "$SLURM_CPUS_PER_TASK"

echo "--- Stage 2: build_nre_database.py ---"
python build_nre_database.py \
    --param-file "$PRIOR_DAT" \
    --halo-catalog-path "$HALO_CATALOG" \
    --catalog-dir "$CATALOG_DIR" \
    --output-dir "$DB_OUTPUT_DIR" \
    --n-workers "$SLURM_CPUS_PER_TASK"

echo "======================================================"
echo "Finished: $(date)"
echo "======================================================"
