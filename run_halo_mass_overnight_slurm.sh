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
# Overnight halo-mass model retrain -- v10.
#
# v9 added --uvlf-reweight to scale each example's training loss by its
# parent theta's UVLF plausibility -- but Ivan's read of
# eval_vs_truth_uvlf5sigma_v9.pdf was right: the residual shrinkage bias,
# WITHIN the UVLF-consistent subset, looked essentially unchanged from v8's.
# diagnose_effective_mass_distribution.py confirmed it quantitatively: the
# max/min bin fraction ratio within that subset was still 153x after v9's
# combined LDS x UVLF weighting (vs 245x raw) -- barely moved, and the
# weight that DID shift went toward the already-dense HIGH-mass end, not
# the genuinely sparse low-mass end.
#
# Root cause, found by that same diagnostic: --reweight-by-mass (LDS)
# computed its density estimate over the RAW, unweighted example counts --
# dominated by the 87% UVLF-implausible majority, where low mass looks
# DENSE (flooded by implausible theta that can scatter low-mass halos
# bright). So LDS was boosting whatever's globally sparse (high mass),
# while UVLF-reweighting independently ALSO favors high mass (same
# correlation, r=+0.38/rho=+0.78 between UVLF log-L and host mass) --
# the two factors compounded in the SAME direction instead of LDS
# targeting what's actually rare among UVLF-plausible theta specifically.
# This was a composition bug between two reweighting schemes, not evidence
# the architecture needed more work.
#
# v10's fix (train_halo_mass.py, no new CLI flags -- same v9 command,
# automatic once both --reweight-by-mass and --uvlf-reweight are on): LDS's
# density estimate is now computed USING the UVLF weight as the reference
# population (np.bincount's weights= argument) instead of a plain count --
# so "sparse" now means "sparse among UVLF-plausible theta," not "sparse
# globally." Verified by simulation against the real empirical structure
# before this run: under the old unweighted LDS, the sparsest-looking bin
# was the HIGHEST mass bin (backwards); under the fixed, UVLF-weighted LDS,
# it's correctly the LOWEST mass bin among plausible theta.
#
# Re-evaluates TWICE at the end, same as v9: the usual full-prior
# eval_vs_truth.pdf, AND the UVLF-consistent-only check
# (plot_eval_vs_truth_uvlf_consistent.py, <=5 sigma) -- the second one is
# the actual test of whether the conditioned-LDS fix worked. Compare its
# per-bin spread directly against v9's (and run
# diagnose_effective_mass_distribution.py again against v10's database
# weighting if the max/min ratio needs rechecking numerically).
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
OUT=$NEIGHBORS/halo_mass_model_multibox4_v10

echo ">>> Training halo-mass model v10 (UVLF-conditioned LDS + Deep Sets + beta-NLL + early stopping)"
python train_halo_mass.py \
    --database-dir $DB1 $DB2 $DB3 $DB4 \
    --only-angular --epochs 300 --max-per-catalog 0 --weight-by-catalog-count \
    --reweight-by-mass --reweight-alpha 0.5 --reweight-bins 30 --lds-kernel-sigma 2.0 \
    --beta-nll 0.5 \
    --uvlf-reweight --uvlf-sigma-scale 5.0 \
    --max-weight-ratio 10.0 --grad-clip 5.0 \
    --early-stop-patience 30 --early-stop-warmup 70 \
    --diag-every 10 --diag-bins 8 \
    --architecture deepset --phi-dims 64 64 --pooling sum \
    --hidden-dims 512 512 512 512 512 512 \
    --batch-size 1024 \
    --output-dir $OUT

echo "======================================================"
echo ">>> Re-running eval_vs_truth (full prior) for direct before/after comparison"
python evaluate_halo_mass_model.py \
    --model-dir $OUT \
    --database-dir $DB1 $DB2 $DB3 $DB4 \
    --n-examples 500 \
    --output $OUT/eval_vs_truth_v10.pdf

echo "======================================================"
echo ">>> Re-running eval_vs_truth (UVLF-consistent theta only, <=5 sigma) -- the actual test"
python plot_eval_vs_truth_uvlf_consistent.py \
    --model-dir $OUT \
    --database-dir $DB1 $DB2 $DB3 $DB4 \
    --n-examples 500 --max-sigma 5.0 \
    --output $OUT/eval_vs_truth_uvlf5sigma_v10.pdf

echo "======================================================"
echo "Finished: $(date)"
echo "======================================================"
