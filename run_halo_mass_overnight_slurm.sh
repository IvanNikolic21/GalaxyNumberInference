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
# Overnight halo-mass model retrain -- v11.
#
# v10 fixed the LDS x UVLF composition bug (max/min mass-bin ratio within
# the UVLF-consistent subset 245x -> 37x), yet the predicted-vs-true S-shape
# was unchanged. diagnose_heldout_calibration.py (held-out val split) showed
# why: binned by PREDICTED mu, the training-weighted val set lies on y = x
# (mean |<true>-<mu>| = 0.043 dex), i.e. the network was already a correct
# posterior mean -- the S-shape in true-binned plots is regression to the
# mean, expected whenever the inputs don't pin down Mh. Three real issues
# remained, all addressed here:
#
#  1. Missing information: the bright galaxy's OWN M_UV was never stored or
#     fed to the network (only the faint neighbors + "passed M_UV < -21.5").
#     -> patch_bright_logmh.py now backfills bright_muv; --use-bright-muv.
#     RUN THE PATCH ON ALL FOUR DATABASE DIRS BEFORE SUBMITTING THIS JOB --
#     files without bright_muv are skipped (and the dataset shrinks).
#  2. Non-Gaussian posterior: the v10 PIT showed a long low-mass tail and a
#     sharp upper edge. -> --n-components 3 (Gaussian-mixture head).
#  3. Wrong prior for the physical population: LDS (--reweight-by-mass) is a
#     weight on the TARGET, so it changes the learned posterior (a weight on
#     theta only cancels at fixed theta). On UVLF-consistent theta it made
#     low-mu predictions up to 0.5 dex too low. -> LDS dropped; UVLF and
#     catalog-count weighting (both functions of theta only) kept.
#
# smoke_test_halo_mass_mixture.py runs first and aborts the job if the new
# loss/heads misbehave. Evaluation at the end: the two usual eval_vs_truth
# plots plus diagnose_heldout_calibration.py (the meaningful test now --
# reverse binning, PIT, central AND one-sided coverage, on held-out data).
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
OUT=$NEIGHBORS/halo_mass_model_multibox4_v11

echo ">>> Smoke test (mixture loss / heads)"
python smoke_test_halo_mass_mixture.py || { echo "Smoke test FAILED -- aborting"; exit 1; }

echo "======================================================"
echo ">>> Training halo-mass model v11 (bright M_UV input + 3-component mixture + UVLF reweight, no LDS)"
python train_halo_mass.py \
    --database-dir $DB1 $DB2 $DB3 $DB4 \
    --only-angular --epochs 300 --max-per-catalog 0 --weight-by-catalog-count \
    --use-bright-muv --n-components 3 \
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
    --output $OUT/eval_vs_truth_v11.pdf

echo "======================================================"
echo ">>> Re-running eval_vs_truth (UVLF-consistent theta only, <=5 sigma) -- the actual test"
python plot_eval_vs_truth_uvlf_consistent.py \
    --model-dir $OUT \
    --database-dir $DB1 $DB2 $DB3 $DB4 \
    --n-examples 500 --max-sigma 5.0 \
    --output $OUT/eval_vs_truth_uvlf5sigma_v11.pdf

echo "======================================================"
echo ">>> Held-out calibration (reverse binning, PIT, central + one-sided coverage)"
python diagnose_heldout_calibration.py \
    --model-dir $OUT \
    --database-dir $DB1 $DB2 $DB3 $DB4 \
    --max-per-catalog 0 --weight-by-catalog-count \
    --uvlf-reweight --uvlf-sigma-scale 5.0 --max-weight-ratio 10.0 \
    --output $OUT/heldout_calibration_v11.pdf

echo "======================================================"
echo "Finished: $(date)"
echo "======================================================"
