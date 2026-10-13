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
# Overnight halo-mass model retrain -- v9.
#
# v8 added the first architecture change (Deep Sets, --architecture
# deepset) to target the persistent shrinkage-to-the-mean S-shape that
# survived every v2-v7 loss/training-loop fix -- but v8's RMS (0.361 dex)
# and tail bias were essentially IDENTICAL to v7's plain MLP. That led to
# checking the TRAINING PRIOR itself rather than the model:
# diagnose_uvlf_stratified_neighbor_count.py found the uniform theta prior
# this whole database was built from is overwhelmingly UVLF-inconsistent
# (sampled log-likelihoods from -1.5 to -56,069,751.6 across just 2000
# theta -- nearly the WHOLE prior is astrophysically impossible), and
# plot_eval_vs_truth_uvlf_consistent.py confirmed the v8 model STILL shows
# a residual shrinkage bias even when EVALUATED only on the UVLF-consistent
# subset (theta within 5 sigma of Donnan+24) -- i.e. the bias partly
# reflects what the model was TRAINED on (overwhelmingly implausible junk
# diluting the gradient), not just what it's evaluated on.
#
# v9's fix: --uvlf-reweight (new in train_halo_mass.py) scales each
# example's training loss by its parent theta's UVLF plausibility, reusing
# the exact log-likelihood infer_nre.py's --use-uvlf path and
# check_uvlf_consistency.py already compute, via a smooth Gaussian taper in
# equivalent-sigma units (--uvlf-sigma-scale, module docstring in
# train_halo_mass.py for why a raw softmax on the log-L would instead
# underflow ~the whole dataset to zero weight given this prior's range).
# Kept the Deep Sets architecture from v8 (not reverting it -- it's a
# separate, independently-motivated fix that may still matter once the
# prior itself isn't drowning out the signal). Everything else (LDS,
# beta-NLL, warmup-protected early stopping) is kept identical to v7/v8 for
# a clean comparison.
#
# Re-evaluates TWICE at the end: the usual full-prior eval_vs_truth.pdf
# (comparable to every prior version), AND the same UVLF-consistent-only
# check (plot_eval_vs_truth_uvlf_consistent.py, <=5 sigma, matching the one
# already run against v8) -- that second one is the actual test of whether
# this fix worked, since that's exactly the plot that showed the residual
# bias in the first place.
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
OUT=$NEIGHBORS/halo_mass_model_multibox4_v9

echo ">>> Training halo-mass model v9 (UVLF-reweighted + Deep Sets + LDS + beta-NLL + early stopping)"
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
    --output $OUT/eval_vs_truth_v9.pdf

echo "======================================================"
echo ">>> Re-running eval_vs_truth (UVLF-consistent theta only, <=5 sigma) -- the actual test"
python plot_eval_vs_truth_uvlf_consistent.py \
    --model-dir $OUT \
    --database-dir $DB1 $DB2 $DB3 $DB4 \
    --n-examples 500 --max-sigma 5.0 \
    --output $OUT/eval_vs_truth_uvlf5sigma_v9.pdf

echo "======================================================"
echo "Finished: $(date)"
echo "======================================================"
