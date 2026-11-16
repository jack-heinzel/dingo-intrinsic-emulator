#!/bin/bash
# Cross-run score (scripts/eval_score.py): every run moved onto run PRIOR's training prior and
# scored by log q'(truth) on the same injections.
#   PRIOR=A NINJ=1000 NSAMP=20000 sbatch slurm/eval_score.sbatch A B D A3 B3 D3
# Writes eval/score_prior<PRIOR>.pkl (saved after each run).
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -C h100-sxm5
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 08:00:00
#SBATCH -J dingo-eval-score
#SBATCH -o /mnt/ceph/users/jheinzel/dingo-intrinsic-emulator/logs/eval_score_%j.log

set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/mnt/home/jheinzel/dingo  # custom priors (emulator.priors) in configs
REPO=/mnt/home/jheinzel/dingo
PRIOR=${PRIOR:-A} NINJ=${NINJ:-1000} NSAMP=${NSAMP:-20000} SEED=${SEED:-2}
echo "start $(TZ=America/New_York date)  node $(hostname)  runs $*  prior $PRIOR  n $NINJ  samples $NSAMP"
$REPO/.venv/bin/python $REPO/scripts/eval_score.py sample --runs "$@" --prior-run "$PRIOR" \
  --n-inj "$NINJ" --n-samples "$NSAMP" --seed "$SEED"
$REPO/.venv/bin/python $REPO/scripts/eval_score.py summary --prior-run "$PRIOR"
echo "end $(TZ=America/New_York date)"
