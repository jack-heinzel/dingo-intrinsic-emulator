#!/bin/bash
# Calibration (P-P) for the given runs with a chosen seed and number of injections.
#   SEED=1 NINJ=2000 sbatch slurm/eval_calib.sbatch A B
# Writes eval/calibration_<run>_seed<SEED>.pkl.
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -C h100-sxm5
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 02:30:00
#SBATCH -J dingo-eval-calib
#SBATCH -o /mnt/ceph/users/jheinzel/dingo-intrinsic-emulator/logs/eval_calib_%j.log

set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/mnt/home/jheinzel/dingo  # custom priors (emulator.priors) in configs
REPO=/mnt/home/jheinzel/dingo
SEED=${SEED:-1} NINJ=${NINJ:-2000}
echo "start $(TZ=America/New_York date)  node $(hostname)  runs $*  seed $SEED  n $NINJ"
for r in "$@"; do
  $REPO/.venv/bin/python $REPO/scripts/eval_calibration.py sample "$r" --n-inj "$NINJ" --seed "$SEED" --tag "_seed$SEED"
done
echo "end $(TZ=America/New_York date)"
