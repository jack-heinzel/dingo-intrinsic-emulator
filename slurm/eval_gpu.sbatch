#!/bin/bash
# GPU part of the A/B evaluation: calibration (P-P) for both runs, then the points and
# emulator values for the reference comparison.
#   sbatch slurm/eval_gpu.sbatch
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -C h100-sxm5
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -J dingo-eval-gpu
#SBATCH -o /mnt/ceph/users/jheinzel/dingo-intrinsic-emulator/logs/eval_gpu_%j.log

set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/mnt/home/jheinzel/dingo  # custom priors (emulator.priors) in configs
REPO=/mnt/home/jheinzel/dingo
PY=$REPO/.venv/bin/python
NEV=${NEV:-10} NPT=${NPT:-10}
echo "start $(TZ=America/New_York date)  node $(hostname)"; nvidia-smi -L
$PY $REPO/scripts/eval_calibration.py sample A --n-inj 1000 --n-samples 5000
$PY $REPO/scripts/eval_calibration.py sample B --n-inj 1000 --n-samples 5000
$PY $REPO/scripts/eval_reference.py prepare --n-events "$NEV" --n-points "$NPT"
echo "end $(TZ=America/New_York date)"
