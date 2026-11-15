#!/bin/bash
# Train run A or B on one H100 (4-GPU Genoa nodes; 24 cores and 360 GB = one GPU's share):  sbatch slurm/train.sbatch A   (or B)
# Resumes from model_latest.pt if the run directory already has one.
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -C h100-sxm5
#SBATCH -c 24
#SBATCH --mem=360G
#SBATCH -t 2-00:00:00
#SBATCH -J dingo-train
#SBATCH -o /mnt/ceph/users/jheinzel/dingo-intrinsic-emulator/logs/train_%x_%j.log

set -euo pipefail
RUN=${1:?usage: sbatch slurm/train.sbatch A|B}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
REPO=/mnt/home/jheinzel/dingo
DATA=/mnt/ceph/users/jheinzel/dingo-intrinsic-emulator
DIR=$DATA/training/run_$RUN
mkdir -p "$DIR"
echo "run $RUN start $(TZ=America/New_York date)  node $(hostname)"
nvidia-smi -L

ARGS=(--settings_file "$REPO/configs/train_$RUN.yaml" --train_dir "$DIR")
if [ -f "$DIR/model_latest.pt" ]; then
  echo "resuming from $DIR/model_latest.pt"
  ARGS=(--train_dir "$DIR" --checkpoint "$DIR/model_latest.pt")
fi
"$REPO/.venv/bin/python" "$REPO/scripts/train.py" "${ARGS[@]}"
echo "run $RUN end $(TZ=America/New_York date)"
