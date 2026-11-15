#!/bin/bash
# CPU part of the A/B evaluation: one dynesty evidence per (event, point), in parallel
# over the node's cores. Resumable: finished points are skipped on resubmission.
#   sbatch --dependency=afterok:<eval_gpu job> slurm/eval_reference.sbatch
#SBATCH -p gen
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH -J dingo-eval-ref
#SBATCH -o /mnt/ceph/users/jheinzel/dingo-intrinsic-emulator/logs/eval_ref_%j.log

set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
REPO=/mnt/home/jheinzel/dingo
echo "start $(TZ=America/New_York date)  node $(hostname)  cores $SLURM_CPUS_ON_NODE"
$REPO/.venv/bin/python $REPO/scripts/eval_reference.py reference --nproc "$SLURM_CPUS_ON_NODE" --nlive "${NLIVE:-500}"
set +u; module load texlive; set -u  # plot_style uses LaTeX
$REPO/.venv/bin/python $REPO/scripts/eval_reference.py analyze
echo "end $(TZ=America/New_York date)"
