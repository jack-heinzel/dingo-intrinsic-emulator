#!/bin/bash
# Regenerate one waveform dataset:  sbatch slurm/gen_waveforms.sbatch B
# Writes to node-local disk, then copies to Ceph (replacing waveforms_<RUN>.hdf5).
#SBATCH -p gen
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 01:30:00
#SBATCH -J dingo-genwf
#SBATCH -o /mnt/ceph/users/jheinzel/dingo-intrinsic-emulator/logs/genwf_%j.log

set -euo pipefail
RUN=${1:?usage: sbatch slurm/gen_waveforms.sbatch A|B}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
REPO=/mnt/home/jheinzel/dingo
DATA=/mnt/ceph/users/jheinzel/dingo-intrinsic-emulator
echo "start $(TZ=America/New_York date)  node $(hostname)  cores $SLURM_CPUS_ON_NODE"
"$REPO/.venv/bin/dingo_generate_dataset" --settings_file "$REPO/configs/waveforms_$RUN.yaml" \
  --out_file "/tmp/waveforms_$RUN.hdf5" --num_processes "$SLURM_CPUS_ON_NODE"
cp "/tmp/waveforms_$RUN.hdf5" "$DATA/waveforms/waveforms_$RUN.hdf5"
echo "end $(TZ=America/New_York date)"
