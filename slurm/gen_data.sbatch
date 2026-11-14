#!/bin/bash
# Data for runs A and B: 1M-waveform datasets A and B, and the O3 H1/L1/V1 ASD dataset
# (900 PSDs from scripts/thin_psd_segments.py, 16 parallel GWOSC download workers).
#SBATCH -p gen
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 06:00:00
#SBATCH -J dingo-gen
#SBATCH -o /mnt/ceph/users/jheinzel/dingo-intrinsic-emulator/logs/gen_%j.log

set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
REPO=/mnt/home/jheinzel/dingo
DATA=/mnt/ceph/users/jheinzel/dingo-intrinsic-emulator
BIN=$REPO/.venv/bin
NWORK=16
NPROC=$(( SLURM_CPUS_ON_NODE - NWORK ))
mkdir -p "$DATA/waveforms" "$DATA/asd/full_work"
echo "start $(TZ=America/New_York date)  node $(hostname)  cores $SLURM_CPUS_ON_NODE"

# ASD downloads in the background (I/O bound, ~90 s per PSD)
pids=()
for c in $(seq -w 0 $(( NWORK - 1 ))); do
  seg="$DATA/asd/full_work/segments_chunk_$c.pkl"
  "$BIN/dingo_estimate_psds" --data_dir "$DATA/asd/full_work" \
    --settings_file "$REPO/configs/asd_O3.yaml" --time_segments_file "$seg" \
    > "$DATA/asd/full_work/chunk_$c.log" 2>&1 &
  pids+=($!)
done

# Waveforms on the remaining cores, written to node-local disk then copied to Ceph
for r in A B; do
  "$BIN/dingo_generate_dataset" --settings_file "$REPO/configs/waveforms_$r.yaml" \
    --out_file "/tmp/waveforms_$r.hdf5" --num_processes "$NPROC"
  cp "/tmp/waveforms_$r.hdf5" "$DATA/waveforms/waveforms_$r.hdf5"
  echo "waveforms $r done $(TZ=America/New_York date)"
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$(( fail + 1 )); done
echo "ASD workers done $(TZ=America/New_York date); failed workers: $fail"

"$BIN/dingo_merge_asd_datasets" --data_dir "$DATA/asd/full_work" \
  --settings_file "$REPO/configs/asd_O3.yaml" \
  --time_segments_file "$DATA/asd/full_work/segments_all.pkl" \
  --out_name "$DATA/asd/asds_O3.hdf5"
echo "end $(TZ=America/New_York date)"
