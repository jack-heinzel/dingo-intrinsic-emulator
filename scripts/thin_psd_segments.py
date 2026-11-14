"""Pick N PSD segments per detector, evenly spaced over the observing run, and split them
into chunk files for parallel `dingo_estimate_psds` workers.

dingo's own selection (num_psds_max) keeps only the first N segments, and time_gap acts
only within a science segment, so on O3 it barely thins (~5700 H1 segments either way).

    python scripts/thin_psd_segments.py configs/asd_O3.yaml OUT_DIR --per-detector 300 --chunks 16

Writes OUT_DIR/segments_all.pkl (for dingo_merge_asd_datasets) and
OUT_DIR/segments_chunk_XX.pkl (one per worker).
"""
import argparse
import os
import pickle

import numpy as np
import yaml

from dingo.gw.noise.utils import get_time_segments

parser = argparse.ArgumentParser()
parser.add_argument("settings_file")
parser.add_argument("out_dir")
parser.add_argument("--per-detector", type=int, default=300)
parser.add_argument("--chunks", type=int, default=16)
args = parser.parse_args()

settings = yaml.safe_load(open(args.settings_file))["dataset_settings"]
segments = get_time_segments(settings)
thinned = {}
for det, segs in segments.items():
    n = min(args.per_detector, len(segs))
    pick = np.unique(np.round(np.linspace(0, len(segs) - 1, n)).astype(int))
    thinned[det] = [segs[i] for i in pick]
    print(f"{det}: {len(thinned[det])} of {len(segs)} segments, "
          f"GPS {thinned[det][0][0]} - {thinned[det][-1][0]}")

os.makedirs(args.out_dir, exist_ok=True)
with open(os.path.join(args.out_dir, "segments_all.pkl"), "wb") as f:
    pickle.dump(thinned, f)
for c in range(args.chunks):
    chunk = {det: segs[c::args.chunks] for det, segs in thinned.items()}
    with open(os.path.join(args.out_dir, f"segments_chunk_{c:02d}.pkl"), "wb") as f:
        pickle.dump(chunk, f)
print(f"wrote {args.chunks} chunk files to {args.out_dir}")
