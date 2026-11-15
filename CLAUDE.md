# dingo-intrinsic-emulator

Build an emulator for the **marginal likelihood of the intrinsic BBH parameters**,
L(m1, q, a1, a2, cos θ1, cos θ2, z | d), marginalized over the extrinsic parameters (sky location, inclination,
polarization, phase, time), using DINGO neural posterior estimation (user, 2026-11-14). It will feed population
inference such as `~/bbh-pixelpop-constraints`. See README.md for the approach. `dingo-tutorial/` is a git submodule of
stephengreen/dingo-tutorial (the CCA, November 2026 tutorial). Don't edit it here; put our own notebooks and code at
the top level.

## Decisions (user, 2026-11-14)

- GitHub: `jack-heinzel/dingo-intrinsic-emulator`, private, no license for now.
- Environment: `dingo-gw==0.10.3` pinned with the tutorial's constraints (`bilby_pipe` 1.8.1, since `dingo_pipe` doesn't
  work with later versions).
- Plots: `plot_style.py` is ported unchanged from bbh-pixelpop-constraints (see Conventions).

## Design: factorized posterior (user, 2026-11-14)

q(θ | d) = q1(θ_int | d) · q2(φ12, φ_jl, θ_ext | θ_int, d). These are two DINGO networks, chained at inference.
The 15 dimensions split 7 + 2 + 6.
- Training: q1 has `inference_parameters: [7 intrinsics]`. Anything omitted is drawn from the prior and so
  marginalized. q2 has `inference_parameters: [phi_12, phi_jl, theta_jn, phase, psi, ra, dec, geocent_time]` and
  `context_parameters: [7 intrinsics]`. This is the generic `data.context_parameters` key (set_train_transforms).
- Inference: `ChainComposer([FlowFactor(q1), FlowFactor(q2), RAToEventFrame()])` (`dingo.core.inference`, docs
  `sampling_chains.md`). `sample_and_log_prob` gives the full 15D log q for importance sampling. `log_prob` sums
  the factors in reverse, so q1's factor alone is the exact intrinsic marginal: the emulator,
  log L(θ_int | d) = log q1 − log π(θ_int) + const.
- Caveats (checked 2026-11-14):
  - Tokenization (DINGO-T1's transformer) raises NotImplementedError with non-empty `context_parameters`, so q2
    can't use the T1 architecture. T1 is a baseline only.
  - Physical parameters as context are untested: the only examples are GNPE time proxies and the BNS chirp-mass
    pin. Run a small test first.

## Coordinates for q1 (masses: user, 2026-11-14; spins: proposed, not yet agreed)

- Masses: use the exchange symmetry of the likelihood as a reflective boundary (user). Relabeling swaps the spins
  too: (m1, a1, θ1) <-> (m2, a2, θ2), and φ12 -> −φ12 in q2. Proposed coordinates are (ln Mc, ln q) with
  ln q ∈ (ln q_min, −ln q_min), so the swap is ln q -> −ln q.
- Principle: only degeneracies (one physical point, many coordinate values) hurt. Jacobian zeros don't, because
  L = q_x/π_train,x is a scalar computed in the same coordinates x, so the Jacobian cancels, and the user's prior
  multiplies L as a function.
- Spins (proposed): the elliptical-grid map from the square (u, v) ∈ [−1, 1]² to the disk:
  χ_z = u √(1 − v²/2), χ_⊥ = v √(1 − u²/2), with χ_⊥ signed. It is one-to-one on the closed square (edges onto
  quarter circles, no collapse). J = (1 − u²/2 − v²/2)/(√(1 − u²/2) √(1 − v²/2)) is zero only at the 4 corners.
  It is smooth, a = 0 is (0, 0), and the mirror is v -> −v. Checked numerically 2026-11-14: closed-form inverse
  round-trips to 4e-14, J matches finite differences. L is smooth in (χ_z, χ_⊥²), hence in (u, v).
- Rules for J: define π_train in box coordinates, so the flow's target is L × smooth. Never divide q_box by J,
  except to plot the training-prior posterior physically. J ≤ 1 everywhere (max 1 at the origin, mean π/4) and
  J ≈ 2.8 × the distance to a corner. A uniform box prior puts ~4% of samples within 0.1 of the 4 corner images
  (2.1× the uniform-disk rate). Reweighting to a physical prior uses w = π_phys · J/π_box, so w -> 0 at the
  corners (safe).
- The singularity that matters for reweighting is the user's prior at the origin: π_user ∝ |sin θ|/a. With π_train
  finite at a = 0, E[w²] ∝ ∫ da/a diverges logarithmically for events whose likelihood allows zero spin, so
  ESS ~ N/log N. The emulator avoids this: evaluate L(θ) directly, with L(a = 0) well defined and tilt-independent.
  For posterior samples under π_user, either sample L·π_user directly (MCMC, or importance sampling with a 1/a
  proposal; preferred) or use the mixture π_train = ½ π_box + ½ π_user (w ≤ 2, but half the spike returns to the
  flow's target).
- Rejected (2026-11-14, user): s = χ_⊥/√(1 − χ_z²). The edges χ_z = ±1 collapse to the poles, which is a
  degeneracy.
- Rejected (2026-11-14, user): the radial stretch u = artanh(a)·n̂, or any map that sends a wall to infinity.
  p(a -> 1) ∝ (1 − a)^(α/2 − 1) for a learned tail e^(−αr): a spike or a zero unless α = 2 exactly. dingo's
  linear-tail splines with a Gaussian base give p(1) = 0.
- Flow: the default dingo NSF (user, 2026-11-14: "try to avoid building any new code"), with
  `tail_bound: 3.0` so the splines cover the whole standardized box (edges at ±1.73 to ±1.86σ; the default 1.0
  leaves ~40% of the box in linear tails). Folding removes the mirror walls, but the outer walls (|χ| = 1, q_min,
  Mc and d_L bounds) leak, as they do in stock dingo. A box-preserving flow (bounded RQ splines, tails=None,
  permutations only, uniform base) would fix them; it is deferred because it needs a custom flow builder.
- Symmetry group G (8 elements): label swap × the sign of v1 × the sign of v2. All 8 map the box onto itself.
  Train with a random g ∈ G (augmentation after waveform generation). Evaluate q_fold(x) = Σ_g q(g·x) on
  ln q ≤ 0, v ≥ 0. Don't mirror at a = 1 (no symmetry there). q2 conditions on these folded intrinsics.
- Augmentation relabels parameters only. Waveforms are always generated with the canonical labels (m1 ≥ m2), and
  the sign of u_⊥ is a coin flip independent of d. So the symmetry is exact by construction and the waveform model
  needs no invariance check. The fold is normalized for any q: Σ_g ∫_{g·D} q = 1, since each g has |det| = 1. The
  training prior must be G-symmetric. This needs a custom training transform that subclasses or patches dingo's
  set_train_transforms.
- Spin prior for inference: uniform a and uniform cos θ (user, 2026-11-14). In box coordinates it has a 1/a spike at
  a = 0 and a kink at v = 0. Run B trains on each spin uniform in the disk, p(a) ∝ a with uniform
  tilt (bilby PowerLaw(1) and Uniform, so no code). That is π_box = J × const: smooth and G-symmetric. The user's
  prior is applied analytically afterwards. The emulator is L ∝ q_fold/π_train, and posteriors are
  ∝ q_fold · π_user/π_train.

## Runs A and B: pilot comparison (user, 2026-11-14)

- Both runs train q1 only. They share the embedding (Stephen's suggestion: the DenseResNet has no funnel, 24
  blocks all 1024 wide), `tail_bound: 3`, O3 H1+L1+V1 noise, BBH data (8 s, 20-1024 Hz, IMRPhenomXPHM, f_ref 20,
  detector-frame Mc 10-100, q 0.125-1, d_L 100-6000 Mpc), and the budget: 1M waveforms, 120 + 60 epochs, batch
  4096, one H100 each.
  - A: stock dingo (`chirp_mass, mass_ratio, a_1, a_2, tilt_1, tilt_2, luminosity_distance`, standard priors).
  - B: (ln Mc, ln q, u1, v1, u2, v2, d_L) with relabeling augmentation (`emulator/box_reparam.py`, a patch of
    `set_train_transforms` loaded by `scripts/train.py`) and a fixed standardization. Mirror means are exactly 0.
- Configs: `configs/`. Slurm scripts: `slurm/gen_data.sbatch` (both waveform sets, plus the ASD set from 16 parallel
  `dingo_estimate_psds` workers merged with `dingo_merge_asd_datasets`) and `slurm/train.sbatch A|B` (resumes from
  model_latest.pt). Data, models and logs are under `/mnt/ceph/users/jheinzel/dingo-intrinsic-emulator/`.
- O3 PSDs: 300 per detector, evenly spread (`scripts/thin_psd_segments.py`). dingo's num_psds_max keeps the first
  N segments, and time_gap only thins within a science segment: O3 has ~5700 H1, ~6200 L1, ~1800 V1 segments.
- Login node: set OPENBLAS_NUM_THREADS=1 (and OMP/MKL), or numpy imports die (per-user thread cap).
- To compare later: the emulator log L on held-out injections, both against a reference marginal likelihood
  (not yet chosen) and via coverage and P-P tests on the intrinsics.

## Open questions

- phi_12 and phi_jl are intrinsic but sit in q2, so the emulator marginalizes them. Confirm this.
- z enters through luminosity_distance with a fixed cosmology. Source-frame m1 = m1_det / (1 + z), so the prior and
  the density need the Jacobian.
- Choice of training prior π(θ_int), waveform model, detectors/PSDs, and whether to use DINGO-T1 as a baseline.

## Environment

- `.venv` (uv, on the `python/3.12.9` module): `module load python/3.12.9 uv && uv sync --python "$(which python3.12)"`.
  Run with `uv run ...` or `.venv/bin/python`. torch 2.9.1+cu128.
- Rusty login nodes cap each user at 1 CPU and 8 GB in total. Build waveform datasets, train and run inference on
  compute/GPU nodes through Slurm, and get the user's approval before every submission.
- Large outputs (waveform/ASD datasets `.hdf5`, trained models `.pt`, samples) go to
  `/mnt/ceph/users/jheinzel/dingo-intrinsic-emulator/`, never into git or GPFS home. `.gitignore` excludes them.

## Conventions (shared with bbh-pixelpop-constraints)

- Plots: `plot_style.apply()`, then `plot_style.save(fig, name)` writes `plots/<name>.pdf`. The style is LaTeX serif
  at 16 pt, legends at 14 pt (`plot_style.LEG_KW`), and ticks inward on all four sides (outward on pcolormesh panels).
  No titles. File names say which model/data made the figure.
- Figures use **all** posterior samples for medians and bands, never a subset chosen for speed. If a full pass is
  slow, propose a job instead. Showing a random subset as thin traces is fine.
- Never `rm -f`/`rm -rf` variable or glob paths. Never `pkill -f`/`pgrep -f` with patterns that match themselves.
  Never edit a script while it runs: launch frozen copies (`launch_frozen_*.sh`, gitignored).
- Report times in US Eastern.
- Git: commit and push to `origin main` at milestones (user, 2026-11-14). A milestone is a completed, verified step,
  such as a working environment, a trained model, or a validated result. Don't commit large data.
- Record user decisions under Decisions with (user, date), and runs under Runs with date, time, job id, and outcome.

## Runs

- 2026-11-14 16:03 EST: environment installed; tutorial imports verified on the rusty3 login node (CPU only).
- 2026-11-14 18:32 EST: `asd/asds_O3_fiducial.hdf5` (one O3 PSD each for H1, L1 and V1, GPS ~1238167000) built
  on the login node.
- 2026-11-14 ~18:45 EST: smoke tests on the login node (300 waveforms, tiny networks, one CPU epoch) pass for A and
  B. They cover the transform chain, the round trip of the relabelings (1e-15), and the folded log-prob. Waveform
  cost is ~25-100 ms each on one core.
- 2026-11-14 18:37 EST: submitted (user approved) data job 7475306 (`slurm/gen_data.sbatch`, gen, 1 node, 6 h)
  and training jobs 7475307 (run A) and 7475308 (run B) (`slurm/train.sbatch`, 1 H100 each, 2 d), afterok on 7475306.
  Logs: `/mnt/ceph/users/jheinzel/dingo-intrinsic-emulator/logs/`.
- 2026-11-14 19:47 EST: data job 7475306 COMPLETED in 1 h 09 min (worker5090, 128 cores). `waveforms/waveforms_{A,B}.hdf5`
  (1M each, 6.5 GB, 200 SVD modes) and `asd/asds_O3.hdf5` (300 PSDs each for H1/L1/V1, GPS 1238166018-1269361170,
  0 of 16 download workers failed). Checks: no non-finite or empty waveforms or ASDs; priors as configured (B: mean
  a = 0.660, A: 0.495); band-median ASDs match the fiducial PSD. SVD mismatch at n = 200 (validation set from each
  run's own prior): A median 2e-8, 99% 2e-4, max 1e-2; B median 9e-7, 99% 2e-3, max 4e-2. B is worse because its
  LogUniform Mc prior puts more waveforms at low Mc (median 32 vs 71), which are longer. This is a confound for the
  A/B comparison; if B underperforms at low Mc, raise the SVD size before blaming the coordinates.
  Training jobs 7475307 (A) and 7475308 (B) now pending (Priority).
- 2026-11-14 20:42/20:44 EST: training jobs 7475308 (B) and 7475307 (A) OUT_OF_MEMORY (128 GB) at the start of
  epoch 1, after 12-15 min. Cause: 30 data-loader workers each prefetching two 4096-sample batches (~1.2 GB each).
  Fix: num_workers 20, and `slurm/train.sbatch` now asks for 1 H100 on the 4-GPU Genoa nodes (`-C h100-sxm5`) with
  24 cores and 360 GB (one GPU's share of the node). Not resubmitted yet (user).
- 2026-11-14 ~21:20 EST: run B now compresses with run A's SVD basis (user: same compression as A).
  `waveforms/svd_A.hdf5` is extracted from waveforms_A.hdf5, and `configs/waveforms_B.yaml` loads it through
  `compression.svd.file`. A 20-waveform test gives a V identical to A's. On 2000 uncompressed run-B-prior
  waveforms: A's basis gives median 5e-7, 99% 3e-3 (Mc < 20: median 1.8e-4, 99% 6e-3); B's own basis gives 9e-7,
  2e-3 (Mc < 20: 8.7e-5, 3e-3). So the compression is identical, but B's low-Mc waveforms are compressed about 2×
  worse than before. The quality gap between the runs comes from the prior, not the basis.
  waveforms_B.hdf5 must be regenerated (`slurm/gen_waveforms.sbatch B`; not submitted yet).
- 2026-11-14 21:14 EST: submitted (user approved) 7475709 (`slurm/gen_waveforms.sbatch B`: regenerate waveforms_B.hdf5
  with A's basis), 7475710 (train A) and 7475711 (train B, afterok 7475709). Training uses the new resources
  (h100-sxm5, 24 cores, 360 GB, 20 workers). The embedding's first-layer SVD initialization is still computed per
  run from each run's own waveforms.
