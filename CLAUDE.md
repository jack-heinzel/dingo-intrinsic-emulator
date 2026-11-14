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
- Spins (proposed): u_i = artanh(a_i)·(cos θ_i, sin θ_i), with u_⊥ signed. The azimuth-marginalized likelihood is
  smooth and even in χ_⊥, so flatness in cos θ as a -> 0 is just continuity at the origin. artanh is odd, so the
  map is smooth at 0 and removes the |χ| = 1 edge; |∂χ/∂u| = sech²(r) tanh(r)/r. The training prior is uniform
  in the (χ_z, χ_⊥) disk, which is smooth in u. The standard prior (uniform in a and cos θ) has a 1/a spike at the
  origin.
- Symmetry group G (8 elements): label swap × the sign of u_1⊥ × the sign of u_2⊥. Train with a random g ∈ G
  (augmentation after waveform generation). Evaluate q_fold(x) = Σ_g q(g·x) on ln q ≤ 0, u_⊥ ≥ 0. q2 conditions
  on these folded intrinsics.
- Augmentation relabels parameters only. Waveforms are always generated with the canonical labels (m1 ≥ m2), and
  the sign of u_⊥ is a coin flip independent of d. So the symmetry is exact by construction and the waveform model
  needs no invariance check. The fold is normalized for any q: Σ_g ∫_{g·D} q = 1, since each g has |det| = 1. The
  training prior must be G-symmetric. This needs a custom training transform that subclasses or patches dingo's
  set_train_transforms.
- Spin prior for inference: uniform a and uniform cos θ (user, 2026-11-14). In u it is ∝ |sin θ| sech²(r)/r: a 1/r
  spike at a = 0 and a kink at sin θ = 0. Proposal, not yet agreed: train on a smooth G-symmetric π_train (Gaussian
  in u) and apply the user's prior analytically. The emulator is L ∝ q_fold/π_train, and posteriors are
  ∝ q_fold · π_user/π_train.

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
