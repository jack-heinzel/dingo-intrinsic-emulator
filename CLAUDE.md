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

## Open questions

- DINGO's default `inference_parameters` are chirp_mass, mass_ratio, phase, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl,
  theta_jn, luminosity_distance, geocent_time, ra, dec, psi. The target set leaves out the precession azimuths
  phi_12 and phi_jl, so they would be marginalized too, even though they are intrinsic. Confirm this.
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
