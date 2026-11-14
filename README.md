# dingo-intrinsic-emulator

Goal: use [DINGO](https://github.com/dingo-gw/dingo) (neural posterior estimation for GW)
to build an emulator for the **marginal likelihood of the intrinsic BBH parameters**

    L(θ_int | d) = ∫ L(d | θ_int, θ_ext) π(θ_ext) dθ_ext,
    θ_int = (m1, q, a1, a2, cos θ1, cos θ2, z)

marginalized over the extrinsic parameters (sky location, inclination, polarization, phase, time),
for use in population inference (e.g. `bbh-pixelpop-constraints`).

## Setup

```bash
git clone --recurse-submodules https://github.com/jack-heinzel/dingo-intrinsic-emulator.git
cd dingo-intrinsic-emulator
module load python/3.12.9 uv
uv sync --python "$(which python3.12)"
uv run jupyter lab dingo-tutorial/DINGO_Tutorial.ipynb
```

`pyproject.toml` pins `dingo-gw==0.10.3` with the dependency constraints from the
tutorial (in particular `bilby_pipe<=1.8`, which `dingo_pipe` needs).

`dingo-tutorial/` is a git submodule of
[stephengreen/dingo-tutorial](https://github.com/stephengreen/dingo-tutorial).

## Approach (sketch)

NPE gives q(θ | d) ≈ p(θ | d). If the network is trained with only the intrinsic parameters
as `inference_parameters`, the extrinsic ones are still drawn from their prior when simulating
training data, so the flow learns the **marginal** posterior q(θ_int | d) directly. Then

    log L(θ_int | d) = log q(θ_int | d) − log π(θ_int) + log Z(d),

so the likelihood emulator is the flow density divided by the (known, analytic) training prior,
up to an event-dependent constant that drops out of population likelihoods.

Points to settle:
- **Redshift / distance.** DINGO samples `luminosity_distance` and detector-frame masses. z must
  be in the inference parameters, and source-frame m1 = m1_det / (1 + z) needs the Jacobian
  when converting the prior and the density.
- **Prior choice.** π(θ_int) should be broad and analytically simple (it divides out), and cover
  the population support.
- **Accuracy.** Unweighted NPE densities are approximate. DINGO-IS (importance sampling) needs the
  full parameter set, so one option is: full-parameter model + IS → weighted intrinsic samples →
  check/correct the intrinsic-only flow density against them.
- **Baseline.** The pre-trained DINGO-T1 model (tutorial §6) gives full posteriors to compare against.
