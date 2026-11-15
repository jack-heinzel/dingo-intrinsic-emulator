"""Shared helpers for evaluating runs A and B (scripts/eval_calibration.py,
scripts/eval_reference.py).

Physical parameters (PHYS) are the detector-frame intrinsics both runs describe:
chirp_mass, mass_ratio (<= 1), a_1, a_2, tilt_1, tilt_2 (in [0, pi]), luminosity_distance.
Run A infers exactly these. Run B infers box coordinates (emulator/box_reparam.py); its
samples are folded to the canonical region (ln q <= 0, v >= 0) before conversion, which
is sampling from q_fold.

The emulated log-likelihood, up to a constant per event, is
    log L(theta) = log q(x(theta) | d) - log pi_train(x(theta)),
in each run's own coordinates x (the Jacobian of x(theta) cancels). For run B, q is
q_fold and pi_train is the augmented training prior; that prior is uniform in
(ln Mc, ln q, d_L) and proportional to J(u, v) for each spin, so
log pi_train = log J(u1, v1) + log J(u2, v2) + const.
"""
import pickle

import numpy as np
import torch

from emulator import box_reparam as br

DATA = "/mnt/ceph/users/jheinzel/dingo-intrinsic-emulator"
ASD_PATH = f"{DATA}/asd/asds_O3.hdf5"
MODEL_PATH = {r: f"{DATA}/training/run_{r}/model_stage_1.pt" for r in "AB"}
PHYS = ["chirp_mass", "mass_ratio", "a_1", "a_2", "tilt_1", "tilt_2", "luminosity_distance"]
EXTRINSIC = ["ra", "dec", "psi", "theta_jn", "phase", "phi_12", "phi_jl"]  # integrated out
# Prior support shared by both runs (configs/waveforms_*.yaml, configs/train_*.yaml).
BOUNDS = dict(chirp_mass=(10.0, 100.0), mass_ratio=(0.125, 1.0), a_1=(0.0, br.A_MAX),
              a_2=(0.0, br.A_MAX), tilt_1=(0.0, np.pi), tilt_2=(0.0, np.pi),
              luminosity_distance=(100.0, 6000.0))


def load_metadata(path):
    """Model metadata without reading the whole checkpoint into memory."""
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)["metadata"]


def load_model(path, device="cuda"):
    from dingo.core.posterior_models.build_model import build_model_from_kwargs
    model = build_model_from_kwargs(filename=path, device=device, load_training_info=False)
    model.network.eval()
    return model


def run_of(model_or_metadata):
    md = getattr(model_or_metadata, "metadata", model_or_metadata)
    params = md["train_settings"]["data"]["inference_parameters"]
    return "B" if "u_1" in params else "A"


def in_bounds(phys):
    ok = np.ones(len(phys["chirp_mass"]), bool)
    for k, (lo, hi) in BOUNDS.items():
        ok &= np.isfinite(phys[k]) & (phys[k] >= lo) & (phys[k] <= hi)
    return ok


# --- coordinate conversions -------------------------------------------------------------

def box_to_phys(box):
    """Run-B samples (dict of arrays) -> folded physical parameters. Points outside the
    box (|u| or |v| > 1, i.e. flow leakage) come back NaN in the spins."""
    lnq = np.asarray(box["ln_mass_ratio"], float)
    u1, v1, u2, v2 = (np.asarray(box[k], float) for k in ("u_1", "v_1", "u_2", "v_2"))
    swap = lnq > 0
    lnq = np.where(swap, -lnq, lnq)
    u1, u2 = np.where(swap, u2, u1), np.where(swap, u1, u2)
    v1, v2 = np.where(swap, v2, v1), np.where(swap, v1, v2)
    out = dict(chirp_mass=np.exp(np.asarray(box["ln_chirp_mass"], float)),
               mass_ratio=np.exp(lnq),
               luminosity_distance=np.asarray(box["luminosity_distance"], float))
    for i, (u, v) in ((1, (u1, v1)), (2, (u2, v2))):
        bad = (np.abs(u) > 1) | (np.abs(v) > 1)
        a, tilt = br.square_to_spin(np.where(bad, 0.0, u), np.where(bad, 0.0, v))
        out[f"a_{i}"] = np.where(bad, np.nan, a)
        out[f"tilt_{i}"] = np.where(bad, np.nan, tilt)
    return out


def phys_to_box(phys):
    """Physical parameters (mass_ratio <= 1, tilt in [0, pi]) -> canonical box point."""
    out = dict(ln_chirp_mass=np.log(phys["chirp_mass"]), ln_mass_ratio=np.log(phys["mass_ratio"]),
               luminosity_distance=np.asarray(phys["luminosity_distance"], float))
    for i in (1, 2):
        u, v = br.spin_to_square(np.asarray(phys[f"a_{i}"], float), np.asarray(phys[f"tilt_{i}"], float))
        out[f"u_{i}"], out[f"v_{i}"] = u, np.abs(v)
    return out


def samples_to_phys(samples, run):
    cols = {k: np.asarray(samples[k]) for k in samples}
    return box_to_phys(cols) if run == "B" else {k: cols[k].astype(float) for k in PHYS}


# --- densities ----------------------------------------------------------------------------

def make_sampler(model, event):
    from dingo.gw.inference.sampler import GWComposedSampler
    return GWComposedSampler.from_model(model, {"waveform": event["waveform"], "asds": event["asds"]})


def _log_q(sampler, cols, chunk=20000):
    """Network log density at given (unstandardized) inference-parameter columns."""
    n = len(next(iter(cols.values())))
    dev = sampler.context.device if hasattr(sampler.context, "device") else "cpu"
    out = []
    with torch.no_grad():
        for s in range(0, n, chunk):
            batch = {k: torch.as_tensor(np.asarray(v[s:s + chunk]), dtype=torch.float32, device=dev)
                     for k, v in cols.items()}
            out.append(sampler.composer.log_prob(batch, sampler.context).double().cpu().numpy())
    return np.concatenate(out)


def log_q_phys(sampler, run, phys):
    """log q at physical points, in the run's own coordinates; q_fold for run B."""
    if run == "A":
        return _log_q(sampler, {k: phys[k] for k in PHYS})
    box = phys_to_box(phys)
    keys = list(br.BOX_KEYS) + ["luminosity_distance"]
    theta = np.stack([box[k] for k in keys], axis=-1)
    index = {k: i for i, k in enumerate(keys)}
    return br.fold_log_prob(lambda th: _log_q(sampler, {k: th[:, i] for k, i in index.items()}),
                            theta, index)


def log_prior_train(sampler, run, phys):
    """log pi_train in the run's own coordinates, up to a constant."""
    if run == "A":
        prior = sampler.context.prior
        return sum(np.asarray(prior[k].ln_prob(np.asarray(phys[k])), float) for k in PHYS)
    box = phys_to_box(phys)
    return (br.log_jacobian_square_to_disk(box["u_1"], box["v_1"])
            + br.log_jacobian_square_to_disk(box["u_2"], box["v_2"]))


def log_L_emulator(sampler, run, phys):
    return log_q_phys(sampler, run, phys) - log_prior_train(sampler, run, phys)


def network_snr(injection, theta, asds):
    """Optimal network SNR of the injected signal, in the injection's ASDs."""
    signal = injection.signal(theta)["waveform"]
    df = injection.data_domain.delta_f
    return float(np.sqrt(sum(np.sum(4 * df * np.abs(signal[i]) ** 2 / asds[i] ** 2) for i in asds)))


def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)
