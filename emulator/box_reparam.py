"""Reparameterized q1 coordinates with symmetry augmentation (run B). See symmetry.md.

Coordinates learned by the flow (all standard dingo machinery otherwise):
    ln_chirp_mass, ln_mass_ratio, u_1, v_1, u_2, v_2, luminosity_distance
where (u_i, v_i) in [-1, 1]^2 is the elliptical-grid ("squircle") preimage of the
spin vector (chi_z, chi_perp) / A_MAX, with chi_z = a cos(tilt), chi_perp = a sin(tilt).

Training: waveforms are generated with canonical labels (m1 >= m2, chi_perp >= 0).
`BoxReparamAugment` maps to the coordinates above and applies a uniformly random
element of the 8-element relabeling group G (label swap x sign(v_1) x sign(v_2)).
Only parameter labels change, never the data, so the symmetry is exact.

Evaluation: the density on the canonical region (ln q <= 0, v_i >= 0) is the sum of the
learned density over the 8 relabelings (`fold_log_prob`).

Usage: `import emulator.box_reparam` before dingo builds training transforms (see
scripts/train.py). The patch is active only when `inference_parameters` contains the
keys in `BOX_KEYS`. Otherwise dingo is unchanged.
"""

import itertools

import numpy as np

import dingo.gw.training.train_builders as _tb
import dingo.gw.training.train_pipeline as _tp
from dingo.gw.transforms import SelectStandardizeRepackageParameters

A_MAX = 0.99  # spin-magnitude prior bound; spins are scaled by it to fill the unit disk
BOX_KEYS = ("ln_chirp_mass", "ln_mass_ratio", "u_1", "v_1", "u_2", "v_2")
SQRT2 = np.sqrt(2.0)


# --- elliptical-grid map between the square (u, v) and the unit disk (x, y) ---------

def square_to_disk(u, v):
    return u * np.sqrt(1 - v**2 / 2), v * np.sqrt(1 - u**2 / 2)


def disk_to_square(x, y):
    """Inverse of square_to_disk, in the form that is accurate near the origin."""
    a, b = 2 + x**2 - y**2, 2 - x**2 + y**2
    u = 2 * SQRT2 * x / (np.sqrt(np.clip(a + 2 * SQRT2 * x, 0, None))
                         + np.sqrt(np.clip(a - 2 * SQRT2 * x, 0, None)))
    v = 2 * SQRT2 * y / (np.sqrt(np.clip(b + 2 * SQRT2 * y, 0, None))
                         + np.sqrt(np.clip(b - 2 * SQRT2 * y, 0, None)))
    return u, v


def log_jacobian_square_to_disk(u, v):
    """log |d(x, y) / d(u, v)|; -inf only at the four corners."""
    return (np.log(1 - u**2 / 2 - v**2 / 2)
            - 0.5 * np.log(1 - u**2 / 2) - 0.5 * np.log(1 - v**2 / 2))


def spin_to_square(a, tilt):
    x, y = a * np.cos(tilt) / A_MAX, a * np.sin(tilt) / A_MAX
    return disk_to_square(x, y)


def square_to_spin(u, v):
    """Returns (a, tilt) with tilt in [0, pi] (the sign of v is dropped, i.e. folded)."""
    x, y = square_to_disk(u, v)
    return A_MAX * np.hypot(x, y), np.arctan2(np.abs(y), x)


# --- training-time transform ---------------------------------------------------------

class BoxReparamAugment:
    """Adds BOX_KEYS to sample['parameters'], with a random relabeling g in G.

    Uses numpy's global RNG, which dingo reseeds per DataLoader worker
    (dingo.core.utils.torchutils.fix_random_seeds), like its other random transforms.
    """

    def __call__(self, input_sample):
        sample = input_sample.copy()
        p = dict(sample["parameters"])
        chirp_mass = np.asarray(p["chirp_mass"], dtype=float)
        shape = chirp_mass.shape
        ln_q = np.log(np.asarray(p["mass_ratio"], dtype=float))
        u1, v1 = spin_to_square(np.asarray(p["a_1"], float), np.asarray(p["tilt_1"], float))
        u2, v2 = spin_to_square(np.asarray(p["a_2"], float), np.asarray(p["tilt_2"], float))

        swap = np.random.random(shape) < 0.5
        s1 = np.where(np.random.random(shape) < 0.5, -1.0, 1.0)
        s2 = np.where(np.random.random(shape) < 0.5, -1.0, 1.0)
        # swap labels: ln q -> -ln q, spin blocks exchanged
        ln_q = np.where(swap, -ln_q, ln_q)
        u1, u2 = np.where(swap, u2, u1), np.where(swap, u1, u2)
        v1, v2 = np.where(swap, v2, v1), np.where(swap, v1, v2)
        # independent sign flips of the in-plane spin components
        v1, v2 = s1 * v1, s2 * v2

        new = dict(ln_chirp_mass=np.log(chirp_mass), ln_mass_ratio=ln_q,
                   u_1=u1, v_1=v1, u_2=u2, v_2=v2)
        if shape == ():
            new = {k: float(val) for k, val in new.items()}
        p.update(new)
        sample["parameters"] = p
        return sample


_original_set_train_transforms = _tb.set_train_transforms


def set_train_transforms(wfd, data_settings, asd_dataset_path, *args, **kwargs):
    """dingo's set_train_transforms, plus BoxReparamAugment before standardization."""
    _original_set_train_transforms(wfd, data_settings, asd_dataset_path, *args, **kwargs)
    if not set(BOX_KEYS) <= set(data_settings["inference_parameters"]):
        return
    ts = wfd.transform.transforms
    idx = [i for i, t in enumerate(ts) if isinstance(t, SelectStandardizeRepackageParameters)]
    if idx:
        ts.insert(idx[0], BoxReparamAugment())


_tb.set_train_transforms = set_train_transforms
_tp.set_train_transforms = set_train_transforms


# --- evaluation: folded density ------------------------------------------------------

def group_elements():
    """The 8 relabelings as (swap, sign_v1, sign_v2)."""
    return list(itertools.product((False, True), (1.0, -1.0), (1.0, -1.0)))


def apply_group(theta, g, index):
    """Apply relabeling g to an array theta[..., D] of *unstandardized* parameters,
    with column positions given by index (dict name -> column)."""
    swap, s1, s2 = g
    out = theta.copy()
    if swap:
        out[..., index["ln_mass_ratio"]] = -theta[..., index["ln_mass_ratio"]]
        for a, b in (("u_1", "u_2"), ("v_1", "v_2")):
            out[..., index[a]], out[..., index[b]] = theta[..., index[b]], theta[..., index[a]]
    out[..., index["v_1"]] *= s1
    out[..., index["v_2"]] *= s2
    return out


def fold_log_prob(log_prob_fn, theta, index):
    """log sum_g q(g theta), for theta on the canonical region. log_prob_fn maps an
    array [N, D] of unstandardized parameters to log q [N] (in those parameters)."""
    lps = np.stack([log_prob_fn(apply_group(theta, g, index)) for g in group_elements()])
    m = lps.max(axis=0)
    return m + np.log(np.exp(lps - m).sum(axis=0))
