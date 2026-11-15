"""Reparameterized q1 coordinates with symmetry augmentation (runs B and C). See symmetry.md.

Run C uses the "disk" chart instead of the square: Cartesian spin components
(chi_z_i, chi_perp_i) in place of (u_i, v_i), with the same relabeling group and fold. With
dingo's default (Gaussian-base, unbounded) flow, the square brings nothing, so the disk
chart drops the map and its Jacobian (user, 2026-11-15). The notes below describe run B.

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


def spin_to_disk(a, tilt):
    """Cartesian spin components (chi_z, chi_perp), chi_perp >= 0 for tilt in [0, pi]."""
    return a * np.cos(tilt), a * np.sin(tilt)


def disk_to_spin(chi_z, chi_perp):
    """Returns (a, tilt) with tilt in [0, pi] (the sign of chi_perp is dropped, i.e. folded)."""
    return np.hypot(chi_z, chi_perp), np.arctan2(np.abs(chi_perp), chi_z)


# Two spin charts share the relabeling group. "box" (run B): the elliptical-grid square.
# "disk" (run C): Cartesian spin components; no map, so no Jacobian. In each, the second
# key of a spin is the one whose sign is a coin flip (the mirror).
DISK_KEYS = ("ln_chirp_mass", "ln_mass_ratio", "chi_z_1", "chi_perp_1", "chi_z_2", "chi_perp_2")
CHARTS = {
    "box": dict(keys=BOX_KEYS, spin=(("u_1", "v_1"), ("u_2", "v_2")), to_chart=spin_to_square),
    "disk": dict(keys=DISK_KEYS, spin=(("chi_z_1", "chi_perp_1"), ("chi_z_2", "chi_perp_2")),
                 to_chart=spin_to_disk),
}


def chart_of(names):
    """'box', 'disk' or None for a list of parameter names."""
    for chart, c in CHARTS.items():
        if set(c["keys"]) <= set(names):
            return chart
    return None


# --- training-time transform ---------------------------------------------------------

class ReparamAugment:
    """Adds the chart's keys to sample['parameters'], with a random relabeling g in G.

    Uses numpy's global RNG, which dingo reseeds per DataLoader worker
    (dingo.core.utils.torchutils.fix_random_seeds), like its other random transforms.
    """

    def __init__(self, chart="box"):
        self.chart = chart

    def __call__(self, input_sample):
        c = CHARTS[self.chart]
        (z1k, p1k), (z2k, p2k) = c["spin"]
        sample = input_sample.copy()
        p = dict(sample["parameters"])
        chirp_mass = np.asarray(p["chirp_mass"], dtype=float)
        shape = chirp_mass.shape
        ln_q = np.log(np.asarray(p["mass_ratio"], dtype=float))
        z1, p1 = c["to_chart"](np.asarray(p["a_1"], float), np.asarray(p["tilt_1"], float))
        z2, p2 = c["to_chart"](np.asarray(p["a_2"], float), np.asarray(p["tilt_2"], float))

        swap = np.random.random(shape) < 0.5
        s1 = np.where(np.random.random(shape) < 0.5, -1.0, 1.0)
        s2 = np.where(np.random.random(shape) < 0.5, -1.0, 1.0)
        # swap labels: ln q -> -ln q, spin blocks exchanged
        ln_q = np.where(swap, -ln_q, ln_q)
        z1, z2 = np.where(swap, z2, z1), np.where(swap, z1, z2)
        p1, p2 = np.where(swap, p2, p1), np.where(swap, p1, p2)
        # independent sign flips of the in-plane spin components
        p1, p2 = s1 * p1, s2 * p2

        new = {"ln_chirp_mass": np.log(chirp_mass), "ln_mass_ratio": ln_q,
               z1k: z1, p1k: p1, z2k: z2, p2k: p2}
        if shape == ():
            new = {k: float(val) for k, val in new.items()}
        p.update(new)
        sample["parameters"] = p
        return sample


class BoxReparamAugment(ReparamAugment):
    """Run B's augmentation (kept for existing code and checkpoints)."""

    def __init__(self):
        super().__init__("box")


_original_set_train_transforms = _tb.set_train_transforms


def set_train_transforms(wfd, data_settings, asd_dataset_path, *args, **kwargs):
    """dingo's set_train_transforms, plus ReparamAugment before standardization."""
    _original_set_train_transforms(wfd, data_settings, asd_dataset_path, *args, **kwargs)
    chart = chart_of(data_settings["inference_parameters"])
    if chart is None:
        return
    ts = wfd.transform.transforms
    idx = [i for i, t in enumerate(ts) if isinstance(t, SelectStandardizeRepackageParameters)]
    if idx:
        ts.insert(idx[0], ReparamAugment(chart))


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
    (z1k, p1k), (z2k, p2k) = CHARTS[chart_of(list(index))]["spin"]
    out = theta.copy()
    if swap:
        out[..., index["ln_mass_ratio"]] = -theta[..., index["ln_mass_ratio"]]
        for a, b in ((z1k, z2k), (p1k, p2k)):
            out[..., index[a]], out[..., index[b]] = theta[..., index[b]], theta[..., index[a]]
    out[..., index[p1k]] *= s1
    out[..., index[p2k]] *= s2
    return out


def fold_log_prob(log_prob_fn, theta, index):
    """log sum_g q(g theta), for theta on the canonical region. log_prob_fn maps an
    array [N, D] of unstandardized parameters to log q [N] (in those parameters)."""
    lps = np.stack([log_prob_fn(apply_group(theta, g, index)) for g in group_elements()])
    m = lps.max(axis=0)
    return m + np.log(np.exp(lps - m).sum(axis=0))
