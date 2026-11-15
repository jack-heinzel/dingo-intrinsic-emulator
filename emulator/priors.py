"""Training priors that bilby/dingo load by module path, e.g. in a waveform config:

    a_1: emulator.priors.SpinMixture(weight=0.5, sigma=0.1, maximum=0.99)

The repository must be importable (PYTHONPATH or sys.path), as in scripts/train.py and the
Slurm scripts.
"""
import numpy as np
from bilby.core.prior import Prior


class CoredUniform(Prior):
    """Spin magnitude a in [0, maximum] with p(a) ∝ a / sqrt(a^2 + a0^2): uniform in a (the
    inference prior) for a >> a0, and ∝ a (uniform in the disk) for a << a0.

    With the tilt uniform in [0, pi], the density of (chi_z, chi_perp) in the disk is
    ∝ 1 / sqrt(a^2 + a0^2): a function of a^2, so smooth, with a mild peak of width ~a0 at the
    origin instead of the 1/a spike of a uniform-a prior. CDF and inverse are closed form.
    a0 = 0.05: 2.2% of draws below a = 0.05, 6.6% below 0.1 (uniform a: 5.1%, 10.1%).
    """

    def __init__(self, a0=0.05, minimum=0.0, maximum=0.99, name=None, latex_label=None,
                 unit=None, boundary=None):
        super().__init__(name=name, latex_label=latex_label, unit=unit, minimum=minimum,
                         maximum=maximum, boundary=boundary)
        if minimum != 0.0:
            raise ValueError("CoredUniform is defined on [0, maximum].")
        self.a0 = a0
        self._norm = np.sqrt(maximum ** 2 + a0 ** 2) - a0

    def prob(self, val):
        a = np.asarray(val, dtype=float)
        p = a / np.sqrt(a ** 2 + self.a0 ** 2) / self._norm
        return np.where((a >= 0) & (a <= self.maximum), p, 0.0)

    def ln_prob(self, val):
        with np.errstate(divide="ignore"):
            return np.log(self.prob(val))

    def cdf(self, val):
        a = np.clip(np.asarray(val, dtype=float), 0.0, self.maximum)
        return (np.sqrt(a ** 2 + self.a0 ** 2) - self.a0) / self._norm

    def rescale(self, val):
        r = np.asarray(val, dtype=float) * self._norm + self.a0
        return np.sqrt(np.clip(r ** 2 - self.a0 ** 2, 0.0, None))

    def disk_ln_density(self, a):
        """log of p(a) / (pi a), the density of (chi_z, chi_perp) with uniform tilt; smooth
        through a = 0. Up to the constant -log(pi)."""
        a = np.asarray(a, dtype=float)
        return -0.5 * np.log(a ** 2 + self.a0 ** 2) - np.log(self._norm)


class SpinMixture(Prior):
    """Spin magnitude a in [0, maximum] such that, with the tilt uniform in [0, pi], the
    spin vector (chi_z, chi_perp) is a mixture in the (half-)disk of
        weight     : uniform over the disk                 (p(a) = 2a / maximum^2)
        1 - weight : a round Gaussian of width sigma at 0  (p(a) Rayleigh(sigma), truncated)
    The density in disk coordinates, p(a) / (pi a), is smooth and finite at a = 0, so the
    training target has no spike at zero spin, while a large fraction of samples sits at
    small a (weight 0.5, sigma 0.1: ~20% below a = 0.1, vs ~1% for the uniform disk).
    """

    def __init__(self, weight=0.5, sigma=0.1, minimum=0.0, maximum=0.99, name=None,
                 latex_label=None, unit=None, boundary=None):
        super().__init__(name=name, latex_label=latex_label, unit=unit, minimum=minimum,
                         maximum=maximum, boundary=boundary)
        if minimum != 0.0:
            raise ValueError("SpinMixture is defined on [0, maximum].")
        self.weight = weight
        self.sigma = sigma
        self._norm = -np.expm1(-maximum ** 2 / (2 * sigma ** 2))  # Rayleigh mass in [0, max]
        self._grid = np.linspace(0.0, maximum, 20001)
        self._cdf_grid = self.cdf(self._grid)

    def prob(self, val):
        a = np.asarray(val, dtype=float)
        p = (self.weight * 2 * a / self.maximum ** 2
             + (1 - self.weight) * a / self.sigma ** 2 * np.exp(-a ** 2 / (2 * self.sigma ** 2)) / self._norm)
        return np.where((a >= 0) & (a <= self.maximum), p, 0.0)

    def ln_prob(self, val):
        with np.errstate(divide="ignore"):
            return np.log(self.prob(val))

    def cdf(self, val):
        a = np.clip(np.asarray(val, dtype=float), 0.0, self.maximum)
        return (self.weight * (a / self.maximum) ** 2
                + (1 - self.weight) * -np.expm1(-a ** 2 / (2 * self.sigma ** 2)) / self._norm)

    def rescale(self, val):
        """Inverse CDF, by interpolation on a fine grid (error < 1e-8 in a)."""
        return np.interp(np.asarray(val, dtype=float), self._cdf_grid, self._grid)

    def disk_ln_density(self, a):
        """log of the density of (chi_z, chi_perp) in the half-disk with uniform tilt,
        p(a) / (pi a), smooth through a = 0. Up to the constant -log(pi)."""
        a = np.asarray(a, dtype=float)
        dens = (self.weight * 2 / self.maximum ** 2
                + (1 - self.weight) / self.sigma ** 2 * np.exp(-a ** 2 / (2 * self.sigma ** 2)) / self._norm)
        return np.log(dens)
