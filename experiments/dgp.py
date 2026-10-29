"""Stationary compact factor covariates and heavy-tailed AR regression errors.

The DGPs are designed for Sections 3--4 of the Numerical Studies manuscript.
Their one-time marginals do not change when the two AR coefficients change.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import levy_stable
from numpy.polynomial.legendre import leggauss


levy_stable.parameterization = "S1"
DEFAULT_P = 1.25
DEFAULT_ALPHA = 1.5
DEFAULT_SCALE = 0.8
SPARSITIES = (4, 8, 16, 32)


@dataclass(frozen=True)
class DGP:
    kind: str
    n: int
    d: int
    rho_x: float = 0.5
    rho_e: float = 0.5
    sparsity: int = 4
    alpha: float = DEFAULT_ALPHA
    beta: float = 0.0
    base_scale: float = DEFAULT_SCALE
    amplitude: float = 1.0

    def validate(self):
        if self.kind not in ("sparse_relu", "factor_sparse"):
            raise ValueError(f"Unknown signal kind: {self.kind}")
        if self.n < 2 or self.d < 64:
            raise ValueError("Require n >= 2 and d >= 64.")
        if not 0 <= self.rho_x < 1 or not 0 <= self.rho_e < 1:
            raise ValueError("AR coefficients must be in [0, 1).")
        if self.kind == "sparse_relu":
            if self.sparsity not in SPARSITIES or self.d <= self.sparsity:
                raise ValueError("sparsity must be 4, 8, 16, or 32 and below d.")
        if not DEFAULT_P < self.alpha < 2 or not -1 <= self.beta <= 1:
            raise ValueError("Require 1.25 < alpha < 2 and beta in [-1,1].")
        if self.base_scale <= 0 or self.amplitude <= 0:
            raise ValueError("Scale and amplitude must be positive.")


def factor_loadings(d):
    if d < 33:
        raise ValueError("The factor construction requires d >= 33.")
    loading = np.full(d, 0.2, dtype=np.float64)
    loading[0] = 1.0
    loading[1:33] = 0.6
    return loading


def _factor_to_x(factor, d, rng):
    loading = factor_loadings(d).astype(np.float32)
    latent = rng.standard_normal((len(factor), d), dtype=np.float32)
    latent *= np.sqrt(1.0 - loading * loading)
    latent += factor.astype(np.float32)[:, None] * loading
    return ndtr(latent).astype(np.float32, copy=False)


def draw_x_series(n, d, rho_x, rng):
    """Generate a strictly stationary factor-AR covariate trajectory."""
    factor = np.empty(n, dtype=np.float64)
    factor[0] = rng.standard_normal()
    shocks = rng.standard_normal(n - 1)
    shock_scale = math.sqrt(1.0 - rho_x * rho_x)
    for t in range(1, n):
        factor[t] = rho_x * factor[t - 1] + shock_scale * shocks[t - 1]
    return _factor_to_x(factor, d, rng)


def draw_x_marginal(n, d, rng):
    """Independent draws from the one-time joint factor law of X_t."""
    return _factor_to_x(rng.standard_normal(n), d, rng)


@lru_cache(maxsize=None)
def stable_scale(alpha=DEFAULT_ALPHA, beta=0.0, base_scale=DEFAULT_SCALE):
    """Match the stationary IQR of symmetric alpha=1.5, scale=base_scale."""
    reference_iqr = base_scale * (
        levy_stable.ppf(0.75, DEFAULT_ALPHA, 0.0)
        - levy_stable.ppf(0.25, DEFAULT_ALPHA, 0.0)
    )
    unit_iqr = (
        levy_stable.ppf(0.75, alpha, beta)
        - levy_stable.ppf(0.25, alpha, beta)
    )
    return float(reference_iqr / unit_iqr)


@lru_cache(maxsize=None)
def absolute_moment(alpha=DEFAULT_ALPHA, beta=0.0,
                    base_scale=DEFAULT_SCALE, p=DEFAULT_P):
    """Stationary p-th absolute moment under the SciPy S1 convention."""
    if not 0 < p < alpha:
        raise ValueError("Absolute p moment requires p < alpha.")
    scale = stable_scale(alpha, beta, base_scale)
    if beta == 0:
        unit = (
            2**p * math.gamma((p + 1.0) / 2.0)
            * math.gamma(1.0 - p / alpha)
            / (math.sqrt(math.pi) * math.gamma(1.0 - p / 2.0))
        )
        return float(scale**p * unit)
    return float(levy_stable.expect(
        lambda x: abs(x)**p, args=(alpha, beta), scale=scale
    ))


def curvature_min_tau(vp, p=DEFAULT_P, output_bound=2.0):
    return max(4.0 * output_bound, (2.0**(p + 1.0) * vp)**(1.0 / p))


def draw_error_series(n, rho_e, alpha, beta, base_scale, rng):
    """Stationary stable AR(1), including exact stationary initialization."""
    scale = stable_scale(alpha, beta, base_scale)
    innovation_scale = scale * (1.0 - rho_e**alpha)**(1.0 / alpha)
    eps = np.empty(n, dtype=np.float64)
    eps[0] = levy_stable.rvs(
        alpha, beta, scale=scale, size=1, random_state=rng
    )[0]
    innovations = levy_stable.rvs(
        alpha, beta, scale=innovation_scale, size=n - 1,
        random_state=rng,
    )
    for t in range(1, n):
        eps[t] = rho_e * eps[t - 1] + innovations[t - 1]
    return eps


def regression_function(x, kind, sparsity=4, amplitude=1.0):
    if kind == "sparse_relu":
        if sparsity not in SPARSITIES or x.shape[1] <= sparsity:
            raise ValueError("Invalid sparse signal dimension.")
        hinges = np.maximum(
            x[:, 1:sparsity + 1:2]
            + x[:, 2:sparsity + 1:2] - 1.0, 0.0,
        )
        return (amplitude * (3.0 / sparsity) * hinges.sum(axis=1)).astype(
            np.float32
        )
    if kind == "factor_sparse":
        if x.shape[1] < 5:
            raise ValueError("The factor-sparse signal requires d >= 5.")
        return (
            0.7 * np.sin(2.0 * np.pi * x[:, 0])
            + 0.8 * np.maximum(x[:, 1] + x[:, 2] - 1.0, 0.0)
            - 0.5 * np.maximum(x[:, 3] + x[:, 4] - 1.0, 0.0)
        ).astype(np.float32)
    raise ValueError(f"Unknown signal kind: {kind}")


def active_coordinates(kind, sparsity=4):
    if kind == "sparse_relu":
        return np.arange(1, sparsity + 1)
    if kind == "factor_sparse":
        return np.arange(5)
    raise ValueError(f"Unknown signal kind: {kind}")


def exact_sparse_parameter_count(sparsity):
    if sparsity not in SPARSITIES:
        raise ValueError("Unsupported sparsity.")
    return 5 * (sparsity // 2)


def make_exact_sparse_network(model, sparsity, amplitude):
    """Fill a two-hidden-layer ClippedNetwork with an exact f_{0,s}."""
    if len(model.layers) != 3 or model.layers[0].weight.shape[0] < sparsity // 2:
        raise ValueError("Exact comparator needs depth=2 and width >= s/2.")
    if model.layers[0].weight.shape[1] <= sparsity:
        raise ValueError("Input dimension is too small.")
    import torch

    coefficient = amplitude * 3.0 / sparsity
    with torch.no_grad():
        for layer in model.layers:
            layer.weight.zero_()
            layer.bias.zero_()
        for k in range(sparsity // 2):
            model.layers[0].weight[k, 2 * k + 1] = 1.0
            model.layers[0].weight[k, 2 * k + 2] = 1.0
            model.layers[0].bias[k] = -1.0
            model.layers[1].weight[k, k] = 1.0
            model.layers[2].weight[0, k] = coefficient
        model.project_parameters()
    return model


def generate_training_series(spec, rng):
    spec.validate()
    x = draw_x_series(spec.n, spec.d, spec.rho_x, rng)
    eps = draw_error_series(
        spec.n, spec.rho_e, spec.alpha, spec.beta,
        spec.base_scale, rng,
    )
    f0 = regression_function(
        x, spec.kind, spec.sparsity, spec.amplitude
    )
    return x, (f0.astype(np.float64) + eps).astype(np.float32)


def calibrate_sparse_amplitudes(seed=20261028, n_reference=2**18):
    """Freeze relative signal strengths using noiseless reference draws."""
    rng = np.random.default_rng(seed)
    x = draw_x_marginal(n_reference, 33, rng)
    variances = {
        s: float(np.var(regression_function(x, "sparse_relu", s), ddof=0))
        for s in SPARSITIES
    }
    factors = {s: math.sqrt(variances[4] / variances[s]) for s in SPARSITIES}
    for s, factor in factors.items():
        if 1.5 * factor > 2.0:
            raise ValueError(f"Calibrated f_0 for s={s} exceeds M=2.")
    return factors, variances


@lru_cache(maxsize=None)
def population_huber_shift(alpha, beta, base_scale, tau):
    """Solve E psi_tau(e-c)=0 using the stationary stable CDF."""
    if beta == 0:
        return 0.0
    scale = stable_scale(alpha, beta, base_scale)
    nodes, weights = leggauss(256)

    def score(c):
        cdf = levy_stable.cdf(c + tau * nodes, alpha, beta, scale=scale)
        return float(tau * (1.0 - np.dot(weights, cdf)))

    lower, upper = -20.0 * (tau + scale), 20.0 * (tau + scale)
    return float(brentq(score, lower, upper, xtol=1e-9))
