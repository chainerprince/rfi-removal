"""Gaussian-bin injection for recorded complex voltages."""

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import truncnorm


def _sample_bins(lower, upper, rng):
    # Reflect positive tails so CDF subtraction does not round them to zero.
    reflected = lower > 0
    a = np.where(reflected, -upper, lower)
    b = np.where(reflected, -lower, upper)
    left, right = ndtr(a), ndtr(b)
    regular = right - left > np.finfo(float).tiny
    p = left + rng.random(len(a)) * (right - left)
    samples = ndtri(np.clip(p, np.nextafter(0., 1.), np.nextafter(1., 0.)))
    if not regular.all():
        # Unlikely recorded codes can lie beyond the representable CDF tail.
        samples[~regular] = truncnorm.rvs(a[~regular], b[~regular], random_state=rng)
    return np.where(reflected, -samples, samples)


def _check_codes(q, limits):
    low, high = limits
    if not -8 <= low < 0 < high <= 7 or int(low) != low or int(high) != high:
        raise ValueError("limits must be integer codes within [-8, 7], spanning zero")
    if not np.iscomplexobj(q) or q.ndim < 1:
        raise ValueError("q must contain complex voltages, with time on axis 0")
    for part in (q.real, q.imag):
        if (not np.isfinite(part).all() or np.any(part != np.rint(part))
                or np.any((part < low) | (part > high))):
            raise ValueError("q must contain integer codes within the stated limits")


def fit_quantized_gaussian(q, *, limits=(-7, 7)):
    """Fit local Gaussian means and sigmas from integer-bin counts along time.

    Both outputs have shape (*q.shape[1:], 2); the last axis is real, imaginary.
    Use a quiet reference segment. This is a marginal fit, not a noise detector.
    """
    _check_codes(q, limits)
    if q.shape[0] < 64:
        raise ValueError("use at least 64 reference samples; several thousand are preferable")
    mean = np.empty((*q.shape[1:], 2), dtype=np.float64)
    sigma = np.empty_like(mean)
    codes = np.arange(limits[0], limits[1])
    edges = (codes + 0.5).reshape(-1, *([1] * (q.ndim - 1)))
    for component, part in enumerate((q.real, q.imag)):
        cdf = np.stack([(part <= code).mean(axis=0) for code in codes])
        usable = (cdf > 0.01) & (cdf < 0.99)
        if np.any(usable.sum(axis=0) < 2):
            raise ValueError("too few populated bins; mask dead or saturated inputs before fitting")
        z = ndtri(np.clip(cdf, 0.001, 0.999))
        # CDF thresholds lie on edge = mean + sigma * z. Tail counts are noisy.
        weight = np.where(usable, np.exp(-z*z) / (cdf * (1-cdf) + 1e-15), 0)
        total = weight.sum(axis=0)
        z_bar = (weight * z).sum(axis=0) / total
        edge_bar = (weight * edges).sum(axis=0) / total
        variance = (weight * (z - z_bar)**2).sum(axis=0)
        if np.any(variance <= 0):
            raise ValueError("reference does not resolve a Gaussian width")
        scale = (weight * (z - z_bar) * (edges - edge_bar)).sum(axis=0) / variance
        mean[..., component] = edge_bar - scale * z_bar
        sigma[..., component] = scale
    return mean, sigma


def inject_quantized(q, rfi, rng, *, mean, sigma, limits=(-7, 7)):
    """Add complex RFI through conditional Gaussian bins; return new integer codes.

    mean and sigma end in (real, imaginary), as returned by fit_quantized_gaussian.
    Samples are drawn independently within bins. Joint correlations are not fitted.
    """
    _check_codes(q, limits)
    rfi = np.broadcast_to(rfi, q.shape)
    mean = np.broadcast_to(np.asarray(mean, dtype=float), (*q.shape, 2))
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (*q.shape, 2))
    if not np.isfinite(rfi).all() or not np.isfinite(mean).all():
        raise ValueError("RFI and mean must be finite")
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("sigma must be finite and positive")
    out = q.astype(np.complex64, copy=True)
    low, high = limits
    for component, (codes, shift, dest) in enumerate(
            ((q.real, rfi.real, out.real), (q.imag, rfi.imag, out.imag))):
        active = shift != 0
        if not active.any():
            continue
        code = codes[active].astype(float)
        mu, sd = mean[..., component][active], sigma[..., component][active]
        # A rail code includes the whole clipped tail, not just a one-count bin.
        lower = np.where(code == low, -np.inf, code - 0.5)
        upper = np.where(code == high, np.inf, code + 0.5)
        latent = mu + sd * _sample_bins((lower-mu)/sd, (upper-mu)/sd, rng)
        dest[active] = np.rint(latent + shift[active]).clip(low, high)
    return out
