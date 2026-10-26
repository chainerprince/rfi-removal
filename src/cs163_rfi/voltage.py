import numpy as np

DT = 32.768e-6
M = 64
CUBE_SAMPLES = 4096
ANTENNAS = 96
CHANNELS = 384
POLARIZATIONS = 2


def noise(shape, rng, sigma=1.43):
    """Complex Gaussian noise. sigma is RMS per component, in ADC counts."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    v = rng.standard_normal(shape, dtype=np.float32).astype(np.complex64)
    v.imag = rng.standard_normal(shape, dtype=np.float32)
    v *= sigma
    return v


def quantize(v, *, limits=(-8, 7)):
    """Round once to signed 4-bit components with a fixed scale of one count."""
    if not np.isfinite(v).all():
        raise ValueError("voltages must be finite")
    low, high = limits
    if not -8 <= low < high <= 7 or int(low) != low or int(high) != high:
        raise ValueError("limits must be integer codes within [-8, 7]")
    out = np.rint(v.real).clip(low, high).astype(np.complex64)
    out.imag = np.rint(v.imag).clip(low, high)
    return out


def inject_rfi(v, rng, *, kind, inr, noise_power, channels=None,
               start=0, stop=None, period=64, duty=0.1, pulse_phase=0, sample_offset=0,
               gains=None):
    """Return (contaminated, RFI) in (time, antenna, channel, polarization).

    INR is on-state mean source power / receiver noise power, per channel
    and polarization at unit gain. Add to floating voltages before quantizing.
    """
    if v.ndim != 4 or not np.iscomplexobj(v):
        raise ValueError("v must be a complex (time, antenna, channel, pol) array")
    if kind not in ("tone", "pulsed_tone", "noise", "impulse"):
        raise ValueError(f"unknown RFI kind: {kind}")
    if inr < 0 or noise_power <= 0 or not 0 < duty <= 1 or period < 1:
        raise ValueError("invalid INR, noise power, period, or duty cycle")
    nt, na, nc, npol = v.shape
    channels = np.arange(nc) if channels is None else np.asarray(channels)
    if channels.ndim != 1 or not np.issubdtype(channels.dtype, np.integer):
        raise ValueError("channels must be integer indices")
    if np.any((channels < 0) | (channels >= nc)):
        raise ValueError("channel index out of range")
    t = np.arange(nt) + sample_offset
    gate = t >= start
    if stop is not None:
        gate &= t < stop
    if kind == "pulsed_tone":
        gate &= (t - start + pulse_phase) % period < round(duty * period)
    source_shape = (nt, 1, len(channels), npol)
    if kind in ("noise", "impulse"):
        source = noise(source_shape, rng, sigma=np.sqrt(noise_power * max(inr, 1e-30) / 2))
        if inr == 0:
            source.fill(0)
    else:
        source = np.broadcast_to(
            np.exp(2j * np.pi * 0.13 * t)[:, None, None, None], source_shape
        ).astype(np.complex64).copy()
        source *= np.sqrt(inr * noise_power)
    source *= gate[:, None, None, None]
    rfi = np.zeros_like(v)
    coupling = 1 if gains is None else np.broadcast_to(gains, (na, nc, npol))[:, channels]
    rfi[:, :, channels, :] = source * coupling
    return v + rfi, rfi


def moments(v, m=M):
    """S1=sum(|v|^2), S2=sum(|v|^4), retaining antenna/channel/pol axes."""
    if m < 2 or v.shape[0] % m:
        raise ValueError("time length must be a multiple of m >= 2")
    power = v.real.astype(np.float32)**2 + v.imag.astype(np.float32)**2
    power = power.reshape(-1, m, *v.shape[1:])
    return power.sum(axis=1), np.square(power).sum(axis=1)


def spectral_kurtosis(s1, s2, m=M):
    """SK from power sums. Zero-power cells return NaN."""
    if m < 2 or s1.shape != s2.shape:
        raise ValueError("invalid m or unequal moment shapes")
    with np.errstate(divide="ignore", invalid="ignore"):
        sk = (m + 1) / (m - 1) * (m * s2 / np.square(s1) - 1)
    return np.where(s1 > 0, sk, np.nan)
