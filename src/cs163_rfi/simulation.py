"""An ideal receiver, pulsed transmitters and digital modulation."""

import numpy as np
from scipy.signal import fftconvolve, firwin

from .voltage import CUBE_SAMPLES, DT, noise, quantize

N_CHANNELS = 128
N_ANTENNAS = 10
CHANNEL_HZ = 1 / DT
SIGMA = 1.43
FAMILIES = ("carrier", "radar", "wireless", "digital", "impulse")
MODULATIONS = ("switched_carrier", "bpsk", "qpsk", "ask", "fsk")


def frequencies():
    """Channel centres in Hz, highest frequency first."""
    return 1.35e9 + np.arange(63, -65, -1) * CHANNEL_HZ


def make_scene(seed, family="mixed", strength=0.3, cubes=30):
    """Describe a scene. Strength is on-state RFI / noise power in its band."""
    if family not in (*FAMILIES, *MODULATIONS, "mixed", "clean", "sky"):
        raise ValueError(f"unknown family: {family}")
    if not np.isfinite(strength) or strength < 0 or not isinstance(cubes, int) or cubes < 1:
        raise ValueError("strength must be nonnegative and cubes must be positive")
    rng = np.random.default_rng(seed)
    duration = cubes * CUBE_SAMPLES * DT
    events = []
    kinds = list(rng.choice(FAMILIES, 3, replace=False)) if family == "mixed" else [family]
    for i, kind in enumerate(kinds):
        if kind in ("clean", "sky"):
            continue
        start = float(rng.uniform(0.05, 0.4) * duration)
        stop = float(rng.uniform(0.65, 0.98) * duration)
        if kind in ("carrier", "digital") and rng.random() < 0.4:
            start, stop = 0.0, duration
        width = int(rng.choice([8, 16, 32, 48])) if kind in ("wireless", "digital") else 1
        if kind == "impulse":
            width = N_CHANNELS
        channel = float(rng.uniform(30, 98))
        gains = rng.uniform(0.5, 2.0, N_ANTENNAS)
        if rng.random() < 0.3:
            keep = rng.choice(N_ANTENNAS, int(rng.integers(2, 6)), replace=False)
            gains[~np.isin(np.arange(N_ANTENNAS), keep)] = 0
        pulse = float(rng.uniform(4e-6, 100e-6))
        period = float(rng.uniform(max(0.003, pulse * 30), 0.027))
        if kind == "wireless":
            period, pulse = float(rng.uniform(0.008, 0.025)), float(rng.uniform(0.001, 0.004))
        if kind == "impulse":
            pulse = float(rng.uniform(10e-6, 0.001))
            stop = min(stop, start + pulse)
        event = dict(kind=kind, seed=int(seed) + 1000 + i, strength=float(strength),
                           start_s=start, stop_s=stop, channel=channel, width=width,
                           period_s=period, pulse_s=pulse, phase=float(rng.uniform(0, 2*np.pi)),
                           delay_s=float(rng.uniform(-1e-8, 1e-8)),
                           power_gains=gains.tolist())
        if kind in MODULATIONS:
            event.update(period_s=64*DT, pulse_s=32*DT, symbol_rate=3000.,
                         channel_separation=4.)
        events.append(event)
    return dict(version="0.7", seed=int(seed), family=family, cubes=int(cubes),
                samples=int(cubes * CUBE_SAMPLES), dt_s=DT, frequency_hz=frequencies().tolist(),
                antennas=[f"A{i:02d}" for i in range(N_ANTENNAS)], polarization=["X"],
                sigma=SIGMA, strength=float(strength), events=events,
                science=family == "sky", label_policy="power_threshold",
                label_noise_fraction=0.01)


def _random_stream(seed, first, count):
    """White complex samples with fixed blocks, including filter history."""
    block_size = CUBE_SAMPLES * N_CHANNELS
    out = np.zeros(count, np.complex64)
    for block in range(max(0, first // block_size), (first + count - 1) // block_size + 1):
        values = noise((block_size,), np.random.default_rng([seed, block]), sigma=1/np.sqrt(2))
        left, right = max(first, block*block_size), min(first+count, (block+1)*block_size)
        out[left-first:right-first] = values[left-block*block_size:right-block*block_size]
    return out


def channelize(wave):
    """Non-overlapping rectangular FFTs, unitary scaling, descending channels."""
    spectra = np.fft.fft(wave.reshape(-1, N_CHANNELS), axis=-1, norm="ortho")
    return np.fft.fftshift(spectra, axes=-1)[:, ::-1].astype(np.complex64)


def _modulation(event, t):
    """Symbol changes, including a phase-continuous two-frequency signal."""
    rate = event["symbol_rate"]
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("symbol_rate must be positive")
    index = np.floor(t*rate).astype(int)
    levels = 4 if event["kind"] in ("qpsk", "ask") else 2
    symbols = np.random.default_rng(event["seed"]).integers(levels, size=index[-1]+1)
    if event["kind"] in ("bpsk", "qpsk"):
        return np.exp(2j*np.pi*symbols[index]/levels)
    if event["kind"] == "ask":
        return 2*symbols[index]/(levels-1) - 1
    offsets = (symbols-0.5)*event["channel_separation"]*CHANNEL_HZ
    cycles = np.r_[0., np.cumsum(offsets/rate)]
    phase = cycles[index] + offsets[index]*(t-index/rate)
    return np.exp(2j*np.pi*phase)


def _transmitter(event, cube, noise_power):
    if event["period_s"] <= 0 or event["pulse_s"] <= 0 or event["stop_s"] <= event["start_s"]:
        raise ValueError("pulse period, pulse length and event duration must be positive")
    count = CUBE_SAMPLES * N_CHANNELS
    first = cube * count
    dt = DT / N_CHANNELS
    t = (first + np.arange(count)) * dt
    gate = (t >= event["start_s"]) & (t < event["stop_s"])
    if event["kind"] in ("radar", "wireless", *MODULATIONS):
        gate &= (t - event["start_s"]) % event["period_s"] < event["pulse_s"]
    if not gate.any():
        return np.zeros((CUBE_SAMPLES, N_CHANNELS), np.complex64)
    frequency = (63 - event["channel"]) * CHANNEL_HZ
    phase = np.exp(1j * (2*np.pi*frequency*t + event["phase"]))
    level = noise_power * event["strength"]
    if event["kind"] in ("carrier", "radar", *MODULATIONS):
        wave = phase * np.sqrt(level / N_CHANNELS)
        if event["kind"] in MODULATIONS[1:]:
            wave *= _modulation(event, t)
    else:
        wave = _random_stream(event["seed"], first - 128, count + 128)
        if event["width"] < N_CHANNELS:
            taps = firwin(129, event["width"] / N_CHANNELS)
            wave = fftconvolve(wave, taps, mode="valid") * phase
        else:
            wave = wave[128:]
        wave *= np.sqrt(level)
    return channelize(wave * gate)


def simulate_cube(scene, cube=0, *, return_voltage=False):
    """Return native power, labels, valid cells, and injected power for one cube.

    Axes are time, antenna, frequency, polarization. New recordings label
    injected power at or above 1% of mean receiver-noise power.
    """
    if not 0 <= cube < scene["cubes"]:
        raise ValueError("cube is outside this scene")
    shape = (CUBE_SAMPLES, N_ANTENNAS, N_CHANNELS, 1)
    rng = np.random.default_rng([scene["seed"], cube, 771])
    receiver = noise(shape, rng, scene["sigma"])
    noise_power = 2 * scene["sigma"]**2
    if scene["science"]:
        t = (cube*CUBE_SAMPLES + np.arange(CUBE_SAMPLES)) * DT
        freq = frequencies() / 1e6
        arrival = scene["samples"]*DT/2 + 4.148808e3*400*(freq**-2 - freq.max()**-2)
        envelope = np.exp(-0.5*((t[:, None]-arrival)/0.002)**2)
        sky = noise((CUBE_SAMPLES, 1, N_CHANNELS, 1),
                    np.random.default_rng([scene["seed"], cube, 772]), scene["sigma"])
        receiver += sky * np.sqrt(0.3*envelope[:, None, :, None])
    rfi = np.zeros(shape, np.complex64)
    labels = np.zeros(shape, bool)
    affected = np.zeros(shape, bool)
    for event in scene["events"]:
        source = _transmitter(event, cube, noise_power)
        if scene.get("label_policy") in ("binary", "power_threshold"):
            # Remove floating-point FFT roundoff before injection, not from scoring.
            source[np.abs(source) < 1e-6*np.abs(source).max()] = 0
        gains = np.sqrt(event["power_gains"])
        # A compact array with a plane-wave phase gradient. No spatial classifier is used.
        phases = np.exp(-2j*np.pi*np.arange(N_ANTENNAS)[:, None]*event["delay_s"]*frequencies())
        field = source[:, None, :, None] * (gains[:, None]*phases)[None, :, :, None]
        source_power = np.abs(field)**2
        reference = (noise_power * event["strength"] * gains**2)[None, :, None, None]
        if scene.get("label_policy") == "binary":
            labels |= source_power > 0
        elif scene.get("label_policy") != "power_threshold":
            labels |= (source_power >= scene["label_tail_fraction"]*reference) & (reference > 0)
            affected |= (source_power > 1e-12*reference) & (reference > 0)
        rfi += field
    voltage = quantize(receiver + rfi)
    power = (voltage.real**2 + voltage.imag**2).astype(np.uint8)
    injected = np.abs(rfi)**2
    if scene.get("label_policy") == "power_threshold":
        labels = injected >= scene["label_noise_fraction"]*noise_power
    clipped = np.mean(((receiver+rfi).real > 7.5) | ((receiver+rfi).imag > 7.5)
                      | ((receiver+rfi).real < -8.5) | ((receiver+rfi).imag < -8.5))
    result = dict(power=power, labels=labels, valid=labels | ~affected,
                  injected_power=injected.astype(np.float32), clipped_fraction=float(clipped))
    if return_voltage:
        result.update(voltage=voltage, clean_voltage=quantize(receiver))
    return result
