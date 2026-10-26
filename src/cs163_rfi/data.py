"""Generate recordings and load time windows with their binary RFI masks."""

import json
import csv
from pathlib import Path

import numpy as np

from .simulation import N_ANTENNAS, N_CHANNELS, simulate_cube
from .simulation import make_scene, MODULATIONS
from .voltage import CUBE_SAMPLES, DT


def write_scene(path, scene, *, cubes=None):
    """Write a new directory. cubes selects whole cubes for a small excerpt."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    indices = list(range(scene["cubes"])) if cubes is None else list(cubes)
    metadata = dict(scene, cube_indices=indices, stored_samples=len(indices)*CUBE_SAMPLES)
    shape = (metadata["stored_samples"], N_ANTENNAS, N_CHANNELS, 1)
    power = np.lib.format.open_memmap(path/"power.npy", mode="w+", dtype="uint8", shape=shape)
    packed_shape = (shape[0], shape[1], N_CHANNELS//8, 1)
    labels = np.lib.format.open_memmap(path/"labels.npy", mode="w+", dtype="uint8", shape=packed_shape)
    valid = np.lib.format.open_memmap(path/"valid.npy", mode="w+", dtype="uint8", shape=packed_shape)
    rfi_power = np.lib.format.open_memmap(path/"rfi_power.npy", mode="w+", dtype="float32",
                                        shape=(shape[0]//64, *shape[1:]))
    clipped = []
    for i, cube in enumerate(indices):
        sample = simulate_cube(scene, cube)
        time = slice(i*CUBE_SAMPLES, (i+1)*CUBE_SAMPLES)
        power[time] = sample["power"]
        labels[time] = np.packbits(sample["labels"], axis=2)
        valid[time] = np.packbits(sample["valid"], axis=2)
        rfi_power[i*64:(i+1)*64] = average_power(sample["injected_power"], 64)
        clipped.append(sample["clipped_fraction"])
    for array in (power, labels, valid, rfi_power):
        array.flush()
    metadata["clipped_fraction"] = float(np.mean(clipped))
    (path/"scene.json").write_text(json.dumps(metadata, indent=2) + "\n")


def read_scene(path, *, start=0, samples=CUBE_SAMPLES, antenna=None):
    """Read native float32 power and Boolean truth. Time is in native samples.

    The default reads 134 ms. samples=None reads to the end. An antenna index
    keeps a length-one antenna axis. File reads are memory-mapped.
    """
    path = Path(path)
    metadata = json.loads((path/"scene.json").read_text())
    stop = metadata["stored_samples"] if samples is None else start + samples
    if not 0 <= start < stop <= metadata["stored_samples"]:
        raise ValueError("requested window is outside the stored scene")
    if antenna is not None and not 0 <= antenna < N_ANTENNAS:
        raise ValueError("antenna index must be between 0 and 9")
    ants = slice(None) if antenna is None else slice(antenna, antenna+1)
    selection = (slice(start, stop), ants, slice(None), slice(None))
    result = {"metadata": metadata}
    result["power"] = np.load(path/"power.npy", mmap_mode="r")[selection].astype(np.float32)
    for name in ("labels", "valid"):
        packed = np.load(path/f"{name}.npy", mmap_mode="r")[selection]
        result[name] = np.unpackbits(packed, axis=2, count=N_CHANNELS).astype(bool)
    local = np.arange(start, stop)
    indices = np.asarray(metadata["cube_indices"])
    result["time_s"] = (indices[local//CUBE_SAMPLES]*CUBE_SAMPLES + local % CUBE_SAMPLES)*DT
    result["frequency_mhz"] = np.asarray(metadata["frequency_hz"])/1e6
    result["antenna_names"] = metadata["antennas"][ants]
    return result


def average_power(power, factor=64):
    """Average the time axis for display. factor=1 keeps native samples."""
    if not isinstance(factor, (int, np.integer)) or factor < 1 or len(power) % factor:
        raise ValueError("factor must be a positive divisor of the time length")
    return np.asarray(power, dtype=np.float32).reshape(-1, factor, *power.shape[1:]).mean(1)


def power_sums(power, m=64):
    """S1=sum(P), S2=sum(P**2). These sums are not squared mean power."""
    if m < 1 or len(power) % m:
        raise ValueError("m must divide the time length")
    cells = np.asarray(power, dtype=np.float32).reshape(-1, m, *power.shape[1:])
    return cells.sum(1), np.square(cells).sum(1)


def simulate_recording(*, seed=1, rfi="radar", strength=1., duration_s=4096*DT,
                       duty_cycle=None, period_s=None, symbol_rate=None):
    """Generate channelized power and binary truth for ten antennas.

    Duration rounds up to a whole 4096-sample block. Strength is reference
    RFI power / noise power, before channel leakage and antenna gains.
    """
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be positive")
    count = int(np.ceil(duration_s/(CUBE_SAMPLES*DT)))
    settings = make_scene(seed, rfi, strength, cubes=count)
    settings["generator"] = dict(seed=seed, rfi=rfi, strength=strength,
                                 duration_s=count*CUBE_SAMPLES*DT)
    for key, value in (("duty_cycle", duty_cycle), ("period_s", period_s),
                       ("symbol_rate", symbol_rate)):
        if value is not None:
            settings["generator"][key] = value
    for event in settings["events"]:
        if period_s is not None:
            if not np.isfinite(period_s) or period_s <= 0:
                raise ValueError("period_s must be positive")
            event["period_s"] = period_s
        if duty_cycle is not None:
            if not np.isfinite(duty_cycle) or not 0 < duty_cycle <= 1:
                raise ValueError("duty_cycle must be in (0, 1]")
            if event["kind"] not in ("radar", "wireless", *MODULATIONS):
                raise ValueError("this waveform has no duty-cycle control")
            event["pulse_s"] = duty_cycle*event["period_s"]
        if symbol_rate is not None:
            if event["kind"] not in MODULATIONS[1:]:
                raise ValueError("symbol_rate is for bpsk, qpsk, ask or fsk")
            event["symbol_rate"] = symbol_rate
    parts = [simulate_cube(settings, i) for i in range(count)]
    return dict(power=np.concatenate([p["power"] for p in parts]),
                rfi_mask=np.concatenate([p["labels"] for p in parts]),
                injected_power=np.concatenate([p["injected_power"] for p in parts]),
                metadata=settings, time_s=np.arange(count*CUBE_SAMPLES)*DT,
                frequency_mhz=np.asarray(settings["frequency_hz"])/1e6,
                antenna_names=settings["antennas"],
                clipped_fraction=float(np.mean([p["clipped_fraction"] for p in parts])))


def save_recording(path, recording):
    """Save a new recording without overwriting another one."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    np.save(path/"power.npy", recording["power"].astype(np.uint8))
    np.save(path/"rfi_mask.npy", np.packbits(recording["rfi_mask"], axis=2))
    np.save(path/"rfi_power.npy", average_power(recording["injected_power"], 64))
    settings = recording["metadata"]
    frequency = settings["frequency_hz"]
    metadata = dict(version="1.0", shape=list(recording["power"].shape),
                    dt_s=settings["dt_s"], first_channel_hz=frequency[0],
                    channel_step_hz=frequency[1]-frequency[0], polarization="X",
                    label_noise_fraction=settings["label_noise_fraction"],
                    generator=settings["generator"])
    (path/"recording.json").write_text(json.dumps(metadata, indent=2)+"\n")


def load_recording(path, *, start=0, samples=None, antenna=None):
    """Read float32 power and Boolean RFI truth; axes are time, antenna, channel, feed."""
    path = Path(path)
    metadata = json.loads((path/"recording.json").read_text())
    shape = np.load(path/"power.npy", mmap_mode="r").shape
    stop = shape[0] if samples is None else start+samples
    if not 0 <= start < stop <= shape[0]:
        raise ValueError("requested samples are outside this recording")
    if antenna is not None and not 0 <= antenna < N_ANTENNAS:
        raise ValueError("antenna must be between 0 and 9")
    ants = slice(None) if antenna is None else slice(antenna, antenna+1)
    selection = (slice(start, stop), ants, slice(None), slice(None))
    power = np.load(path/"power.npy", mmap_mode="r")[selection].astype(np.float32)
    packed = np.load(path/"rfi_mask.npy", mmap_mode="r")[selection]
    if "frequency_hz" in metadata:
        frequency = np.asarray(metadata["frequency_hz"])
        antennas = metadata["antennas"]
    else:
        frequency = metadata["first_channel_hz"] + np.arange(shape[2])*metadata["channel_step_hz"]
        antennas = [f"A{i:02d}" for i in range(shape[1])]
    return dict(power=power, rfi_mask=np.unpackbits(packed, axis=2, count=shape[2]).astype(bool),
                time_s=np.arange(start, stop)*metadata["dt_s"],
                frequency_mhz=frequency/1e6,
                antenna_names=antennas[ants], metadata=metadata, start_sample=start)


def load_sample(dataset, sample_id, *, antenna=0, start=0, samples=CUBE_SAMPLES):
    """Load one antenna's time × frequency input and its matching RFI mask."""
    dataset = Path(dataset)
    with (dataset/"dataset.csv").open() as stream:
        row = next((r for r in csv.DictReader(stream) if int(r["sample_id"]) == sample_id), None)
    if row is None:
        raise ValueError(f"sample {sample_id} is not in dataset.csv")
    if not 0 <= antenna < N_ANTENNAS:
        raise ValueError("antenna must be between 0 and 9")
    recording = load_recording(dataset/row["path"], start=start, samples=samples, antenna=antenna)
    for name in ("power", "rfi_mask"):
        recording[name] = recording[name][:, 0, :, 0]
    recording["split"] = row["split"]
    recording["sample_id"] = sample_id
    return recording
