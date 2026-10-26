"""Per-antenna baselines from the pinned DSA-110 flagger."""

import numpy as np
import torch

from dsart.rfi.bandpass_outlier import bandpass_outlier_mask
from dsart.rfi.sk import sk_combined_masks
from dsart.rfi.sum_threshold import sum_threshold_1d

from .data import power_sums
from .voltage import CUBE_SAMPLES


def flag_rfi(power, *, method="sk-bandpass", far=1e-4, m_values=(64, 256, 1024, 4096), start_sample=0):
    """Return native Boolean flags without changing power.

    power has axes (time, channel) or (time, antenna, channel, polarization).
    Each aligned 4096-sample block is tested separately. A flagged channel
    removes that whole block. start_sample is the offset in the recording.
    sk-bandpass adds the production channel expansion to the union.
    """
    if method not in ("sk", "bandpass", "sk-bandpass"):
        raise ValueError("method must be sk, bandpass, or sk-bandpass")
    power = np.asarray(power)
    if start_sample % CUBE_SAMPLES:
        raise ValueError("baseline windows must start on a 4096-sample boundary")
    if power.ndim == 2:
        return flag_rfi(power[:, None, :, None], method=method, far=far,
                        m_values=m_values, start_sample=start_sample)[:, 0, :, 0]
    if power.ndim != 4 or not len(power) or len(power) % CUBE_SAMPLES:
        raise ValueError("power must have four axes and a whole number of 4096-sample blocks")
    if not np.isfinite(power).all() or np.any(power < 0) or not 0 < far < 1:
        raise ValueError("power must be finite and nonnegative; far must be in (0, 1)")
    if not m_values or any(m < 4 or CUBE_SAMPLES % m for m in m_values):
        raise ValueError("SK lengths must divide 4096 and be at least 4")
    mask = np.zeros(power.shape, bool)
    with torch.no_grad():
        for start in range(0, len(power), CUBE_SAMPLES):
            block = power[start:start+CUBE_SAMPLES]
            flags = torch.zeros(block.shape[1:], dtype=torch.bool)
            if method in ("sk", "sk-bandpass"):
                s1, s2 = {}, {}
                for m in m_values:
                    a, b = power_sums(block, m)
                    s1[m], s2[m] = torch.from_numpy(a), torch.from_numpy(b)
                flags |= sk_combined_masks(s1, s2, far=far)[0]
            if method in ("bandpass", "sk-bandpass"):
                full = torch.from_numpy(block.sum(0, dtype=np.float32))
                flags |= bandpass_outlier_mask(full, far=far, m_acc=CUBE_SAMPLES)
            if method == "sk-bandpass":
                flags = sum_threshold_1d(flags.permute(0, 2, 1).contiguous(), max_m=8,
                                         eta=1.5).permute(0, 2, 1)
            mask[start:start+CUBE_SAMPLES] = flags.numpy()
    return mask


def flag_power(power, threshold):
    """Native power threshold; choose threshold on separate clean validation data."""
    return np.asarray(power) > threshold
