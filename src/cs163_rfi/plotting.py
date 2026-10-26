"""Views of the same samples used by the detectors."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch
import numpy as np

from .data import average_power
from .simulation import FAMILIES, N_CHANNELS, make_scene, simulate_cube
from .voltage import DT

NAMES = {"carrier": "Carrier · satellite or telemetry",
         "radar": "Pulse train · radar-like",
         "wireless": "Bursts · wireless link",
         "digital": "Steady band · digital transmitter",
         "impulse": "Broadband impulse · electrical discharge"}


def _extent(sample):
    freq, time = sample["frequency_mhz"], sample["time_s"]
    df = abs(freq[1]-freq[0])
    return [time[0]*1000, (time[-1]+DT)*1000, freq[-1]-df/2, freq[0]+df/2]


def plot_antenna_grid(sample, factor=64, time_ms=None):
    power = average_power(sample["power"], factor)
    visible = power
    if time_ms is not None:
        time = average_power(sample["time_s"], factor)*1000
        visible = power[(time >= time_ms[0]) & (time <= time_ms[1])]
    high = np.quantile(visible, 0.99)
    fig, axes = plt.subplots(2, 5, figsize=(13, 5), sharex=True, sharey=True, layout="constrained")
    for ant, ax in enumerate(axes.flat):
        if ant >= power.shape[1]:
            ax.set_visible(False)
            continue
        image = ax.imshow(power[:, ant, :, 0].T, aspect="auto", extent=_extent(sample),
                          origin="upper", vmin=0, vmax=high, cmap="magma", interpolation="nearest")
        ax.set_title(sample["antenna_names"][ant])
        if time_ms is not None:
            ax.set_xlim(time_ms)
    fig.supxlabel("Time (ms)")
    fig.supylabel("Frequency (MHz)")
    fig.colorbar(image, ax=axes, label="Mean power (counts²)", shrink=0.8, extend="max")
    fig.suptitle(f"One recording, ten antennas · display average {factor*DT*1000:.3f} ms")
    return fig


def plot_sample(sample, mask=None, antenna=0, factor=64, time_ms=None):
    select = lambda x: x if x.ndim == 2 else x[:, antenna, :, 0]
    power = average_power(select(sample["power"]), factor)
    labels = average_power(select(sample.get("rfi_mask", sample.get("labels"))), factor)
    fig, axes = plt.subplots(1, 2 if mask is None else 3, figsize=(13, 3.8),
                             sharex=True, sharey=True, layout="constrained")
    im = axes[0].imshow(power.T, origin="upper", aspect="auto", extent=_extent(sample),
                        cmap="magma", interpolation="nearest")
    axes[0].set_title(f"{sample['antenna_names'][antenna]} · measured power")
    fig.colorbar(im, ax=axes[0], label="Counts²", shrink=0.8)
    im = axes[1].imshow(labels.T, origin="upper", aspect="auto", extent=_extent(sample),
                        vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
    axes[1].set_title("Target mask · RFI fraction" if factor > 1 else "Target mask · 1 = RFI")
    fig.colorbar(im, ax=axes[1], label="RFI fraction" if factor > 1 else "RFI label",
                 ticks=None if factor > 1 else [0, 1], shrink=0.8)
    if mask is not None:
        view = average_power(select(mask), factor)
        im = axes[2].imshow(view.T, origin="upper", aspect="auto", extent=_extent(sample),
                            vmin=0, vmax=1, cmap="Reds")
        axes[2].set_title("Detector flags · fraction of samples" if factor > 1 else "Detector mask · 1 = remove")
        fig.colorbar(im, ax=axes[2], label="Flag fraction" if factor > 1 else "Flag",
                     ticks=None if factor > 1 else [0, 1], shrink=0.8)
    for ax in axes:
        ax.set_xlabel("Time (ms)")
        if time_ms is not None:
            ax.set_xlim(time_ms)
    axes[0].set_ylabel("Frequency (MHz)")
    return fig


def plot_errors(sample, mask, antenna=0, first=0, samples=512):
    selection = slice(first, first+samples) if mask.ndim == 2 else (slice(first, first+samples), antenna, slice(None), 0)
    truth = sample.get("rfi_mask", sample.get("labels"))[selection]
    flags = mask[selection]
    values = np.zeros(truth.shape, np.uint8)
    values[truth & flags] = 1
    values[truth & ~flags] = 2
    values[~truth & flags] = 3
    fig, ax = plt.subplots(figsize=(11, 3.5), layout="constrained")
    colors = ["#f3f4f5", "#2d947b", "#edbb42", "#cf5177"]
    extent = _extent(sample)
    extent[:2] = [sample["time_s"][first]*1000, (sample["time_s"][first+samples-1]+DT)*1000]
    image = ax.imshow(values.T, aspect="auto", extent=extent, origin="upper",
                      cmap=ListedColormap(colors), vmin=-0.5, vmax=3.5, interpolation="nearest")
    cb = fig.colorbar(image, ax=ax, ticks=range(4))
    cb.ax.set_yticklabels(["Clean", "Caught", "Missed", "False flag"])
    ax.set(xlabel="Time (ms)", ylabel="Frequency (MHz)", title="Errors at native resolution")
    return fig


def plot_telescopes(assets="assets"):
    """Local photographs; source links and image rights are in assets/CREDITS.md."""
    fig = plt.figure(figsize=(12, 4.5), layout="constrained")
    photos = [("parkes.jpg", "Parkes · single dish", "One 64-m dish collects radio waves.",
               "David McClenaghan / CSIRO · CC BY 3.0"),
              ("dsa-110.webp", "DSA-110 · interferometer", "Correlate signals from separate antennas.",
               "Gregg Hallinan / Caltech")]
    images = [plt.imread(Path(assets)/name) for name, *_ in photos]
    ratios = [im.shape[1]/im.shape[0] for im in images]
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 0.17), width_ratios=ratios)
    for column, (name, title, caption, credit) in enumerate(photos):
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(images[column])
        ax.set_title(title, loc="left", fontweight="semibold", color="#292522", pad=12)
        ax.set_anchor("N")
        ax.axis("off")
        caption_ax = fig.add_subplot(grid[1, column])
        caption_ax.axis("off")
        caption_ax.text(0, 0.85, caption, va="top", fontsize=11)
        caption_ax.text(0, 0.25, credit, va="top", fontsize=9, color="#69625c")
    return fig


def plot_radio_science(assets="assets"):
    """Three discoveries; image sources are in assets/CREDITS.md."""
    fig = plt.figure(figsize=(13, 4.7), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 0.26])
    examples = [
        ("m87.jpg", "M87's black hole", "Image its shadow. Study gravity.",
         "EHT Collaboration / ESO"),
        ("double-pulsar.jpg", "The double pulsar", "Test strong-field gravity.",
         "Illustration: John Rowe Animations / CSIRO"),
        ("lorimer-burst.png", "The Lorimer burst", "FRBs probe matter between galaxies.",
         "Lorimer et al. (2007), Figure 2"),
    ]
    for column, (name, title, caption, credit) in enumerate(examples):
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(plt.imread(Path(assets)/name))
        ax.set_anchor("N")
        ax.set_title(title, fontweight="semibold")
        ax.axis("off")
        text = fig.add_subplot(grid[1, column])
        text.axis("off")
        text.text(0.5, 0.9, caption, ha="center", va="top", fontsize=11)
        text.text(0.5, 0.35, credit, ha="center", va="top", fontsize=9, color="#69625c")
    return fig


def plot_kurtosis(sample, m=64):
    """Show power and SK in the brightest channel of one antenna."""
    from dsart.rfi.sk import sk_thresholds
    from .data import power_sums
    from .voltage import spectral_kurtosis

    power = sample["power"]
    channel = int(power.mean(axis=0).argmax())
    s1, s2 = power_sums(power[:, channel], m)
    sk = spectral_kurtosis(s1, s2, m)
    time = average_power(sample["time_s"], m)*1000
    truth = average_power(sample["rfi_mask"][:, channel], m) > 0
    low, high = sk_thresholds(m, 1e-4)
    detected = (sk < low) | (sk > high)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True, layout="constrained")
    axes[0].plot(time, s1/m, color="#292522", lw=1.5)
    axes[0].set_ylabel("Mean power (counts²)")
    axes[1].plot(time, sk, ".-", color="#292522", lw=1, ms=4)
    axes[1].scatter(time[detected], sk[detected], color="#a3482b", s=45,
                    zorder=4, label="Outside SK limits")
    axes[1].axhline(1, color="#69625c", ls=":", label="Ideal noise: SK = 1")
    for value in (low, high):
        axes[1].axhline(value, color="#a3482b", ls="--", lw=1)
    for ax in axes:
        ax.fill_between(time, 0, 1, where=truth, step="mid", color="#e6b980",
                        alpha=0.25, transform=ax.get_xaxis_transform(), label="RFI present")
        ax.set_xlim(sample["time_s"][0]*1000, (sample["time_s"][-1]+DT)*1000)
    axes[1].set(xlabel="Time (ms)", ylabel="Spectral kurtosis",
                ylim=(0, max(high, float(np.nanmax(sk)))*1.12))
    axes[1].legend(loc="upper right", fontsize=9, ncol=3)
    axes[0].set_title(f"{sample['antenna_names'][0]} · channel {channel} · "
                      f"{sample['frequency_mhz'][channel]:.3f} MHz · {m*DT*1000:.3f} ms intervals")
    return fig


def plot_signal_chain():
    """Imaging uses complex voltages; power alone does not keep baseline phase."""
    fig, ax = plt.subplots(figsize=(14, 4.6), layout="constrained", facecolor="white")
    ax.set(xlim=(-0.25, 15.65), ylim=(-0.15, 4.8), aspect="equal")
    ax.axis("off")
    width, height, step = 2.45, 1.05, 3.2
    x = 1.3 + step*np.arange(5)
    top, bottom = 3.2, 0.85
    ink, muted, accent = "#292522", "#69625c", "#a3482b"

    def box(cx, cy, title, detail, task=False):
        fill, edge = ("#fbf5ee", accent) if task else ("#ffffff", "#958b82")
        ax.add_patch(FancyBboxPatch((cx-width/2, cy-height/2), width, height,
                                   boxstyle="round,pad=0,rounding_size=0.02",
                                   facecolor=fill, edgecolor=edge, linewidth=1.2))
        ax.text(cx, cy+0.15, title, ha="center", va="center", color=ink,
                fontsize=12, fontweight="semibold")
        ax.text(cx, cy-0.20, detail, ha="center", va="center", color=muted, fontsize=10)

    def arrow(a, b, color=muted):
        ax.annotate("", b, a, arrowprops=dict(arrowstyle="-|>", color=color,
                                             lw=1.4, shrinkA=4, shrinkB=4))

    box(x[0], top, "Antennas + receiver", "Digitize and channelize")
    box(x[1], top, "Complex voltages", "v = I + iQ")
    box(x[2], top, "Apply mask", "Keep unflagged samples", task=True)
    box(x[3], top, "Visibilities", "Antenna-pair correlations")
    box(x[4], top, "Dirty image", "A view of the radio sky")
    for left, right in zip(x[:-1], x[1:]):
        arrow((left+width/2, top), (right-width/2, top))
    for i, label in [(2, "Correlate\nand average"), (3, "Calibrate\nand image")]:
        ax.text((x[i]+x[i+1])/2, top+0.70, label, ha="center", color=muted, fontsize=10)
    ax.text(x[1], top+1.15, "36 GB/s · 129.6 TB/hour", ha="center", color=ink, fontsize=10)
    ax.text(x[3], top+1.15, "~7.52 MB/s · ~27 GB/hour", ha="center", color=ink, fontsize=10)

    box(x[1], bottom, "Power", "P = I² + Q²", task=True)
    box(x[2], bottom, "RFI mask", "One flag per sample", task=True)
    arrow((x[1], top-height/2), (x[1], bottom+height/2), accent)
    arrow((x[1]+width/2, bottom), (x[2]-width/2, bottom), accent)
    arrow((x[2], bottom+height/2), (x[2], top-height/2), accent)
    ax.text(x[2]+0.18, (top+bottom)/2, "Flag before\naveraging", va="center", color=accent, fontsize=10)
    ax.text(x[3]-width/2, bottom+0.15, "OUR TASK", color=accent, fontsize=10, fontweight="semibold")
    ax.text(x[3]-width/2, bottom-0.23, "Predict the mask from power.", color=ink, fontsize=11)
    ax.text(x[3]-width/2, bottom-0.65, "DSA-110: packed voltages; stored visibility files.",
            color=muted, fontsize=9)
    return fig


def plot_duty_gallery():
    """Paper-inspired modulation examples, all on the same time scale."""
    cases = [("switched_carrier", 0.1), ("switched_carrier", 0.5),
             ("switched_carrier", 0.85), ("bpsk", 0.85), ("ask", 1.), ("fsk", 1.)]
    fig, axes = plt.subplots(3, 2, figsize=(12, 7), sharex=True, sharey=True, layout="constrained")
    for ax, (kind, duty) in zip(axes.flat, cases):
        settings = make_scene(221, kind, strength=2, cubes=1)
        event = settings["events"][0]
        event.update(start_s=0., stop_s=4096*DT, channel=64.25,
                     pulse_s=duty*event["period_s"], symbol_rate=30000.)
        power = simulate_cube(settings)["injected_power"][:512, 0, :, 0]
        view = power[:, 56:74]
        im = ax.imshow(np.log10(1+view).T, origin="upper", aspect="auto", cmap="magma",
                       vmin=0, vmax=1, extent=[0, 512*DT*1000, 73.5, 55.5])
        name = {"switched_carrier": "Switched carrier", "bpsk": "Binary phase-shift keying",
                "ask": "Amplitude-shift keying", "fsk": "Frequency-shift keying"}[kind]
        ax.set_title(f"{name} · transmitter on {duty:.0%}")
    fig.supxlabel("Time (ms)")
    fig.supylabel("Channel index")
    fig.colorbar(im, ax=axes, label="log₁₀(1 + injected power / count²)", shrink=0.7)
    fig.suptitle("Duty cycle and modulation · channelized signals, without receiver noise")
    return fig


def plot_rfi_gallery():
    scenes = [make_scene(80+i, family, strength=1, cubes=1) for i, family in enumerate(FAMILIES)]
    scenes[1]["events"][0].update(strength=20, start_s=0.02, stop_s=0.12, pulse_s=100e-6, period_s=0.006)
    scenes[-1]["events"][0].update(strength=10, start_s=0.07, stop_s=0.0704)
    mixed = make_scene(99, "clean", cubes=1)
    mixed["events"] = [s["events"][0] for s in scenes]
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, sharey=True, layout="constrained")
    for scene, ax, title in zip(scenes+[mixed], axes.flat, [NAMES[k] for k in FAMILIES]+["All five waveforms together"]):
        field = simulate_cube(scene)["injected_power"].mean(axis=(1, 3))
        view = average_power(field, 16)
        freq = np.asarray(scene["frequency_hz"])/1e6
        im = ax.imshow(np.log10(1+view).T, aspect="auto", origin="upper", cmap="magma",
                       vmin=0, vmax=1.4, extent=[0, len(field)*DT*1000, freq[-1], freq[0]])
        ax.set_title(title, fontsize=11)
    fig.supxlabel("Time (ms)")
    fig.supylabel("Frequency (MHz)")
    fig.colorbar(im, ax=axes, label="log₁₀(1 + injected power / count²)", shrink=0.7)
    fig.suptitle("Injected signals · 0.524 ms display pixels · receiver noise omitted")
    return fig


def satellite_pass():
    """Mean carrier power for a straight overhead pass and an isotropic receiver."""
    time = np.arange(-300, 300.01, 0.25)
    height, speed, carrier, c = 650e3, 7500., 1.35e9, 299792458.
    distance = np.sqrt(height**2 + (speed*time)**2)
    doppler = -carrier/c * speed**2*time/distance
    bins = np.arange(-4, 5)
    offset = doppler[:, None]*DT - bins[None, :]
    response = (np.sinc(offset)/np.sinc(offset/N_CHANNELS))**2
    power = response * (height/distance[:, None])**2
    return time, bins/DT/1000, power, doppler/1000


def plot_satellite_pass():
    time, frequency, power, doppler = satellite_pass()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), layout="constrained")
    axes[0].plot(time/60, doppler, color="#1c7f8b")
    axes[0].set(xlabel="Time from closest approach (min)", ylabel="Doppler shift (kHz)",
                title="One carrier during a ten-minute pass")
    im = axes[1].imshow(power.T, origin="lower", aspect="auto", cmap="magma", interpolation="nearest",
                       extent=[time[0]/60, time[-1]/60, frequency[0]-15.26, frequency[-1]+15.26])
    axes[1].set(xlabel="Time from closest approach (min)", ylabel="Offset from carrier (kHz)",
                title="The same pass in 30.52 kHz channels")
    fig.colorbar(im, ax=axes[1], label="Relative mean power")
    return fig
