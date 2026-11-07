"""Shared grid workflow, extracted from the team notebook.

Authors: CS163 RFI team. Historical protocol and checkpoint tensor names retained.
"""

import os
import sys
import json
import random
import hashlib
import platform
import time
import uuid
import copy
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from IPython.display import display
from cs163_rfi import load_sample, load_recording, flag_rfi, score_mask
from cs163_rfi.plotting import plot_sample, plot_errors

torch.set_num_threads(1)
plt.rcParams.update({"font.size": 11, "figure.dpi": 110,
                     "axes.spines.top": False, "axes.spines.right": False})
METHODS = ("sk", "bandpass", "sk-bandpass")
BASELINE_FAR = 1e-4
WORKFLOW_VERSION = "rfi-team-1.2-content-check"
REFERENCE_ZIP_SHA256 = "a6a2d54877d4aafcc92e7b7c2d19c4dba7751e14843a4f380b37ab9163d4fb09"
# Fingerprint of required relative names, uncompressed sizes and ZIP CRC32 values.
EXPECTED_MANIFEST_SHA256 = "77fa8c04edaad3ce75d7f578c8b5b2708a0a3b2585723ce0e593d55308c0fab7"

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def json_safe(value):
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, Path): return str(value)
    return value

def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(json_safe(obj), indent=2, allow_nan=False))
    os.replace(tmp, path)

def atomic_torch_save(payload, path):
    path = Path(path)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)

from cs163_rfi import paths

def checkpoint_dir(run_dir):
    """Mirror a results folder under the external checkpoint root."""
    folder = paths.CHECKPOINT_DIR / Path(run_dir).resolve().relative_to(paths.RESULTS_DIR.resolve())
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def inspect_dataset(root, data, archive_path):
    root, data = Path(root), Path(data)
    archive_hash = manifest_hash = None
    required = {}
    if archive_path is not None:
        archive_hash = sha256_file(archive_path)
        assert zipfile.is_zipfile(archive_path), "Upload a complete, valid starter ZIP."
        prefix = "cs163-rfi-starter-v1.0/"
        with zipfile.ZipFile(archive_path) as z:
            required = {}
            for member in z.infolist():
                if member.is_dir() or not member.filename.startswith(prefix):
                    continue
                relative = member.filename[len(prefix):]
                if relative.startswith(("data/", "src/", "vendor/")) or relative == "pyproject.toml":
                    assert relative not in required, f"Duplicate archive entry: {relative}"
                    required[relative] = member
            manifest = {name: [m.file_size, m.CRC] for name, m in required.items()}
            manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            assert manifest_hash == EXPECTED_MANIFEST_SHA256, (
                f"Starter contents differ: found {len(required)} required entries; expected 831. "
                f"Uploaded ZIP SHA256={archive_hash}; contents fingerprint={manifest_hash}. "
                "Check that you uploaded the inner cs163-rfi-starter-v1.0.zip from the supplied bundle. "
                "Missing/changed files or a different starter release cannot be accepted automatically.")
            if archive_hash != REFERENCE_ZIP_SHA256:
                print("ZIP packaging differs, but all 831 required file sizes/checksums match the reference.")
            # Verify extracted bytes too: stale extraction must not silently supply training data.
            for relative, member in required.items():
                if not relative.startswith("data/"):
                    continue  # Refactored source is versioned in git, not the starter archive.
                target = (data.parent.parent if relative.startswith("data/") else root) / relative
                assert target.is_file() and target.stat().st_size == member.file_size, f"Missing/stale file: {target}"
                crc = 0
                with target.open("rb") as f:
                    for block in iter(lambda: f.read(1024 * 1024), b""):
                        crc = zlib.crc32(block, crc)
                assert (crc & 0xffffffff) == member.CRC, f"Modified data/source: {target}"
    catalog = pd.read_csv(data / "dataset.csv")
    assert len(catalog) == 200 and catalog.sample_id.is_unique
    assert catalog.groupby("split").size().to_dict() == {"test": 40, "train": 100, "validation": 60}
    assert catalog.path.is_unique
    checked = 0
    validation_positives = 0
    for row in catalog.itertuples(index=False):
        folder = (data / row.path).resolve()
        assert folder.is_relative_to(data.resolve())
        raw = np.load(folder / "power.npy", mmap_mode="r")
        packed = np.load(folder / "rfi_mask.npy", mmap_mode="r")
        meta = json.loads((folder / "recording.json").read_text())
        assert raw.shape == (4096, 10, 128, 1) and raw.dtype == np.uint8
        assert packed.shape == (4096, 10, 16, 1) and packed.dtype == np.uint8
        assert np.isclose(meta["dt_s"], 32.768e-6)
        assert np.isclose(meta["label_noise_fraction"], 0.01)
        labels = np.unpackbits(packed, axis=2, count=128).astype(bool)
        assert np.isclose(labels.mean(), row.labelled_fraction, rtol=1e-5, atol=1e-8)
        if row.rfi_type in ("clean", "sky"): assert not labels.any()
        if row.split == "validation": validation_positives += int(labels.sum())
        checked += 1
    assert validation_positives == 5799220
    # Verify the supplied loader against native stored bytes, not just shape.
    row = catalog.set_index("sample_id").loc[18]
    raw = np.load(data / row.path / "power.npy", mmap_mode="r")
    packed = np.load(data / row.path / "rfi_mask.npy", mmap_mode="r")
    s = load_sample(data, 18, antenna=0)
    np.testing.assert_array_equal(s["power"], raw[:, 0, :, 0].astype(np.float32))
    np.testing.assert_array_equal(s["rfi_mask"], np.unpackbits(packed, axis=2, count=128)[:, 0, :, 0].astype(bool))
    report = {"archive_sha256": archive_hash, "required_content_manifest_sha256": manifest_hash,
              "required_files_checked": len(required), "recordings_checked": checked,
              "validation_positive_cells": validation_positives,
              "catalog_sha256": sha256_file(data / "dataset.csv"),
              "shape_per_recording": [4096, 10, 128, 1]}
    return data, catalog, report

def fit_training_normalization(data, catalog):
    # Exact preprocessing from the earlier notebook: pool ALL noise-only
    # TRAINING recordings across time, antennas, frequency and feed.
    clean_train = catalog.query("split == 'train' and rfi_type == 'clean'")
    assert len(clean_train) == 5
    n_values = 0; total = 0.0; squared_total = 0.0
    for row in clean_train.itertuples():
        raw = load_recording(data / row.path)["power"].astype(np.float64)
        n_values += raw.size
        total += raw.sum()
        squared_total += np.square(raw).sum()
    mean = float(total / n_values)
    std = float(np.sqrt(max(squared_total / n_values - mean**2, 0)))
    assert std > 0
    np.testing.assert_allclose([mean, std], [4.257218322753906, 4.2564571117143295], rtol=1e-10)
    return mean, std

class AntennaDataset(Dataset):
    def __init__(self, data, frame, mean, std):
        self.data, self.mean, self.std = Path(data), mean, std
        self.pairs = [(int(row.sample_id), antenna)
                      for row in frame.itertuples() for antenna in range(10)]
    def __len__(self): return len(self.pairs)
    def __getitem__(self, index):
        sample_id, antenna = self.pairs[index]
        s = load_sample(self.data, sample_id, antenna=antenna)
        assert s["power"].shape == s["rfi_mask"].shape == (4096, 128)
        x = torch.from_numpy((s["power"] - self.mean) / self.std).unsqueeze(0)
        y = torch.from_numpy(s["rfi_mask"].astype(np.float32)).unsqueeze(0)
        return x, y

def quantize_probability(probability, bins):
    p = np.asarray(probability)
    assert bins >= 2 and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    # Integer scores make histogram calibration, PR and hard masks agree exactly.
    return np.floor(p.astype(np.float64) * (bins - 1)).astype(np.int64)

def probability_histograms(probability, truth, bins):
    assert truth.dtype == np.bool_ and truth.shape == probability.shape
    q = quantize_probability(probability, bins)
    return (np.bincount(q[truth], minlength=bins), np.bincount(q[~truth], minlength=bins))

def precision_recall_from_histograms(positive, negative):
    """Let scikit-learn compute PR/AP; histogram counts are sample weights."""
    positive, negative = np.asarray(positive), np.asarray(negative)
    assert positive.shape == negative.shape and (positive >= 0).all() and (negative >= 0).all()
    if positive.sum() == 0:
        # Recall/AP are undefined when this family contains no positive labels.
        return {"precision": np.array([1.0]), "recall": np.array([np.nan]), "ap": np.nan}
    bins = len(positive)
    labels = np.r_[np.ones(bins), np.zeros(bins)]
    scores = np.tile(np.arange(bins), 2)
    weights = np.r_[positive, negative]
    present = weights > 0  # At most 2*bins observations; never expand millions of cells.
    labels, scores, weights = labels[present], scores[present], weights[present]
    precision, recall, _ = precision_recall_curve(labels, scores, sample_weight=weights)
    ap = average_precision_score(labels, scores, sample_weight=weights)
    return {"precision": precision[::-1], "recall": recall[::-1], "ap": float(ap)}

def calibrate_cut(negative_histogram, target_rate):
    assert 0 <= target_rate <= 1 and negative_histogram.sum() > 0
    # cut in [0,bins]; predict quantized_score >= cut. bins means no flags.
    tails = np.r_[np.cumsum(negative_histogram[::-1], dtype=np.int64)[::-1], 0]
    rates = tails / negative_histogram.sum()
    cut = int(np.flatnonzero(rates <= target_rate)[0])
    return cut, float(rates[cut])

def threshold_mask(probability, cut, bins):
    assert 0 <= cut <= bins
    return quantize_probability(probability, bins) >= cut

def pooled_metrics(frame, groups):
    table = frame.groupby(groups)[["tp", "fp", "fn", "tn"]].sum()
    table["precision"] = table.tp / (table.tp + table.fp).replace(0, np.nan)
    table["recall"] = table.tp / (table.tp + table.fn).replace(0, np.nan)
    table["false_positive_rate"] = table.fp / (table.fp + table.tn).replace(0, np.nan)
    table["removed"] = (table.tp + table.fp) / (table.tp + table.fp + table.fn + table.tn)
    if {"power_sum", "retained_power_sum"}.issubset(frame.columns):
        energy = frame.groupby(groups)[["power_sum", "retained_power_sum"]].sum()
        table["measured_power_retained"] = energy.retained_power_sum / energy.power_sum.replace(0, np.nan)
    return table

def whole_mask_ap50(frame):
    # Same EXPLICIT custom definition as Week1: one entire binary mask per
    # antenna/recording, one foreground class, all detections in ONE tied group.
    union = frame.tp + frame.fp + frame.fn
    iou = frame.tp / union.replace(0, np.nan)
    targets = int(((frame.tp + frame.fn) > 0).sum())
    detections = int(((frame.tp + frame.fp) > 0).sum())
    matches = int((iou >= 0.5).sum())
    p = matches / detections if detections else 0.0
    r = matches / targets if targets else np.nan
    ap = float(np.where(np.linspace(0, 1, 101) <= r + 1e-12, p, 0.0).mean()) if targets else np.nan
    return {"targets": targets, "detections": detections, "matches": matches,
            "precision50": p, "recall50": r, "custom_whole_mask_mAP50": ap}


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True))
    def forward(self, x): return self.layers(x)

class SegmentationModel(nn.Module):
    """Two-level U-Net; use_skips=False gives the matched no-skip CNN."""
    def __init__(self, base=8, use_skips=True):
        super().__init__()
        self.use_skips = use_skips
        self.enc1 = DoubleConv(1, base)
        self.enc2 = DoubleConv(base, 2*base)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(2*base, 4*base)
        self.up2 = nn.ConvTranspose2d(4*base, 2*base, 2, stride=2)
        self.dec2 = DoubleConv(4*base if use_skips else 2*base, 2*base)
        self.up1 = nn.ConvTranspose2d(2*base, base, 2, stride=2)
        self.dec1 = DoubleConv(2*base if use_skips else base, base)
        self.output = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        z = self.bottleneck(self.pool(e2))
        z = self.up2(z)
        z = self.dec2(torch.cat((z, e2), dim=1) if self.use_skips else z)
        z = self.up1(z)
        z = self.dec1(torch.cat((z, e1), dim=1) if self.use_skips else z)
        return self.output(z)

def loss_value(logits, target, loss_name):
    assert logits.shape == target.shape
    if loss_name == "bce": return nn.functional.binary_cross_entropy_with_logits(logits, target)
    if loss_name == "mse": return nn.functional.mse_loss(logits.sigmoid(), target)
    if loss_name == "l1": return nn.functional.l1_loss(logits.sigmoid(), target)
    raise ValueError("LOSS must be bce, mse, or l1; MSE already covers squared L2.")

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def build_training_state(config, data, catalog, mean, std, device):
    seed_everything(config["seed"])
    model = SegmentationModel(config["base_channels"], config["architecture"] == "unet").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    train_generator = torch.Generator().manual_seed(config["seed"])
    val_generator = torch.Generator().manual_seed(config["seed"] + 1)
    train = AntennaDataset(data, catalog.query("split == 'train'"), mean, std)
    validation = AntennaDataset(data, catalog.query("split == 'validation'"), mean, std)
    train_loader = DataLoader(train, batch_size=config["batch_size"], shuffle=True,
                              num_workers=0, generator=train_generator)
    val_loader = DataLoader(validation, batch_size=config["batch_size"], shuffle=False,
                            num_workers=0, generator=val_generator)
    return model, optimizer, train_loader, val_loader, train_generator, val_generator

def rng_state(train_generator, val_generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "train_generator": train_generator.get_state(), "val_generator": val_generator.get_state()}

def restore_rng(state, train_generator, val_generator):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("GPU count changed; exact RNG resume requires the same device count.")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    train_generator.set_state(state["train_generator"].cpu())
    val_generator.set_state(state["val_generator"].cpu())

def cpu_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def run_epoch(model, loader, device, loss_name, optimizer=None, bins=65536):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0; pixels = 0
    ph = np.zeros(bins, np.int64); nh = np.zeros(bins, np.int64)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = loss_value(logits, y, loss_name)
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite training/validation loss")
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        loss_sum += loss.item() * y.numel(); pixels += y.numel()
        if not training:
            probability = logits.detach().sigmoid().cpu().numpy()
            truth = y.cpu().numpy().astype(bool)
            a, b = probability_histograms(probability, truth, bins)
            ph += a; nh += b
    assert pixels > 0
    ap = precision_recall_from_histograms(ph, nh)["ap"] if not training else np.nan
    return loss_sum / pixels, ap

def train_to_plateau(model, optimizer, train_loader, val_loader, train_generator, val_generator,
                     config, run_dir, device, normalization, max_epochs=100, resume=False):
    run_dir = Path(run_dir); ckpt_dir = checkpoint_dir(run_dir)
    history = []; best_ap = -np.inf; best_epoch = 0
    anchor_ap = -np.inf; anchor_epoch = 0; start_epoch = 1
    if resume:
        # Load only checkpoints created by this workflow in your own shared folder.
        ck = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
        assert ck["config"] == config and ck["normalization"] == normalization
        assert ck["device_type"] == device.type, "Resume on the same device type for reproducibility."
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        history = ck["history"]; best_ap = ck["best_ap"]; best_epoch = ck["best_epoch"]
        anchor_ap, anchor_epoch = ck["anchor_ap"], ck["anchor_epoch"]
        start_epoch = ck["epoch"] + 1
        restore_rng(ck["rng"], train_generator, val_generator)
        if ck["status"] == "validation_plateau":
            print("This run already met the stopping rule; loading its existing result.")
            return history, ck["status"]
    assert max_epochs >= start_epoch, "Increase MAX_EPOCHS to extend this unfinished trial, then resume the same grid."
    status = "budget_exhausted_not_converged"
    for epoch in range(start_epoch, max_epochs + 1):
        started = time.perf_counter()
        train_loss, _ = run_epoch(model, train_loader, device, config["loss"], optimizer,
                                  bins=config["probability_bins"])
        val_loss, val_ap = run_epoch(model, val_loader, device, config["loss"],
                                     bins=config["probability_bins"])
        assert np.isfinite(val_ap), "Validation must include positive RFI labels."
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                        "validation_pixel_ap_quantized": val_ap,
                        "seconds": time.perf_counter() - started})
        if val_ap > best_ap:
            best_ap, best_epoch = float(val_ap), epoch
            atomic_torch_save({"model_state_dict": cpu_state_dict(model), "config": config,
                               "normalization": normalization, "epoch": epoch,
                               "validation_ap": best_ap}, ckpt_dir / "best.pt")
        if val_ap > anchor_ap + config["min_ap_improvement"]:
            anchor_ap, anchor_epoch = float(val_ap), epoch
        plateau = epoch >= config["min_epochs"] and epoch - anchor_epoch >= config["patience"]
        status = "validation_plateau" if plateau else "budget_exhausted_not_converged"
        payload = {"model_state_dict": cpu_state_dict(model),
                   "optimizer_state_dict": optimizer.state_dict(), "config": config,
                   "normalization": normalization, "epoch": epoch, "history": history,
                   "best_ap": best_ap, "best_epoch": best_epoch,
                   "anchor_ap": anchor_ap, "anchor_epoch": anchor_epoch,
                   "device_type": device.type,
                   "rng": rng_state(train_generator, val_generator), "status": status}
        atomic_torch_save(payload, ckpt_dir / "last.pt")
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        write_json(run_dir / "training_status.json", {"status": status, "last_epoch": epoch,
                    "best_epoch": best_epoch, "best_validation_ap": best_ap})
        print(f"Epoch {epoch}: train={train_loss:.6f}, val={val_loss:.6f}, AP={val_ap:.5f}; saved {run_dir.name}")
        if plateau: break
    return history, status

@torch.inference_mode()
def predict_probability(model, raw_power, normalization, device):
    model.eval()
    x = torch.from_numpy((raw_power.astype(np.float32) - normalization["mean"]) / normalization["std"])
    return model(x[None, None].to(device)).sigmoid()[0, 0].cpu().numpy()

def calibrate_on_clean_validation(model, data, catalog, normalization, device, bins, checkpoint_sha):
    clean = catalog.query("split == 'validation' and rfi_type == 'clean'")
    assert len(clean) == 3
    negative_hist = np.zeros(bins, np.int64)
    baseline_counts = {m: [0, 0] for m in METHODS}
    for row in clean.itertuples():
        rec = load_recording(data / row.path)
        assert not rec["rfi_mask"].any()
        for method in METHODS:
            flags = flag_rfi(rec["power"], method=method, far=BASELINE_FAR)
            baseline_counts[method][0] += int(flags.sum())
            baseline_counts[method][1] += flags.size
        for antenna in range(10):
            p = predict_probability(model, rec["power"][:, antenna, :, 0], normalization, device)
            negative_hist += np.bincount(quantize_probability(p, bins).ravel(), minlength=bins)
    # Same target for ALL architectures/losses: measured combined-baseline clean FPR.
    target = baseline_counts["sk-bandpass"][0] / baseline_counts["sk-bandpass"][1]
    cut, achieved = calibrate_cut(negative_hist, target)
    return {"calibration_split": "validation", "calibration_family": "clean",
            "clean_sample_ids": clean.sample_id.astype(int).tolist(),
            "probability_bins": bins, "quantized_cut": cut,
            "decision_rule": "floor(sigmoid(logit)*(bins-1)) >= quantized_cut",
            "probability_lower_bound": cut / (bins - 1) if cut < bins else None,
            "target_clean_fpr": target, "achieved_clean_fpr": achieved,
            "baseline_clean_fpr": {m: a / b for m, (a, b) in baseline_counts.items()},
            "best_checkpoint_sha256": checkpoint_sha}, negative_hist

def evaluate_split(model, data, frame, normalization, device, bins, cut):
    positive = np.zeros(bins, np.int64); negative = np.zeros(bins, np.int64)
    family_hist = {f: [np.zeros(bins, np.int64), np.zeros(bins, np.int64)] for f in frame.rfi_type.unique()}
    rows = []
    for number, row in enumerate(frame.itertuples(), 1):
        rec = load_recording(data / row.path)
        baseline = {m: flag_rfi(rec["power"], method=m, far=BASELINE_FAR) for m in METHODS}
        for antenna in range(10):
            raw = rec["power"][:, antenna, :, 0]
            truth = rec["rfi_mask"][:, antenna, :, 0]
            before = time.perf_counter()
            p = predict_probability(model, raw, normalization, device)
            inference_seconds = time.perf_counter() - before
            ph, nh = probability_histograms(p, truth, bins)
            positive += ph; negative += nh
            family_hist[row.rfi_type][0] += ph; family_hist[row.rfi_type][1] += nh
            predictions = {m: baseline[m][:, antenna, :, 0] for m in METHODS}
            predictions["model_matched_clean_fpr"] = threshold_mask(p, cut, bins)
            for method, mask in predictions.items():
                result = score_mask(mask, truth)
                rows.append({"sample_id": int(row.sample_id), "antenna": antenna,
                             "split": row.split, "rfi_type": row.rfi_type, "method": method,
                             "power_sum": float(raw.sum(dtype=np.float64)),
                             "retained_power_sum": float(raw[~mask].sum(dtype=np.float64)),
                             "model_inference_seconds": inference_seconds if method == "model_matched_clean_fpr" else np.nan,
                             **result})
        if number % 10 == 0: print(f"Evaluated {number}/{len(frame)} {frame.iloc[0].split} recordings")
    return pd.DataFrame(rows), positive, negative, family_hist

def save_evaluation(results, positive, negative, family_hist, out, title, config, calibration):
    out = Path(out); out.mkdir(exist_ok=True, parents=True)
    results.to_csv(out / "per_antenna.csv", index=False)
    overall = pooled_metrics(results, "method")
    family = pooled_metrics(results, ["rfi_type", "method"])
    overall.to_csv(out / "metrics_overall.csv"); family.to_csv(out / "metrics_by_family.csv")
    rows = []
    for name, group in [("overall", results), *list(results.groupby("rfi_type"))]:
        for method, data in group.groupby("method"):
            rows.append({"rfi_type": name, "method": method, **whole_mask_ap50(data)})
    map50 = pd.DataFrame(rows)
    map50.to_csv(out / "custom_whole_mask_AP50.csv", index=False)
    pr = precision_recall_from_histograms(positive, negative)
    np.savez(out / "pixel_pr_histograms.npz", positive=positive, negative=negative,
             bins=config["probability_bins"])
    family_ap = []
    for name, (ph, nh) in family_hist.items():
        family_ap.append({"rfi_type": name, "pixel_ap_quantized": precision_recall_from_histograms(ph, nh)["ap"]})
    pd.DataFrame(family_ap).to_csv(out / "model_pixel_AP_by_family.csv", index=False)
    pd.DataFrame({"recall": pr["recall"], "precision": pr["precision"]}).to_csv(out / "pixel_PR_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    ax.plot(pr["recall"], pr["precision"], label=f"Model, quantized pixel AP={pr['ap']:.4f}")
    for method in METHODS:
        point = overall.loc[method]
        ax.scatter(point.recall, point.precision, label=method, s=50)
    ax.set(xlabel="Native-pixel recall", ylabel="Native-pixel precision",
           title=title, xlim=(0, 1.02), ylim=(0, 1.02))
    ax.legend(); ax.grid(alpha=.2)
    for ext in ["png", "svg"]: fig.savefig(out / f"precision_recall.{ext}", dpi=300)
    plt.show(); plt.close(fig)
    controls = family.loc[family.index.get_level_values("rfi_type").isin(["clean", "sky"])]
    controls.to_csv(out / "controls.csv")
    summary = {"pixel_ap_quantized": pr["ap"], "probability_bins": config["probability_bins"],
               "recordings": int(results.sample_id.nunique()), "antenna_pairs": int(len(results) // 4),
               "positive_cells": int(positive.sum()), "negative_cells": int(negative.sum()),
               "model_mean_inference_seconds": float(results.loc[results.method.eq("model_matched_clean_fpr"), "model_inference_seconds"].mean()),
               "calibration": calibration,
               "metric_note": "Pixel AP is exact for quantized probabilities; approximate for unquantized model scores. Whole-mask AP50 is a custom single-tied-group metric, not pixel AP or a standard instance benchmark. Measured power retained includes noise, not isolated astronomical flux."}
    write_json(out / "summary.json", summary)
    display(overall[["precision", "recall", "false_positive_rate", "removed"]])
    display(map50.query("rfi_type == 'overall'"))
    with pd.option_context("display.max_rows", None):
        display(family[["precision", "recall", "false_positive_rate", "removed"]])
    display(controls[["false_positive_rate", "removed", "measured_power_retained"]])
    return overall, family, map50

def plot_examples(model, results, data, normalization, device, bins, cut, out):
    model_rows = results.loc[results.method.eq("model_matched_clean_fpr") & ((results.tp + results.fn) > 0)].copy()
    model_rows["iou"] = model_rows.tp / (model_rows.tp + model_rows.fp + model_rows.fn)
    selected = []
    if len(model_rows):
        selected = [("highest_IoU", model_rows.loc[model_rows.iou.idxmax()]),
                    ("lowest_IoU", model_rows.loc[model_rows.iou.idxmin()])]
    sky = results.loc[results.method.eq("model_matched_clean_fpr") & results.rfi_type.eq("sky")]
    if len(sky): selected.append(("astronomical_control", sky.iloc[0]))
    for label, row in selected:
        s = load_sample(data, int(row.sample_id), antenna=int(row.antenna))
        p = predict_probability(model, s["power"], normalization, device)
        flags = threshold_mask(p, cut, bins)
        fig = plot_sample(s, flags, factor=16)
        fig.suptitle(f"{label}: recording {int(row.sample_id)}, A{int(row.antenna):02d}; selected examples, not typical performance")
        fig.savefig(Path(out) / f"example_{label}.png", dpi=300, bbox_inches="tight")
        plt.show(); plt.close(fig)

def initialize_grid(save_root, owner, protocol, losses, learning_rates, resume_dir=""):
    """One manifest per person/grid; never share a grid directory between writers."""
    root = Path(save_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    grid_spec = {"owner": owner, "protocol": protocol,
                 "losses": list(losses), "learning_rates": list(learning_rates)}
    if resume_dir:
        grid_dir = Path(resume_dir).resolve()
        assert grid_dir.is_relative_to(root), "GRID_RESUME_DIR must be inside the team runs folder."
        manifest = json.loads((grid_dir / "grid_manifest.json").read_text())
        assert manifest["spec"] == grid_spec, "Grid configuration changed. Restore it or start a new grid."
    else:
        safe_owner = "".join(c if c.isalnum() or c in "_-" else "_" for c in owner)
        assert safe_owner
        grid_dir = root / (f"grid_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{safe_owner}_"
                           f"{protocol['architecture']}_b{protocol['base_channels']}_{uuid.uuid4().hex[:8]}")
        grid_dir.mkdir()
        manifest = {"spec": grid_spec, "trials": {}}
        for loss, lr in product(losses, learning_rates):
            key = f"{loss}_lr{lr:g}"
            config = dict(protocol, loss=loss, learning_rate=float(lr))
            trial_dir = grid_dir / key
            trial_dir.mkdir()
            write_json(trial_dir / "config.json", config)
            manifest["trials"][key] = {"config": config, "status": "pending"}
        write_json(grid_dir / "grid_manifest.json", manifest)
    return grid_dir, manifest


def save_grid_summary(grid_dir, manifest):
    rows = []
    for key, entry in manifest["trials"].items():
        config = entry["config"]
        rows.append({"trial": key, "architecture": config["architecture"],
                     "base_channels": config["base_channels"], "loss": config["loss"],
                     "learning_rate": config["learning_rate"], "status": entry["status"],
                     **entry.get("metrics", {}), "run_dir": str(Path(grid_dir) / key)})
    table = pd.DataFrame(rows)
    # This file belongs to one grid, so other teammates never append to it.
    tmp = Path(grid_dir) / f"summary_{uuid.uuid4().hex}.tmp"
    table.to_csv(tmp, index=False)
    os.replace(tmp, Path(grid_dir) / "grid_summary.csv")
    write_json(Path(grid_dir) / "grid_summary.json", rows)
    return table


def evaluate_trained_trial(model, data, catalog, normalization, device, config, run_dir, split="validation"):
    run_dir = Path(run_dir)
    status = json.loads((run_dir / "training_status.json").read_text())["status"]
    assert status == "validation_plateau", "Resume training until the stopping rule is met first."
    best = torch.load(checkpoint_dir(run_dir) / "best.pt", map_location="cpu", weights_only=False)
    assert best["config"] == config and best["normalization"] == normalization
    model.load_state_dict(best["model_state_dict"])
    model.eval()
    checkpoint_sha = sha256_file(checkpoint_dir(run_dir) / "best.pt")
    bins = config["probability_bins"]
    if split == "validation":
        calibration, histogram = calibrate_on_clean_validation(
            model, data, catalog, normalization, device, bins, checkpoint_sha)
        write_json(run_dir / "calibration.json", calibration)
        np.save(run_dir / "clean_validation_negative_histogram.npy", histogram)
    else:
        assert split == "test"
        calibration = json.loads((run_dir / "calibration.json").read_text())
    assert calibration["best_checkpoint_sha256"] == checkpoint_sha
    assert calibration["probability_bins"] == bins
    frame = catalog.loc[catalog.split.eq(split)].copy()
    assert len(frame) == (60 if split == "validation" else 40)
    results, ph, nh, family_hist = evaluate_split(
        model, data, frame, normalization, device, bins, calibration["quantized_cut"])
    assert len(results) == len(frame) * 10 * 4
    assert results.groupby("method").sample_id.nunique().eq(len(frame)).all()
    if split == "validation":
        assert int(ph.sum()) == 5799220 and int(ph.sum() + nh.sum()) == 314572800
    out = run_dir / f"evaluation_{split}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}"
    overall, family, ap50 = save_evaluation(
        results, ph, nh, family_hist, out,
        f"{config['architecture']}, b={config['base_channels']}, {config['loss']}, lr={config['learning_rate']:g} · {split}",
        config, calibration)
    parameters = sum(p.numel() for p in model.parameters())
    write_json(out / "run_metadata.json", {"config": config, "normalization": normalization,
               "training_status": status, "selected_epoch": best["epoch"],
               "parameters": parameters, "evaluation_split": split})
    plot_examples(model, results, data, normalization, device, bins, calibration["quantized_cut"], out)
    model_metrics = overall.loc["model_matched_clean_fpr"]
    ap50_row = ap50.loc[ap50.rfi_type.eq("overall") & ap50.method.eq("model_matched_clean_fpr")].iloc[0]
    summary = json.loads((out / "summary.json").read_text())
    return {"evaluation_dir": str(out), "selected_epoch": best["epoch"], "parameters": parameters,
            f"{split}_pixel_ap_quantized": summary["pixel_ap_quantized"],
            "precision": model_metrics.precision, "recall": model_metrics.recall,
            "false_positive_rate": model_metrics.false_positive_rate,
            "removal_fraction": model_metrics.removed, "custom_whole_mask_map50": ap50_row["custom_whole_mask_mAP50"],
            "clean_fpr_target": calibration["target_clean_fpr"],
            "clean_fpr_achieved": calibration["achieved_clean_fpr"]}


def train_one_trial(config, run_dir, data, catalog, normalization, device, max_epochs):
    """Start independently, or restore this trial's last completed epoch."""
    model, optimizer, train_loader, val_loader, train_generator, val_generator = build_training_state(
        config, data, catalog, normalization["mean"], normalization["std"], device)
    assert len(train_loader.dataset) == 1000 and len(val_loader.dataset) == 600
    with torch.inference_mode():
        assert model(torch.zeros(1, 1, 4096, 128, device=device)).shape == (1, 1, 4096, 128)
    write_json(run_dir / f"environment_{uuid.uuid4().hex[:8]}.json",
               {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "parameters": sum(p.numel() for p in model.parameters()), "max_epochs_this_session": max_epochs})
    history, status = train_to_plateau(
        model, optimizer, train_loader, val_loader, train_generator, val_generator,
        config, run_dir, device, normalization, max_epochs=max_epochs,
        resume=(checkpoint_dir(run_dir) / "last.pt").is_file())
    return model, history, status


def save_training_curves(history, config, run_dir):
    """Show loss within this trial and AP for comparison across trials."""
    best = torch.load(checkpoint_dir(run_dir) / "best.pt", map_location="cpu", weights_only=False)
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    axes[0].plot(h.epoch, h.train_loss, label="Train")
    axes[0].plot(h.epoch, h.validation_loss, label="Validation")
    axes[0].set(xlabel="Epoch", ylabel=f"Mean {config['loss']} loss")
    axes[0].legend()
    axes[1].plot(h.epoch, h.validation_pixel_ap_quantized)
    axes[1].axvline(best["epoch"], linestyle="--", color="grey")
    axes[1].set(xlabel="Epoch", ylabel="Validation pixel AP (quantized)")
    fig.savefig(run_dir / "training_curves.png", dpi=300)
    plt.show()
    plt.close(fig)


def run_six_trial_grid(grid_dir, manifest, data, catalog, normalization, device, max_epochs):
    grid_dir = Path(grid_dir)
    print("GRID DIRECTORY — keep this path to resume after a reset:", grid_dir, flush=True)
    print("Run only one writer for this grid directory.", flush=True)
    save_grid_summary(grid_dir, manifest)
    for number, (key, entry) in enumerate(manifest["trials"].items(), 1):
        if entry["status"] == "complete":
            out = Path(entry["metrics"]["evaluation_dir"])
            assert (out / "summary.json").is_file() and (checkpoint_dir(grid_dir / key) / "best.pt").is_file(), "Completed files missing."
            print(f"[{number}/6] {key}: already complete; skipping.", flush=True)
            continue
        config = entry["config"]
        run_dir = grid_dir / key
        assert json.loads((run_dir / "config.json").read_text()) == config
        entry["status"] = "running"
        write_json(grid_dir / "grid_manifest.json", manifest)
        print(f"[{number}/6] {key}: training/resuming in {run_dir}", flush=True)
        try:
            model, history, status = train_one_trial(
                config, run_dir, data, catalog, normalization, device, max_epochs)
            if status != "validation_plateau":
                entry["status"] = status
                print(f"PAUSED at {key}: stopping rule not met. Increase MAX_EPOCHS, keep this GRID_DIRECTORY, and rerun.", flush=True)
                break
            save_training_curves(history, config, run_dir)
            entry["metrics"] = evaluate_trained_trial(model, data, catalog, normalization, device, config, run_dir)
            entry["status"] = "complete"
            print(f"[{number}/6] {key}: checkpoints, figures and metrics saved.", flush=True)
        except BaseException as exc:
            entry["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
            entry["last_error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            write_json(grid_dir / "grid_manifest.json", manifest)
            save_grid_summary(grid_dir, manifest)
            # Discard the previous model/optimizer before allocating the next trial.
            if "model" in locals():
                del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return save_grid_summary(grid_dir, manifest)
