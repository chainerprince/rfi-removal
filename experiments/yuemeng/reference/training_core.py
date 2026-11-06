import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
import random
import time
import uuid
import shutil
from datetime import datetime, timezone
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from cs163_rfi import load_sample, load_recording, flag_rfi, score_mask

METHODS = ("sk", "bandpass", "sk-bandpass")
BASELINE_FAR = 1e-4
WORKFLOW_VERSION = "rfi-team-1.2-content-check"
ALLOW_CUDA_TO_MPS_RESUME = False


def settings_match(saved, current):
    return all(key in saved and saved[key] == value for key, value in current.items())


def training_settings(config):
    keys = ("architecture", "base_channels", "seed", "batch_size", "probability_bins",
            "min_epochs", "patience", "min_ap_improvement", "loss", "learning_rate")
    return {key: config[key] for key in keys}

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

def load_catalog(data):
    data = Path(data).resolve()
    catalog = pd.read_csv(data / "dataset.csv")
    if not catalog.sample_id.is_unique or not catalog.path.is_unique:
        raise ValueError("Each recording must have a unique ID and path")
    if not {"train", "validation"} <= set(catalog.split):
        raise ValueError("The catalog must contain training and validation recordings")
    if any(not (data / path).resolve().is_relative_to(data) for path in catalog.path):
        raise ValueError("Recording paths must be inside the dataset directory")
    return catalog

def fit_training_normalization(data, catalog):
    clean_train = catalog.query("split == 'train' and rfi_type == 'clean'")
    if clean_train.empty:
        raise ValueError("Normalization requires clean training recordings")
    n_values = 0; total = 0.0; squared_total = 0.0
    for row in clean_train.itertuples():
        raw = load_recording(data / row.path)["power"].astype(np.float64)
        n_values += raw.size
        total += raw.sum()
        squared_total += np.square(raw).sum()
    mean = float(total / n_values)
    std = float(np.sqrt(max(squared_total / n_values - mean**2, 0)))
    assert std > 0
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
    return np.floor(p.astype(np.float64) * (bins - 1)).astype(np.int64)

def probability_histograms(probability, truth, bins):
    assert truth.dtype == np.bool_ and truth.shape == probability.shape
    q = quantize_probability(probability, bins)
    return (np.bincount(q[truth], minlength=bins), np.bincount(q[~truth], minlength=bins))

def precision_recall_from_histograms(positive, negative):
    positive, negative = np.asarray(positive), np.asarray(negative)
    assert positive.shape == negative.shape and (positive >= 0).all() and (negative >= 0).all()
    if positive.sum() == 0:
        return {"precision": np.array([1.0]), "recall": np.array([np.nan]), "ap": np.nan}
    bins = len(positive)
    labels = np.r_[np.ones(bins), np.zeros(bins)]
    scores = np.tile(np.arange(bins), 2)
    weights = np.r_[positive, negative]
    present = weights > 0
    labels, scores, weights = labels[present], scores[present], weights[present]
    precision, recall, _ = precision_recall_curve(labels, scores, sample_weight=weights)
    ap = average_precision_score(labels, scores, sample_weight=weights)
    return {"precision": precision[::-1], "recall": recall[::-1], "ap": float(ap)}

def calibrate_cut(negative_histogram, target_rate):
    assert 0 <= target_rate <= 1 and negative_histogram.sum() > 0
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
            "train_generator": train_generator.get_state(), "val_generator": val_generator.get_state(),
            "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None}

def restore_rng(state, train_generator, val_generator):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("GPU count changed; exact RNG resume requires the same device count.")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    if state.get("mps") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())
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
                     config, run_dir, device, normalization, max_epochs=100, resume=False, *, keep_previous_best=False):
    run_dir = Path(run_dir)
    history = []; best_ap = -np.inf; best_epoch = 0
    anchor_ap = -np.inf; anchor_epoch = 0; start_epoch = 1
    if resume:
        ck = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        assert settings_match(ck["config"], config) and ck["normalization"] == normalization
        assert ck["device_type"] == device.type or (ALLOW_CUDA_TO_MPS_RESUME and ck["device_type"] == "cuda" and device.type == "mps"), "Cross-device resume requires explicit approval; numerical identity is not guaranteed."
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
            best_path = run_dir / "best.pt"
            if keep_previous_best and best_path.is_file():
                backup = run_dir / f"previous_best.{uuid.uuid4().hex}.tmp"
                shutil.copyfile(best_path, backup)
                os.replace(backup, run_dir / "previous_best.pt")
            atomic_torch_save({"model_state_dict": cpu_state_dict(model), "config": config,
                               "normalization": normalization, "epoch": epoch,
                               "validation_ap": best_ap}, best_path)
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
        atomic_torch_save(payload, run_dir / "last.pt")
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

def calibrate_on_clean_validation(model, data, catalog, normalization, device, bins, selected_epoch):
    clean = catalog.query("split == 'validation' and rfi_type == 'clean'")
    if clean.empty:
        raise ValueError("Calibration requires clean validation recordings")
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
    target = baseline_counts["sk-bandpass"][0] / baseline_counts["sk-bandpass"][1]
    cut, achieved = calibrate_cut(negative_hist, target)
    return {"calibration_split": "validation", "calibration_family": "clean",
            "clean_sample_ids": clean.sample_id.astype(int).tolist(),
            "probability_bins": bins, "quantized_cut": cut,
            "decision_rule": "floor(sigmoid(logit)*(bins-1)) >= quantized_cut",
            "probability_lower_bound": cut / (bins - 1) if cut < bins else None,
            "target_clean_fpr": target, "achieved_clean_fpr": achieved,
            "baseline_clean_fpr": {m: a / b for m, (a, b) in baseline_counts.items()},
            "selected_epoch": selected_epoch}, negative_hist

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

def save_evaluation(results, positive, negative, family_hist, out, config, calibration):
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
    controls = family.loc[family.index.get_level_values("rfi_type").isin(["clean", "sky"])]
    controls.to_csv(out / "controls.csv")
    summary = {"pixel_ap_quantized": pr["ap"], "probability_bins": config["probability_bins"],
               "recordings": int(results.sample_id.nunique()), "antenna_pairs": int(len(results) // 4),
               "positive_cells": int(positive.sum()), "negative_cells": int(negative.sum()),
               "model_mean_inference_seconds": float(results.loc[results.method.eq("model_matched_clean_fpr"), "model_inference_seconds"].mean()),
               "calibration": calibration,
               "metric_note": "Pixel AP is exact for quantized probabilities; approximate for unquantized model scores. Whole-mask AP50 is a custom single-tied-group metric, not pixel AP or a standard instance benchmark. Measured power retained includes noise, not isolated astronomical flux."}
    write_json(out / "summary.json", summary)
    return overall, family, map50

def initialize_grid(save_root, owner, protocol, losses, learning_rates, resume_dir=""):
    root = Path(save_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    grid_spec = {"owner": owner, "protocol": protocol,
                 "losses": list(losses), "learning_rates": list(learning_rates)}
    if resume_dir:
        grid_dir = Path(resume_dir).resolve()
        assert grid_dir.is_relative_to(root), "GRID_RESUME_DIR must be inside the team runs folder."
        manifest = json.loads((grid_dir / "grid_manifest.json").read_text())
        saved = manifest["spec"]
        if (any(saved[key] != grid_spec[key] for key in ("owner", "losses", "learning_rates"))
                or not settings_match(saved["protocol"], protocol)):
            raise ValueError("Grid configuration changed. Restore it or start a new grid.")
        saved["protocol"] = protocol
        for entry in manifest["trials"].values():
            config = entry["config"]
            entry["config"] = dict(protocol, loss=config["loss"], learning_rate=config["learning_rate"])
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
    tmp = Path(grid_dir) / f"summary_{uuid.uuid4().hex}.tmp"
    table.to_csv(tmp, index=False)
    os.replace(tmp, Path(grid_dir) / "grid_summary.csv")
    write_json(Path(grid_dir) / "grid_summary.json", rows)
    return table

def evaluate_trained_trial(model, data, catalog, normalization, device, config, run_dir, split="validation"):
    run_dir = Path(run_dir)
    status = json.loads((run_dir / "training_status.json").read_text())
    assert status["status"] == "validation_plateau", "Resume training until the stopping rule is met first."
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    assert best["epoch"] == status["best_epoch"]
    assert settings_match(best["config"], config) and best["normalization"] == normalization
    model.load_state_dict(best["model_state_dict"])
    model.eval()
    bins = config["probability_bins"]
    if split == "validation":
        calibration, histogram = calibrate_on_clean_validation(
            model, data, catalog, normalization, device, bins, best["epoch"])
        write_json(run_dir / "calibration.json", calibration)
        np.save(run_dir / "clean_validation_negative_histogram.npy", histogram)
    else:
        assert split == "test"
        calibration = json.loads((run_dir / "calibration.json").read_text())
    assert calibration.get("selected_epoch", best["epoch"]) == best["epoch"]
    assert calibration["probability_bins"] == bins
    frame = catalog.loc[catalog.split.eq(split)].copy()
    if frame.empty:
        raise ValueError(f"No recordings in the {split} split")
    results, ph, nh, family_hist = evaluate_split(
        model, data, frame, normalization, device, bins, calibration["quantized_cut"])
    out = run_dir / f"evaluation_{split}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}"
    overall, _, ap50 = save_evaluation(results, ph, nh, family_hist, out, config, calibration)
    parameters = sum(p.numel() for p in model.parameters())
    write_json(out / "run_metadata.json", {"config": config, "normalization": normalization,
               "training_status": status["status"], "selected_epoch": best["epoch"],
               "parameters": parameters, "evaluation_split": split})
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
    model, optimizer, train_loader, val_loader, train_generator, val_generator = build_training_state(
        config, data, catalog, normalization["mean"], normalization["std"], device)
    history, status = train_to_plateau(
        model, optimizer, train_loader, val_loader, train_generator, val_generator,
        config, run_dir, device, normalization, max_epochs=max_epochs,
        resume=(run_dir / "last.pt").is_file())
    return model, history, status

def run_six_trial_grid(grid_dir, manifest, data, catalog, normalization, device, max_epochs):
    grid_dir = Path(grid_dir)
    print("GRID DIRECTORY — keep this path to resume after a reset:", grid_dir, flush=True)
    print("Run only one writer for this grid directory.", flush=True)
    save_grid_summary(grid_dir, manifest)
    for number, (key, entry) in enumerate(manifest["trials"].items(), 1):
        if entry["status"] == "complete":
            out = Path(entry["metrics"]["evaluation_dir"])
            assert (out / "summary.json").is_file() and (grid_dir / key / "best.pt").is_file(), "Completed files missing."
            print(f"[{number}/6] {key}: already complete; skipping.", flush=True)
            continue
        config = entry["config"]
        run_dir = grid_dir / key
        assert settings_match(json.loads((run_dir / "config.json").read_text()), config)
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
            entry["metrics"] = evaluate_trained_trial(model, data, catalog, normalization, device, config, run_dir)
            entry["status"] = "complete"
            print(f"[{number}/6] {key}: checkpoints and metrics saved.", flush=True)
        except BaseException as exc:
            entry["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
            entry["last_error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            write_json(grid_dir / "grid_manifest.json", manifest)
            save_grid_summary(grid_dir, manifest)
            if "model" in locals():
                del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return save_grid_summary(grid_dir, manifest)
