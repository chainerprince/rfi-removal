import argparse
import fcntl
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train_groups_legacy as training


def divide(a, b):
    return float(a / b) if b else None


def bin_name(ratio):
    for group, (lo, hi) in training.GROUPS.items():
        if lo <= ratio and (ratio < hi or (group == "G4" and ratio <= hi)):
            return group
    return None


def update_comparison(root):
    frames = []
    for name in ["baseline", *training.GROUPS]:
        folder = root / "evaluations" / name
        done = folder / "complete.json"
        if not done.exists():
            continue
        info = json.loads(done.read_text())
        metrics = pd.read_csv(folder / "group_metrics.csv")
        metrics.insert(0, "model", name)
        metrics.insert(1, "checkpoint_epoch", info["selected_epoch"])
        metrics.insert(2, "training_status", info["training_status"])
        frames.append(metrics)
    if not frames:
        return
    table = pd.concat(frames, ignore_index=True)
    tmp = root / ("comparison." + str(time.time_ns()) + ".tmp")
    table.to_csv(tmp, index=False)
    tmp.replace(root / "comparison.csv")


def evaluate(root, name):
    root = Path(root).resolve()
    catalog = training.load_catalog()
    original_calibration = json.loads((training.BASELINE / "calibration.json").read_text())
    if name == "baseline":
        checkpoint_path = training.BASELINE / "best.pt"
        training_status = "historical_validation_plateau"
    else:
        checkpoint_path = root / name / "best.pt"
        status_path = root / name / "training_status.json"
        status = json.loads(status_path.read_text())
        training_status = status["status"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["normalization"] == training.NORMALIZATION
    config = checkpoint["config"]
    assert (config["architecture"], config["base_channels"]) == ("cnn", 8)
    if name == "baseline":
        assert original_calibration.get("selected_epoch", checkpoint["epoch"]) == checkpoint["epoch"]
    else:
        assert config["group"] == name and checkpoint["epoch"] == status["best_epoch"]
    bins = int(config["probability_bins"])
    torch.set_num_threads(1)
    assert torch.backends.mps.is_available()
    device = torch.device("mps")
    model = training.SegmentationModel(base=8, use_skips=False).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    out = root / "evaluations" / name
    out.mkdir(parents=True, exist_ok=True)
    complete = out / "complete.json"
    files = ("calibration.json", "clean_validation_negative_histogram.npy",
             "per_antenna.csv", "group_metrics.csv", "group_histograms.npz")
    if complete.exists():
        info = json.loads(complete.read_text())
        assert info["selected_epoch"] == checkpoint["epoch"]
        assert all((out / file).is_file() for file in files)
        update_comparison(root)
        print(f"EVAL_ALREADY_COMPLETE model={name}", flush=True)
        return
    validation = catalog.loc[catalog.split.eq("validation")].sort_values("sample_id")
    clean = validation.loc[validation.rfi_type.eq("clean")]
    assert clean.sample_id.astype(int).tolist() == sorted(original_calibration["clean_sample_ids"])
    target = float(original_calibration["target_clean_fpr"])
    started = time.perf_counter()
    neg_hist = np.zeros(bins, np.int64)
    print(f"EVAL_START model={name} epoch={checkpoint['epoch']} calibration_clean={len(clean)} device=mps", flush=True)
    for row in clean.itertuples():
        rec = training.load_recording(training.DATA / row.path)
        assert not rec["rfi_mask"].any()
        for ant in range(10):
            probability = training.predict_probability(model, rec["power"][:, ant, :, 0], training.NORMALIZATION, device)
            neg_hist += np.bincount(training.quantize_probability(probability, bins).ravel(), minlength=bins)
    cut, achieved = training.calibrate_cut(neg_hist, target)
    calibration = {
        "clean_ids": clean.sample_id.astype(int).tolist(), "quantized_cut": int(cut),
        "probability_threshold": float(cut / (bins - 1)) if cut < bins else None,
        "target_clean_fpr": target, "achieved_clean_fpr": achieved,
        "decision_rule": "floor(sigmoid(logit)*(bins-1)) >= cut", "bins": bins,
        "selected_epoch": checkpoint["epoch"], "validation_calibration_not_independent": True,
    }
    training.write_json(out / "calibration.json", calibration)
    np.save(out / "clean_validation_negative_histogram.npy", neg_hist)
    hist = {group: {"positive": np.zeros(bins, np.int64), "negative": np.zeros(bins, np.int64)} for group in training.GROUPS}
    records = []
    for index, row in enumerate(validation.itertuples(), 1):
        rec = training.load_recording(training.DATA / row.path)
        group = bin_name(float(row.reference_power_ratio))
        for ant in range(10):
            truth = rec["rfi_mask"][:, ant, :, 0]
            probability = training.predict_probability(model, rec["power"][:, ant, :, 0], training.NORMALIZATION, device)
            pos, neg = training.probability_histograms(probability, truth, bins)
            tp, fp = int(pos[cut:].sum()), int(neg[cut:].sum())
            records.append({
                "sample_id": int(row.sample_id), "antenna": ant, "rfi_type": row.rfi_type,
                "reference_power_ratio": float(row.reference_power_ratio), "group": group or row.rfi_type,
                "tp": tp, "fp": fp, "fn": int(pos.sum()) - tp, "tn": int(neg.sum()) - fp,
            })
            if group is not None:
                hist[group]["positive"] += pos
                hist[group]["negative"] += neg
        if index % 10 == 0:
            print(f"EVAL_PROGRESS model={name} validation={index}/{len(validation)} seconds={time.perf_counter()-started:.1f}", flush=True)
    counts = pd.DataFrame(records)
    metrics = []
    for group in training.GROUPS:
        group_rows = counts.loc[counts.group.eq(group)]
        tp, fp, fn, tn = (int(group_rows[col].sum()) for col in ["tp", "fp", "fn", "tn"])
        ph, nh = hist[group]["positive"], hist[group]["negative"]
        ap = training.precision_recall_from_histograms(ph, nh)["ap"]
        metrics.append({
            "group": group, "recordings": int(group_rows.sample_id.nunique()),
            "pixel_ap": float(ap), "recall": divide(tp, tp + fn), "precision": divide(tp, tp + fp),
            "fpr": divide(fp, fp + tn), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "threshold": calibration["probability_threshold"], "clean_fpr_calibration": achieved,
        })
    counts.to_csv(out / "per_antenna.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "group_metrics.csv", index=False)
    np.savez_compressed(out / "group_histograms.npz", **{f"{group}_{kind}": hist[group][kind] for group in hist for kind in ("positive", "negative")})
    training.write_json(complete, {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "model": name,
        "selected_epoch": int(checkpoint["epoch"]), "training_status": training_status,
        "evaluation_split": "validation", "device": "mps",
        "seconds": time.perf_counter() - started,
    })
    update_comparison(root)
    print(f"EVAL_COMPLETE model={name} seconds={time.perf_counter()-started:.1f} cutoff={cut} clean_fpr={achieved:.6f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=["baseline", *training.GROUPS])
    args = parser.parse_args()
    lock_dir = args.run_dir.resolve() / "evaluations" / args.model
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / ".evaluate.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        evaluate(args.run_dir, args.model)
