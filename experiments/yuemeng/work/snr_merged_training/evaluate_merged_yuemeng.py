import argparse
import fcntl
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train_merged_yuemeng as merged

t = merged.original
ORIGINAL_BINS = {"G1": (0.01, 0.1), "G2": (0.1, 1.0),
                 "G3": (1.0, 10.0), "G4": (10.0, 100.0)}
SUBGROUPS = {"G1": ["G1"], "G2": ["G2"], "G3_G4": ["G3", "G4"]}


def subgroup(ratio):
    for name, (lo, hi) in ORIGINAL_BINS.items():
        if lo <= ratio and (ratio <= hi if name == "G4" else ratio < hi):
            return name
    return None


def ratio(a, b):
    return float(a / b) if b else None


def load_model(root, group):
    status = json.loads((root / group / "training_status.json").read_text())
    assert status["status"] == "validation_plateau"
    ckpt_path = root / group / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == status["best_epoch"]
    assert ckpt["normalization"] == t.NORMALIZATION
    c = ckpt["config"]
    assert (c["group"], c["architecture"], c["base_channels"]) == (group, "cnn", 8)
    model = t.SegmentationModel(base=8, use_skips=False).to("mps").eval()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model, ckpt


def accumulate_record(model, row, bins):
    rec = t.load_recording(t.DATA / row.path)
    pos = np.zeros(bins, np.int64)
    neg = np.zeros(bins, np.int64)
    for ant in range(10):
        truth = rec["rfi_mask"][:, ant, :, 0]
        p = t.predict_probability(model, rec["power"][:, ant, :, 0], t.NORMALIZATION, torch.device("mps"))
        a, b = t.probability_histograms(p, truth, bins)
        pos += a
        neg += b
    return pos, neg


def calibrate(model, catalog, group, selected_epoch, out, bins):
    clean = merged.selected(catalog, group, "validation")
    clean = clean.loc[clean.rfi_type.eq("clean")].sort_values("sample_id")
    baseline = json.loads((t.BASELINE / "calibration.json").read_text())
    assert clean.sample_id.astype(int).tolist() == sorted(baseline["clean_sample_ids"])
    target = float(baseline["target_clean_fpr"])
    neg = np.zeros(bins, np.int64)
    for row in clean.itertuples():
        pos, counts = accumulate_record(model, row, bins)
        assert pos.sum() == 0
        neg += counts
    cut, achieved = t.calibrate_cut(neg, target)
    calibration = {"split": "validation", "clean_sample_ids": clean.sample_id.astype(int).tolist(),
                   "target_clean_fpr": target, "achieved_clean_fpr": achieved,
                   "bins": bins, "quantized_cut": int(cut),
                   "threshold": float(cut / (bins - 1)) if cut < bins else None,
                   "selected_epoch": selected_epoch,
                   "decision_rule": "floor(probability*(bins-1)) >= cut"}
    t.write_json(out / "calibration.json", calibration)
    np.save(out / "clean_validation_negative_histogram.npy", neg)
    return calibration


def evaluate_split(model, catalog, group, split, calibration, out):
    bins, cut = calibration["bins"], calibration["quantized_cut"]
    rows = merged.selected(catalog, group, split).sort_values("sample_id")
    names = [group] + (SUBGROUPS[group] if group == "G3_G4" else [])
    hist = {name: {"positive": np.zeros(bins, np.int64), "negative": np.zeros(bins, np.int64)} for name in names}
    record_rows = []
    for index, row in enumerate(rows.itertuples(), 1):
        pos, neg = accumulate_record(model, row, bins)
        tp, fp = int(pos[cut:].sum()), int(neg[cut:].sum())
        label = subgroup(float(row.reference_power_ratio))
        item = {"sample_id": int(row.sample_id), "split": split, "rfi_type": row.rfi_type,
                "original_group": label or row.rfi_type,
                "reference_power_ratio": float(row.reference_power_ratio),
                "tp": tp, "fp": fp, "fn": int(pos.sum()) - tp, "tn": int(neg.sum()) - fp}
        record_rows.append(item)
        if row.rfi_type not in ("clean", "sky"):
            assert label in SUBGROUPS[group]
            for name in [group] + ([label] if group == "G3_G4" else []):
                hist[name]["positive"] += pos
                hist[name]["negative"] += neg
        elif row.rfi_type == "clean":
            assert pos.sum() == 0
        if index % 10 == 0 or index == len(rows):
            print(f"EVAL_PROGRESS group={group} split={split} recordings={index}/{len(rows)}", flush=True)
    counts = pd.DataFrame(record_rows)
    clean = counts.loc[counts.rfi_type.eq("clean")]
    sky = counts.loc[counts.rfi_type.eq("sky")]
    clean_fpr = ratio(int(clean.fp.sum()), int(clean.fp.sum() + clean.tn.sum()))
    sky_fpr = ratio(int(sky.fp.sum()), int(sky.fp.sum() + sky.tn.sum()))
    metrics = []
    for name in names:
        if name == group:
            subset = counts.loc[~counts.rfi_type.isin(["clean", "sky"])]
        else:
            subset = counts.loc[counts.original_group.eq(name)]
        tp, fp, fn, tn = (int(subset[c].sum()) for c in ("tp", "fp", "fn", "tn"))
        ph, nh = hist[name]["positive"], hist[name]["negative"]
        ap = float(t.precision_recall_from_histograms(ph, nh)["ap"]) if ph.sum() else None
        metrics.append({"trained_group": group, "evaluation_group": name, "split": split,
                        "rfi_recordings": int(subset.sample_id.nunique()), "pixel_ap": ap,
                        "recall": ratio(tp, tp + fn), "precision": ratio(tp, tp + fp),
                        "rfi_recording_fpr": ratio(fp, fp + tn), "clean_control_fpr": clean_fpr,
                        "sky_control_fpr": sky_fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                        "quantized_cut": cut})
    counts.to_csv(out / "per_recording.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    np.savez_compressed(out / "histograms.npz", **{f"{name}_{kind}": arr for name, h in hist.items() for kind, arr in h.items()})
    return metrics


def validate(root, catalog, group):
    out = root / "validation" / group
    out.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_model(root, group)
    done = out / "complete.json"
    if done.exists():
        info = json.loads(done.read_text())
        assert info["selected_epoch"] == ckpt["epoch"]
        assert all((out / name).is_file() for name in (
            "calibration.json", "clean_validation_negative_histogram.npy",
            "per_recording.csv", "metrics.csv", "histograms.npz"))
        return
    print(f"VALIDATION_START {group} epoch={ckpt['epoch']}", flush=True)
    calibration = calibrate(model, catalog, group, ckpt["epoch"], out, ckpt["config"]["probability_bins"])
    evaluate_split(model, catalog, group, "validation", calibration, out)
    t.write_json(done, {"selected_epoch": ckpt["epoch"],
                        "completed_utc": datetime.now(timezone.utc).isoformat()})
    del model
    torch.mps.empty_cache()
    print(f"VALIDATION_COMPLETE {group}", flush=True)


def freeze(root, catalog):
    audit = json.loads((root / "split_audit.json").read_text())
    assert audit["recording_level_splits_disjoint"] and audit["generated_records_used"] == 0
    models = []
    for group in merged.GROUPS:
        done = json.loads((root / "validation" / group / "complete.json").read_text())
        calibration = root / "validation" / group / "calibration.json"
        info = json.loads(calibration.read_text())
        epoch = done["selected_epoch"]
        assert info.get("selected_epoch", epoch) == epoch
        membership = json.loads((root / group / "membership.json").read_text())
        ids = merged.selected(catalog, group, "test").sample_id.astype(int).tolist()
        assert ids == membership["test_ids"]
        assert not set(ids) & (set(membership["train_ids"]) | set(membership["validation_ids"]))
        models.append({"group": group, "checkpoint_epoch": epoch,
                       "quantized_cut": info["quantized_cut"], "test_ids": ids})
    return {"models": models,
            "selection_policy": "Checkpoint and threshold selected on validation, fixed before this test evaluation",
            "prior_test_caveat": "Original test set has already been examined in previous analyses; not a fresh blind test",
            "G4_test": "No original G4 RFI test recordings"}


def test_one(root, catalog, spec):
    group = spec["group"]
    out = root / "test" / group
    out.mkdir(parents=True, exist_ok=True)
    calibration_file = root / "validation" / group / "calibration.json"
    calibration = json.loads(calibration_file.read_text())
    epoch, cut = spec["checkpoint_epoch"], spec["quantized_cut"]
    assert calibration.get("selected_epoch", epoch) == epoch and calibration["quantized_cut"] == cut
    done = out / "complete.json"
    if done.exists():
        info = json.loads(done.read_text())
        assert info.get("selected_epoch", epoch) == epoch and info["quantized_cut"] == cut
        assert all((out / name).is_file() for name in ("per_recording.csv", "metrics.csv", "histograms.npz"))
        return
    model, ckpt = load_model(root, group)
    assert ckpt["epoch"] == epoch and ckpt["config"]["probability_bins"] == calibration["bins"]
    print(f"TEST_START {group} fixed_cut={cut}", flush=True)
    evaluate_split(model, catalog, group, "test", calibration, out)
    t.write_json(done, {"selected_epoch": epoch, "quantized_cut": cut,
                        "completed_utc": datetime.now(timezone.utc).isoformat()})
    del model
    torch.mps.empty_cache()
    print(f"TEST_COMPLETE {group}", flush=True)


def write_results(root):
    rows = []
    for split in ("validation", "test"):
        for group in merged.GROUPS:
            df = pd.read_csv(root / split / group / "metrics.csv")
            df.insert(0, "best_epoch", json.loads((root / group / "training_status.json").read_text())["best_epoch"])
            rows.append(df)
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(root / "results.csv", index=False)


def run(root, trainer_pid):
    root = Path(root).resolve()
    with (root / ".finalizer.lock").open("a+") as lock, (root / ".writer.lock").open("a+") as writer_lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f"FINALIZER_WAITING training_pid={trainer_pid}", flush=True)
        while not (root / "training_complete.json").exists():
            try:
                import os
                os.kill(trainer_pid, 0)
            except ProcessLookupError:
                raise RuntimeError("Training stopped without training_complete.json; inspect checkpoints before resuming")
            time.sleep(10)
        fcntl.flock(writer_lock.fileno(), fcntl.LOCK_EX)
        assert torch.backends.mps.is_available()
        torch.set_num_threads(1)
        catalog = t.load_catalog()
        merged.audit(catalog)
        for group in merged.GROUPS: validate(root, catalog, group)
        frozen = freeze(root, catalog)
        frozen_path = root / "frozen_before_test.json"
        if frozen_path.exists():
            saved_models = json.loads(frozen_path.read_text())["models"]
            assert len(saved_models) == len(frozen["models"])
            assert all(t.settings_match(saved, model) for saved, model in zip(saved_models, frozen["models"]))
        else:
            t.write_json(frozen_path, frozen)
        print("TEST_INPUTS_FROZEN", flush=True)
        for spec in frozen["models"]: test_one(root, catalog, spec)
        write_results(root)
        completion = {"status": "complete", "groups": list(merged.GROUPS),
                      "completed_utc": datetime.now(timezone.utc).isoformat(),
                      "limitation": "G4 has zero original RFI test records; previous test has been examined"}
        t.write_json(root / "completion.json", completion)
        print("FINALIZER_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--trainer-pid", type=int, required=True)
    args = parser.parse_args()
    run(args.run_dir, args.trainer_pid)
