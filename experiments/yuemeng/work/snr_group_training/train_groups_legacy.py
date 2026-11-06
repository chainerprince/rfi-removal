import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
from reference.training_core import (
    SegmentationModel, build_training_state, calibrate_cut, load_catalog as read_catalog,
    load_recording, loss_value, precision_recall_from_histograms, predict_probability,
    probability_histograms, quantize_probability, settings_match, training_settings,
    train_to_plateau, write_json,
)
DATA = Path(os.environ.get("RFI_DATA_DIR", str(WORKSPACE / "data/starter-v1.0"))).expanduser().resolve()
BASELINE = Path(os.environ.get("RFI_BASELINE_DIR", str(WORKSPACE / "inputs/baseline"))).expanduser().resolve()
EVALUATOR = Path(__file__).with_name("evaluate_groups.py")
NORMALIZATION = {"mean": 4.257218322753906, "std": 4.2564571117143295}
GROUPS = {"G1": (0.01, 0.1), "G2": (0.1, 1.0), "G3": (1.0, 10.0), "G4": (10.0, 100.0)}


def load_catalog():
    return read_catalog(DATA)


def selected_catalog(catalog, group):
    lo, hi = GROUPS[group]
    ratio = catalog.reference_power_ratio
    in_range = ratio.ge(lo) & (ratio.le(hi) if group == "G4" else ratio.lt(hi))
    controls = catalog.rfi_type.isin(["clean", "sky"])
    return catalog.loc[(in_range | controls) & catalog.split.isin(["train", "validation"])].copy()


def config_for(group):
    config = training_settings(json.loads((BASELINE / "config.json").read_text()))
    if (config["architecture"], config["base_channels"]) != ("cnn", 8):
        raise ValueError("Group training requires the base-8 CNN configuration")
    config.update(group=group, reference_ratio_interval=list(GROUPS[group]), max_epochs=100)
    return config


def verify_resume(run_dir, config):
    last_path = run_dir / "last.pt"
    if not last_path.exists():
        return False
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    assert settings_match(last["config"], config) and last["normalization"] == NORMALIZATION and last["device_type"] == "mps"
    assert len(last["history"]) == last["epoch"] and last["history"][-1]["epoch"] == last["epoch"]
    best_path = run_dir / "best.pt"
    try:
        best = torch.load(best_path, map_location="cpu", weights_only=False)
    except (OSError, EOFError, RuntimeError):
        best = {}
    if best.get("epoch") != last["best_epoch"]:
        prev = run_dir / "previous_best.pt"
        assert prev.exists(), "No matching best checkpoint for the last completed epoch"
        old = torch.load(prev, map_location="cpu", weights_only=False)
        assert old["epoch"] == last["best_epoch"] and settings_match(old["config"], config) and old["normalization"] == NORMALIZATION
        temp = best_path.with_name("best." + uuid.uuid4().hex + ".tmp")
        shutil.copyfile(prev, temp)
        os.replace(temp, best_path)
    else:
        assert settings_match(best["config"], config) and best["normalization"] == NORMALIZATION
    pd.DataFrame(last["history"]).to_csv(run_dir / "history.csv", index=False)
    write_json(run_dir / "training_status.json", {"status": last["status"], "last_epoch": last["epoch"],
               "best_epoch": last["best_epoch"], "best_validation_ap": last["best_ap"]})
    return True


def smoke_check(catalog):
    assert torch.backends.mps.is_available(), "Apple MPS is unavailable"
    group = selected_catalog(catalog, "G1")
    model, optimizer, loader, _, _, _ = build_training_state(config_for("G1"), DATA, group,
                                                               NORMALIZATION["mean"], NORMALIZATION["std"], torch.device("mps"))
    x, y = next(iter(loader))
    x, y = x.to("mps"), y.to("mps")
    output = model(x)
    loss = loss_value(output, y, "bce")
    loss.backward()
    assert output.shape == y.shape and torch.isfinite(loss).item()
    assert all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
    print(f"SMOKE_OK device=mps output={tuple(output.shape)} loss={loss.item():.6f} groups=4; no optimizer update", flush=True)
    del model, optimizer, loader, x, y, output, loss
    torch.mps.empty_cache()


def run(run_root):
    run_root = Path(run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / ".writer.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        catalog = load_catalog()
        torch.set_num_threads(1)
        metadata = {
            "group_boundaries": {name: list(bounds) for name, bounds in GROUPS.items()},
            "normalization": NORMALIZATION,
        }
        meta_path = run_root / "run_metadata.json"
        if meta_path.exists():
            saved = json.loads(meta_path.read_text())
            if not settings_match(saved, metadata):
                raise ValueError("Run data, groups, or normalization changed")
        else:
            write_json(meta_path, metadata)
        device = torch.device("mps")
        print(f"RUN_START dir={run_root} device={device} UTC={datetime.now(timezone.utc).isoformat()}", flush=True)
        for group in GROUPS:
            selected = selected_catalog(catalog, group)
            config = config_for(group)
            directory = run_root / group
            directory.mkdir(exist_ok=True)
            cfg = directory / "config.json"
            membership = {
                "group": group, "reference_ratio_interval": list(GROUPS[group]),
                "train_ids": selected.loc[selected.split == "train", "sample_id"].astype(int).tolist(),
                "validation_ids": selected.loc[selected.split == "validation", "sample_id"].astype(int).tolist(),
                "test_ids": [],
            }
            if cfg.exists():
                assert settings_match(json.loads(cfg.read_text()), config)
                assert json.loads((directory / "membership.json").read_text()) == membership
            else:
                write_json(cfg, config)
                write_json(directory / "membership.json", membership)
            resume = verify_resume(directory, config)
            model, optimizer, train_loader, val_loader, train_gen, val_gen = build_training_state(
                config, DATA, selected, NORMALIZATION["mean"], NORMALIZATION["std"], device)
            print(f"GROUP_START {group} train_recordings={len(train_loader.dataset)//10} val_recordings={len(val_loader.dataset)//10} resume={resume}", flush=True)
            previous = torch.load(directory / "last.pt", map_location="cpu", weights_only=False) if resume else None
            if previous and previous["epoch"] >= 100 and previous["status"] != "validation_plateau":
                history, status = previous["history"], previous["status"]
                print(f"GROUP_CAP_REACHED {group}: preserving original 100-epoch cap", flush=True)
            else:
                history, status = train_to_plateau(model, optimizer, train_loader, val_loader, train_gen, val_gen,
                                                   config, directory, device, NORMALIZATION, max_epochs=100,
                                                   resume=resume, keep_previous_best=True)
            print(f"GROUP_FINISHED {group} status={status} epochs={len(history)}", flush=True)
            del model, optimizer, train_loader, val_loader
            torch.mps.empty_cache()
            if EVALUATOR.exists():
                models = ["baseline", group] if group == "G1" else [group]
                for name in models:
                    result = subprocess.run([sys.executable, str(EVALUATOR), "--run-dir", str(run_root), "--model", name])
                    if result.returncode:
                        print(f"EVALUATION_FAILED model={name} exit={result.returncode}; continue remaining training", flush=True)
            else:
                print(f"EVALUATOR_NOT_READY {group}", flush=True)
        write_json(run_root / "worker_finished.json", {"finished_utc": datetime.now(timezone.utc).isoformat()})
        print("WORKER_FINISHED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        torch.set_num_threads(1)
        smoke_check(load_catalog())
    else:
        assert args.run_dir is not None
        run(args.run_dir)
