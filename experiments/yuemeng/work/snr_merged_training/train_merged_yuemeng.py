import argparse
import fcntl
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work/snr_group_training"))
import train_groups_legacy as original

GROUPS = {"G1": (0.01, 0.1), "G2": (0.1, 1.0), "G3_G4": (1.0, 100.0)}
SPLITS = ("train", "validation", "test")
original.GROUPS = GROUPS


def in_group(frame, group):
    lo, hi = GROUPS[group]
    ratio = frame.reference_power_ratio
    return ratio.ge(lo) & (ratio.le(hi) if group == "G3_G4" else ratio.lt(hi))


def selected(frame, group, split):
    return frame.loc[frame.split.eq(split) &
                     (in_group(frame, group) | frame.rfi_type.isin(["clean", "sky"]))].copy()


def audit(catalog):
    if not catalog.sample_id.is_unique or set(catalog.split) != set(SPLITS):
        raise ValueError("Training, validation, and test need distinct recordings")
    controls = catalog.rfi_type.isin(["clean", "sky"])
    rfi = catalog.loc[~controls]
    membership = {group: in_group(rfi, group) for group in GROUPS}
    if not sum(membership.values()).eq(1).all():
        raise ValueError("Each RFI recording must belong to exactly one group")
    counts = {group: rfi.loc[mask].groupby("split").size().reindex(SPLITS, fill_value=0).to_dict()
              for group, mask in membership.items()}
    shared_controls = catalog.loc[controls].groupby("split").size().reindex(SPLITS, fill_value=0).to_dict()
    return {"recording_level_splits_disjoint": True, "original_recordings": len(catalog),
            "rfi_by_group": counts, "shared_controls": shared_controls,
            "generated_records_used": 0, "ten_antennas_stay_with_recording": True}


def metadata():
    return {"groups": {name: list(bounds) for name, bounds in GROUPS.items()},
            "normalization": original.NORMALIZATION, "device": "mps", "max_epochs": 100}


def preflight(catalog):
    report = audit(catalog)
    assert torch.backends.mps.is_available(), "Apple MPS is unavailable"
    torch.set_num_threads(1)
    group = "G1"
    frame = pd.concat([selected(catalog, group, s) for s in ("train", "validation")])
    config = original.config_for(group)
    model, optimizer, loader, _, _, _ = original.build_training_state(
        config, original.DATA, frame, original.NORMALIZATION["mean"], original.NORMALIZATION["std"], torch.device("mps"))
    x, y = next(iter(loader)); x, y = x.to("mps"), y.to("mps")
    logits = model(x); loss = original.loss_value(logits, y, "bce"); loss.backward()
    assert logits.shape == y.shape and torch.isfinite(loss).item()
    assert all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
    print("PREFLIGHT_OK", json.dumps(report), "shape", tuple(logits.shape), "loss", loss.item(), flush=True)
    del model, optimizer, loader, x, y, logits, loss
    torch.mps.empty_cache()


def run(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".writer.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert torch.backends.mps.is_available(), "Apple MPS is unavailable"
        torch.set_num_threads(1)
        catalog = original.load_catalog()
        split_audit = audit(catalog)
        run_meta = metadata()
        manifest = root / "run_metadata.json"
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            if not original.settings_match(saved, run_meta):
                raise ValueError("Run data, groups, or training settings changed")
        else:
            original.write_json(manifest, run_meta)
            original.write_json(root / "split_audit.json", split_audit)
        print(f"RUN_START {datetime.now(timezone.utc).isoformat()} root={root} device=mps", flush=True)
        for group in GROUPS:
            directory = root / group
            directory.mkdir(exist_ok=True)
            train = selected(catalog, group, "train")
            val = selected(catalog, group, "validation")
            frame = pd.concat([train, val])
            config = original.config_for(group)
            membership = {"group": group, "reference_ratio_interval": list(GROUPS[group]),
                          "train_ids": train.sample_id.astype(int).tolist(),
                          "validation_ids": val.sample_id.astype(int).tolist(),
                          "test_ids": selected(catalog, group, "test").sample_id.astype(int).tolist()}
            cfgpath = directory / "config.json"
            if cfgpath.exists():
                assert original.settings_match(json.loads(cfgpath.read_text()), config)
                assert json.loads((directory / "membership.json").read_text()) == membership
            else:
                original.write_json(cfgpath, config)
                original.write_json(directory / "membership.json", membership)
            resume = original.verify_resume(directory, config)
            model, optimizer, train_loader, val_loader, train_gen, val_gen = original.build_training_state(
                config, original.DATA, frame, original.NORMALIZATION["mean"], original.NORMALIZATION["std"], torch.device("mps"))
            print(f"GROUP_START {group} train={len(train)} val={len(val)} resume={resume}", flush=True)
            previous = torch.load(directory / "last.pt", map_location="cpu", weights_only=False) if resume else None
            if previous and previous["epoch"] >= 100 and previous["status"] != "validation_plateau":
                status, history = previous["status"], previous["history"]
            else:
                history, status = original.train_to_plateau(
                    model, optimizer, train_loader, val_loader, train_gen, val_gen, config,
                    directory, torch.device("mps"), original.NORMALIZATION, max_epochs=100,
                    resume=resume, keep_previous_best=True)
            print(f"GROUP_FINISHED {group} status={status} epochs={len(history)}", flush=True)
            del model, optimizer, train_loader, val_loader, train_gen, val_gen
            torch.mps.empty_cache()
            if status != "validation_plateau":
                original.write_json(root / "paused.json", {"group": group, "status": status,
                    "last_epoch": len(history), "reason": "Original 100-epoch limit reached; preserved for review"})
                raise RuntimeError(f"{group} stopped at the original 100-epoch cap")
        original.write_json(root / "training_complete.json", {"groups": list(GROUPS),
            "completed_utc": datetime.now(timezone.utc).isoformat()})
        print("TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        preflight(original.load_catalog())
    else:
        assert args.run_dir
        run(args.run_dir)
