"""Selina's original validation-F1 imbalance comparison, adapted to repo paths.

CNN tensor names, sampled-pixel budget, seed, loss and selection protocol are
retained from the Drive notebook. This is distinct from clean-FPR calibration.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import random, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score
from cs163_rfi import paths, load_sample
from cs163_rfi.experiment import checkpoint_dir, fit_training_normalization, inspect_dataset, sha256_file, write_json

OUT = paths.RESULTS_DIR / "imbalance_f1_comparison"
DATA = paths.DATA_DIR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 163
LR = 3e-4
BASE = 8
BINS = 65536

MIN_EPOCHS = 20
PATIENCE = 10
MIN_AP_IMPROVEMENT = 0.001
MAX_EPOCHS = 100

PATCH = (1024, 64)
CROPS_PER_PARENT = 8
FOREGROUND_PROB = 0.33
FOCAL_GAMMA = 2.0
POS_WEIGHTS = (5.0, 10.0, 20.0, 40.0)

HISTORICAL_BASELINE_AP = 0.640891781812333
class FullDataset(Dataset):
    def __init__(self, frame):
        self.items = [(int(r.sample_id), a)
                      for r in frame.itertuples() for a in range(10)]

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        sid, ant = self.items[i]
        s = load_sample(DATA, sid, antenna=ant)
        x = s["power"].astype(np.float32)
        x = (x - np.float32(MEAN)) / np.float32(STD)
        y = s["rfi_mask"].astype(np.float32)
        return torch.from_numpy(x)[None], torch.from_numpy(y)[None]


class CropDataset(Dataset):
    def __init__(self, frame):
        self.base = FullDataset(frame)
        self.epoch = 0

    def __len__(self): return len(self.base)
    def set_epoch(self, epoch): self.epoch = epoch

    def __getitem__(self, i):
        x, y = self.base[i]
        _, H, W = y.shape
        ph, pw = PATCH
        rng = np.random.default_rng(SEED + 1000003*self.epoch + i)
        pos = torch.nonzero(y[0] > 0, as_tuple=False).numpy()
        xs, ys = [], []

        for _ in range(CROPS_PER_PARENT):
            biased = len(pos) and rng.random() < FOREGROUND_PROB
            if biased:
                r, c = pos[rng.integers(len(pos))]
                r0 = rng.integers(max(0, r-ph+1), min(r, H-ph)+1)
                c0 = rng.integers(max(0, c-pw+1), min(c, W-pw)+1)
            else:
                r0 = rng.integers(0, H-ph+1)
                c0 = rng.integers(0, W-pw+1)

            xs.append(x[:, r0:r0+ph, c0:c0+pw])
            ys.append(y[:, r0:r0+ph, c0:c0+pw])

        return torch.stack(xs), torch.stack(ys)



class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class CNN(nn.Module):
    def __init__(self, b=8):
        super().__init__()
        self.enc1 = DoubleConv(1, b)
        self.enc2 = DoubleConv(b, 2*b)
        self.pool = nn.MaxPool2d(2)
        self.mid = DoubleConv(2*b, 4*b)
        self.up2 = nn.ConvTranspose2d(4*b, 2*b, 2, 2)
        self.dec2 = DoubleConv(2*b, 2*b)
        self.up1 = nn.ConvTranspose2d(2*b, b, 2, 2)
        self.dec1 = DoubleConv(b, b)
        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(self.pool(x))
        x = self.mid(self.pool(x))
        x = self.dec2(self.up2(x))
        x = self.dec1(self.up1(x))
        return self.out(x)

assert sum(p.numel() for p in CNN(BASE).parameters()) == 26441

def focal_loss(logits, target):
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none")
    p = logits.sigmoid()
    pt = torch.where(target > 0.5, p, 1-p)
    return (((1-pt)**FOCAL_GAMMA) * bce).mean()
def make_loader(ds, shuffle, seed):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=1, shuffle=shuffle,
                      num_workers=0, generator=g)

def add_hist(logits, y, pos_hist, neg_hist):
    p = logits.sigmoid().detach().cpu().numpy()
    truth = y.detach().cpu().numpy().astype(bool)
    q = np.floor(p.astype(np.float64) * (BINS-1)).astype(np.int64)
    pos_hist += np.bincount(q[truth], minlength=BINS)
    neg_hist += np.bincount(q[~truth], minlength=BINS)

def ap_from_hist(pos_hist, neg_hist):
    labels = np.r_[np.ones(BINS), np.zeros(BINS)]
    scores = np.tile(np.arange(BINS), 2)
    weights = np.r_[pos_hist, neg_hist]
    keep = weights > 0
    return float(average_precision_score(
        labels[keep], scores[keep], sample_weight=weights[keep]))

def best_f1_cut(pos_hist, neg_hist):
    tp = np.r_[np.cumsum(pos_hist[::-1])[::-1], 0]
    fp = np.r_[np.cumsum(neg_hist[::-1])[::-1], 0]
    fn = pos_hist.sum() - tp
    denom = 2*tp + fp + fn
    f1 = np.divide(2*tp, denom, out=np.zeros_like(tp, dtype=float),
                   where=denom > 0)
    best = f1.max()
    cut = int(np.flatnonzero(np.isclose(f1, best))[-1])  # conservative tie
    return cut, float(best)

@torch.inference_mode()
def score_dataset(model, dataset, device):
    model.eval()
    ph = np.zeros(BINS, np.int64)
    nh = np.zeros(BINS, np.int64)
    for x, y in make_loader(dataset, False, SEED+1):
        x, y = x.to(device), y.to(device)
        add_hist(model(x), y, ph, nh)
    return ph, nh

def metrics_at_cut(ph, nh, cut):
    tp = int(ph[cut:].sum()) if cut < BINS else 0
    fp = int(nh[cut:].sum()) if cut < BINS else 0
    fn = int(ph.sum() - tp)
    tn = int(nh.sum() - fp)
    precision = tp/(tp+fp) if tp+fp else np.nan
    recall = tp/(tp+fn) if tp+fn else np.nan
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan
    fpr = fp/(fp+tn) if fp+tn else np.nan
    return dict(precision=precision, recall=recall, f1=f1, fpr=fpr)


def set_seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

def train(name, loss_kind, pos_weight=None):
    run = OUT / name
    run.mkdir(parents=True, exist_ok=True)
    result_file = run / "result.json"
    if result_file.exists() and (checkpoint_dir(run)/"best.pt").is_file():
        return json.loads(result_file.read_text())

    set_seed()
    model = CNN(BASE).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    if loss_kind == "crop":
        ds = train_crop
    else:
        ds = train_full

    loader = make_loader(ds, True, SEED)

    if loss_kind in ("bce", "crop"):
        criterion = nn.BCEWithLogitsLoss()
    elif loss_kind == "focal":
        criterion = focal_loss
    else:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device))

    best_ap = -np.inf
    best_epoch = 0
    anchor_ap = -np.inf
    anchor_epoch = 0
    hist = []

    for epoch in range(1, MAX_EPOCHS+1):
        if hasattr(ds, "set_epoch"): ds.set_epoch(epoch)
        model.train()
        loss_sum = 0.0; pixels = 0

        for x, y in loader:
            if x.ndim == 5:  # crop arm: [1,8,1,h,w] -> [8,1,h,w]
                x, y = x.squeeze(0), y.squeeze(0)
            x, y = x.to(device), y.to(device)

            loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            loss_sum += loss.item() * y.numel()
            pixels += y.numel()

        ph, nh = score_dataset(model, val_set, device)
        val_ap = ap_from_hist(ph, nh)
        hist.append(dict(epoch=epoch, train_loss=loss_sum/pixels,
                         validation_ap=val_ap))

        if val_ap > best_ap:
            best_ap = val_ap
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_dir(run)/"best.pt")

        if val_ap > anchor_ap + MIN_AP_IMPROVEMENT:
            anchor_ap, anchor_epoch = val_ap, epoch

        pd.DataFrame(hist).to_csv(run/"history.csv", index=False)
        print(f"{name:14s} epoch {epoch:02d} | AP {val_ap:.5f} | best {best_ap:.5f}")

        if epoch >= MIN_EPOCHS and epoch-anchor_epoch >= PATIENCE:
            break

    if epoch < MIN_EPOCHS or epoch-anchor_epoch < PATIENCE:
        raise RuntimeError("Epoch budget exhausted before plateau; increase MAX_EPOCHS and rerun.")

    result = dict(name=name, loss_kind=loss_kind, pos_weight=pos_weight,
                  best_epoch=best_epoch, validation_ap=float(best_ap))
    result_file.write_text(json.dumps(result, indent=2))
    return result


def load_best(name):
    model = CNN(BASE).to(device)
    model.load_state_dict(torch.load(
        checkpoint_dir(OUT/name)/"best.pt", map_location=device, weights_only=True))
    model.eval()
    return model


def prepare():
    """Verify original data and fit normalization on noise-only training records."""
    global MEAN, STD, train_full, train_crop, val_set, test_set
    _, catalog, report = inspect_dataset(paths.REPO_ROOT, DATA, None)
    MEAN, STD = fit_training_normalization(DATA, catalog)
    train_full = FullDataset(catalog.query("split == 'train'"))
    train_crop = CropDataset(catalog.query("split == 'train'"))
    val_set = FullDataset(catalog.query("split == 'validation'"))
    test_set = FullDataset(catalog.query("split == 'test'"))
    assert CROPS_PER_PARENT * PATCH[0] * PATCH[1] == 4096 * 128
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data_checks.json").write_text(json.dumps(report, indent=2))
    return catalog

CANDIDATES = [("baseline_bce", "bce", None), ("crop_mix", "crop", None),
              ("focal_g2", "focal", None)] + [
              (f"weighted_{int(w)}", "weighted", w) for w in POS_WEIGHTS]

def run_candidates():
    return pd.DataFrame([train(*candidate) for candidate in CANDIDATES])

def evaluate_selected(validation_runs):
    """Freeze choices on validation, then report test metrics for each remedy."""
    # Pick the weighted-BCE weight using validation only.
    weighted_rows = validation_runs[
        validation_runs.loss_kind.eq("weighted")]
    best_weighted = weighted_rows.loc[
        weighted_rows.validation_ap.idxmax()]

    selected = validation_runs[
        validation_runs.name.isin(["baseline_bce", "crop_mix", "focal_g2"])
    ].copy()
    selected = pd.concat([selected, best_weighted.to_frame().T],
                         ignore_index=True)

    frozen = []
    for r in selected.itertuples():
        model = load_best(r.name)
        vph, vnh = score_dataset(model, val_set, device)
        cut, val_f1 = best_f1_cut(vph, vnh)
        folder = OUT / r.name
        np.savez_compressed(folder / "validation_histograms.npz", positive=vph, negative=vnh)
        frozen.append(dict(method=r.name, pos_weight=None if pd.isna(r.pos_weight) else float(r.pos_weight),
                           validation_ap=ap_from_hist(vph, vnh), validation_f1=val_f1,
                           quantized_cut=cut, probability_bins=BINS,
                           checkpoint_sha256=sha256_file(checkpoint_dir(folder) / "best.pt")))
    # Record all validation choices before the first test evaluation.
    freeze_file = OUT / "frozen_selection.json"
    if freeze_file.exists():
        assert json.loads(freeze_file.read_text()) == frozen, "Frozen validation choices changed"
    else:
        write_json(freeze_file, frozen)
    rows = []
    for record in frozen:
        name, cut = record["method"], record["quantized_cut"]
        model = load_best(name)
        assert sha256_file(checkpoint_dir(OUT / name) / "best.pt") == record["checkpoint_sha256"]
        tph, tnh = score_dataset(model, test_set, device)
        np.savez_compressed(OUT / name / "test_histograms.npz", positive=tph, negative=tnh)
        test_metrics = metrics_at_cut(tph, tnh, cut)
        rows.append(dict(**record,
                         threshold=cut/(BINS-1) if cut < BINS else None,
                         test_ap=ap_from_hist(tph, tnh),
                         **{"test_"+key: value for key, value in test_metrics.items()}))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "final_comparison.csv", index=False)
    return summary
