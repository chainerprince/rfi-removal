"""Selina's nested training-set scaling experiment; original ordering retained."""
import json, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from cs163_rfi import paths, imbalance as shared
from cs163_rfi.experiment import checkpoint_dir, write_json, sha256_file
from cs163_rfi.imbalance import BINS, CNN, FullDataset as RFIDataset, make_loader as loader
from cs163_rfi.imbalance import score_dataset as score, ap_from_hist as ap, best_f1_cut as f1_cut
SEED = 163
LR = 3e-4
BASE = 8
SIZES = tuple(range(10, 101, 10))
MIN_EPOCHS, PATIENCE, MIN_AP_IMPROVEMENT, MAX_EPOCHS = 20, 10, .001, 100
OUT = paths.RESULTS_DIR / "dataset_scaling"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def hard_metrics(ph, nh, cut):
    m = shared.metrics_at_cut(ph, nh, cut)
    return m["precision"], m["recall"], m["f1"], m["fpr"]

def prepare():
    global catalog, subsets, val_set, test_set
    catalog = shared.prepare()
    val_set, test_set = shared.val_set, shared.test_set
    OUT.mkdir(parents=True, exist_ok=True)
    train_frame = catalog.query("split == 'train'").copy()

    # Shuffle within each RFI family, then spread each family's examples
    # evenly through one deterministic global order.
    parts = []
    for family, g in train_frame.groupby("rfi_type", sort=True):
        g = g.sample(frac=1, random_state=SEED).reset_index(drop=True)
        g["order_key"] = (np.arange(len(g)) + 0.5) / len(g)
        parts.append(g)

    ordered_train = (pd.concat(parts)
                     .sort_values(["order_key", "rfi_type", "sample_id"])
                     .reset_index(drop=True))

    assert len(ordered_train) == 100
    subsets = {n: ordered_train.iloc[:n].copy() for n in SIZES}

    # Nestedness check.
    for a, b in zip(SIZES[:-1], SIZES[1:]):
        assert set(subsets[a].sample_id).issubset(set(subsets[b].sample_id))

    return subsets

def set_seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

def train_size(nrec):
    run=OUT/f"n_{nrec:03d}"
    run.mkdir(parents=True,exist_ok=True)
    result_file=run/"result.json"
    if result_file.exists():
        return json.loads(result_file.read_text())

    set_seed()
    model=CNN(BASE).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    criterion=nn.BCEWithLogitsLoss()
    train_set=RFIDataset(subsets[nrec])
    train_loader=loader(train_set,True,SEED)

    best_ap=-np.inf; best_epoch=0; anchor_ap=-np.inf; anchor_epoch=0
    hist=[]

    for epoch in range(1,MAX_EPOCHS+1):
        model.train(); loss_sum=0.0; pixels=0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device)
            loss=criterion(model(x),y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            loss_sum += loss.item()*y.numel(); pixels += y.numel()

        vph,vnh=score(model,val_set,device)
        vap=ap(vph,vnh)
        hist.append(dict(epoch=epoch,train_loss=loss_sum/pixels,
                         validation_ap=vap))

        if vap>best_ap:
            best_ap=vap; best_epoch=epoch
            torch.save(model.state_dict(),checkpoint_dir(run)/"best.pt")

        if vap>anchor_ap+MIN_AP_IMPROVEMENT:
            anchor_ap,anchor_epoch=vap,epoch

        pd.DataFrame(hist).to_csv(run/"history.csv",index=False)
        print(f"n={nrec:3d} epoch {epoch:02d} | AP {vap:.5f} | best {best_ap:.5f}")

        if epoch>=MIN_EPOCHS and epoch-anchor_epoch>=PATIENCE:
            break

    if epoch-anchor_epoch < PATIENCE:
        raise RuntimeError("Epoch budget exhausted before plateau")

    # Freeze validation-F1 threshold, then evaluate test.
    model.load_state_dict(torch.load(checkpoint_dir(run)/"best.pt",
                                     map_location=device,weights_only=True))
    vph,vnh=score(model,val_set,device)
    cut,val_f1=f1_cut(vph,vnh)
    np.savez_compressed(run / "validation_histograms.npz", positive=vph, negative=vnh)
    write_json(run / "frozen_selection.json", dict(quantized_cut=cut, probability_bins=BINS,
               validation_ap=ap(vph,vnh), checkpoint_sha256=sha256_file(checkpoint_dir(run)/"best.pt")))
    tph,tnh=score(model,test_set,device)
    np.savez_compressed(run / "test_histograms.npz", positive=tph, negative=tnh)
    precision,recall,test_f1,fpr=hard_metrics(tph,tnh,cut)

    result=dict(
        train_recordings=nrec,
        best_epoch=best_epoch,
        validation_ap=ap(vph,vnh),
        validation_f1=val_f1,
        quantized_cut=cut, probability_bins=BINS,
        threshold=cut/(BINS-1) if cut<BINS else None,
        test_ap=ap(tph,tnh),
        test_f1=test_f1,
        test_precision=precision,
        test_recall=recall,
        test_fpr=fpr)
    result_file.write_text(json.dumps(result, indent=2))
    return result
