import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--starter-root", type=Path, required=True,
                        help="Extracted original cs163-rfi-starter-v1.0 folder")
    parser.add_argument("--starter-archive", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "runs/unet_b16")
    parser.add_argument("--resume-dir", type=Path, help="An existing grid created by this package")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--allow-cuda-to-mps-resume", action="store_true")
    parser.add_argument("--check", action="store_true", help="Verify data and one training-sample forward pass; no optimization")
    args = parser.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from reference import training_core as c

    torch.set_num_threads(1)
    c.ALLOW_CUDA_TO_MPS_RESUME = args.allow_cuda_to_mps_resume
    name = args.device
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    if name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    device = torch.device(name)
    data = args.starter_root.resolve() / "data/starter-v1.0"
    catalog = c.load_catalog(data)
    mean, std = c.fit_training_normalization(data, catalog)
    normalization = {"mean": mean, "std": std}
    if args.check:
        c.seed_everything(163)
        model = c.SegmentationModel(base=16, use_skips=True).to(device).eval()
        dataset = c.AntennaDataset(data, catalog[catalog.split.eq("train")], mean, std)
        x, y = dataset[0]
        with torch.inference_mode():
            logits = model(x[None].to(device))
        assert logits.shape == (1, *y.shape) and torch.isfinite(logits).all()
        print("UNET_PREFLIGHT_OK", tuple(logits.shape), "device", name)
        return
    protocol = {
        "workflow_version": c.WORKFLOW_VERSION, "architecture": "unet", "base_channels": 16,
        "seed": 163, "batch_size": 1, "probability_bins": 65536,
        "min_epochs": 20, "patience": 10, "min_ap_improvement": .001,
        "baseline_far": c.BASELINE_FAR,
    }
    grid, manifest = c.initialize_grid(args.output_dir.resolve(), "Yuemeng", protocol,
                                       ("bce", "mse", "l1"), (.0003, .001),
                                       str(args.resume_dir.resolve()) if args.resume_dir else "")
    summary = c.run_six_trial_grid(grid, manifest, data, catalog, normalization, device, 100)
    print(summary.to_string(index=False))
    print("GRID", grid)


if __name__ == "__main__":
    main()
