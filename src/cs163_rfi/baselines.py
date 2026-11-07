"""Pooled native-pixel validation of the unmodified starter flaggers."""
import pandas as pd
from . import paths, load_recording, flag_rfi, score_mask
from .experiment import inspect_dataset, pooled_metrics


def evaluate_validation(far=1e-4, output_dir=None):
    """Score all 60 validation recordings; pool counts before computing ratios."""
    _, catalog, _ = inspect_dataset(paths.REPO_ROOT, paths.DATA_DIR, None)
    rows = []
    for recording in catalog.query("split == 'validation'").itertuples():
        sample = load_recording(paths.DATA_DIR / recording.path)
        for method in ("sk", "bandpass", "sk-bandpass"):
            flags = flag_rfi(sample['power'], method=method, far=far)
            for antenna in range(10):
                rows.append(dict(sample_id=int(recording.sample_id), antenna=antenna,
                                 method=method, rfi_type=recording.rfi_type,
                                 **score_mask(flags[:, antenna, :, 0], sample['rfi_mask'][:, antenna, :, 0])))
    counts = pd.DataFrame(rows)
    summary = pooled_metrics(counts, ['method'])
    out = paths.RESULTS_DIR / 'week_01' if output_dir is None else output_dir
    out.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out / 'baseline_per_antenna.csv', index=False)
    summary.to_csv(out / 'baseline_validation.csv')
    print('Saved:', out)
    return summary
