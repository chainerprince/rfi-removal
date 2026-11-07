# RFI detection — CS163 Weeks 1–3

Detect interference in native time × frequency power images. Read the notebooks in week order. Week 1 is supplied starter code; Weeks 2–3 contain the team's experiments.

| Week | Notebook | Purpose |
| --- | --- | --- |
| 1 | [Starter and baselines](notebooks/week_01/starter.ipynb) | Load data; run SK, bandpass and combined flagging; inspect errors. |
| 1 | [Baseline comparison](notebooks/week_01/baselines.ipynb) | Corrected pooled validation totals, error examples and optional FAR sweep. |
| 2 | [Grid search](notebooks/week_02/grid_search.ipynb) | CNN/U-Net × widths 8/16 × BCE/MSE/L1 × rates 0.0003/0.001. |
| 2 | [Results](notebooks/week_02/results.ipynb) | Rank the 24 validation results and compare factor means. |
| 3 | [Duty-cycle evaluation](notebooks/week_03/duty_cycle_A_eval_sweep.ipynb) | Compare a fixed U-Net and statistical methods across duty cycles. |
| 3 | [Duty-group training](notebooks/week_03/duty_cycle_B_train_mix.ipynb) | Train with low, high or combined duty cycles. |
| 3 | [Family analysis](notebooks/week_03/family_analysis.ipynb) | Compare pooled error rates with recording-level bootstrap intervals. |
| 3 | [Source strength](notebooks/week_03/source_strength.ipynb) | Analyze the fixed full-data model by power band; optionally train balanced groups. |
| 3 | [Imbalance remedies](notebooks/week_03/imbalance_remedies.ipynb) | Compare foreground crops, focal loss and weighted BCE. |
| 3 | [Dataset scaling](notebooks/week_03/dataset_scaling.ipynb) | Train on nested subsets of 10–100 recordings. |

## Run

Run `bash setup.sh`, configure [.env.example](.env.example) as `.env`, then activate `conda activate ./.conda` and start `python -m jupyter lab`. Use the same kernel and run each notebook from its first cell. See [notebook setup](notebooks/README.md) for Colab and [CONTRIBUTING.md](CONTRIBUTING.md) for the cluster.

`RFI_DATA_DIR` must contain `dataset.csv`. Checkpoints stay outside git under `RFI_CHECKPOINTS_ROOT/RFI_NAME`; metrics and plots go in `results/`. Reusable experiment code lives in `src/cs163_rfi/`; original notebooks are preserved in `notebooks/archive/`.

For the full grid, run all four architecture/width pairs. On the cluster:

```sh
RFI_ARCHITECTURE=cnn RFI_BASE_CHANNELS=8 sbatch slurm/run_unet_grid.sh
```

Repeat for CNN/16, U-Net/8 and U-Net/16. Choose GPU for training. Test evaluation is opt-in where indicated.

## Protocol

- **Data:** 200 recordings, ten antennas each; split 100/60/40 by recording. Each antenna input/target is `[1, 4096, 128]`.
- **Normalization:** one mean/std from the five clean training recordings: `4.257218322753906` / `4.2564571117143295`. Statistical flaggers receive raw power.
- **Models:** predict logits. BCE uses logits; MSE/L1 use sigmoid probabilities. CNN/8 has 26,441 parameters; U-Net/8 has 29,321.
- **Selection:** checkpoints use validation AP. The grid matches clean-validation FPR; imbalance/scaling choose validation-F1 thresholds. Test does not select models or thresholds.
- **Metrics:** precision = TP/(TP+FP), recall = TP/(TP+FN), FPR = FP/(FP+TN). Pool native-pixel counts before ratios. Pixel AP summarizes ranking across thresholds; the grid uses 65,536 probability bins.

The best grid CNN uses BCE and learning rate 0.0003: validation AP `0.6408918`, recall `0.6849259`. Compact historical tables are in `results/reference/`; floating-point results can vary across GPU/software versions. Source-strength analysis uses RFI-only groups and a frozen clean-FPR cutoff; the balanced-band training section records its separate validation-F1 protocol. Global-context and antenna-averaging experiments are outside this update.
