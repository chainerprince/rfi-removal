# Week 3 — weaknesses and remedies

- `duty_cycle_A_eval_sweep.ipynb`: fixed historical U-Net and flaggers over simulated duty/strength/family combinations; 95% recording-seed bootstrap intervals. This U-Net is distinct from the winning grid CNN.
- `duty_cycle_B_train_mix.ipynb`: train low/high/combined duty sets using seeds 163/164; compare on shared synthetic evaluations. The combined control is newly trained.
- `family_analysis.ipynb`: Lei’s pooled validation count analysis with recording-level bootstrap intervals. The reference archive lacks per-family score histograms; this copy reports error rates, not recomputed per-family AP.
- `source_strength.ipynb`: fixed full-data CNN evaluated on RFI-only source-power groups at its saved clean-FPR cutoff. The separate, opt-in balanced-band experiment uses 18 RFI training records per band plus controls and validation-F1 cutoffs. Ratios come from simulator metadata, not estimated observational SNR.
- `imbalance_remedies.ipynb`: Selina’s full-image BCE, eight 1024×64 crop mixture, focal gamma 2, and BCE weights 5/10/20/40. Each crop is foreground-directed with probability 0.33 when positives exist; it need not be centered or have a fixed foreground fraction. Select weight by validation AP; select thresholds by validation F1; test evaluation is opt-in.
- `dataset_scaling.ipynb`: Selina’s deterministic nested 10–100-recording subsets, family-interleaved using seed 163. The original full-training clean normalization is held fixed. Reference plots run by default; training/test runs are opt-in.

Simple imbalance remedies give small differences, not a decisive universal winner. Small-data scaling is noisy and nonmonotonic. Source-strength/family differences are descriptive and do not isolate a cause.
