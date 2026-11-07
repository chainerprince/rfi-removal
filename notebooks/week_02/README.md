# Week 2 — grid search

`grid_search.ipynb` runs six independent loss/rate combinations for one architecture/width pair. Run all four pairs for the full 24 configurations. Shared code is in `cs163_rfi.experiment`: data checks → normalization → histogram metrics → models/losses → training/resume → calibration → evaluation → grid orchestration.

CNN/8 has 26,441 parameters; U-Net/8 has 29,321. Both use two pooling levels and transposed-convolution upsampling. Best CNN: BCE, Adam 0.0003, seed 163, epoch 35, validation AP 0.640891781812333. Checkpoint selection uses highest validation AP; meaningful-improvement patience uses 0.001, minimum 20 epochs, patience 10. Unfinished runs must resume until plateau.

`results.ipynb` ranks the 24 saved validation settings and presents equal-weight marginal means. All 16 MSE/L1 runs have zero recall at their saved clean-FPR-calibrated thresholds. This does not imply all raw scores were identical or literally zero.
