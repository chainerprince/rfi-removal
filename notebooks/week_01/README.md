# Week 1 — supplied starter

`starter.ipynb` is the instructor-provided tutorial with environment/output paths adapted to this repo. The original notebook and rendered HTML remain in `../archive/`. It explains native power/masks, SK, bandpass, combined flagging, controls and simulation. The final pooled-validation section evaluates all 60 validation recordings with the original flaggers. Expected validation precision/recall/FPR: SK 1.96%/5.06%/4.77%; bandpass 68.39%/33.91%/0.29%; combined 11.57%/51.32%/7.37%.

`baselines.ipynb` contains the corrected `Week1_code.ipynb` analysis: pooled validation totals, illustrative error masks and an optional FAR sweep. Its three methods' native confusion counts match fresh inference exactly. The combined method includes production channel expansion after the SK/bandpass union. Single-example scores are separate from pooled totals.
