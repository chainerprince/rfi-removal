# Notebook setup

Weekly folders follow the project sequence. `archive/` preserves the provided starter notebook/HTML and the original team Colab notebook; these are provenance references. `RFI_unet_team.ipynb` redirects to the current grid notebook.

All active notebooks import the installed package; none modifies `sys.path`. Configure `.env` before opening them. Outputs go through `cs163_rfi.paths`; checkpoints stay outside the repo.

## Fresh Google Colab runtime

1. Clone the repository and install it, rather than uploading a notebook alone:

   ```python
   !git clone https://github.com/chainerprince/rfi-removal.git
   %cd /content/rfi-removal
   %pip install -e '.[notebook]'
   ```

2. Mount Drive if needed, then extract the original starter ZIP into `/content` or another folder outside the clone. Do not upload data to git. A failed or incomplete ZIP extraction must be fixed before training.
3. Write `.env` in the cloned repository, adapting these example paths:

   ```python
   from pathlib import Path
   Path('.env').write_text(
       'RFI_NAME=my-run\n'
       'RFI_DATA_DIR=/content/cs163-rfi-starter-v1.0/data/starter-v1.0\n'
       'RFI_CHECKPOINTS_ROOT=/content/drive/MyDrive/rfi_checkpoints\n'
   )
   ```

4. Open a weekly notebook from this checkout using Colab’s Files panel. Run top-to-bottom in a fresh kernel. For a new architecture/width, set the notebook’s configuration before running. Data checks enforce the original splits, shapes, labels and normalization.
5. Persist `results/` and external checkpoints before the runtime ends. Colab’s local files are temporary. Choose GPU for full training; exact cross-device floating-point agreement is not promised.

`week_02/results.ipynb`, `week_03/family_analysis.ipynb`, and the default reference mode of `week_03/dataset_scaling.ipynb` run without raw data/checkpoints after package/path setup. Their successful execution verifies saved-artifact analysis, not new model inference.
