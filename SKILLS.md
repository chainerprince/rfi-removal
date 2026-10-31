# Working in this repo (for AI coding assistants)

ML project: detect radio-frequency interference (RFI) in radio telescope data.
Python package `cs163_rfi` in `src/`, experiments as notebooks in `notebooks/`,
HPC jobs in `slurm/`. Human-facing guide: [CONTRIBUTING.md](CONTRIBUTING.md).

## Environment

- Conda env lives in `.conda/` at the repo root. Set up or update it with `bash setup.sh`.
- Run Python as `conda activate ./.conda` then `python ...`, or call `.conda/bin/python` directly.
- `cs163_rfi` is installed in editable mode, so `import cs163_rfi` works everywhere.
  Don't add `sys.path` hacks.
- To add a dependency: add it to `environment.yml` (conda-forge name, or under
  `pip:`), then run `bash setup.sh`. Don't `pip install` ad hoc.

## Paths: always from `.env`

- `.env` at the repo root (git-ignored, one per person) holds `RFI_NAME`,
  `RFI_DATA_DIR` and `RFI_CHECKPOINTS_ROOT`. Template: `.env.example`.
- In code, use `from cs163_rfi import paths`:
  - `paths.DATA_DIR`: dataset (contains `dataset.csv`)
  - `paths.CHECKPOINT_DIR`: this person's checkpoint folder (`RFI_CHECKPOINTS_ROOT / RFI_NAME`)
  - `paths.RESULTS_DIR`: `results/` in the repo
  - `paths.REPO_ROOT`, `paths.NAME`
- Never hard-code absolute paths or usernames. If a path is missing, add a
  setting to `.env.example` and `src/cs163_rfi/paths.py`.

## Where outputs go

- Metrics, plots, history (CSV/JSON/PNG): under `paths.RESULTS_DIR`. These are committed.
- Model checkpoints (`*.pt`): under `paths.CHECKPOINT_DIR`, at the same relative
  path as the run's results folder (e.g. `results/runs/<grid>/<trial>/` →
  `CHECKPOINT_DIR/runs/<grid>/<trial>/best.pt`). Never save `.pt` files inside the repo.

## Adding an experiment

1. Copy the closest notebook in `notebooks/` and its script in `slurm/`.
2. Keep the first code cell's pattern: `from cs163_rfi import paths`, then build
   every folder from `paths`.
3. Put reusable code in `src/cs163_rfi/`, not copied between notebooks.
4. Slurm scripts are submitted from the repo root (`sbatch slurm/<script>.sh`).
   They `cd "$SLURM_SUBMIT_DIR"`, `conda activate ./.conda` and write logs to `logs/`.

## Don't

- Don't edit `vendor/dsa110/` (third-party code).
- Don't commit `.env`, `.conda/`, `logs/`, `executed_notebooks/`, `*.pt` or data.
- Don't modify the dataset in `RFI_DATA_DIR`. It's shared by the team.
