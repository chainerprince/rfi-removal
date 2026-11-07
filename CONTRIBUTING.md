# Contributing

A quick guide to how this repo is organized and how to get set up.
For background on the project and the dataset, see the [README](README.md).
If you use an AI coding assistant, point it at [SKILLS.md](SKILLS.md).

## Setup (once)

From inside the `rfi-removal/` folder:

```sh
bash setup.sh
```

This does three things:
1. creates a conda environment in `.conda/` with everything the code needs,
   including this repo's own package (`cs163_rfi`)
2. creates your `.env` file from `.env.example`
3. creates the `logs/` folder

Then open `.env` and set `RFI_NAME` to any name you like (for example
`RFI_NAME=prince`). Your checkpoints are saved under that name.

Activate the environment whenever you work:

```sh
conda activate ./.conda
```

In VS Code or Jupyter, pick `.conda/bin/python` as the kernel.

### Adding a package

Add it to `environment.yml`, then run `bash setup.sh` again. Use the conda-forge
name under `dependencies:`, or put it under `pip:` if it's only on PyPI. Commit
`environment.yml` so everyone gets the package.

## The `.env` file

All code reads its folders from `.env`, so if something moves, you only change
`.env`. It isn't committed, so each person has their own.

| Setting | What it is |
|---|---|
| `RFI_NAME` | Your name. Your checkpoints go in `RFI_CHECKPOINTS_ROOT/RFI_NAME`. |
| `RFI_DATA_DIR` | The dataset folder (the one that contains `dataset.csv`) |
| `RFI_CHECKPOINTS_ROOT` | The shared checkpoint folder, one subfolder per person |

In Python, read these through `cs163_rfi.paths` rather than typing paths:

```python
from cs163_rfi import paths

paths.DATA_DIR         # the dataset
paths.CHECKPOINT_DIR   # your checkpoint folder: RFI_CHECKPOINTS_ROOT / RFI_NAME
paths.RESULTS_DIR      # results/ in the repo
```

## Where things live

On the HPC, the repo sits inside a shared project folder:

```
/resnick/groups/mlprojects/rfi-removal-project/
├── data/starter-v1.0/     # the dataset (not in git)
├── model_checkpoints/     # saved models (not in git)
│   └── <RFI_NAME>/        # e.g. model_checkpoints/prince/
└── rfi-removal/           # this repo
```

`.env.example` already points at these folders. If you work somewhere else,
copy the data to your machine and change the paths in your `.env`.

Inside the repo:

| Folder | What it holds |
|---|---|
| `src/cs163_rfi/` | Our Python code: loading data, simulating RFI, flagging, metrics, plots, paths |
| `vendor/dsa110/` | Code copied from the DSA-110 telescope project. Don't edit it. |
| `notebooks/` | Weekly experiments: `week_01/`, `week_02/`, `week_03/`; originals in `archive/` |
| `slurm/` | Scripts that run the notebooks as jobs on the HPC |
| `results/` | Outputs we keep: metrics (CSV/JSON) and plots |
| `assets/` | Images used in the notebooks |

These are created when you run things and are not in git: `.conda/` (the
environment), `logs/` (job logs) and `executed_notebooks/` (notebooks after a
job has run them).

### Results vs. checkpoints

Metrics, plots and training history go in `results/` and are committed.
Checkpoints (`*.pt`) are too big for git, so they go in your checkpoint folder
instead, at the same relative path. For example, the checkpoint for
`results/runs/<grid>/<trial>/` is
`model_checkpoints/<RFI_NAME>/runs/<grid>/<trial>/best.pt`.

## Running jobs on the HPC

Long runs (such as training) go through Slurm. Submit from the repo root:

```sh
sbatch slurm/run_unet_grid.sh     # submit a job
squeue -u $USER                   # check on your jobs
```

When the job finishes:
- the log is in `logs/`
- the notebook with all its output is in `executed_notebooks/`
- the results are in `results/`, and checkpoints are in your checkpoint folder

To add a new experiment, copy an existing notebook and its Slurm script, then
change what you need to.

## What to commit

**Do commit:** code, notebooks, `environment.yml`, and small result files (CSV, JSON, PNG).

**Don't commit:** `.env`, `.conda/`, `logs/`, `executed_notebooks/`, or model
checkpoints (`*.pt`). `.gitignore` already skips these. Run `git status` before
every commit to make sure.

Pull before you start working. Make your changes on a branch and open a pull
request into `master`.
