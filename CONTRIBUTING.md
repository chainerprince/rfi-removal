# Contributing

A quick guide to how this repo is organized and how to get set up.
For background on the project and the dataset, see the [README](README.md).

## What's in the repo

| Folder | What it holds |
|---|---|
| `src/cs163_rfi/` | Our Python code: loading data, simulating RFI, flagging, metrics, plots |
| `vendor/dsa110/` | Code copied from the DSA-110 telescope project. Don't edit it. |
| `notebooks/` | Experiments: one notebook per experiment |
| `slurm/` | Scripts that run the notebooks as jobs on the HPC |
| `results/` | Outputs we keep: metrics (CSV/JSON) and plots |
| `assets/` | Images used in the notebooks |
| `data/` | The dataset. **Not in git.** |

These folders are created when you run things, and they are also not in git:
`logs/` (job logs) and `executed_notebooks/` (notebooks after a job has run them).

## The data

The dataset is too big for git. The code looks for it here:

```
rfi-removal/data/starter-v1.0/
```

- **On the HPC:** the data is already in this location in the shared repo folder.
  You don't need to do anything.
- **Not on the HPC:** get a copy of the `data/` folder and put it inside your
  `rfi-removal/` folder, next to `src/` and `notebooks/`.

The notebooks look for the repo at its HPC location by default. If you work
somewhere else, tell them where your copy is before you start Jupyter:

```sh
export RFI_PROJECT_ROOT=/path/to/your/rfi-removal
```

## Setup

You need Python 3.11 or newer. From inside the `rfi-removal/` folder, run:

```sh
pip install -e ".[notebook]"
```

## Running jobs on the HPC

Long runs (such as training) go through Slurm:

```sh
sbatch slurm/run_unet_grid.sh     # submit a job
squeue -u $USER                   # check on your jobs
```

When the job finishes:
- the log is in `logs/`
- the notebook with all its output is in `executed_notebooks/`
- the results are in `results/`

To add a new experiment, copy an existing notebook and its Slurm script, then
change what you need to.

## What to commit

**Do commit:** code, notebooks, and small result files (CSV, JSON, PNG).

**Don't commit:** `data/`, `logs/`, `executed_notebooks/`, or model checkpoints (`*.pt`).
`.gitignore` already skips these. Run `git status` before every commit to make sure.

Pull before you start working. Make your changes on a branch and open a pull
request into `master`.
