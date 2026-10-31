"""Project folders, read from the .env file at the repo root.

Copy .env.example to .env and edit it; .env is not in git. A variable already
set in the environment (for example by a Slurm script) wins over .env.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"


def _read_env(path):
    values = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("\"'")
    return values


def _setting(key):
    value = os.environ.get(key) or _ENV.get(key)
    if not value or value == "your-name":
        raise RuntimeError(f"Set {key} in {ENV_FILE} (run `bash setup.sh` or copy .env.example).")
    return value


_ENV = _read_env(ENV_FILE)
NAME = _setting("RFI_NAME")
DATA_DIR = Path(_setting("RFI_DATA_DIR")).expanduser()
CHECKPOINTS_ROOT = Path(_setting("RFI_CHECKPOINTS_ROOT")).expanduser()
CHECKPOINT_DIR = CHECKPOINTS_ROOT / NAME  # your .pt files; mirrors the layout of results/
RESULTS_DIR = REPO_ROOT / "results"  # metrics and plots, committed to git
