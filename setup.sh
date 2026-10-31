#!/bin/bash
# One-time setup, and rerun it after editing environment.yml:
#   bash setup.sh
# Creates (or updates) the conda environment in .conda/ and your .env file.
set -euo pipefail
cd "$(dirname "$0")"
# Login nodes have no GPU; tell conda which CUDA to solve for so it picks the GPU build of PyTorch.
export CONDA_OVERRIDE_CUDA=${CONDA_OVERRIDE_CUDA:-12.9}

if [ -d .conda ]; then
    conda env update -p ./.conda -f environment.yml --prune
else
    conda env create -p ./.conda -f environment.yml
fi

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env: open it and set RFI_NAME (and the folders, if yours differ)."
fi
mkdir -p logs

echo "Done. Activate with: conda activate $PWD/.conda"
