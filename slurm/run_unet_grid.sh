#!/bin/bash
#SBATCH --job-name=rfi_unet_grid
#SBATCH --time=24:00:00         # Six trials, up to MAX_EPOCHS each; resume with RFI_GRID_RESUME_DIR if it times out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/rfi_unet_grid_%j.log

# Usage (from the repo root, so logs/ and .conda/ are found):
#   sbatch slurm/run_unet_grid.sh                     # new grid, BASE_CHANNELS=8
#   RFI_BASE_CHANNELS=16 sbatch slurm/run_unet_grid.sh
#   RFI_GRID_RESUME_DIR=/path/to/results/runs/grid_... sbatch slurm/run_unet_grid.sh

# 1. Initialize Conda for this script
source ~/.bashrc

# 2. Activate this repo's conda environment (created by: bash setup.sh)
cd "$SLURM_SUBMIT_DIR"
conda activate ./.conda

# 3. Folders come from .env; these only set per-job options (read by the notebook's first cell)
export RFI_BASE_CHANNELS=${RFI_BASE_CHANNELS:-8}
export RFI_GRID_RESUME_DIR=${RFI_GRID_RESUME_DIR:-}

mkdir -p executed_notebooks

# 4. Run the notebook top to bottom; the executed copy keeps all printed output and figures
echo "=== Starting U-Net grid (base=$RFI_BASE_CHANNELS) on $(hostname) ==="
nvidia-smi
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 \
    --output-dir executed_notebooks \
    --output "RFI_unet_team_${SLURM_JOB_ID}.ipynb" \
    notebooks/RFI_unet_team.ipynb

echo "=== U-Net grid finished ==="
