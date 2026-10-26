#!/bin/bash
#SBATCH --job-name=rfi_unet_grid
#SBATCH --time=24:00:00         # Six trials, up to MAX_EPOCHS each; resume with RFI_GRID_RESUME_DIR if it times out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/resnick/groups/CS156b/from_central/2026/njack2/rfi-removal/logs/rfi_unet_grid_%j.log

# Usage:
#   sbatch slurm/run_unet_grid.sh                     # new grid, BASE_CHANNELS=8
#   RFI_BASE_CHANNELS=16 sbatch slurm/run_unet_grid.sh
#   RFI_GRID_RESUME_DIR=/path/to/results/runs/grid_... sbatch slurm/run_unet_grid.sh

# 1. Initialize Conda for this script
source ~/.bashrc

# 2. Activate your shared team environment
conda activate cs156b_env

# 3. Paths (read by the notebook's first cell)
export RFI_PROJECT_ROOT=/resnick/groups/CS156b/from_central/2026/njack2/rfi-removal
export RFI_RESULTS_DIR=${RFI_RESULTS_DIR:-$RFI_PROJECT_ROOT/results}
export RFI_OWNER=${RFI_OWNER:-$USER}
export RFI_BASE_CHANNELS=${RFI_BASE_CHANNELS:-8}
export RFI_GRID_RESUME_DIR=${RFI_GRID_RESUME_DIR:-}

cd "$RFI_PROJECT_ROOT"
mkdir -p executed_notebooks

# 4. Run the notebook top to bottom; the executed copy keeps all printed output and figures
echo "=== Starting U-Net grid (owner=$RFI_OWNER, base=$RFI_BASE_CHANNELS) on $(hostname) ==="
nvidia-smi
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 \
    --output-dir executed_notebooks \
    --output "RFI_unet_team_${SLURM_JOB_ID}.ipynb" \
    notebooks/RFI_unet_team.ipynb

echo "=== U-Net grid finished ==="
