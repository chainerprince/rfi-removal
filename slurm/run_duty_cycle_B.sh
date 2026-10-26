#!/bin/bash
#SBATCH --job-name=rfi_duty_B
#SBATCH --time=24:00:00         # 3 conditions x 2 seeds, ~20-40 min each, plus the sweep; resubmit to continue
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/resnick/groups/CS156b/from_central/2026/njack2/rfi-removal/logs/rfi_duty_B_%j.log

# Experiment B: retrain the U-Net with low / high / all duty-cycle training mixes,
# then evaluate every model on the same generated duty-cycle sweep.
# Usage:
#   sbatch slurm/run_duty_cycle_B.sh
#   RFI_DUTY_B_DIR=/path/to/results/duty_cycle/B_train_mix_<job> sbatch slurm/run_duty_cycle_B.sh   # resume:
#       runs with done.json are skipped; an unfinished run restarts from epoch 1

# 1. Initialize Conda for this script
source ~/.bashrc

# 2. Activate your shared team environment
conda activate cs156b_env

# 3. Paths (read by the notebook's first cell)
export RFI_PROJECT_ROOT=/resnick/groups/CS156b/from_central/2026/njack2/rfi-removal
export RFI_DUTY_B_DIR=${RFI_DUTY_B_DIR:-$RFI_PROJECT_ROOT/results/duty_cycle/B_train_mix_${SLURM_JOB_ID}}

cd "$RFI_PROJECT_ROOT"
mkdir -p executed_notebooks

# 4. Run the notebook top to bottom; the executed copy keeps all printed output and figures
echo "=== Starting duty-cycle experiment B on $(hostname), output $RFI_DUTY_B_DIR ==="
nvidia-smi
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 \
    --output-dir executed_notebooks \
    --output "duty_cycle_B_train_mix_${SLURM_JOB_ID}.ipynb" \
    notebooks/duty_cycle_B_train_mix.ipynb

echo "=== Duty-cycle experiment B finished ==="
