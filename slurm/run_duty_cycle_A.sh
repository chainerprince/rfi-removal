#!/bin/bash
#SBATCH --job-name=rfi_duty_A
#SBATCH --time=06:00:00         # ~430 recordings at ~10 s each on CPU (baselines dominate)
#SBATCH --partition=expansion
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/rfi_duty_A_%j.log

# Experiment A: evaluate the trained grid U-Net and the baselines on a generated duty-cycle sweep.
# No training and no GPU needed.
# Usage (from the repo root, so logs/ and .conda/ are found):
#   sbatch slurm/run_duty_cycle_A.sh
#   RFI_MODEL_RUN_DIR=/path/to/results/trial_dir RFI_MODEL_CHECKPOINT_DIR=/path/to/checkpoint_dir \
#       sbatch slurm/run_duty_cycle_A.sh   # evaluate another trial

# 1. Initialize Conda for this script
source ~/.bashrc

# 2. Activate this repo's conda environment (created by: bash setup.sh)
cd "$SLURM_SUBMIT_DIR"
conda activate ./.conda

# 3. Folders come from .env; these only set per-job options (read by the notebook's first cell)
export RFI_DUTY_A_DIR=${RFI_DUTY_A_DIR:-$PWD/results/duty_cycle/A_eval_sweep_${SLURM_JOB_ID}}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
[ -n "$RFI_MODEL_RUN_DIR" ] && export RFI_MODEL_RUN_DIR
[ -n "$RFI_MODEL_CHECKPOINT_DIR" ] && export RFI_MODEL_CHECKPOINT_DIR

mkdir -p executed_notebooks

# 4. Run the notebook top to bottom; the executed copy keeps all printed output and figures
echo "=== Starting duty-cycle experiment A on $(hostname), output $RFI_DUTY_A_DIR ==="
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 \
    --output-dir executed_notebooks \
    --output "duty_cycle_A_eval_sweep_${SLURM_JOB_ID}.ipynb" \
    notebooks/duty_cycle_A_eval_sweep.ipynb

echo "=== Duty-cycle experiment A finished ==="
