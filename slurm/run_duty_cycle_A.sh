#!/bin/bash
#SBATCH --job-name=rfi_duty_A
#SBATCH --time=06:00:00         # ~430 recordings at ~10 s each on CPU (baselines dominate)
#SBATCH --partition=expansion
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/resnick/groups/CS156b/from_central/2026/njack2/rfi-removal/logs/rfi_duty_A_%j.log

# Experiment A: evaluate the trained grid U-Net and the baselines on a generated duty-cycle sweep.
# No training and no GPU needed.
# Usage:
#   sbatch slurm/run_duty_cycle_A.sh
#   RFI_MODEL_RUN_DIR=/path/to/trial_dir sbatch slurm/run_duty_cycle_A.sh   # evaluate another trial

# 1. Initialize Conda for this script
source ~/.bashrc

# 2. Activate your shared team environment
conda activate cs156b_env

# 3. Paths (read by the notebook's first cell)
export RFI_PROJECT_ROOT=/resnick/groups/CS156b/from_central/2026/njack2/rfi-removal
export RFI_DUTY_A_DIR=${RFI_DUTY_A_DIR:-$RFI_PROJECT_ROOT/results/duty_cycle/A_eval_sweep_${SLURM_JOB_ID}}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
[ -n "$RFI_MODEL_RUN_DIR" ] && export RFI_MODEL_RUN_DIR

cd "$RFI_PROJECT_ROOT"
mkdir -p executed_notebooks

# 4. Run the notebook top to bottom; the executed copy keeps all printed output and figures
echo "=== Starting duty-cycle experiment A on $(hostname), output $RFI_DUTY_A_DIR ==="
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 \
    --output-dir executed_notebooks \
    --output "duty_cycle_A_eval_sweep_${SLURM_JOB_ID}.ipynb" \
    notebooks/duty_cycle_A_eval_sweep.ipynb

echo "=== Duty-cycle experiment A finished ==="
