#!/bin/bash
# Submit both duty-cycle experiments: A (evaluation sweep, CPU) and B (training mix, GPU).
# Usage:
#   bash slurm/submit_duty_cycle.sh        # both
#   bash slurm/submit_duty_cycle.sh A      # only A
#   bash slurm/submit_duty_cycle.sh B      # only B

cd /resnick/groups/CS156b/from_central/2026/njack2/rfi-removal
for experiment in ${@:-A B}; do
    sbatch slurm/run_duty_cycle_${experiment}.sh
done
