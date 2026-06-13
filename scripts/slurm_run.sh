#!/bin/bash
#SBATCH --job-name=maze-agent
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

set -euo pipefail

PROJECT_DIR="/mnt/nw/home/b.sturgeon/activation-multiplexing-maze-agent"

# Ensure log directory exists
mkdir -p "${PROJECT_DIR}/slurm_logs"

# Activate venv
source "${PROJECT_DIR}/.venv/bin/activate"
cd "${PROJECT_DIR}"

# Set environment for headless rendering
export MUJOCO_GL=osmesa
export MESA_GL_VERSION_OVERRIDE=3.3

# Print job info
echo "=== Job Info ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Python: $(python --version)"
echo "Torch CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "================"

# Run the script passed as argument
# Usage: sbatch scripts/slurm_run.sh python your_script.py --args
"$@"
