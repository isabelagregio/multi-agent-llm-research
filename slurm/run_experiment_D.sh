#!/bin/bash
#SBATCH --account=def-sigmagao
#SBATCH --job-name=expD-adaptive
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=a100:1
#SBATCH --time=06:00:00
#SBATCH --output=/lustre06/project/6087921/isa1909/logs/experiment_D_%j.out
#SBATCH --error=/lustre06/project/6087921/isa1909/logs/experiment_D_%j.err

set -euo pipefail

PROJECT_ROOT="/lustre06/project/6087921/isa1909"

cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/results/experiment_D_adaptive_stopping"

module purge

module load StdEnv/2023
module load gcc/12.3
module load arrow/21.0.0
module load cuda/12.9


source ~/envs/multiagent/bin/activate

export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "CUDA_HOME=${CUDA_HOME:-unset}"
which nvcc
nvcc --version

python -c "import torch; print('torch:', torch.__version__); print('torch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"


echo "=========================================="
echo "Experiment D - Adaptive Self-Refine"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Project root: ${PROJECT_ROOT}"
echo "=========================================="

python "${PROJECT_ROOT}/experiments/experiment_D_adaptive_stopping.py" \
    --repetitions 3

# Optional confirmatory run with TRUE adaptive execution and measured wall time.
# This performs extra inference, so leave it commented for the first run.
#
# python "${PROJECT_ROOT}/experiments/experiment_D_adaptive_stopping.py" \
#     --repetitions 3 \
#     --runtime-validation

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
