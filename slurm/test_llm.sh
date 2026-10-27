#!/bin/bash

#SBATCH --job-name=agent_forest_test
#SBATCH --account=def-sigmagao
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G

#SBATCH --output=results/agent_forest_%j.out
#SBATCH --error=results/agent_forest_%j.err

module load python/3.11.5
module load cuda/12.9
module load opencv/4.14.0

source ~/envs/multiagent/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p results

echo "=========================================="
echo "=== Agent Forest experiment ==="
echo "=========================================="

echo
echo "Node:"
hostname

echo
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo
echo "CUDA_VISIBLE_DEVICES:"
echo "$CUDA_VISIBLE_DEVICES"

echo
echo "nvcc:"
which nvcc
nvcc --version

echo
echo "CUDA_HOME:"
echo "$CUDA_HOME"

echo
echo "Python:"
python --version

echo
echo "PyTorch / CUDA:"
python -c "import torch; \
print('PyTorch:', torch.__version__); \
print('CUDA build:', torch.version.cuda); \
print('CUDA available:', torch.cuda.is_available()); \
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo
echo "vLLM:"
python -c "import vllm; print('vLLM:', vllm.__version__)"

echo
echo "=========================================="
echo "Starting Python experiment..."
echo "=========================================="

python -u experiments/test_llm.py

EXIT_CODE=$?

echo
echo "=========================================="
echo "Experiment finished"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE