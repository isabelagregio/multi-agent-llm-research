#!/bin/bash

#SBATCH --account=def-sigmagao
#SBATCH --job-name=qwen25-100
#SBATCH --partition=gpubase_bygpu_b2

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

#SBATCH --gpus-per-node=a100:1
#SBATCH --time=06:00:00

#SBATCH --output=/home/isa1909/projects/def-sigmagao/isa1909/logs/qwen25-100-%j.out
#SBATCH --error=/home/isa1909/projects/def-sigmagao/isa1909/logs/qwen25-100-%j.err

set -euo pipefail

PROJECT_ROOT="/home/isa1909/projects/def-sigmagao/isa1909"
cd "$PROJECT_ROOT"
mkdir -p logs
mkdir -p results/experiment_4_role_specialization_qwen25_7b

module load StdEnv/2023
module load python/3.11
module load arrow/25.0.0
source "$PROJECT_ROOT/.venv/bin/activate"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export N_QUESTIONS="${N_QUESTIONS:-100}"
export N_REPETITIONS="${N_REPETITIONS:-1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
export MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-8192}"
export TEMPERATURE="${TEMPERATURE:-0.7}"
export GENERATION_SEED="${GENERATION_SEED:-42}"

echo "============================================================"
echo "SLURM job: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Project: $PROJECT_ROOT"
echo "Python: $(which python)"
python --version
echo "N_QUESTIONS: $N_QUESTIONS"
echo "============================================================"

nvidia-smi

python - <<'PY2'
import torch, transformers, datasets
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY2

python "$PROJECT_ROOT/experiments/experiment_role_specialization_qwen25_7b_narval.py"
