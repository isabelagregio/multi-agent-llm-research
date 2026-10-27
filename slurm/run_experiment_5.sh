#!/bin/bash
#SBATCH --account=def-sigmagao
#SBATCH --job-name=qwen25-controlled-mas
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/isa1909/projects/def-sigmagao/isa1909/logs/qwen25-controlled-mas-%j.out
#SBATCH --error=/home/isa1909/projects/def-sigmagao/isa1909/logs/qwen25-controlled-mas-%j.err

set -euo pipefail

PROJECT_ROOT="/home/isa1909/projects/def-sigmagao/isa1909"
SCRIPT_NAME="experiment_5_single_forest_selfrefine_studentcritic.py"
RESULTS_SUBDIR="experiment_5_single_forest_selfrefine_studentcritic"

cd "$PROJECT_ROOT"
mkdir -p logs
mkdir -p "results/$RESULTS_SUBDIR"

module --force purge

module load StdEnv/2023
module load python/3.11
module load gcc
module load arrow/25.0.0
module load cuda/12.9
module load opencv/4.14.0

source ~/envs/multiagent/bin/activate


export PROJECT_ROOT="$PROJECT_ROOT"
export MODEL_PATH="$PROJECT_ROOT/models/Qwen2.5-7B-Instruct"
export RESULTS_SUBDIR="$RESULTS_SUBDIR"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Final-run defaults.
# Smoke test:
#   N_QUESTIONS=20 N_REPETITIONS=1 sbatch run_experiment_5.sh
export N_QUESTIONS="${N_QUESTIONS:-200}"
export N_REPETITIONS="${N_REPETITIONS:-3}"

# Matched 5-call budget:
# Forest=5; Self-Refine=1+2*(feedback+refine); Student-Critic=1+2*(critic+update)
export FOREST_N="${FOREST_N:-5}"
export REFINE_ROUNDS="${REFINE_ROUNDS:-2}"
export CRITIC_ROUNDS="${CRITIC_ROUNDS:-2}"

export TEMPERATURE="${TEMPERATURE:-0.7}"
export TOP_P="${TOP_P:-0.95}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export DATASET_SEED="${DATASET_SEED:-42}"
export GENERATION_SEED="${GENERATION_SEED:-42}"

echo "============================================================"
echo "SLURM job: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Python: $(which python)"
python --version
echo "N_QUESTIONS: $N_QUESTIONS"
echo "N_REPETITIONS: $N_REPETITIONS"
echo "FOREST_N: $FOREST_N"
echo "REFINE_ROUNDS: $REFINE_ROUNDS"
echo "CRITIC_ROUNDS: $CRITIC_ROUNDS"
echo "MAX_NUM_SEQS: $MAX_NUM_SEQS"
echo "============================================================"

nvidia-smi

python - <<'PY'
import torch, transformers, datasets, vllm, math_verify
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("vllm:", vllm.__version__)
print("math_verify: OK")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

START_TIME=$(date +%s)
python -u "$PROJECT_ROOT/experiments/$SCRIPT_NAME"
END_TIME=$(date +%s)

echo "============================================================"
echo "Experiment completed."
echo "Elapsed seconds: $((END_TIME - START_TIME))"
echo "Results: $PROJECT_ROOT/results/$RESULTS_SUBDIR"
echo "============================================================"
