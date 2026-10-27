#!/bin/bash
#SBATCH --account=def-sigmagao
#SBATCH --job-name=expB2-protocol
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/expB2_%j.out
#SBATCH --error=logs/expB2_%j.err

set -euo pipefail

PROJECT_ROOT="/lustre06/project/6087921/isa1909"
SCRIPT="${PROJECT_ROOT}/experiments/experiment_B2_protocol_validation.py"
RAW_RESULTS_B="${PROJECT_ROOT}/results/experiment_7_round_scaling_difficulty/raw_results.jsonl"
RESULTS_DIR="${PROJECT_ROOT}/results/experiment_B2_protocol_validation"

cd "${PROJECT_ROOT}"
mkdir -p logs "${RESULTS_DIR}"

module --force purge
module load StdEnv/2023
module load python/3.11
module load gcc
module load arrow/25.0.0
module load cuda/12.9

source ~/envs/multiagent/bin/activate

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PROJECT_ROOT
export RAW_RESULTS_B
export RESULTS_DIR
export N_QUESTIONS="${N_QUESTIONS:-300}"
export N_REPETITIONS="${N_REPETITIONS:-3}"
export MAX_CRITIC_ROUNDS="${MAX_CRITIC_ROUNDS:-3}"

if [ ! -f "${SCRIPT}" ]; then
    echo "ERROR: Python script not found: ${SCRIPT}"
    exit 1
fi

if [ ! -f "${RAW_RESULTS_B}" ]; then
    echo "ERROR: Experiment B traces not found: ${RAW_RESULTS_B}"
    exit 1
fi

echo "============================================================"
echo "B2 - Protocol Validation"
echo "Start: $(date -Is)"
echo "Job ID: ${SLURM_JOB_ID:-unset}"
echo "Questions: ${N_QUESTIONS}"
echo "Repetitions: ${N_REPETITIONS}"
echo "Critic rounds: ${MAX_CRITIC_ROUNDS}"
echo "Input: ${RAW_RESULTS_B}"
echo "Output: ${RESULTS_DIR}"
echo "============================================================"

START=$(date +%s)
nvidia-smi || true

python "${SCRIPT}"

END=$(date +%s)
echo "============================================================"
echo "Completed: $(date -Is)"
echo "Elapsed seconds: $((END-START))"
echo "Results: ${RESULTS_DIR}"
echo "============================================================"
