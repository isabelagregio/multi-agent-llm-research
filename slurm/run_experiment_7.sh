#!/bin/bash
#SBATCH --account=def-sigmagao
#SBATCH --job-name=exp7-rounds
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/isa1909/projects/def-sigmagao/isa1909/logs/exp7_%j.out
#SBATCH --error=/home/isa1909/projects/def-sigmagao/isa1909/logs/exp7_%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023
module load python/3.11
module load gcc
module load arrow/25.0.0
module load cuda/12.9

source ~/envs/multiagent/bin/activate
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

PROJECT_ROOT=/home/isa1909/projects/def-sigmagao/isa1909
SCRIPT=$PROJECT_ROOT/experiments/experiment_7_round_scaling_difficulty.py
mkdir -p "$PROJECT_ROOT/logs"

export N_QUESTIONS=${N_QUESTIONS:-500}
export BASE_A_QUESTIONS=${BASE_A_QUESTIONS:-200}
export BASE_A_SEED=${BASE_A_SEED:-42}
export N_REPETITIONS=${N_REPETITIONS:-3}
export ROUND_VALUES=${ROUND_VALUES:-0,1,2,3}
export RESULTS_SUBDIR=${RESULTS_SUBDIR:-experiment_7_round_scaling_difficulty}

START=$(date +%s)
echo "Host: $(hostname)"
echo "Start: $(date -Is)"
echo "Job ID: ${SLURM_JOB_ID:-unset}"
echo "GPU allocation: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "N_QUESTIONS=$N_QUESTIONS"
echo "BASE_A_QUESTIONS=$BASE_A_QUESTIONS"
echo "N_REPETITIONS=$N_REPETITIONS"
echo "ROUND_VALUES=$ROUND_VALUES"
nvidia-smi || true

time python "$SCRIPT"

END=$(date +%s)
echo "End: $(date -Is)"
echo "Total shell elapsed seconds: $((END-START))"
nvidia-smi || true
