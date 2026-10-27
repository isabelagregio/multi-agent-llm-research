#!/bin/bash
#SBATCH --account=def-sigmagao
#SBATCH --job-name=exp6-mas
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/isa1909/projects/def-sigmagao/isa1909/logs/exp6_%j.out
#SBATCH --error=/home/isa1909/projects/def-sigmagao/isa1909/logs/exp6_%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023
module load python/3.11
module load gcc
module load arrow/25.0.0
module load cuda/12.9
# Load the exact OpenCV version available on Narval before activating the venv.
# Example only: module load opencv/<version>

source ~/envs/multiagent/bin/activate
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

PROJECT_ROOT=/home/isa1909/projects/def-sigmagao/isa1909
SCRIPT=$PROJECT_ROOT/experiments/experiment_6_single_forest_selfrefine_studentcritic_debate.py
mkdir -p "$PROJECT_ROOT/logs"

export N_QUESTIONS=${N_QUESTIONS:-200}
export N_REPETITIONS=${N_REPETITIONS:-3}
export FOREST_N=${FOREST_N:-5}
export REFINE_ROUNDS=${REFINE_ROUNDS:-2}
export CRITIC_ROUNDS=${CRITIC_ROUNDS:-2}
export DEBATE_ROUNDS=${DEBATE_ROUNDS:-1}

START=$(date +%s)
echo "Host: $(hostname)"
echo "Start: $(date -Is)"
echo "GPU allocation: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

time python "$SCRIPT"

END=$(date +%s)
echo "End: $(date -Is)"
echo "Total shell elapsed seconds: $((END-START))"
nvidia-smi || true
