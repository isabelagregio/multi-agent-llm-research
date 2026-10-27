#!/bin/bash
#SBATCH --account=def-sigmagao
#SBATCH --job-name=expG-rgad
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=a100:1
#SBATCH --time=06:00:00
#SBATCH --output=/lustre06/project/6087921/isa1909/logs/experiment_G_%j.out
#SBATCH --error=/lustre06/project/6087921/isa1909/logs/experiment_G_%j.err

set -euo pipefail
PROJECT_ROOT="/lustre06/project/6087921/isa1909"
cd "${PROJECT_ROOT}"
mkdir -p logs results/experiment_G_risk_gated_adaptive_debate

module load StdEnv/2023
module load gcc/12.3
module load arrow/21.0.0
module load cuda/12.9
source ~/envs/multiagent/bin/activate

export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MPLBACKEND=Agg

echo "Experiment G — RGAD"
echo "Node: $(hostname)"
echo "CUDA_HOME=${CUDA_HOME:-unset}"
which nvcc
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

python "${PROJECT_ROOT}/experiments/experiment_G_risk_gated_adaptive_debate.py" \
  --b-raw "${PROJECT_ROOT}/results/experiment_7_round_scaling_difficulty/raw_results.jsonl" \
  --questions "${PROJECT_ROOT}/results/experiment_7_round_scaling_difficulty/sampled_questions.json" \
  --output-dir "${PROJECT_ROOT}/results/experiment_G_risk_gated_adaptive_debate"
