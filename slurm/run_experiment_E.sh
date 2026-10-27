#!/bin/bash
#SBATCH --account=def-sigmagao
#SBATCH --job-name=expE-cost
#SBATCH --partition=cpubase_bycore_b2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre06/project/6087921/isa1909/logs/experiment_E_%j.out
#SBATCH --error=/lustre06/project/6087921/isa1909/logs/experiment_E_%j.err

set -euo pipefail

PROJECT_ROOT="/lustre06/project/6087921/isa1909"

cd "${PROJECT_ROOT}"

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/results/experiment_E_cost_aware"

# Experiment E is offline/CPU-only.
# Arrow is loaded because the existing research environment may import
# packages that depend on pyarrow.
module load StdEnv/2023
module load gcc/12.3
module load arrow/21.0.0

source ~/envs/multiagent/bin/activate

export MPLBACKEND=Agg

echo "=========================================="
echo "Experiment E — Cost-Aware Evaluation"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

python "${PROJECT_ROOT}/experiments/experiment_E_cost_aware_evaluation.py" \
    --input-dir "${PROJECT_ROOT}/results/experiment_D_adaptive_stopping" \
    --output-dir "${PROJECT_ROOT}/results/experiment_E_cost_aware" \
    --bootstrap-samples 2000

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
