#!/bin/bash
#SBATCH --account=def-sigmagao
#SBATCH --job-name=expH-rair
#SBATCH --partition=cpubase_bycore_b2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/lustre06/project/6087921/isa1909/logs/experiment_H_rair_%j.out
#SBATCH --error=/lustre06/project/6087921/isa1909/logs/experiment_H_rair_%j.err

set -euo pipefail

PROJECT_ROOT="/lustre06/project/6087921/isa1909"
SCRIPT="${PROJECT_ROOT}/experiments/experiment_H_risk_aware_interaction_router.py"
B_RAW="${PROJECT_ROOT}/results/experiment_7_round_scaling_difficulty/raw_results.jsonl"
OUT="${PROJECT_ROOT}/results/experiment_H_risk_aware_interaction_router"

cd "${PROJECT_ROOT}"
mkdir -p logs "${OUT}"

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load arrow/21.0.0
source ~/envs/multiagent/bin/activate

export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
export PROJECT_ROOT

echo "============================================================"
echo "Experiment H -- Risk-Aware Interaction Router (RAIR)"
echo "Start: $(date -Is)"
echo "Job ID: ${SLURM_JOB_ID:-unset}"
echo "Host: $(hostname)"
echo "Input: ${B_RAW}"
echo "Output: ${OUT}"
echo "No new LLM inference; CPU cross-fitted analysis of Experiment B traces."
echo "============================================================"

if [ ! -f "${SCRIPT}" ]; then
    echo "ERROR: script not found: ${SCRIPT}"
    exit 1
fi

if [ ! -f "${B_RAW}" ]; then
    echo "ERROR: Experiment B trace file not found: ${B_RAW}"
    exit 1
fi

python -c "import sklearn, numpy, matplotlib; print('sklearn', sklearn.__version__, 'numpy', numpy.__version__)"

python "${SCRIPT}" \
  --b-raw "${B_RAW}" \
  --output-dir "${OUT}" \
  --bootstrap-samples 2000

echo "============================================================"
echo "Completed: $(date -Is)"
echo "Results: ${OUT}"
echo "Quick read: ${OUT}/quick_read.txt"
echo "============================================================"
