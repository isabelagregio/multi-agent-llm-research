#!/bin/bash

#SBATCH --account=def-sigmagao
#SBATCH --job-name=expC-signals
#SBATCH --partition=cpubase_bycore_b2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/expC_%j.out
#SBATCH --error=logs/expC_%j.err

set -euo pipefail

PROJECT_ROOT="/lustre06/project/6087921/isa1909"

PYTHON_SCRIPT="${PROJECT_ROOT}/experiments/experiment_C_interaction_utility_signals.py"

RAW_RESULTS="${PROJECT_ROOT}/results/experiment_7_round_scaling_difficulty/raw_results.jsonl"

RESULTS_DIR="${PROJECT_ROOT}/results/experiment_C_interaction_utility_signals"

export RAW_RESULTS
export RESULTS_DIR
export BOOTSTRAP_SAMPLES=2000
export BOOTSTRAP_SEED=20261005

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${RESULTS_DIR}"

source ~/envs/multiagent/bin/activate

echo "============================================================"
echo "Experiment C - Interaction Utility Signals"
echo "Python:  ${PYTHON_SCRIPT}"
echo "Input:   ${RAW_RESULTS}"
echo "Output:  ${RESULTS_DIR}"
echo "============================================================"

if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "ERROR: Python script not found:"
    echo "${PYTHON_SCRIPT}"
    exit 1
fi

if [ ! -f "${RAW_RESULTS}" ]; then
    echo "ERROR: Experiment 7 raw results not found:"
    echo "${RAW_RESULTS}"
    exit 1
fi

python "${PYTHON_SCRIPT}"

echo "============================================================"
echo "Experiment C completed."
echo "Results saved to:"
echo "${RESULTS_DIR}"
echo "============================================================"