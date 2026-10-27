#!/bin/bash
#SBATCH --account=def-sigmagao
#SBATCH --job-name=expF-risk
#SBATCH --partition=cpubase_bycore_b2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre06/project/6087921/isa1909/logs/experiment_F_%j.out
#SBATCH --error=/lustre06/project/6087921/isa1909/logs/experiment_F_%j.err

set -euo pipefail
PROJECT_ROOT="/lustre06/project/6087921/isa1909"
cd "${PROJECT_ROOT}"
mkdir -p logs results/experiment_F_risk_failure_propagation

module load StdEnv/2023
module load gcc/12.3
module load arrow/21.0.0
source ~/envs/multiagent/bin/activate
export MPLBACKEND=Agg

python "${PROJECT_ROOT}/experiments/experiment_F_risk_failure_propagation.py"   --output-dir "${PROJECT_ROOT}/results/experiment_F_risk_failure_propagation"
