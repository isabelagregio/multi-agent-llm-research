#!/bin/bash

#SBATCH --job-name=test_torch
#SBATCH --account=def-sigmagao
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=results/test_torch_%j.out
#SBATCH --error=results/test_torch_%j.err

module load python
source ~/envs/multiagent/bin/activate

echo "=== PyTorch GPU test ==="
hostname

python -u experiments/test_torch.py