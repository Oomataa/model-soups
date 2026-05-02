#!/bin/bash
# Variant of the evaluation template for an alternate checkpoint directory.

#SBATCH --job-name=eval_models70
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_models70_%j.out
#SBATCH --error=logs/eval_models70_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate model_soups

export DATA_DIR="${DATA_DIR:-/scratch/$USER/imagenet_workspace}"
export MODEL_DIR="${MODEL_DIR:-/scratch/$USER/model_soups_models_70}"
export WORK_DIR="${WORK_DIR:-$SLURM_SUBMIT_DIR}"

mkdir -p "$WORK_DIR" logs
cd "$WORK_DIR"

python "$SLURM_SUBMIT_DIR/main.py" \
  --eval-individual-models \
  --greedy-soup \
  --data-location "$DATA_DIR" \
  --model-location "$MODEL_DIR"
