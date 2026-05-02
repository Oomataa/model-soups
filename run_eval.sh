#!/bin/bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate model_soups

export DATA_DIR="${DATA_DIR:-/scratch/$USER/imagenet_workspace}"
export MODEL_DIR="${MODEL_DIR:-/scratch/$USER/model_soups_models}"

python main.py \
  --eval-individual-models \
  --data-location "$DATA_DIR" \
  --model-location "$MODEL_DIR"
