#!/bin/bash
# Generic SLURM template for running the full soup pipeline.

#SBATCH --job-name=model_soups
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/model_soups_%j.out
#SBATCH --error=logs/model_soups_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate model_soups

export DATA_DIR="${DATA_DIR:-/scratch/$USER/model_soups_data}"
export MODELS_DIR="${MODELS_DIR:-/scratch/$USER/model_soups_models}"

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

python main.py --eval-individual-models --data-location "$DATA_DIR" --model-location "$MODELS_DIR"
python main.py --uniform-soup --data-location "$DATA_DIR" --model-location "$MODELS_DIR"
python main.py --greedy-soup --data-location "$DATA_DIR" --model-location "$MODELS_DIR"
python main.py --plot
