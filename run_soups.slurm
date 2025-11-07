#!/bin/bash
#SBATCH --job-name=model-soups
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:0
#SBATCH --output=msoups_%j.out
set -euo pipefail

source ~/.bashrc
conda activate model_soups

export DATA_DIR=/scratch/$USER/model-soups-data
export MODELS_DIR=/scratch/$USER/model-soups-models
mkdir -p "$DATA_DIR" "$MODELS_DIR"

cd "$SLURM_SUBMIT_DIR"

python main.py --eval-individual-models --data-location "$DATA_DIR" --model-location "$MODELS_DIR"
python main.py --uniform-soup         --data-location "$DATA_DIR" --model-location "$MODELS_DIR"
python main.py --greedy-soup          --data-location "$DATA_DIR" --model-location "$MODELS_DIR"
python main.py --plot
