#!/bin/bash
#SBATCH --job-name=eval_soups              # Job name

#SBATCH --gres=gpu:1                       # Request 1 GPU
#SBATCH --cpus-per-task=8                  # Number of CPU cores
#SBATCH --mem=32G                          # Memory
#SBATCH --time=4:00:00                     # Walltime (4 hours)
#SBATCH --account=def-ebrahimi-ab          # Your PI’s account
#SBATCH --output=logs/model_soup_output_%j.log
#SBATCH --error=logs/model_soup_error_%j.log
# Create logs directory if it doesn't exist
mkdir -p logs
# -----------------------------
# Load modules and activate environment
module load python/3.10
module load cuda/12.1

source ~/miniconda3/bin/activate model_soups

# -----------------------------
# Define paths
export DATA_DIR=/scratch/$USER/imagenet      # adjust if needed
export MODEL_DIR=/scratch/$USER/model-soups-models

# -----------------------------
# Run your command
python main.py \
  --eval-individual-models \
  --data-location "$DATA_DIR" \
  --model-location "$MODEL_DIR"

