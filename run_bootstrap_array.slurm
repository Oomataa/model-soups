#!/bin/bash
# Generic SLURM template for bootstrap fine-tuning jobs.

#SBATCH --job-name=bootstrap_soups
#SBATCH --array=0-1399%50
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/bootstrap_%A_%a.out
#SBATCH --error=logs/bootstrap_%A_%a.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate model_soups

NUM_MODELS=70
TRIALS_PER_MODEL=20
GLOBAL_IDX=${SLURM_ARRAY_TASK_ID}
MODEL_ID=$((GLOBAL_IDX / TRIALS_PER_MODEL))
TRIAL_ID=$((GLOBAL_IDX % TRIALS_PER_MODEL))

if (( MODEL_ID >= NUM_MODELS )); then
  echo "MODEL_ID ${MODEL_ID} is out of range; skipping."
  exit 0
fi

export DATA_DIR="${DATA_DIR:-/scratch/$USER/imagenet_workspace}"
export PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
export CKPT_ROOT="${CKPT_ROOT:-$PROJECT_DIR/checkpoints/bootstrap_runs}"

mkdir -p "$CKPT_ROOT" logs
CKPT_DIR="${CKPT_ROOT}/model_${MODEL_ID}/trial_${TRIAL_ID}"
mkdir -p "$CKPT_DIR"

cd "$PROJECT_DIR"

python finetune.py \
  --data-location "$DATA_DIR" \
  --model-location "$CKPT_DIR" \
  --name "ft_m${MODEL_ID}_t${TRIAL_ID}" \
  --batch-size 256 \
  --workers 8 \
  --model-id "$MODEL_ID" \
  --trial-id "$TRIAL_ID" \
  --bootstrap \
  --bootstrap-seed "$MODEL_ID"
