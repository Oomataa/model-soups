#!/bin/bash
#SBATCH --job-name=bootsoups
#SBATCH --array=0-5                  
#SBATCH --time=20:00:00
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/bootsoups_%A_%a.out
#SBATCH --error=logs/bootsoups_%A_%a.err

WORKDIR=$(pwd)
WORK_DATA_ROOT="/work/kahou_lab/omata/imagenet_work"
mkdir -p "$WORK_DATA_ROOT"
DATA_ROOT="/home/ikhianosen.ehizokhal/code/model-soups-data/imagenet"


mkdir -p "$WORKDIR/logs"

#account=rrg-ebrahimi 
#partition=gpubase_bygpu_b2
#gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1

# --- Activate your environment (edit to match your setup) ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate model_soups
cd $WORK_DATA_ROOT
echo "[INFO] Using WORK_DATA_ROOT=$WORK_DATA_ROOT"

if [ ! -d "${WORK_DATA_ROOT}/imagenet/train" ]; then
    echo "[INFO] ImageNet not found in ${WORK_DATA_ROOT}, copying tars and extracting..."

    cp "$DATA_ROOT/ILSVRC2012_img_train.tar" .
    cp "$DATA_ROOT/ILSVRC2012_img_val.tar" .
    cp "$DATA_ROOT/imagenet/extract_ILSVRC.sh" .
    cp "$DATA_ROOT/imagenet/valprep.sh" .

    chmod +x ./extract_ILSVRC.sh 
    ./extract_ILSVRC.sh

    cd imagenet/val
    chmod +x "$WORK_DATA_ROOT/valprep.sh" || true
    bash "$WORK_DATA_ROOT/valprep.sh"
    cd "$WORK_DATA_ROOT"
else
    echo "[INFO] Found existing extracted ImageNet at ${WORK_DATA_ROOT}/imagenet, skipping extraction."
fi

echo "[INFO] After extraction, structure is:"
find . -maxdepth 2 -type d | head

echo "[INFO] After valprep:"
find imagenet/val -mindepth 1 -maxdepth 1 -type d | head

ls "$WORK_DATA_ROOT"     >> "$WORKDIR/logs/scratch_listing.txt" || true
ls imagenet              >> "$WORKDIR/logs/scratch_listing.txt" || true
ls imagenet/train        >> "$WORKDIR/logs/scratch_listing.txt" || true
ls imagenet/val          >> "$WORKDIR/logs/scratch_listing.txt" || true

MODEL="ViT-B/32"
DATA="$WORK_DATA_ROOT" 


cd ~/code/model-soups

# NEW: separate location for bootstrap checkpoints
CKPT_ROOT="$WORKDIR/model-soups-bootstraps-70hparamtest" 
SEED=${SLURM_ARRAY_TASK_ID}               # unique per job: 0..70
CKPT_DIR=${CKPT_ROOT}/seed_${SEED}        # seed-specific subfolder
mkdir -p "$CKPT_DIR" "$WORKDIR/logs"

NAME="ft_bootS${SEED}"                    # unique filename stem per model

python finetune.py \
    --data-location "$DATA" \
    --model-location "$CKPT_DIR" \
    --name "$NAME" \
    --bootstrap \
    --bootstrap-seed $SEED \
    --search-hparams \
    --num-trials 5
  


echo "[INFO] Evaluation job finished at $(date)"



# Optional cleanup (ARC will delete in 5 days anyway)
#rm -rf $WORK_DATA_ROOT/*
