#!/bin/bash
#SBATCH --job-name=bootsoups
#SBATCH --account=rrg-ebrahimi
#SBATCH --array=0-71                     
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/bootsoups_%A_%a.out
#SBATCH --error=logs/bootsoups_%A_%a.err

# --- Activate your environment (edit to match your setup) ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate model_soups

SCRATCH="/scratch/omata/model-soups-data"
cp $SCRATCH/imagenet/ILSVRC2012_img_train.tar $SLURM_TMPDIR/
cp $SCRATCH/imagenet/ILSVRC2012_img_val.tar $SLURM_TMPDIR/
cp $SCRATCH/imagenet/imagenet/extract_ILSVRC.sh $SLURM_TMPDIR/
cp $SCRATCH/imagenet/imagenet/valprep.sh $SLURM_TMPDIR/


cd $SLURM_TMPDIR
./extract_ILSVRC.sh
cd imagenet/val
chmod +x "$SLURM_TMPDIR/valprep.sh" || true
bash $SLURM_TMPDIR/valprep.sh
cd $SLURM_TMPDIR

echo "[INFO] After extraction, structure is:"
find . -maxdepth 2 -type d | head

echo "[INFO] After valprep:"
find imagenet/val -mindepth 1 -maxdepth 1 -type d | head

ls $SLURM_TMPDIR >> $SCRATCH/log.txt
ls imagenet >> $SCRATCH/log.txt
ls imagenet/train >> $SCRATCH/log.txt
echo "This is validation class:"
ls imagenet/val >> $SCRATCH/log.txt

MODEL="ViT-B/32"
DATA="$SLURM_TMPDIR" 
EPOCHS=8
BATCH=256

cd ~/links/scratch/model-soups-mine

# NEW: separate location for bootstrap checkpoints
CKPT_ROOT=/scratch/omata/model-soups-bootstraps
SEED=${SLURM_ARRAY_TASK_ID}               # unique per job: 0..70
CKPT_DIR=${CKPT_ROOT}/seed_${SEED}        # seed-specific subfolder
mkdir -p "${CKPT_DIR}"

NAME="ft_bootS${SEED}"                    # unique filename stem per model

python finetune.py \
  --data-location "$DATA" \
  --model-location "$CKPT_DIR" \
  --name "$NAME" \
  --bootstrap \
  --bootstrap-seed $SEED \
  --save-last-only


echo "[INFO] Evaluation job finished at $(date)"

