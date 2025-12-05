#!/bin/bash
#SBATCH --job-name=eval_models_70test
#SBATCH --account=rrg-ebrahimi
#SBATCH --time=2-00:00:00       # up to 2 days
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1             # remove or adjust if CPU-only
#SBATCH --output=eval_models_70test.out
#SBATCH --error=eval_models_70test.err


echo $SLURM_TMPDIR
echo $CONDA_DEFAULT_ENV
echo "[Step 1] Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate model_soups

echo "[Step 2] Copying datasets to $SLURM_TMPDIR..."
# change this path if needed
SCRATCH="/scratch/omata/model-soups-data"
cp $SCRATCH/imagenet/ILSVRC2012_img_train.tar $SLURM_TMPDIR/
cp $SCRATCH/imagenet/ILSVRC2012_img_val.tar $SLURM_TMPDIR/
cp $SCRATCH/imagenet/imagenet/extract_ILSVRC.sh $SLURM_TMPDIR/
cp $SCRATCH/imagenet-a.tar $SLURM_TMPDIR/
cp $SCRATCH/imagenet-r.tar $SLURM_TMPDIR/
cp $SCRATCH/imagenetv2-matched-frequency.tar.gz $SLURM_TMPDIR/
cp $SCRATCH/objectnet-1.0.zip $SLURM_TMPDIR/
cp $SCRATCH/sketch.zip $SLURM_TMPDIR/
cp $SCRATCH/imagenet/imagenet/valprep.sh $SLURM_TMPDIR/


ls $SLURM_TMPDIR >> $SCRATCH/log.txt
#python test.py
module load httpproxy
#python test.py
cd $SLURM_TMPDIR
echo "[Step 3] Extracting ImageNet..."
./extract_ILSVRC.sh
cd imagenet/val
chmod +x "$SLURM_TMPDIR/valprep.sh" || true
echo "[Step 4] Preparing val directory..."
bash $SLURM_TMPDIR/valprep.sh
cd $SLURM_TMPDIR
tar -xvf imagenet-a.tar
tar -xvf imagenet-r.tar
#tar -xvf imagenetv2-matched-frequency.tar.gz extracted by script
unzip -P objectnetisatestset objectnet-1.0.zip

unzip sketch.zip
mv imagenet-sketch/sketch sketch/
rmdir imagenet-sketch

echo "[INFO] Listing contents of SLURM_TMPDIR" >> $SCRATCH/log.txt
ls $SLURM_TMPDIR >> $SCRATCH/log.txt

echo "[INFO] Listing contents of imagenet root" >> $SCRATCH/log.txt
ls imagenet >> $SCRATCH/log.txt

echo "[INFO] Listing contents of imagenet/train" >> $SCRATCH/log.txt
ls imagenet/train >> $SCRATCH/log.txt

echo "[INFO] Listing contents of imagenet/val" >> $SCRATCH/log.txt
ls imagenet/val >> $SCRATCH/log.txt

# Define paths
DATA_DIR="$SLURM_TMPDIR"
#MODEL_DIR="/scratch/omata/model-soups-bootstrap-stage"
MODEL_DIR="~/code/model-soups/model-soups-bootstraps-70p"

#Move into your project directory
#WORK=~/links/scratch/model-soups-runs/bootstrap_greedy_1
WORK=~/code/model-soups/model-soups-runs/bootstrap_greedy_1
mkdir -p "$WORK"
cd "$WORK"
#python test.py

if [ ! -f ./imagenet_98_idxs.npy ]; then
    cp /home/omata/code/model-soups/imagenet_98_idxs.npy ./
    echo "Copied imagenet_98_idxs.npy to current directory."
else
    echo "File already exists. Skipping copy."
fi

# Run your command
echo "[Step 4] Preparing val directory..."
python ~/code/model-soups/main.py \
  --eval-individual-models \
  --greedy-soup \
  --data-location "$DATA_DIR" \
  --model-location "$MODEL_DIR"

echo "[INFO] Evaluation job finished at $(date)"

