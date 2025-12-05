#!/bin/bash
#SBATCH --job-name=eval_models
#SBATCH --account=rrg-ebrahimi
#SBATCH --time=2-00:00:00       # up to 2 days
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=h100_1g.10gb:1             # remove or adjust if CPU-only
#SBATCH --output=eval_models.out
#SBATCH --error=eval_models.err


echo $DATA_DIR
echo $CONDA_DEFAULT_ENV

source ~/miniconda3/etc/profile.d/conda.sh
conda activate model_soups

DATA_DIR="/work/kahou_lab/omata/imagenet_work"

# change this path if needed
SCRATCH="/home/ikhianosen.ehizokhal/code/model-soups-data"
cp $SCRATCH/imagenet/ILSVRC2012_img_train.tar $DATA_DIR/
cp $SCRATCH/imagenet/ILSVRC2012_img_val.tar $DATA_DIR/
cp $SCRATCH/imagenet/imagenet/extract_ILSVRC.sh $DATA_DIR/
cp $SCRATCH/imagenet-a.tar $DATA_DIR/
cp $SCRATCH/imagenet-r.tar $DATA_DIR/
cp $SCRATCH/imagenetv2-matched-frequency.tar.gz $DATA_DIR/
cp $SCRATCH/objectnet-1.0.zip $DATA_DIR/
cp $SCRATCH/sketch.zip $DATA_DIR/
cp $SCRATCH/imagenet/imagenet/valprep.sh $DATA_DIR/


ls $DATA_DIR >> $SCRATCH/log.txt

module load httpproxy

cd $DATA_DIR
./extract_ILSVRC.sh
cd imagenet/val
chmod +x "$DATA_DIR/valprep.sh" || true
bash $DATA_DIR/valprep.sh
cd $DATA_DIR
tar -xvf imagenet-a.tar
tar -xvf imagenet-r.tar

unzip -P objectnetisatestset objectnet-1.0.zip
#tar -xvf imagenetv2-matched-frequency.tar.gz extracted by script
unzip sketch.zip
mv imagenet-sketch/sketch sketch/
rmdir imagenet-sketch

ls $DATA_DIR >> $SCRATCH/log.txt
ls imagenet >> $SCRATCH/log.txt
ls imagenet/train >> $SCRATCH/log.txt
ls imagenet/val >> $SCRATCH/log.txt

# Define paths
DATA_DIR="/work/kahou_lab/omata/imagenet_work"
#MODEL_DIR="/scratch/omata/model-soups-bootstrap-stage"
MODEL_DIR="~/code/model-soups/model-soups-bootstraps-70p"

#Move into your project directory
#WORK=~/links/scratch/model-soups-runs/bootstrap_greedy_1
WORK=~/code/model-soups/model-soups-runs/bootstrap_greedy_1
mkdir -p "$WORK"
cd "$WORK"
#python test.py

if [ ! -f ./imagenet_98_idxs.npy ]; then
    cp /home/ikhianosen.ehizokhal/code/model-soups/imagenet_98_idxs.npy ./
    echo "Copied imagenet_98_idxs.npy to current directory."
else
    echo "File already exists. Skipping copy."
fi

# Run your command

python /home/ikhianosen.ehizokhal/code/model-soups/main.py \
  --eval-individual-models \
  --greedy-soup \
  --data-location "$DATA_DIR" \
  --model-location "$MODEL_DIR"

echo "[INFO] Evaluation job finished at $(date)"

