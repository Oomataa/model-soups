#!/bin/bash
#SBATCH --job-name=eval_models
#SBATCH --account=rrg-ebrahimi
#SBATCH --time=2-00:00:00       # up to 2 days
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=h100_1g.10gb:1             # remove or adjust if CPU-only
#SBATCH --output=eval_models.out
#SBATCH --error=eval_models.err


echo $SLURM_TMPDIR
echo $CONDA_DEFAULT_ENV

source ~/miniconda3/etc/profile.d/conda.sh
conda activate model_soups




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


ls $SLURM_TMPDIR >> $SCRATCH/log.txt
python test.py
module load httpproxy
python test.py
cd $SLURM_TMPDIR
./extract_ILSVRC.sh
tar -xvf imagenet-a.tar
tar -xvf imagenet-r.tar
#tar -xvf imagenetv2-matched-frequency.tar.gz extracted by script
unzip -P objectnetisatestset objectnet-1.0.zip

unzip sketch.zip
mv imagenet-sketch/sketch sketch/
rmdir imagenet-sketch

ls $SLURM_TMPDIR >> $SCRATCH/log.txt
ls imagenet >> $SCRATCH/log.txt
ls imagenet/train >> $SCRATCH/log.txt
ls imagenet/val >> $SCRATCH/log.txt

# Define paths
DATA_DIR="$SLURM_TMPDIR"
MODEL_DIR="/scratch/omata/model-soups-models"

#Move into your project directory
cd ~/links/scratch/model-soups-mine
python test.py
# Run your command
python main.py \
  --eval-individual-models \
  --data-location "$DATA_DIR" \
  --model-location "$MODEL_DIR"

echo "[INFO] Evaluation job finished at $(date)"

