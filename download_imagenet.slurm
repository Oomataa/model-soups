#!/bin/bash
# Generic SLURM template for downloading ImageNet archives.

#SBATCH --job-name=imagenet_download
#SBATCH --output=%x_%j.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

URL="https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
FILE="ILSVRC2012_img_train.tar"
EXPECTED_MD5="1d675b47d978889d74fa0da5fadfb00e"

DEST="${SCRATCH:-/scratch/$USER}/imagenet"
mkdir -p "$DEST"
cd "$DEST"

if ! curl -C - -L --retry 20 --retry-all-errors --connect-timeout 30 "$URL" -o "$FILE"; then
  wget -c --tries=100 --timeout=30 --read-timeout=30 "$URL" -O "$FILE"
fi

MD5=$(md5sum "$FILE" | awk '{print $1}')
echo "Computed MD5: $MD5"
echo "Expected MD5: $EXPECTED_MD5"

if [[ "$MD5" != "$EXPECTED_MD5" ]]; then
  echo "MD5 mismatch."
  exit 3
fi

echo "ImageNet train archive verified."
