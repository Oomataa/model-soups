#!/bin/bash
#SBATCH --job-name=imagenet_download
#SBATCH --output=%x_%j.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=def-ebrahimi-ab   # or rrg-ebrahimi-ab if that's your group

set -euo pipefail

URL="https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
FILE="ILSVRC2012_img_train.tar"
EXPECTED_MD5="1d675b47d978889d74fa0da5fadfb00e"

DEST="${SCRATCH}/imagenet"
mkdir -p "$DEST"
cd "$DEST"

echo "[$(date)] Node: $(hostname)"
echo "[$(date)] Working dir: $(pwd)"
echo "[$(date)] Free space:"
df -h .

module load httpproxy

echo "[$(date)] Starting/resuming download..."
if ! curl -C - -L \
  --retry 200 --retry-all-errors \
  --connect-timeout 30 --max-time 0 \
  "$URL" -o "$FILE"; then
  echo "[$(date)] curl failed, trying wget fallback..."
  wget -c --tries=1000 --timeout=30 --read-timeout=30 --progress=dot:giga "$URL" -O "$FILE"
fi

echo "[$(date)] Download finished:"
ls -lh "$FILE"

echo "[$(date)] Probing archive readability..."
tar -tf "$FILE" | head -n 5

echo "[$(date)] Computing MD5..."
MD5=$(md5sum "$FILE" | awk '{print $1}')
echo "Computed MD5: $MD5"
echo "Expected MD5: $EXPECTED_MD5"
if [[ "$MD5" != "$EXPECTED_MD5" ]]; then
  echo "[$(date)] WARNING: MD5 mismatch — file likely incomplete or corrupted."
  exit 3
fi

echo "[$(date)] SUCCESS: ImageNet train tar verified."
