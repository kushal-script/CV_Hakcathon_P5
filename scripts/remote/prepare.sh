#!/bin/bash
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${WORK:-/home/work}"
DATA="$WORK/data/stanford_cars"
URL="https://www.kaggle.com/api/v1/datasets/download/rickyyyyyyy/torchvision-stanford-cars"

pip install -q -r "$PROJECT/requirements-train.txt"
pip install -q -e "$PROJECT" --no-deps

if [ ! -f "$DATA/devkit/cars_meta.mat" ]; then
  mkdir -p "$WORK/data"
  curl -sSL --retry 5 -o "$WORK/data/cars.zip" "$URL"
  unzip -q "$WORK/data/cars.zip" -d "$WORK/data"
  rm "$WORK/data/cars.zip"
fi

python "$PROJECT/scripts/build_assets.py" --data "$DATA" --out "$WORK/assets_check" --skip-box-check
diff -q "$WORK/assets_check/splits/val_ids.txt" "$PROJECT/assets/splits/val_ids.txt"

if [ ! -f "$WORK/cache/index_test.csv" ]; then
  python "$PROJECT/scripts/build_cache.py" --data "$DATA" --out "$WORK/cache" --workers "$(nproc)"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "prepare done"
