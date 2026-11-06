#!/bin/bash
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${WORK:-/home/work}"
NAME="$1"
GPU="$2"
shift 2

pip install -q -r "$PROJECT/requirements-train.txt"
pip install -q -e "$PROJECT" --no-deps
cd "$PROJECT"
export CUDA_VISIBLE_DEVICES="$GPU"
python -c "import sys, torch; print('cuda', torch.cuda.is_available(), torch.__version__); sys.exit(0 if torch.cuda.is_available() else 1)"
WORKERS="$(( ($(nproc) - 2) / 2 ))"
python -m p5.classification.train --assets "$PROJECT/assets/india" --split-csv "$PROJECT/assets/india/split.csv" \
  --crops "$WORK/india/crops" --out "$WORK/runs/$NAME" --workers "$WORKERS" "$@"
python -m p5.classification.export --ckpt "$WORK/runs/$NAME/model.pt" --out "$WORK/runs/$NAME/export"
python -m p5.classification.predict_csv --ckpt "$WORK/runs/$NAME/model.pt" --split-csv "$PROJECT/assets/india/split.csv" \
  --crops "$WORK/india/crops" --out "$WORK/runs/$NAME/logits" --workers "$WORKERS"
OUT="$WORK/runs/$NAME/export_int8"
mkdir -p "$OUT"
cp "$WORK/runs/$NAME/export/classifier.onnx" "$WORK/runs/$NAME/export/classifier_meta.json" "$WORK/runs/$NAME/export/head.npz" "$OUT/"
python scripts/quantize.py --export "$OUT" --assets "$PROJECT/assets/india" --split-csv "$PROJECT/assets/india/split.csv" \
  --crops "$WORK/india/crops" --threads 8 --calib 320 --save-test-logits
echo "job done: $NAME"
