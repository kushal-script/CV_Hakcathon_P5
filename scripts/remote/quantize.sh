#!/bin/bash
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${WORK:-/home/work}"
NAME="${1:-e05_tresnet_l_368}"
CALIB="${CALIB:-256}"

cd "$PROJECT"
OUT="$WORK/runs/$NAME/export_int8"
mkdir -p "$OUT"
cp "$WORK/runs/$NAME/export/classifier.onnx" "$WORK/runs/$NAME/export/classifier_meta.json" "$WORK/runs/$NAME/export/head.npz" "$OUT/"
python scripts/quantize.py --export "$OUT" --cache "$WORK/cache" --assets "$PROJECT/assets" --threads 2 --calib "$CALIB"
echo "job done: quantize $NAME"
