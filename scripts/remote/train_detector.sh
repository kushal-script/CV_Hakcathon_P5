#!/bin/bash
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${WORK:-/home/work}"
COCO="$WORK/coco"

bash "$PROJECT/scripts/remote/prepare.sh"
cd "$PROJECT"
python -c "import sys, torch; print('cuda', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)"

if [ ! -d "$WORK/coco_neg/train" ]; then
  mkdir -p "$COCO"
  curl -sSL --retry 5 -o "$COCO/ann.zip" http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  curl -sSL --retry 5 -o "$COCO/val2017.zip" http://images.cocodataset.org/zips/val2017.zip
  unzip -q "$COCO/ann.zip" -d "$COCO" && unzip -q "$COCO/val2017.zip" -d "$COCO"
  rm "$COCO/ann.zip" "$COCO/val2017.zip"
  python scripts/coco_negatives.py --coco "$COCO" --assets "$WORK/assets_out" --out "$WORK/coco_neg"
fi

mkdir -p "$WORK/runs/e06_detector"
cd "$WORK/runs/e06_detector"
python -m p5.detection.train --cache "$WORK/cache" --assets "$PROJECT/assets" --out "$WORK/runs/e06_detector" \
  --negatives "$WORK/coco_neg" --workers "$(( $(nproc) - 2 ))" "$@"
echo "job done: e06_detector"
