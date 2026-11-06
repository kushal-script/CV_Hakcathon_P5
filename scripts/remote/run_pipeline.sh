#!/bin/bash
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${WORK:-/home/work}"
WORKERS="${WORKERS:-40}"

cd "$PROJECT"
python scripts/run_pipeline.py --images "$WORK/data/stanford_cars/cars_test" --models "$PROJECT/models" \
  --out "$WORK/runs/e15_pipeline/test_pipeline.jsonl" --workers "$WORKERS"
python scripts/run_pipeline.py --images "$WORK/coco_neg/test" --models "$PROJECT/models" \
  --out "$WORK/runs/e15_pipeline/coco_neg_test_pipeline.jsonl" --workers 16
echo "job done: pipeline"
