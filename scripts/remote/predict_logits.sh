#!/bin/bash
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${WORK:-/home/work}"
NAME="${1:-e05_tresnet_l_368}"

cd "$PROJECT"
python -m p5.classification.predict --ckpt "$WORK/runs/$NAME/model.pt" --cache "$WORK/cache" --assets "$PROJECT/assets" \
  --out "$WORK/runs/$NAME/logits" --det-dir "$WORK/runs/e06_detector" --workers "$(( $(nproc) - 2 ))"
echo "job done: logits $NAME"
