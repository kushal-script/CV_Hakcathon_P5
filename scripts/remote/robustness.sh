#!/bin/bash
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${WORK:-/home/work}"
NAME="${1:-e05_tresnet_l_368}"

cd "$PROJECT"
python -m p5.eval.robustness --ckpt "$WORK/runs/$NAME/model.pt" --detector "$WORK/runs/e06_detector/detector.pt" \
  --cache "$WORK/cache" --out "$WORK/runs/e14_robustness/plain" --workers "$(( $(nproc) - 2 ))"
python -m p5.eval.robustness --ckpt "$WORK/runs/$NAME/model.pt" --detector "$WORK/runs/e06_detector/detector.pt" \
  --cache "$WORK/cache" --out "$WORK/runs/e14_robustness/m2" --workers "$(( $(nproc) - 2 ))" --m2
echo "job done: robustness"
