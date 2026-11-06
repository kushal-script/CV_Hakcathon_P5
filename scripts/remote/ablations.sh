#!/bin/bash
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${WORK:-/home/work}"
EPOCHS="${EPOCHS:-15}"
BACKBONE="${BACKBONE:-tresnet_m.miil_in21k_ft_in1k}"

bash "$PROJECT/scripts/remote/prepare.sh"
cd "$PROJECT"
run() {
  local name="$1"; shift
  if [ ! -f "$WORK/runs/$name/metrics.json" ]; then
    python -m p5.classification.train --cache "$WORK/cache" --assets "$PROJECT/assets" --out "$WORK/runs/$name" \
      --backbone "$BACKBONE" --size 368 --epochs "$EPOCHS" --batch 48 --workers "$(( $(nproc) - 2 ))" "$@"
  fi
  echo "ablation done: $name"
}
run e09_control_crops
run e09_full_image --full-image
run e10_no_aug --no-aug
python -m p5.classification.export --ckpt "$WORK/runs/e09_control_crops/model.pt" --out "$WORK/runs/e09_control_crops/export"
echo "job done: ablations"
