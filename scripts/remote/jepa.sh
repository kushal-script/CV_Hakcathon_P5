#!/bin/bash
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${WORK:-/home/work}"
BACKBONE="${BACKBONE:-hf-ijepa:facebook/ijepa_vith14_1k}"
SIZE="${SIZE:-280}"
EPOCHS="${EPOCHS:-15}"
NAME="e17_ijepa_vith14"
DET="$WORK/runs/e06_detector"

pip install -q "transformers>=4.48"
cd "$PROJECT"
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"

if [ ! -f "$WORK/runs/$NAME/probe/metrics.json" ]; then
  python -m p5.classification.linear_probe --backbone "$BACKBONE" --cache "$WORK/cache" --assets "$PROJECT/assets" \
    --out "$WORK/runs/$NAME/probe" --size 224 --det-dir "$DET" --workers 12
fi
echo "probe done"

python -m p5.classification.train --cache "$WORK/cache" --assets "$PROJECT/assets" --out "$WORK/runs/$NAME/finetune" \
  --backbone "$BACKBONE" --size "$SIZE" --epochs "$EPOCHS" --batch 32 --lr 1e-4 --head-lr 1e-3 --llrd 0.85 \
  --warmup 1 --wd 0.05 --workers 12 --eval-every 3
python -m p5.classification.predict --ckpt "$WORK/runs/$NAME/finetune/model.pt" --cache "$WORK/cache" \
  --assets "$PROJECT/assets" --out "$WORK/runs/$NAME/finetune/logits" --det-dir "$DET" --inputs gt pred --workers 12
echo "job done: $NAME"
