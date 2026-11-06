#!/bin/bash
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${WORK:-/home/work}"
NAME="$1"
shift

bash "$PROJECT/scripts/remote/prepare.sh"
cd "$PROJECT"
python -c "import sys, torch; print('cuda', torch.cuda.is_available(), torch.__version__); sys.exit(0 if torch.cuda.is_available() else 1)"
python -m p5.classification.train --cache "$WORK/cache" --assets "$PROJECT/assets" --out "$WORK/runs/$NAME" \
  --workers "$(( $(nproc) - 2 ))" "$@"
python -m p5.classification.export --ckpt "$WORK/runs/$NAME/model.pt" --out "$WORK/runs/$NAME/export"
echo "job done: $NAME"
