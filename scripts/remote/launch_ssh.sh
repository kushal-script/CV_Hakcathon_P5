#!/bin/bash
set -euo pipefail
HOST="$1"
NAME="$2"
shift 2
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -o LogLevel=ERROR"

rsync -az --delete -e "$SSH" --exclude /data --exclude /cache --exclude /runs --exclude /models --exclude /.venv \
  --exclude /.git --exclude '*.pdf' --exclude __pycache__ "$ROOT/" "root@$HOST:/home/p5/"
CMD="$*"
$SSH "root@$HOST" "mkdir -p /home/jl-runs/$NAME && cd /home/p5 && nohup bash -lc '. .venv/bin/activate && $CMD' \
  > /home/jl-runs/$NAME/output.log 2>&1 < /dev/null & echo \$! > /home/jl-runs/$NAME/pid; \
  (while kill -0 \$(cat /home/jl-runs/$NAME/pid) 2>/dev/null; do sleep 5; done) >/dev/null 2>&1 &"
echo "launched $NAME on $HOST"
