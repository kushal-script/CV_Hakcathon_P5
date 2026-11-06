#!/bin/bash
set -euo pipefail
HOST="$1"
WHAT="${2:-all}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$ROOT/runs/remote"
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15"
mkdir -p "$DEST"

if [[ "$HOST" =~ ^[0-9]+$ ]]; then
  TMP="$(mktemp -d "$DEST/.sync.XXXX")"
  jl download "$HOST" /home/jl-runs "$TMP/jl-runs" -r
  mkdir -p "$DEST/jl-runs"
  rsync -a "$TMP/jl-runs/" "$DEST/jl-runs/"
  rm -rf "$TMP"
  [ "$WHAT" = "all" ] && jl download "$HOST" /home/work/runs "$DEST/work-runs" -r
else
  rsync -az -e "$SSH" "root@$HOST:/home/jl-runs/" "$DEST/jl-runs/"
  if [ "$WHAT" = "all" ]; then
    rsync -az -e "$SSH" --exclude 'yolo_data/' "root@$HOST:/home/work/runs/" "$DEST/work-runs/"
    rsync -az -e "$SSH" "root@$HOST:/home/work/assets_out/" "$DEST/assets_out/" 2>/dev/null || true
  fi
fi
du -sh "$DEST"/* 2>/dev/null
