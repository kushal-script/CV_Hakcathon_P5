#!/usr/bin/env bash
set -euo pipefail

uvicorn app.server:app --host 127.0.0.1 --port "${P5_API_PORT:-8001}" --workers 1 --no-server-header &
api=$!
vahana-gateway &
gateway=$!

trap 'kill -TERM "$api" "$gateway" 2>/dev/null || true' TERM INT
set +e
wait -n "$api" "$gateway"
status=$?
kill -TERM "$api" "$gateway" 2>/dev/null
wait
exit "$status"
