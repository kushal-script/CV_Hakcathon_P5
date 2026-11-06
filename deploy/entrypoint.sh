#!/usr/bin/env bash
set -euo pipefail

if [ -z "${P5_INTERNAL_SECRET:-}" ]; then
  P5_INTERNAL_SECRET="$(head -c 48 /dev/urandom | base64 | tr -d '\n=+/')"
  export P5_INTERNAL_SECRET
fi

uvicorn app.server:app --host 127.0.0.1 --port "${P5_API_PORT:-8001}" --workers 1 --no-server-header --log-level warning &
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
