#!/usr/bin/env bash
set -uo pipefail

D=${VAHANA_DEPLOY_DIR:-/home/ubuntu/deploy}
S=/run/vahana-watchdog
mkdir -p "$S"

fails=$(cat "$S/fails" 2>/dev/null || echo 0)
if curl -fsS -m 10 http://127.0.0.1:8080/api/health >/dev/null; then
  fails=0
else
  fails=$((fails + 1))
  echo "local health check failed, $fails in a row"
  if [ "$fails" -ge 3 ]; then
    echo "restarting the vahana container"
    docker restart vahana >/dev/null
    fails=0
  fi
fi
echo "$fails" > "$S/fails"

url=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$D/tunnel.log" 2>/dev/null | tail -1)
if [ -n "$url" ]; then
  last=$(cat "$D/public_url" 2>/dev/null || true)
  if [ "$url" != "$last" ]; then
    echo "quick tunnel address is $url (previously ${last:-not recorded})"
    echo "$url" > "$D/public_url"
  fi
fi
for host in $(sed -n 's/^ *- hostname: *//p' "$D/cloudflared/config.yml" 2>/dev/null) ${url#https://}; do
  if ! curl -fsS -m 20 "https://$host/api/health" >/dev/null; then
    echo "public address https://$host did not answer its health check"
  fi
done
