#!/usr/bin/env bash
set -euo pipefail

D=${VAHANA_DEPLOY_DIR:-/home/ubuntu/deploy}
B=${VAHANA_BACKUP_DIR:-/home/ubuntu/backups}
KEEP=${VAHANA_BACKUP_KEEP:-28}
umask 077
mkdir -p "$B"

ts=$(date -u +%Y%m%dT%H%M%SZ)
out="$B/$ts"
prev=$(ls -1d "$B"/2*Z 2>/dev/null | tail -1 || true)
rm -rf "$B"/*.partial
mkdir "$out.partial"

python3 - "$D" "$out.partial" <<'EOF'
import sqlite3
import sys

src, dst = sys.argv[1], sys.argv[2]
for name, path in (("gateway.db", f"{src}/state/gateway.db"), ("vahana.db", f"{src}/data/vahana.db")):
    s = sqlite3.connect(path, timeout=30)
    d = sqlite3.connect(f"{dst}/{name}")
    s.backup(d)
    s.close()
    result = d.execute("pragma integrity_check").fetchone()[0]
    d.close()
    if result != "ok":
        raise SystemExit(f"{name} failed its integrity check: {result}")
EOF

if [ -d "$D/data/cases" ]; then
  rsync -a ${prev:+--link-dest="$prev/cases"} "$D/data/cases/" "$out.partial/cases/"
fi
mv "$out.partial" "$out"
ls -1d "$B"/2*Z | sort | awk -v keep="$KEEP" '{ d[NR] = $0 } END { for (i = 1; i <= NR - keep; i++) print d[i] }' | xargs -r rm -rf
echo "backup written to $out ($(du -sh "$out" | cut -f1))"
