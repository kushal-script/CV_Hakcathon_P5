#!/bin/bash
set -uo pipefail
PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${OUT:-/home/work/india}"
cd "$PROJECT"
TOTAL=$(python -c "import json, sys; sys.path.insert(0, 'scripts'); from build_india_dataset import unique_categories; print(len(unique_categories(json.load(open('assets/india/candidate_classes.json')))))")
while true; do
  python scripts/build_india_dataset.py --out "$OUT" --stage collect --offline --workers 4 --downloads 4 --fetch-concurrency "${FETCH:-6}" | grep -v "not listed yet"
  DONE=$(ls "$OUT/collected" 2>/dev/null | wc -l)
  echo "collected $DONE/$TOTAL"
  [ "$DONE" -ge "$TOTAL" ] && break
  [ -f "$OUT/listing_done" ] && [ "$DONE" -ge "$(ls "$OUT/listed" | wc -l)" ] && break
  sleep 20
done
python scripts/build_india_dataset.py --out "$OUT" --stage clean && python scripts/build_india_dataset.py --out "$OUT" --stage split
echo "exit $?"
