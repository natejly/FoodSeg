#!/bin/bash
# Push code + scripts + data (scans, csvs) from the Mac to Bouchet project storage.
#   scripts/bouchet/sync.sh          # code + scripts + data
#   scripts/bouchet/sync.sh code     # code + scripts only
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DEST=bouchet:/nfs/roberts/project/cpsc4900/cpsc4900_njl36

rsync -a --delete --exclude .venv --exclude __pycache__ "$ROOT/pipeline" "$DEST/"
rsync -a --delete "$ROOT/scripts" "$DEST/"
rsync -a "$ROOT/docs" "$DEST/"
if [ "${1:-all}" = "all" ]; then
  rsync -a --exclude fdc_cache.json "$ROOT/data" "$DEST/"
fi
echo synced
