#!/bin/bash
# Copy all FoodScan scans from a USB-connected iPad into ./data/scans.
# Needs the app installed from Xcode (developer-signed) and iPadOS 17+.
#
#   ./scripts/pull_scans.sh            # copies to Thesis/data/scans
#   ./scripts/pull_scans.sh /some/dir  # custom destination
set -euo pipefail

BUNDLE_ID="edu.yale.njl36.foodscan"
DEST="${1:-$(cd "$(dirname "$0")/.." && pwd)/data/scans}"

DEVICE=$(xcrun devicectl list devices --json-output /dev/stdout 2>/dev/null \
  | python3 -c 'import json,sys; d=json.load(sys.stdin)["result"]["devices"]; print(d[0]["identifier"] if d else "")')
if [ -z "$DEVICE" ]; then
  echo "No iPad found. Plug it in, unlock it, and tap Trust." >&2
  exit 1
fi

mkdir -p "$DEST"
echo "Copying scans from device $DEVICE to $DEST ..."
xcrun devicectl device copy from \
  --device "$DEVICE" \
  --domain-type appDataContainer \
  --domain-identifier "$BUNDLE_ID" \
  --source Documents/scans \
  --destination "$DEST"
echo "Done:"
ls -1 "$DEST"
