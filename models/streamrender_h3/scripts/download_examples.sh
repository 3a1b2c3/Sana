#!/usr/bin/env bash
# Downloads the StreamRender-H3 example media (semantic reference videos +
# reference pictures used by requests_A.jsonl / requests_B.jsonl) from the
# GitHub Release asset -- this is not in Git history, and not something
# setup_env.sh or download_models.sh fetches.
#
# Usage:
#   ./scripts/download_examples.sh
#
# Extracts examples/media/ (30 semantic videos, 60 pictures, Git-ignored)
# and examples/media_inventory.json. See examples/README.md for how to then
# select, prepare, and replay an example against a running worker.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

URL="https://github.com/NVlabs/Sana/releases/download/streamrender-h3-examples-v1/streamrender-h3-examples.tar"
TAR="streamrender-h3-examples.tar"

echo "=========================================="
echo "Downloading example media -> $HERE/$TAR"
echo "=========================================="
curl -fL --retry 3 -o "$TAR" "$URL"

echo
echo "=========================================="
echo "Extracting (adds examples/media/, examples/media_inventory.json)"
echo "=========================================="
tar -xf "$TAR"
rm -f "$TAR"

echo
echo "Done. See examples/README.md for selecting and replaying an example."
