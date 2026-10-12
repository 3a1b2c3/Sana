#!/usr/bin/env bash
# Downloads the StreamRender-H3 checkpoint bundle and prepares the default
# example (B10_e-track-3_bluehour). Needs the venv from setup_env.sh.
#
# Sizes: ~206.87 GB for the checkpoint bundle, plus ~66.28 GB of scratch
# space while download_assets.py reconstructs the merged EMA backbone from
# its shards (SHA-256 verified). Budget ~275 GB free before running this.
#
# Usage:
#   ./scripts/download_models.sh [/path/to/models/StreamRender-H3]
#   (default checkpoint dir: ../StreamRender-H3, sibling to this repo's
#   streamrender_h3 directory, matching the README's "/your/code/models/..." layout)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

VENV_DIR="${VENV_DIR:-.venv}"
PY="$VENV_DIR/bin/python"
if [ ! -x "$PY" ]; then
    echo "No venv at $VENV_DIR. Run ./scripts/setup_env.sh first." >&2
    exit 1
fi

CKPT_DIR="${1:-$HERE/../StreamRender-H3}"
mkdir -p "$CKPT_DIR"

echo "=========================================="
echo "Downloading Efficient-Large-Model/StreamRender-H3 -> $CKPT_DIR"
echo "(~206.87 GB; resumable -- safe to re-run if interrupted)"
echo "=========================================="
"$PY" -m pip show huggingface_hub >/dev/null 2>&1 || uv pip install --python "$PY" "huggingface_hub[cli]"
"$PY" -m huggingface_hub.commands.huggingface_cli download \
    Efficient-Large-Model/StreamRender-H3 --local-dir "$CKPT_DIR"

echo
echo "=========================================="
echo "Reconstructing merged EMA backbone from shards (SHA-256 verified)"
echo "(~66.28 GB additional scratch space)"
echo "=========================================="
"$PY" "$CKPT_DIR/download_assets.py"

echo
echo "Checkpoint manifest written: $CKPT_DIR/assets.json"
echo
echo "=========================================="
echo "Next: set up the local asset manifest"
echo "=========================================="
if [ ! -f configs/assets.local.json ]; then
    cp configs/assets.example.json configs/assets.local.json
    echo "Copied configs/assets.example.json -> configs/assets.local.json"
    echo "Edit it: fill in the asset_roles paths (see README.md's \"Assets and"
    echo "initialization\" section), pointing at $CKPT_DIR/assets.json entries."
else
    echo "configs/assets.local.json already exists -- not overwriting."
fi

echo
echo "=========================================="
echo "Preparing the default example (B10_e-track-3_bluehour)"
echo "=========================================="
echo "Run once configs/assets.local.json is filled in:"
echo "  $PY scripts/prepare_example.py --assets $CKPT_DIR/assets.json"
