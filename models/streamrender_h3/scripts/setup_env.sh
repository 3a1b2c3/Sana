#!/usr/bin/env bash
# Creates the Python 3.11 venv for StreamRender-H3 and installs its full CUDA
# kernel stack (torch, flash-attn FA4, magi-attention, nvidia-cutlass-dsl,
# triton), pinned to the exact versions in configs/environment-tested.json.
# requirements.txt alone is NOT a complete environment -- see README.md's
# "Environment" section -- so this script covers the parts requirements.txt
# deliberately leaves out.
#
# This is a separate, dedicated venv. Do not reuse the top-level Sana repo's
# own (different) Torch/CUDA environment for this profile (README warning).
#
# Usage:
#   ./scripts/setup_env.sh              (creates .venv here)
#   VENV_DIR=/path/.venv ./scripts/setup_env.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found on PATH. Install it first: https://docs.astral.sh/uv/" >&2
    exit 1
fi

echo "=========================================="
echo "StreamRender-H3: creating venv ($VENV_DIR, Python 3.11)"
echo "=========================================="
uv venv --python 3.11 "$VENV_DIR"
PY="$VENV_DIR/bin/python"

echo
echo "=========================================="
echo "Installing torch 2.9.1+cu128 (pinned to match environment-tested.json)"
echo "=========================================="
uv pip install --python "$PY" torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128

echo
echo "=========================================="
echo "Installing requirements.txt"
echo "=========================================="
uv pip install --python "$PY" -r requirements.txt

echo
echo "=========================================="
echo "Installing Triton 3.5.1"
echo "=========================================="
uv pip install --python "$PY" triton==3.5.1

echo
echo "=========================================="
echo "Installing nvidia-cutlass-dsl 4.6.2"
echo "=========================================="
uv pip install --python "$PY" nvidia-cutlass-dsl==4.6.2

echo
echo "=========================================="
echo "Installing magi-attention 1.1.1"
echo "=========================================="
uv pip install --python "$PY" magi-attention==1.1.1

echo
echo "=========================================="
echo "Building flash-attn 2.8.4 (FA4 CuTe backend) from source"
echo "=========================================="
# No PyPI wheel exists for this exact pinned build (2.8.4+g0251105a2fb1 --
# a specific Dao-AILab/flash-attention commit, not a tagged release) or for
# newer GPU architectures in general. This compiles from source; set
# TORCH_CUDA_ARCH_LIST to the actual target GPU's compute capability first,
# e.g. TORCH_CUDA_ARCH_LIST='9.0' for H100/GB200-class, '10.3+PTX' for
# GB300-class Blackwell (nvidia-smi --query-gpu=compute_cap --format=csv).
# This is a large, slow compile (many kernel variants); expect 30+ minutes.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    echo "TORCH_CUDA_ARCH_LIST not set -- checking nvidia-smi for the GPU's compute capability..." >&2
    DETECTED_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)"
    if [ -z "$DETECTED_ARCH" ]; then
        echo "Could not detect GPU compute capability. Set TORCH_CUDA_ARCH_LIST explicitly and re-run." >&2
        exit 1
    fi
    export TORCH_CUDA_ARCH_LIST="${DETECTED_ARCH}+PTX"
    echo "Detected compute_cap=$DETECTED_ARCH -> TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
fi
uv pip install --python "$PY" --no-build-isolation --no-cache \
    "git+https://github.com/Dao-AILab/flash-attention.git@0251105a2fb1"

echo
echo "=========================================="
echo "Installing Node 22 dependencies (frontend)"
echo "=========================================="
node --version
npm --prefix coding/threejs-racing ci

echo
echo "Done. Venv: $HERE/$VENV_DIR"
echo "Next: configure configs/assets.local.json (see download_models.sh), then run"
echo "  $PY scripts/preflight.py --assets configs/assets.local.json"
echo "  $PY scripts/preflight.py --assets configs/assets.local.json --sha256"
