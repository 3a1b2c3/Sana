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
# Absolute, so it still resolves correctly after the magi-attention build
# step below cd's into its own clone directory.
case "$VENV_DIR" in
    /*) VENV_ABS="$VENV_DIR" ;;
    *) VENV_ABS="$HERE/$VENV_DIR" ;;
esac
PY="$VENV_ABS/bin/python"

echo
echo "=========================================="
echo "Installing torch (environment-tested.json pins 2.9.1+cu128; this repo"
echo "has since moved to 2.14.1+cu130 manually -- no CUDA extension (magi-"
echo "attention, flash-attn) had been built against either yet when that"
echo "changed, so this is the new baseline, not a regression)"
echo "=========================================="
uv pip install --python "$PY" torch==2.14.1+cu130 --index-url https://download.pytorch.org/whl/cu130

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
echo "Building magi-attention 1.1.1 from source"
echo "=========================================="
# Not on PyPI -- SandAI-org/MagiAttention is installed from its own source
# tree (https://sandai-org.github.io/MagiAttention/docs/main/user_guide/install.html).
# Supports only Hopper and Blackwell (not Ampere). Needs CUDA 13+ in the
# active toolchain (nvcc --version); set MAGI_ATTENTION_ALLOW_BUILD_WITH_CUDA12=1
# to force CUDA 12 at a likely performance cost. First build is ~10-20 min
# and uses most of the CPU.
MAGI_BUILD_DIR="${MAGI_BUILD_DIR:-$HERE/.build/MagiAttention}"
if [ ! -d "$MAGI_BUILD_DIR" ]; then
    git clone https://github.com/SandAI-org/MagiAttention.git "$MAGI_BUILD_DIR"
fi
(
    cd "$MAGI_BUILD_DIR"
    git fetch --tags
    git checkout v1.1.1
    git submodule update --init --recursive
    uv pip install --python "$PY" -r requirements.txt
    # Blackwell-specific build flags (GB200/GB300-class); use the Hopper
    # one-liner `pip install --no-build-isolation .` instead on Hopper GPUs.
    export MAGI_ATTENTION_PREBUILD_FFA=0
    uv pip install --python "$PY" --no-build-isolation .
    export MAGI_ATTENTION_FA4_BACKEND=1
)
echo "NOTE: MAGI_ATTENTION_FA4_BACKEND=1 must also be set in the shell that"
echo "      *runs* StreamRender-H3, not just during this install."

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
