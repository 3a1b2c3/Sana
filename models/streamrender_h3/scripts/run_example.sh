#!/usr/bin/env bash
# Launches the StreamRender-H3 worker (8 GPUs, local single-node) for the
# prepared default example, then replays the matching semantic video once
# the worker reports readiness. Needs setup_env.sh, download_models.sh, and
# prepare_example.py already run (see README.md / examples/README.md).
#
# Usage:
#   ./scripts/run_example.sh                       (B10_e-track-3_bluehour, from configs/example.local.json)
#   ./scripts/run_example.sh --assets configs/snow.local.json --example A11_g-track-2_snow
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

VENV_DIR="${VENV_DIR:-.venv}"
PY="$HERE/$VENV_DIR/bin/python"
if [ ! -x "$PY" ]; then
    echo "No venv at $VENV_DIR. Run ./scripts/setup_env.sh first." >&2
    exit 1
fi

# launch_local.sh (called below) runs `npm`. setup_env.sh installs a local
# Node under .tools/node (no sudo/apt -- common on shared DGX hosts with no
# root access) rather than relying on a system Node; put it on PATH so
# launch_local.sh's npm calls resolve.
if [ -x "$HERE/.tools/node/bin/node" ]; then
    export PATH="$HERE/.tools/node/bin:$PATH"
fi

ASSETS="configs/example.local.json"
EXAMPLE="B10_e-track-3_bluehour"
GPUS="${GPUS:-1}"
EXTRA_REPLAY_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --assets) ASSETS="$2"; shift 2 ;;
        --example) EXAMPLE="$2"; shift 2 ;;
        *) EXTRA_REPLAY_ARGS+=("$1"); shift ;;
    esac
done

if [ ! -f "$ASSETS" ]; then
    echo "$ASSETS not found. Run scripts/prepare_example.py first (see README.md" >&2
    echo "section 4, \"Prepare the default example\")." >&2
    exit 1
fi

echo "=========================================="
echo "Launching worker: $GPUS GPUs, assets=$ASSETS"
echo "=========================================="
GPUS="$GPUS" PYTHON_BIN="$PY" bash scripts/launch_local.sh --assets "$ASSETS" &
WORKER_PID=$!
trap 'kill "$WORKER_PID" 2>/dev/null || true' EXIT

echo
echo "Waiting for the worker at http://127.0.0.1:8765/health to report readiness..."
# /health always returns 200, even while still loading -- the body's "ready"
# field is what actually matters (runtime/server.py).
while true; do
    if ! kill -0 "$WORKER_PID" 2>/dev/null; then
        echo "Worker process exited before becoming ready -- check its output above." >&2
        exit 1
    fi
    HEALTH="$(curl -sf http://127.0.0.1:8765/health 2>/dev/null || true)"
    case "$HEALTH" in
        *'"ready": true'*|*'"ready":true'*) break ;;
    esac
    sleep 5
done
echo "Worker ready."

echo
echo "=========================================="
echo "Replaying example: $EXAMPLE"
echo "=========================================="
"$PY" scripts/replay_example.py --example "$EXAMPLE" "${EXTRA_REPLAY_ARGS[@]}"

echo
echo "Done. Open http://127.0.0.1:5191 to view the live render; WASD or autopilot to drive."
echo "The worker (PID $WORKER_PID) is still running in the background -- Ctrl-C this script to stop it."
wait "$WORKER_PID"
