#!/bin/bash
# Automatic1112 — Headless CLI Launcher (Linux)
#
# Examples:
#   ./hunyuan_cli.sh list-models
#   ./hunyuan_cli.sh i2i --model distil --prompt "Watercolor" --image ref.png
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Virtual environment — auto-detect or use A1112_VENV_PATH
VENV_PATH="${A1112_VENV_PATH:-}"

if [ -z "$VENV_PATH" ]; then
    for candidate in "$PROJECT_ROOT/venv" "$PROJECT_ROOT/.venv" "$PROJECT_ROOT/hunyuan_env"; do
        if [ -d "$candidate" ]; then
            VENV_PATH="$candidate"
            break
        fi
    done
fi

if [ -n "$VENV_PATH" ] && [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
fi

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export TORCH_HOME="$PROJECT_ROOT/.torch_cache"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export QT_QPA_PLATFORM=offscreen

cd "$PROJECT_ROOT"
exec python -m hunyuan_desktop.cli "$@"
