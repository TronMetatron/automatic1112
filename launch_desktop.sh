#!/bin/bash
# HunyuanImage Desktop - PySide6 Application Launcher
# Standalone version — uses LM Studio for prompt enhancement, local GPUs free

set -e

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo "=== HunyuanImage Desktop (Standalone) ==="
echo "Project: $PROJECT_ROOT"
echo "Prompt enhancement: LM Studio @ http://192.168.50.30:1234"

# Activate virtual environment (shared with hun3d)
VENV_PATH="/home/james/hun3d/hunyuan_env"
if [ -d "$VENV_PATH" ]; then
    echo "Activating: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# GPU Configuration — both GPUs visible for model loading
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
fi

# CUDA memory config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Cache directories
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export TORCH_HOME="$PROJECT_ROOT/.torch_cache"

export HF_HUB_OFFLINE=1

# Qt platform settings
export QT_QPA_PLATFORMTHEME=qt6ct
export QT_AUTO_SCREEN_SCALE_FACTOR=1

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Launching desktop application..."
echo ""

# Launch
cd "$PROJECT_ROOT"
python -m hunyuan_desktop.main "$@"
