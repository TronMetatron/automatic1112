#!/bin/bash
# HunyuanImage Desktop - Debug Launcher
# Shows all debug output in terminal

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "  HunyuanImage Desktop - DEBUG MODE"
echo "=========================================="
echo "Project: $PROJECT_ROOT"
echo ""

# Activate virtual environment
VENV_PATH="$PROJECT_ROOT/hunyuan_env"
if [ -d "$VENV_PATH" ]; then
    echo "✓ Activating venv: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    echo "✗ ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# GPU Configuration
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=1
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HUB_OFFLINE=1

# Qt platform settings
export QT_QPA_PLATFORMTHEME=qt6ct
export QT_AUTO_SCREEN_SCALE_FACTOR=1

echo "✓ CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""
echo "Environment Check:"
echo "  Python: $(python --version)"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  Visible GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
"
echo ""
echo "=========================================="
echo "Watch for debug messages:"
echo "  [INIT] - Initialization"
echo "  [LOAD] - Model loading"
echo "  [GEN]  - Generation process"
echo "=========================================="
echo ""

# Launch with unbuffered output
cd "$PROJECT_ROOT"
exec python -u -m hunyuan_desktop.main "$@"
