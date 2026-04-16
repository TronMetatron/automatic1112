# Automatic1112

A desktop application for AI image generation — the spiritual successor to Automatic1111. Built with PySide6, currently supporting Tencent's HunyuanImage-3.0 models with more backends planned.

<!-- ![Screenshot](docs/screenshot.png) -->

## Features

- **Text-to-Image** generation with multiple model variants
- **Image-to-Image** editing and transformation
- **Batch generation** with progress tracking, wildcards, and prompt templates
- **LLM prompt enhancement** via LM Studio or Ollama (keeps your GPU free for generation)
- **Think/Recaption modes** for advanced generation with chain-of-thought reasoning
- **Dual-GPU support** — split large models across two GPUs to fit in VRAM
- **Dataset preparation** tools for training workflows
- **Dark theme** desktop UI with real-time GPU monitoring
- **Headless CLI** for scripted/batch workflows

## Supported Models

| Model | Type | VRAM | Steps | Notes |
|-------|------|------|-------|-------|
| HunyuanImage-3.0 Base (SDNQ uint4) | T2I | ~93GB peak | 20 | Fastest, text-to-image only |
| HunyuanImage-3.0 Instruct (SDNQ) | T2I + I2I | ~93GB peak | 50 | Full feature set |
| HunyuanImage-3.0 Distil (SDNQ) | T2I + I2I | ~93GB peak | 8 | Fast distilled variant |
| HunyuanImage-3.0 Instruct (NF4) | T2I + I2I | ~48GB peak | 50 | Single GPU friendly |
| HunyuanImage-3.0 Distil (NF4) | T2I + I2I | ~48GB peak | 8 | Single GPU, fast |

## Hardware Requirements

### Minimum
- 1x GPU with 48GB+ VRAM (NF4 models) or 96GB+ (SDNQ models)
- 32GB system RAM
- ~150GB disk space for models

### Recommended (Dual-GPU, SDNQ)
- 1x large GPU (80GB+) for transformer/MoE layers (e.g., RTX PRO 6000 Blackwell 96GB)
- 1x secondary GPU (16GB+) for VAE and vision model (e.g., RTX 5090 32GB)
- 64GB system RAM

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/automatic1112.git
cd automatic1112
```

### 2. Install system dependencies

**Linux (Ubuntu/Debian):**
```bash
sudo apt install -y python3.12-venv python3.12-dev libxcb-cursor0
```

**Windows:**
- Install [Python 3.12](https://www.python.org/downloads/)
- Install [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-downloads)

### 3. Create Python environment

```bash
python3 -m venv venv
source venv/bin/activate   # Linux
# venv\Scripts\activate    # Windows
```

### 4. Install PyTorch with CUDA

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

> **Important**: This project requires `transformers==4.57.3`. Newer versions (5.x) have incompatible Cache API changes.

### 6. Download models

Set your model directory (default: `~/automatic1112_models`):

```bash
export A1112_MODEL_DIR="$HOME/automatic1112_models"  # Linux
# set A1112_MODEL_DIR=%USERPROFILE%\automatic1112_models  # Windows
```

Download from HuggingFace:

```bash
# NF4 models (recommended for single GPU, ~48GB VRAM)
huggingface-cli download Disty0/HunyuanImage3-Instruct-NF4-v2 \
    --local-dir "$A1112_MODEL_DIR/HunyuanImage3-Instruct-NF4-v2"

# SDNQ models (for dual-GPU setups, ~93GB peak VRAM)
huggingface-cli download Disty0/HunyuanImage3-Instruct-SDNQ \
    --local-dir "$A1112_MODEL_DIR/HunyuanImage3-Instruct-SDNQ"
```

See [SETUP.md](SETUP.md) for all model variants and dual-GPU patch instructions.

### 7. Launch

**Linux:**
```bash
./launch_desktop.sh
```

**Windows:**
```cmd
launch_desktop.bat
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `A1112_MODEL_DIR` | `~/automatic1112_models` | Directory containing downloaded model weights |
| `A1112_VENV_PATH` | Auto-detected | Path to Python virtual environment |
| `A1112_LMSTUDIO_URL` | `http://localhost:1234` | LM Studio server URL for prompt enhancement |
| `CUDA_VISIBLE_DEVICES` | `0,1` | GPUs to use |
| `HF_HUB_OFFLINE` | `0` | Set to `1` to prevent HuggingFace network calls |

### Prompt Enhancement

The app can use **LM Studio** or **Ollama** running on any machine to enhance your prompts with an LLM before sending them to the image model. This keeps your GPU VRAM free for generation.

Configure the URL in the app's settings panel or via the `A1112_LMSTUDIO_URL` environment variable.

## CLI Usage

```bash
# List available models
./hunyuan_cli.sh list-models

# Generate with CLI
./hunyuan_cli.sh i2i --model distil --prompt "Watercolor painting" --image input.png
```

## Project Structure

```
automatic1112/
├── launch_desktop.sh / .bat   # Platform launchers
├── hunyuan_cli.sh / .bat      # CLI launchers
├── requirements.txt           # Python dependencies
├── hunyuan_desktop/           # Main application package
│   ├── main.py                # GUI entry point
│   ├── cli.py                 # Headless CLI
│   ├── core/                  # Model management, workers, settings
│   ├── widgets/               # PySide6 UI components
│   ├── models/                # Data models
│   ├── dialogs/               # Modal dialogs
│   └── theme/                 # Dark theme
├── ui/                        # Constants, presets, app state
├── patches/                   # Dual-GPU compatibility patches
├── ollama_prompts.py          # LLM prompt enhancement
├── lmstudio_client.py         # LM Studio API client
└── wildcard_utils.py          # Wildcard template system
```

## Credits

- [HunyuanImage-3.0](https://github.com/Tencent/HunyuanImage) by Tencent
- [SDNQ quantization](https://huggingface.co/Disty0) by Disty0
- Inspired by [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

## License

[Apache License 2.0](LICENSE)
