# HunyuanImage-3.0 Desktop — Setup & Dual-GPU Guide

## Overview

This is a PySide6 desktop application for AI image generation using Tencent's HunyuanImage-3.0 model (80B parameter Mixture-of-Experts, SDNQ uint4 quantized). It supports text-to-image, image editing, think/recaption modes, and batch generation with LLM prompt enhancement via LM Studio.

## Hardware Requirements

### Minimum
- **1x GPU with 96GB+ VRAM** (e.g., NVIDIA RTX PRO 6000 Blackwell)
- 32GB system RAM
- ~150GB disk space for models

### Recommended (Dual-GPU)
- **1x large GPU (80GB+)** for transformer layers (e.g., RTX PRO 6000 Blackwell 96GB)
- **1x secondary GPU (16GB+)** for VAE and vision model (e.g., RTX 5090 32GB)
- 64GB system RAM
- ~150GB disk space for models

### Why So Much VRAM?
The quantized model weights are ~48GB, but peak memory during generation reaches ~93GB:
- Model weights: **48GB**
- KV cache during text generation: **~15GB** (freed before diffusion)
- MoE gating activation spike during diffusion: **~35GB per step**

The 35GB spike comes from the Mixture-of-Experts gating function creating large one_hot dispatch tensors across 64 experts for ~262K image tokens (at 1024x1024).

## Installation

### 1. Clone/Copy the Project

```bash
# Project structure:
/home/james/hunyuan_desktop/     # Desktop app (standalone)
/home/james/hun3d/               # Models + web UI + shared venv
```

### 2. Install System Dependencies

```bash
sudo apt install -y python3.12-venv python3.12-dev libxcb-cursor0
```

### 3. Create Python Environment

```bash
cd /home/james/hun3d
python3 -m venv hunyuan_env
source hunyuan_env/bin/activate

# PyTorch with CUDA 12.8
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# Dependencies
pip install "transformers==4.57.3" "accelerate" "gradio>=4.21.0" \
    "einops>=0.8.0" "numpy==1.26.4" "pillow==11.3.0" \
    "diffusers>=0.32.0" "safetensors==0.4.5" "tokenizers>=0.21.0" \
    "huggingface_hub[cli]" "sdnq>=0.1.4" psutil PySide6
```

> **Important**: Use `transformers==4.57.3` — newer versions (5.x) have incompatible Cache API changes that cause errors with this model's custom code.

### 4. Download Models

Download from HuggingFace (requires ~48GB each):

```bash
# Base model (text-to-image only, fastest)
huggingface-cli download Disty0/HunyuanImage3-SDNQ-uint4-svd-r32 \
    --local-dir /home/james/hun3d/HunyuanImage3-SDNQ

# Instruct model (text-to-image + image editing + think/recaption)
huggingface-cli download Disty0/HunyuanImage3-Instruct-SDNQ \
    --local-dir /home/james/hun3d/HunyuanImage3-Instruct-SDNQ

# Distil model (fast generation, 8 steps)
huggingface-cli download Disty0/HunyuanImage3-Instruct-Distil-SDNQ \
    --local-dir /home/james/hun3d/HunyuanImage3-Distil-SDNQ
```

### 5. Apply Dual-GPU Patches (Required for multi-GPU setups)

The model's custom code has device mismatch issues when components are split across GPUs. These patches fix that:

```bash
cd /home/james/hunyuan_desktop
python patches/dual_gpu_patch.py --model-dir /home/james/hun3d/HunyuanImage3-Instruct-SDNQ
```

This patches `modeling_hunyuan_image_3.py` to:
1. **Bridge patch**: Add `.to(target_device)` to all `scatter_()` operations so tensors from different GPUs are aligned before operations
2. **Forward device alignment**: Align all input tensors (masks, indices, timesteps) to the same device at the start of the forward pass
3. **KV cache clear**: Free ~15GB of KV cache between the text generation phase and diffusion phase via `torch.cuda.empty_cache()`

Originals are backed up as `modeling_hunyuan_image_3.py.orig`.

### 6. Launch

```bash
# Desktop app
./launch_desktop.sh

# Or the web UI (Gradio)
cd /home/james/hun3d && ./launch_ui.sh
```

## GPU Configuration

### Dual-GPU Setup (Recommended)

The model is loaded with a custom `device_map` that places components on specific GPUs:

```python
device_map = {
    # Secondary GPU (e.g., RTX 5090 31GB) — non-MoE components
    "vae": 1,              # VAE encoder/decoder (~1.5GB)
    "vision_model": 1,     # SigLIP2 vision encoder (~1GB)
    "vision_aligner": 1,   # Vision projector (~0.1GB)

    # Primary GPU (e.g., Blackwell 95GB) — transformer + diffusion
    "model": 0,            # All 32 MoE transformer layers (~44GB)
    "patch_embed": 0,      # UNet down-sampling
    "timestep_emb": 0,     # Timestep embeddings
    "time_embed": 0,       # Time embeddings
    "time_embed_2": 0,     # Time embeddings (final layer)
    "final_layer": 0,      # UNet up-sampling
    "lm_head": 0,          # Text generation head
}
```

**Why this split works:**
- All 32 transformer layers stay on the large GPU because the MoE gating creates ~35GB activation spikes during each diffusion step
- The VAE and vision model don't have MoE layers, so they don't spike — safe on the smaller GPU
- The bridge patch ensures tensors are moved to the correct device before `scatter_()` operations

### GPU Index Warning

**PyTorch/CUDA device ordering may differ from nvidia-smi!** On our test machine:
- nvidia-smi: GPU 0 = RTX 5090, GPU 1 = Blackwell
- PyTorch cuda: GPU 0 = Blackwell, GPU 1 = RTX 5090

The code auto-detects the largest GPU and assigns it as primary. Verify your ordering:

```python
import torch
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
    print(f"cuda:{i} = {name} ({mem:.0f}GB)")
```

### Single GPU Setup

If you only have one GPU (96GB+), the model loads entirely on it. Peak memory is ~93GB during diffusion. With only 95-96GB total, this is very tight — consider:
- Using the **Base** model instead of Instruct (simpler pipeline, less KV cache)
- Using the **Distil** model (8 steps instead of 50)
- Reducing image size to 768x768
- Using `bot_task="image"` instead of `"think_recaption"` (no KV cache from thinking phase)

### CPU Offload Fallback

If dual-GPU isn't available and single GPU OOMs, use CPU offloading:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    max_memory={0: "38GiB", "cpu": "32GiB"},
    ...
)
```

This offloads ~10 layers to CPU RAM. Text generation becomes ~8x slower (~38 min for think_recaption) but diffusion stays fast (~8s/step).

## Performance Benchmarks

Instruct model, think_recaption mode, 1024x1024, 50 steps:

| Configuration | Text Gen | Diffusion | Total | Peak VRAM |
|---------------|----------|-----------|-------|-----------|
| Dual-GPU (patched) | ~5 min | 4:27 (5.2s/step) | **~11 min** | 93GB |
| CPU offload | ~38 min | 6:56 (8.3s/step) | **~45 min** | 56GB |
| Single Blackwell | ~5 min | OOM | N/A | >95GB |

## Prompt Enhancement

The desktop app uses **LM Studio** on a remote machine for prompt enhancement (no local GPU usage for LLM):

- **URL**: Configured in `ui/constants.py` → `LMSTUDIO_URL`
- **Protocol**: OpenAI-compatible API (`/v1/chat/completions`)
- **Default**: `http://192.168.50.30:1234`

To use local Ollama instead, modify `hunyuan_desktop/core/ollama_worker.py` to use `OllamaManager` instead of `LMStudioClient`.

## Project Structure

```
hunyuan_desktop/
├── launch_desktop.sh          # Launch script
├── test_headless.py           # Headless test script (no Qt UI needed)
├── patches/
│   └── dual_gpu_patch.py      # Applies dual-GPU patches to model code
├── hunyuan_desktop/           # PySide6 desktop app package
│   ├── main.py                # Entry point
│   ├── core/
│   │   ├── app_state.py       # Qt state wrapper
│   │   ├── model_manager.py   # Model loading with dual-GPU support
│   │   ├── generation_worker.py
│   │   ├── batch_worker.py
│   │   ├── ollama_worker.py   # LM Studio prompt enhancement
│   │   └── settings.py        # Persistent settings
│   ├── widgets/
│   │   ├── main_window.py     # Main UI window
│   │   ├── system_bar.py      # Model/GPU selector toolbar
│   │   ├── prompt_panel.py    # Prompt editor
│   │   └── ...
│   ├── models/                # Data models (generation params, batch config)
│   ├── theme/                 # Dark theme
│   └── dialogs/               # Editor dialogs
├── ui/
│   ├── constants.py           # Model paths, presets, LM Studio URL
│   └── state.py               # Global app state, GPU detection
├── ollama_prompts.py          # Prompt enhancement logic
├── lmstudio_client.py         # LM Studio API client
├── wildcard_utils.py          # Wildcard template system
└── HunyuanImage-3.0/         # Original model code (for sdnq module)

hun3d/                         # Shared models + web UI
├── HunyuanImage3-SDNQ/       # Base model (48GB, 11 shards)
├── HunyuanImage3-Instruct-SDNQ/  # Instruct model (48GB, 11 shards)
├── HunyuanImage3-Distil-SDNQ/    # Distil model (48GB, 11 shards)
├── hunyuan_env/               # Shared Python virtual environment
├── hunyuan_ui.py              # Gradio web UI
└── launch_ui.sh               # Web UI launcher
```

## Troubleshooting

### "CUDA out of memory" during diffusion
The MoE activation spike needs ~35GB free VRAM beyond model weights. Solutions:
1. Use dual-GPU setup (recommended)
2. Use CPU offload (`max_memory={0: "38GiB", "cpu": "32GiB"}`)
3. Use smaller image size (768x768 reduces spike to ~20GB)
4. Use `bot_task="image"` instead of `"think_recaption"`

### "Expected all tensors to be on the same device"
The dual-GPU patches haven't been applied. Run:
```bash
python patches/dual_gpu_patch.py --model-dir /path/to/HunyuanImage3-Instruct-SDNQ
```

Also clear the HuggingFace cache if the model was loaded before patching:
```bash
rm -rf ~/.cache/huggingface/modules/transformers_modules/HunyuanImage3*
```

### "SDNQ: Triton test failed"
Install `python3.12-dev` for Triton JIT compilation:
```bash
sudo apt install python3.12-dev
```
Without it, SDNQ falls back to PyTorch eager mode (works fine, slightly slower).

### Model loading hangs or is very slow
- Loading 48GB across 11 shards takes 7-8 minutes — this is normal
- Ensure `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set (launch scripts set this)
- Ensure `HF_HUB_OFFLINE=1` is set to prevent network calls

### "Invalid model type" error
Stale QSettings from a previous installation. The settings validator will auto-fall back to "base". Clear settings:
```bash
rm ~/.config/HunyuanImage/Desktop.conf
```

## Technical Deep Dive: Why Dual-GPU Is Hard for This Model

The HunyuanImage-3.0 architecture is a **unified text+image model** — text generation and image diffusion happen in the same forward pass. This creates two challenges for multi-GPU:

### 1. Device Mismatch in scatter_ Operations
The model's `instantiate_vae_image_tokens()` method uses `hidden_states.scatter_()` to insert VAE-encoded image embeddings into the transformer's hidden state sequence. If the VAE is on GPU 1 and the transformer is on GPU 0, the scatter fails because `src` and `hidden_states` are on different devices.

**Solution**: The bridge patch adds `.to(target_device)` before every scatter operation, allowing the VAE output to be transferred to the transformer's device transparently.

### 2. MoE Activation Spike
Each of the 32 transformer layers has a Mixture-of-Experts module with 64 experts. During the gating computation, `F.one_hot(token_priority, expert_capacity)` creates a boolean tensor of shape `[tokens, experts, capacity]`. For a 1024x1024 image with 262K tokens, this tensor is ~35GB.

This spike happens on **whichever GPU the layer runs on**, so:
- You can't put MoE layers on the 5090 (31GB) — it OOMs
- You can't split MoE layers across GPUs — each layer needs 35GB headroom
- All MoE layers must stay on the Blackwell (95GB)

### 3. KV Cache Persistence
During think/recaption, the model generates text tokens and builds a KV cache (~15GB). This cache persists into the diffusion phase, consuming memory that's needed for the MoE spike. The KV cache clear patch calls `torch.cuda.empty_cache()` between phases, freeing this memory just in time.
