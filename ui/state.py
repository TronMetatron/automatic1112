"""
Global shared state for HunyuanImage-3.0 UI.
All UI modules import this for cross-module state access.
"""

import os
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict
from pathlib import Path


@dataclass
class AppState:
    """Central application state container."""

    # Model state
    model: Any = None
    model_loaded: bool = False
    model_load_lock: threading.Lock = field(default_factory=threading.Lock)
    model_type: str = "base"  # "base" or "instruct"

    # GPU configuration
    selected_gpu: int = 0  # Default GPU for image generation
    selected_ollama_gpu: int = -1  # -1 = no local GPU for LLM (using remote LM Studio)
    available_gpus: List[Dict] = field(default_factory=list)

    # CPU Offloading configuration
    cpu_offload_enabled: bool = False  # Enable CPU/RAM offloading
    max_gpu_memory_gb: int = 80  # Max GPU memory before offloading to CPU
    max_cpu_memory_gb: int = 64  # Max CPU/RAM memory for offloaded layers
    offload_folder: str = "/tmp/hunyuan_offload"  # Disk offload folder

    # Global generation mode (applies to all tabs)
    global_bot_task: str = "image"  # "image", "think", "think_recaption", "recaption"
    global_drop_think: bool = False  # Whether to hide thinking output

    # Batch state
    batch_running: bool = False
    batch_stop_requested: bool = False
    batch_results: List[Dict] = field(default_factory=list)

    # Single generation state
    single_generation_stop: bool = False

    # Generation queue state
    generation_queue: List[Dict] = field(default_factory=list)
    queue_lock: threading.Lock = field(default_factory=threading.Lock)
    queue_worker_running: bool = False
    queue_stop_requested: bool = False

    # Session state
    current_session_dir: Optional[Path] = None
    session_counter: int = 0
    last_generated_image: Optional[str] = None  # Path to most recently generated image
    last_seed_used: int = 0  # Seed of most recently generated image

    # Custom batch directory
    custom_batch_base_dir: Optional[str] = None

    # Ollama state
    ollama_available: bool = False
    ollama_enhancer: Any = None
    ollama_generator: Any = None
    ollama_manager: Any = None

    # Wildcard state
    wildcard_available: bool = False
    wildcard_manager: Any = None

    # Style presets (loaded dynamically)
    style_presets: Dict[str, str] = field(default_factory=dict)


# Global singleton instance
app_state = AppState()


def get_state() -> AppState:
    """Get the global application state."""
    return app_state


def detect_gpus() -> List[Dict]:
    """Detect available CUDA GPUs using PyTorch indexing (not nvidia-smi).

    Note: PyTorch GPU indices may differ from nvidia-smi ordering.
    We use PyTorch's view since that's what the model uses.
    Also maps to nvidia-smi indices for Ollama (CUDA_VISIBLE_DEVICES).
    """
    gpus = []
    try:
        import torch
        import subprocess

        # Get nvidia-smi GPU names for mapping
        nvidia_smi_gpus = {}
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    idx, name = line.split(', ')
                    nvidia_smi_gpus[name.strip()] = int(idx)
        except Exception:
            pass

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                # Find nvidia-smi index by matching GPU name
                nvidia_idx = nvidia_smi_gpus.get(props.name, i)
                gpus.append({
                    'index': i,  # PyTorch index (for model loading)
                    'nvidia_index': nvidia_idx,  # nvidia-smi index (for CUDA_VISIBLE_DEVICES)
                    'name': props.name,
                    'memory_gb': props.total_memory / (1024**3),
                    'display': f"GPU {i}: {props.name} ({props.total_memory / (1024**3):.0f} GB)"
                })
    except Exception as e:
        print(f"[GPU] Error detecting GPUs: {e}")
    return gpus


def set_gpu(gpu_index: int) -> str:
    """Set the active GPU for image generation.

    Note: We don't set CUDA_VISIBLE_DEVICES anymore.
    Instead, we use device_map='cuda:N' when loading the model.
    This allows all GPUs to remain visible.
    """
    state = get_state()
    state.selected_gpu = gpu_index

    # Find GPU name for confirmation
    for gpu in state.available_gpus:
        if gpu['index'] == gpu_index:
            return f"Image GPU set to: GPU {gpu_index} ({gpu['name']})"
    return f"Image GPU set to index {gpu_index}"


def set_ollama_gpu(gpu_index: int) -> str:
    """Set the GPU for Ollama LLM operations."""
    state = get_state()
    state.selected_ollama_gpu = gpu_index

    # Ollama uses CUDA_VISIBLE_DEVICES when it starts
    # We'll pass this when restarting Ollama
    for gpu in state.available_gpus:
        if gpu['index'] == gpu_index:
            return f"Ollama GPU set to: {gpu['name']}"
    return f"Ollama GPU set to index {gpu_index}"


def init_gpus() -> None:
    """Initialize GPU detection and set default."""
    state = get_state()
    state.available_gpus = detect_gpus()

    if state.available_gpus:
        # Default to the GPU with the most memory (Blackwell 95GB)
        default_idx = max(range(len(state.available_gpus)),
                         key=lambda i: state.available_gpus[i].get('memory_gb', 0))
        set_gpu(default_idx)
        gpu_name = state.available_gpus[default_idx]['name']
        print(f"[GPU] Detected {len(state.available_gpus)} GPU(s), using GPU {default_idx} ({gpu_name})")
    else:
        print("[GPU] No CUDA GPUs detected")
