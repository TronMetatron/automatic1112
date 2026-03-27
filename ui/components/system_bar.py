"""
System Bar Component for HunyuanImage-3.0 UI.
Contains GPU selectors (Image + LLM), model status, and load/unload controls.
Always visible at the top of the interface.
"""

import gradio as gr
from dataclasses import dataclass
from typing import List

from ui.state import get_state, detect_gpus, set_gpu, set_ollama_gpu
from core.model_manager import (
    load_model, unload_model, get_model_status, is_model_loaded,
    get_model_type, set_model_type, get_available_model_types
)


@dataclass
class SystemBarComponents:
    """Container for system bar UI components."""
    model_type_dropdown: gr.Dropdown
    gpu_dropdown: gr.Dropdown
    ollama_gpu_dropdown: gr.Dropdown
    model_status: gr.Markdown
    load_btn: gr.Button
    unload_btn: gr.Button
    refresh_btn: gr.Button
    # Ollama model management
    ollama_status: gr.Textbox
    ollama_unload_btn: gr.Button


def get_gpu_choices() -> List[str]:
    """Get list of GPU choices for dropdown (PyTorch indexing)."""
    state = get_state()
    if not state.available_gpus:
        state.available_gpus = detect_gpus()
    return [gpu['display'] for gpu in state.available_gpus]


def get_default_image_gpu() -> str:
    """Get default GPU for image generation."""
    choices = get_gpu_choices()
    state = get_state()

    for gpu in state.available_gpus:
        if gpu['index'] == state.selected_gpu:
            return gpu['display']

    return choices[0] if choices else "No GPU"


def get_default_ollama_gpu() -> str:
    """Get default GPU for Ollama LLM."""
    choices = get_gpu_choices()
    state = get_state()

    for gpu in state.available_gpus:
        if gpu['index'] == state.selected_ollama_gpu:
            return gpu['display']

    return choices[0] if choices else "No GPU"


def handle_model_type_change(model_type_choice: str) -> str:
    """Handle model type selection change."""
    # Extract the model type from the choice (e.g., "HunyuanImage-3.0 Base (T2I)" -> "base")
    model_types = get_available_model_types()
    for label, model_type in model_types:
        if label == model_type_choice:
            result = set_model_type(model_type)
            return get_model_status() + f"\n\n*{result}*"
    return "Model type selection failed"


def handle_image_gpu_change(gpu_choice: str) -> str:
    """Handle image generation GPU selection change."""
    state = get_state()

    for gpu in state.available_gpus:
        if gpu['display'] == gpu_choice:
            gpu_idx = gpu['index']

            if is_model_loaded():
                return f"**WARNING**: Model loaded on GPU {state.selected_gpu}. Unload first, then change GPU and reload."

            result = set_gpu(gpu_idx)
            return get_model_status() + f"\n\n*{result}*"

    return "GPU selection failed"


def handle_ollama_gpu_change(gpu_choice: str) -> str:
    """Handle Ollama LLM GPU selection change.

    This will restart Ollama on the new GPU using nvidia-smi indexing
    (CUDA_VISIBLE_DEVICES follows nvidia-smi, not PyTorch).
    """
    state = get_state()

    for gpu in state.available_gpus:
        if gpu['display'] == gpu_choice:
            pytorch_idx = gpu['index']
            nvidia_idx = gpu.get('nvidia_index', pytorch_idx)  # Use nvidia-smi index for Ollama
            result = set_ollama_gpu(pytorch_idx)

            # Restart Ollama on new GPU (using nvidia-smi index)
            if state.ollama_manager:
                try:
                    success, msg = state.ollama_manager.restart(gpu_index=nvidia_idx)
                    if success:
                        return get_model_status() + f"\n\n*{result}*\n✓ Ollama restarted on nvidia-smi GPU {nvidia_idx} ({gpu['name']})"
                    else:
                        return get_model_status() + f"\n\n*{result}*\n⚠️ Ollama restart failed: {msg}"
                except Exception as e:
                    return get_model_status() + f"\n\n*{result}*\n⚠️ Error: {e}"

            return get_model_status() + f"\n\n*{result}*"

    return "Ollama GPU selection failed"


def handle_load_model():
    """Handle model load button click."""
    for status in load_model():
        yield status


def handle_unload_model() -> str:
    """Handle model unload button click."""
    return unload_model()


def handle_refresh_status() -> str:
    """Handle refresh button click."""
    return get_model_status()


def get_ollama_loaded_model() -> str:
    """Get the currently loaded Ollama model name and VRAM usage."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/ps", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            if models:
                model = models[0]
                name = model.get('name', 'unknown')
                size_vram = model.get('size_vram', 0)
                vram_gb = size_vram / (1024**3)
                return f"Loaded: {name} ({vram_gb:.1f}GB VRAM)"
            return "No model loaded"
        return "Ollama not responding"
    except Exception as e:
        return f"Ollama offline"


def handle_ollama_unload() -> str:
    """Unload the currently loaded Ollama model."""
    try:
        import requests
        # First get the loaded model name
        response = requests.get("http://localhost:11434/api/ps", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            if not models:
                return "No model to unload"

            model_name = models[0].get('name', '')
            if not model_name:
                return "Could not get model name"

            # Unload by setting keep_alive to 0
            unload_response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model_name, "keep_alive": 0},
                timeout=30
            )
            if unload_response.status_code == 200:
                return f"Unloaded: {model_name}"
            return f"Unload failed: {unload_response.status_code}"
        return "Ollama not responding"
    except Exception as e:
        return f"Error: {str(e)}"


def create_system_bar() -> SystemBarComponents:
    """Create the system bar UI component with dual GPU selectors.

    Returns:
        SystemBarComponents with references to all widgets.
    """
    state = get_state()
    if not state.available_gpus:
        state.available_gpus = detect_gpus()

    with gr.Group():
        gr.Markdown("### System (PyTorch GPU Indices)")
        with gr.Row():
            # Model Type Selection
            model_types = get_available_model_types()
            model_type_choices = [label for label, _ in model_types]
            current_type = get_model_type()
            current_label = next((label for label, t in model_types if t == current_type), model_type_choices[0])
            
            model_type_dropdown = gr.Dropdown(
                label="Model",
                choices=model_type_choices,
                value=current_label,
                scale=2,
                info="Base (T2I only) or Instruct (T2I + editing)"
            )

            # Image Generation GPU
            gpu_dropdown = gr.Dropdown(
                label="Image GPU",
                choices=get_gpu_choices(),
                value=get_default_image_gpu(),
                scale=2,
                info="GPU for HunyuanImage model"
            )

            # Ollama/LLM GPU
            ollama_gpu_dropdown = gr.Dropdown(
                label="LLM GPU",
                choices=get_gpu_choices(),
                value=get_default_ollama_gpu(),
                scale=2,
                info="GPU for Ollama LLM"
            )

            # Model Status
            model_status = gr.Markdown(
                value=get_model_status(),
                elem_id="model_status"
            )

        with gr.Row():
            load_btn = gr.Button(
                "Load Image Model",
                variant="primary",
                size="sm",
                scale=1
            )
            unload_btn = gr.Button(
                "Unload",
                variant="stop",
                size="sm",
                scale=1
            )
            refresh_btn = gr.Button(
                "Refresh",
                variant="secondary",
                size="sm",
                scale=1
            )

        # Ollama/LLM status row
        with gr.Row():
            ollama_status = gr.Textbox(
                label="Ollama LLM",
                value=get_ollama_loaded_model(),
                interactive=False,
                scale=3
            )
            ollama_unload_btn = gr.Button(
                "Unload LLM",
                variant="stop",
                size="sm",
                scale=1
            )

    return SystemBarComponents(
        model_type_dropdown=model_type_dropdown,
        gpu_dropdown=gpu_dropdown,
        ollama_gpu_dropdown=ollama_gpu_dropdown,
        model_status=model_status,
        load_btn=load_btn,
        unload_btn=unload_btn,
        refresh_btn=refresh_btn,
        ollama_status=ollama_status,
        ollama_unload_btn=ollama_unload_btn,
    )


def wire_system_bar_events(components: SystemBarComponents) -> None:
    """Wire up event handlers for system bar components."""

    # Model type selection
    components.model_type_dropdown.change(
        fn=handle_model_type_change,
        inputs=[components.model_type_dropdown],
        outputs=[components.model_status]
    )

    # Image GPU selection
    components.gpu_dropdown.change(
        fn=handle_image_gpu_change,
        inputs=[components.gpu_dropdown],
        outputs=[components.model_status]
    )

    # Ollama GPU selection
    components.ollama_gpu_dropdown.change(
        fn=handle_ollama_gpu_change,
        inputs=[components.ollama_gpu_dropdown],
        outputs=[components.model_status]
    )

    # Load model button
    components.load_btn.click(
        fn=handle_load_model,
        inputs=[],
        outputs=[components.model_status]
    )

    # Unload model button
    components.unload_btn.click(
        fn=handle_unload_model,
        inputs=[],
        outputs=[components.model_status]
    )

    # Refresh status button - updates both model and Ollama status
    def refresh_all():
        return get_model_status(), get_ollama_loaded_model()

    components.refresh_btn.click(
        fn=refresh_all,
        inputs=[],
        outputs=[components.model_status, components.ollama_status]
    )

    # Ollama unload button
    components.ollama_unload_btn.click(
        fn=handle_ollama_unload,
        inputs=[],
        outputs=[components.ollama_status]
    )
