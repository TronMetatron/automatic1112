"""
Constants and configuration presets for Automatic1112 Desktop UI.
"""

import os
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("A1112_MODEL_DIR", str(Path.home() / "automatic1112_models")))
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Model variants (pointing to hun3d model copies)
MODEL_PATHS = {
    "base": MODEL_DIR / "HunyuanImage3-SDNQ",
    "instruct": MODEL_DIR / "HunyuanImage3-Instruct-SDNQ",
    "distil": MODEL_DIR / "HunyuanImage3-Distil-SDNQ",
    "nf4": MODEL_DIR / "HunyuanImage3-Instruct-NF4-v2",
    "distil_nf4": MODEL_DIR / "HunyuanImage3-Distil-NF4-v2",
    "firered": MODEL_DIR / "FireRed-Image-Edit-1.1",
}

MODEL_INFO = {
    "base": {
        "name": "HunyuanImage-3.0 Base",
        "description": "Standard T2I model, fastest",
        "default_steps": 20,
        "supports_img2img": False,
    },
    "instruct": {
        "name": "HunyuanImage-3.0 Instruct",
        "description": "T2I + Image editing capabilities",
        "default_steps": 50,
        "supports_img2img": True,
    },
    "distil": {
        "name": "HunyuanImage-3.0 Instruct-Distil",
        "description": "Fast T2I + I2I with think mode (8 steps)",
        "default_steps": 8,
        "supports_img2img": True,
    },
    "nf4": {
        "name": "HunyuanImage-3.0 Instruct NF4",
        "description": "NF4 4-bit quantized Instruct (single GPU, ~48GB VRAM)",
        "default_steps": 50,
        "supports_img2img": True,
    },
    "distil_nf4": {
        "name": "HunyuanImage-3.0 Distil NF4",
        "description": "NF4 4-bit quantized Distil — fast 8-step T2I+I2I (single GPU, ~48GB VRAM)",
        "default_steps": 8,
        "supports_img2img": True,
    },
    "firered": {
        "name": "FireRed Image Edit 1.1",
        "description": "Image editing model (Qwen-Image backbone, T2I + I2I, up to 3 ref images)",
        "default_steps": 40,
        "supports_img2img": True,
    },
}

# Config files
STYLE_PRESETS_FILE = OUTPUT_DIR / "style_presets.json"
HISTORY_FILE = OUTPUT_DIR / "generation_history.json"
UI_CONFIG_FILE = PROJECT_DIR / "ui_config.json"
WILDCARDS_FILE = PROJECT_DIR / "wildcards.json"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# Image Generation Settings
# =============================================================================

IMAGE_SIZES = [
    "auto",
    "1024x1024",
    "1536x1536",
    "1280x768",
    "768x1280",
    "1152x896",
    "896x1152",
    "1344x768",
    "768x1344",
    "1536x640",
    "640x1536",
    "1280x720",
    "832x480",
]

ASPECT_RATIOS = {
    "1:1 Square (1024)": "1024x1024",
    "1:1 Square (1536)": "1536x1536",
    "16:9 (Landscape)": "1280x768",
    "16:9 (720p Video)": "1280x720",
    "16:9 (480p Video)": "832x480",
    "9:16 (Portrait)": "768x1280",
    "4:3 (Standard)": "1152x896",
    "3:4 (Portrait Standard)": "896x1152",
    "21:9 (Ultrawide)": "1536x640",
    "9:21 (Tall)": "640x1536",
    "Auto (Model decides)": "auto",
}

DEEPGEN_ASPECT_RATIOS = {
    "1:1 Square (512)": "512x512",
    "1:1 Square (768)": "768x768",
    "16:9 (Landscape)": "768x512",
    "9:16 (Portrait)": "512x768",
    "4:3 (Standard)": "640x512",
    "3:4 (Portrait Standard)": "512x640",
    "21:9 (Ultrawide)": "768x320",
    "9:21 (Tall)": "320x768",
}

QUALITY_PRESETS = {
    "Distil (8 steps)": {"steps": 8, "description": "Fast distilled model quality"},
    "Draft (Fast)": {"steps": 15, "description": "Quick preview, lower quality"},
    "Standard": {"steps": 20, "description": "Good balance of speed and quality"},
    "High Quality": {"steps": 30, "description": "Better details, slower"},
    "Maximum": {"steps": 50, "description": "Best quality, slowest"},
}

DEEPGEN_QUALITY_PRESETS = {
    "Draft (25 steps)": {"steps": 25, "description": "Quick preview"},
    "Standard": {"steps": 50, "description": "Good balance (recommended)"},
    "High Quality": {"steps": 75, "description": "Better details, slower"},
    "Maximum": {"steps": 100, "description": "Best quality, slowest"},
}

DISTIL_QUALITY_PRESETS = {
    "Distil (8 steps)": {"steps": 8, "description": "Distilled model default (recommended)"},
}

FIRERED_QUALITY_PRESETS = {
    "Draft (20 steps)": {"steps": 20, "description": "Quick preview edit"},
    "Standard (30 steps)": {"steps": 30, "description": "Good balance"},
    "High Quality": {"steps": 40, "description": "Best quality (recommended)"},
}

# Default quality preset per model type
MODEL_DEFAULT_QUALITY = {
    "base": "Standard",
    "instruct": "Maximum",
    "nf4": "Maximum",
    "distil_nf4": "Distil (8 steps)",
    "distil": "Distil (8 steps)",
    "deepgen": "Standard",
    "firered": "High Quality",
    "base_int8": "Maximum",
    "instruct_int8": "Maximum",
    "distil_int8": "Distil (8 steps)",
}

# =============================================================================
# Style Presets
# =============================================================================

DEFAULT_STYLE_PRESETS = {
    "None": "",
    "Photorealistic": ", photorealistic, hyperrealistic, 8k, highly detailed, professional photography",
    "Cinematic": ", cinematic lighting, dramatic atmosphere, movie still, 35mm film grain",
    "Anime": ", anime style, vibrant colors, detailed linework, studio quality anime",
    "Digital Art": ", digital art, concept art, artstation trending, highly detailed illustration",
    "Oil Painting": ", oil painting style, classical art, rich textures, museum quality",
    "Watercolor": ", watercolor painting, soft edges, flowing colors, artistic",
    "3D Render": ", 3D render, octane render, unreal engine 5, high quality CGI",
    "Fantasy Art": ", fantasy art, epic fantasy, magical atmosphere, detailed fantasy illustration",
    "Minimalist": ", minimalist style, clean lines, simple composition, elegant",
    "Vintage": ", vintage photograph, retro style, nostalgic, aged film quality",
    "Comic Book": ", comic book style, bold lines, dynamic composition, graphic novel art",
    "Studio Portrait": ", studio portrait photography, professional lighting, sharp focus, bokeh background",
    "Nature Photography": ", national geographic style, nature photography, stunning natural light",
}

# =============================================================================
# LM Studio / Prompt Enhancement Settings
# =============================================================================

# LM Studio — default URL, overridable via settings or env var
LMSTUDIO_URL = os.environ.get("A1112_LMSTUDIO_URL", "http://localhost:1234")


def get_lmstudio_url() -> str:
    """Get the LM Studio URL from settings, falling back to LMSTUDIO_URL default."""
    try:
        from core.settings import get_settings
        url = get_settings().lmstudio_url
        if url:
            return url
    except Exception:
        pass
    return LMSTUDIO_URL

# Fallback model list
OLLAMA_MODELS = ["lmstudio"]

# Default model
DEFAULT_OLLAMA_MODEL = "lmstudio"

# Prompt enhancement options
OLLAMA_LENGTH_OPTIONS = ["random", "minimal", "short", "medium", "long", "detailed", "cinematic", "experimental"]
OLLAMA_COMPLEXITY_OPTIONS = ["random", "simple", "moderate", "detailed", "complex", "cinematic", "experimental"]

# =============================================================================
# UI Configuration
# =============================================================================

DEFAULT_UI_COLORS = {
    "background": "#1a1a1a",
    "text": "#e0e0e0",
    "primary": "#4a90d9",
    "secondary": "#2d2d2d",
    "accent": "#66bb6a",
    "error": "#ef5350",
    "button_primary": "#4a90d9",
    "button_secondary": "#424242",
    "button_danger": "#ef5350",
}

# =============================================================================
# Generation Defaults
# =============================================================================

DEFAULT_SEED = -1  # -1 means random
DEFAULT_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 5.0
DEFAULT_ASPECT_RATIO = "1:1 Square (1024)"
DEFAULT_QUALITY = "Standard"
DEEPGEN_DEFAULT_GUIDANCE = 4.0


def get_quality_presets(model_type=None):
    """Get quality presets for the given model type."""
    if model_type in ("distil", "distil_int8", "distil_nf4"):
        return DISTIL_QUALITY_PRESETS
    if model_type == "deepgen":
        return DEEPGEN_QUALITY_PRESETS
    if model_type == "firered":
        return FIRERED_QUALITY_PRESETS
    return QUALITY_PRESETS


def get_aspect_ratios(model_type=None):
    """Get aspect ratios for the given model type."""
    if model_type == "deepgen":
        return DEEPGEN_ASPECT_RATIOS
    return ASPECT_RATIOS


def get_default_guidance(model_type=None):
    """Get default guidance scale for the given model type."""
    if model_type == "deepgen":
        return DEEPGEN_DEFAULT_GUIDANCE
    if model_type == "firered":
        return 4.0
    if model_type in ("nf4", "distil_nf4"):
        return 2.5  # NF4-v2 model's recommended guidance scale
    return DEFAULT_GUIDANCE_SCALE


def is_firered_model(model_type):
    """Check if a model type is the FireRed image editor."""
    return model_type == "firered"


def is_int8_model(model_type):
    """Check if a model type uses BnB INT8 quantization."""
    return model_type in ("base_int8", "instruct_int8", "distil_int8")


def is_instruct_like(model_type):
    """Check if a model type supports instruct features (I2I, think mode, bot_task)."""
    return model_type in ("instruct", "distil", "nf4", "distil_nf4", "instruct_int8", "distil_int8")
