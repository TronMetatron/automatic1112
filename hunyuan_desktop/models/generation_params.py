"""Data model for single image generation parameters."""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class GenerationParams:
    """All parameters needed for a single image generation."""

    # Prompt
    prompt: str = ""
    negative_prompt: str = ""
    style: str = "None"

    # Generation settings
    aspect_ratio: str = "1:1 Square (1024)"
    quality: str = "Standard"
    seed: int = -1  # -1 = random
    batch_count: int = 1

    # Ollama enhancement
    use_ollama: bool = False
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_length: str = "medium"
    ollama_complexity: str = "detailed"
    max_prompt_length: int = 0

    # Image-to-Image (instruct/distil only)
    input_image_path: Optional[str] = None  # Legacy single image (backward compat)
    input_image_paths: List[str] = field(default_factory=list)  # Multi-image (up to 2)
    i2i_strength: float = 0.8
    guidance_scale: float = 5.0

    # Bot task / think mode (instruct/distil only)
    bot_task: str = "image"  # "image", "think", "think_recaption", "recaption"
    drop_think: bool = False  # If True, discard chain-of-thought output

    # Output
    output_dir: Optional[str] = None

    def get_i2i_images(self) -> List[str]:
        """Get the list of input images for I2I, merging legacy and new fields."""
        if self.input_image_paths:
            return self.input_image_paths
        if self.input_image_path:
            return [self.input_image_path]
        return []

    def get_steps(self, model_type=None) -> int:
        """Get inference steps from quality preset."""
        from ui.constants import get_quality_presets, MODEL_INFO
        presets = get_quality_presets(model_type)
        preset = presets.get(self.quality, {})
        if not preset:
            # Quality key didn't match (e.g. Edit tab uses default "Standard"
            # but model presets have "Standard (30 steps)"). Use model default.
            info = MODEL_INFO.get(model_type, {})
            return info.get("default_steps", 20)
        return preset.get("steps", 20)

    def get_image_size(self, model_type=None) -> str:
        """Get image size string from aspect ratio."""
        from ui.constants import get_aspect_ratios
        ratios = get_aspect_ratios(model_type)
        # FireRed should default to auto to match input image dimensions
        if model_type == "firered":
            return ratios.get(self.aspect_ratio, "auto")
        return ratios.get(self.aspect_ratio, "512x512" if model_type == "deepgen" else "1024x1024")

    def get_style_suffix(self) -> str:
        """Get the style suffix text."""
        from ui.constants import DEFAULT_STYLE_PRESETS
        return DEFAULT_STYLE_PRESETS.get(self.style, "")

    def to_metadata(self) -> dict:
        """Convert to metadata dict for JSON sidecar."""
        meta = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "style": self.style,
            "aspect_ratio": self.aspect_ratio,
            "quality": self.quality,
            "image_size": self.get_image_size(),
            "steps": self.get_steps(),
            "seed": self.seed,
            "batch_count": self.batch_count,
            "use_ollama": self.use_ollama,
            "ollama_model": self.ollama_model,
            "ollama_length": self.ollama_length,
            "ollama_complexity": self.ollama_complexity,
            "guidance_scale": self.guidance_scale,
        }
        i2i_images = self.get_i2i_images()
        if i2i_images:
            meta["input_images"] = i2i_images
            meta["bot_task"] = self.bot_task
            meta["drop_think"] = self.drop_think
        return meta
