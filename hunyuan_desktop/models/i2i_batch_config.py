"""Data model for image-to-image batch generation configuration."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import re


# Bot task display names -> model values
BOT_TASK_OPTIONS = {
    "Direct (image)": "image",
    "Think": "think",
    "Think + Rewrite": "think_recaption",
    "Rewrite Only": "recaption",
}

BOT_TASK_REVERSE = {v: k for k, v in BOT_TASK_OPTIONS.items()}


@dataclass
class I2IBatchConfig:
    """All parameters needed for an image-to-image batch generation run."""

    # Basic config
    batch_name: str = ""
    prompts: List[str] = field(default_factory=list)

    # Global reference images (apply to all prompts unless overridden)
    # Legacy flat list of non-empty single images (kept for back-compat)
    global_images: List[str] = field(default_factory=list)  # max 3 paths

    # Slot-aligned (always length 3). Each entry is "" or a path.
    # global_image_folders[i] takes precedence over global_image_slots[i].
    # When a folder is set, the worker cycles through its images per generation.
    global_image_slots: List[str] = field(
        default_factory=lambda: ["", "", ""]
    )
    global_image_folders: List[str] = field(
        default_factory=lambda: ["", "", ""]
    )

    # Per-prompt image overrides: {prompt_index: [image_paths]}
    prompt_image_overrides: Dict[int, List[str]] = field(default_factory=dict)

    # Bot task / think mode
    bot_task: str = "image"  # "image", "think", "think_recaption", "recaption"
    drop_think: bool = False

    # Iteration settings
    variations_per_prompt: int = 1
    images_per_combo: int = 1

    # Style settings
    styles: List[str] = field(default_factory=lambda: ["None"])

    # Generation settings
    quality: str = "Standard"  # Quality preset name (maps to steps via constants)
    guidance_scale: float = 5.0
    random_seeds: bool = True

    # Ollama enhancement
    enhance: bool = False
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_length: str = "medium"
    ollama_complexity: str = "detailed"
    max_prompt_length: int = 0

    def get_steps(self, model_type: str = None) -> int:
        """Get inference steps from the quality preset for the given model type."""
        from ui.constants import get_quality_presets
        presets = get_quality_presets(model_type)
        for name, info in presets.items():
            if self.quality in name or name in self.quality:
                return info["steps"]
        # Fallback: use model default
        from ui.constants import MODEL_INFO
        return MODEL_INFO.get(model_type, {}).get("default_steps", 50)

    def get_images_for_prompt(self, prompt_index: int) -> List[str]:
        """Get the image paths for a specific prompt, falling back to global."""
        if prompt_index in self.prompt_image_overrides:
            return self.prompt_image_overrides[prompt_index]
        return self.global_images

    def total_images(self) -> int:
        """Calculate total number of images this batch will generate."""
        n_prompts = len(self.prompts)
        n_styles = max(len(self.styles), 1)
        return n_prompts * self.variations_per_prompt * n_styles * self.images_per_combo

    def preview_text(self) -> str:
        """Generate a human-readable preview of the batch configuration."""
        n_prompts = len(self.prompts)
        n_styles = len(self.styles)
        n_global = len(self.global_images)
        n_overrides = len(self.prompt_image_overrides)
        total = self.total_images()

        parts = [f"{n_prompts} prompt{'s' if n_prompts != 1 else ''}"]
        parts.append(f"{n_styles} style{'s' if n_styles != 1 else ''}")
        parts.append(f"{self.variations_per_prompt} variation{'s' if self.variations_per_prompt != 1 else ''}")
        parts.append(f"{self.images_per_combo} img/combo")

        img_info = f"{n_global} global image{'s' if n_global != 1 else ''}"
        if n_overrides:
            img_info += f", {n_overrides} prompt override{'s' if n_overrides != 1 else ''}"

        bot_display = BOT_TASK_REVERSE.get(self.bot_task, self.bot_task)
        return f"{total} images total ({' x '.join(parts)}) | {img_info} | Quality: {self.quality} | Mode: {bot_display}"

    def to_dict(self) -> dict:
        """Convert to dict for saving."""
        return {
            "batch_name": self.batch_name,
            "prompts": self.prompts,
            "global_images": self.global_images,
            "global_image_slots": self.global_image_slots,
            "global_image_folders": self.global_image_folders,
            "prompt_image_overrides": {
                str(k): v for k, v in self.prompt_image_overrides.items()
            },
            "bot_task": self.bot_task,
            "drop_think": self.drop_think,
            "variations_per_prompt": self.variations_per_prompt,
            "images_per_combo": self.images_per_combo,
            "styles": self.styles,
            "quality": self.quality,
            "guidance_scale": self.guidance_scale,
            "random_seeds": self.random_seeds,
            "enhance": self.enhance,
            "ollama_model": self.ollama_model,
            "ollama_length": self.ollama_length,
            "ollama_complexity": self.ollama_complexity,
            "max_prompt_length": self.max_prompt_length,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "I2IBatchConfig":
        """Create from a saved config dict."""
        prompts = data.get("prompts", [])
        if isinstance(prompts, str):
            prompts = [p.strip() for p in prompts.split("\n") if p.strip()]

        overrides_raw = data.get("prompt_image_overrides", {})
        overrides = {int(k): v for k, v in overrides_raw.items()}

        return cls(
            batch_name=data.get("batch_name", ""),
            prompts=prompts,
            global_images=data.get("global_images", []),
            global_image_slots=(
                list(data.get("global_image_slots", ["", "", ""]))
                + ["", "", ""]
            )[:3],
            global_image_folders=(
                list(data.get("global_image_folders", ["", "", ""]))
                + ["", "", ""]
            )[:3],
            prompt_image_overrides=overrides,
            bot_task=data.get("bot_task", "image"),
            drop_think=data.get("drop_think", False),
            variations_per_prompt=data.get("variations_per_prompt", 1),
            images_per_combo=data.get("images_per_combo", 1),
            styles=data.get("styles", ["None"]),
            quality=data.get("quality", "Standard"),
            guidance_scale=data.get("guidance_scale", 5.0),
            random_seeds=data.get("random_seeds", True),
            enhance=data.get("enhance", False),
            ollama_model=data.get("ollama_model", "qwen2.5:7b-instruct"),
            ollama_length=data.get("ollama_length", "medium"),
            ollama_complexity=data.get("ollama_complexity", "detailed"),
            max_prompt_length=data.get("max_prompt_length", 0),
        )

    @staticmethod
    def parse_prompt_lines(raw_text: str, global_images: List[str]) -> tuple:
        """Parse prompt text with optional per-prompt image overrides.

        Syntax:
            Regular prompt: Transform this into a cyberpunk scene
            Per-prompt override: [img:/path/to/image.png] Transform with this reference
            Dual images: [img1:/path/a.png] [img2:/path/b.png] Some instruction

        Returns:
            (prompts, prompt_image_overrides)
        """
        prompts = []
        overrides = {}
        img_pattern = re.compile(r'\[img(?:\d)?:([^\]]+)\]')

        for line in raw_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            matches = img_pattern.findall(line)
            clean_prompt = img_pattern.sub("", line).strip()

            if not clean_prompt:
                continue

            idx = len(prompts)
            prompts.append(clean_prompt)

            if matches:
                overrides[idx] = [m.strip() for m in matches[:3]]  # max 3 images

        return prompts, overrides
