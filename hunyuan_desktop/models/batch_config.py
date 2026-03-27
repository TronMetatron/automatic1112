"""Data model for batch generation configuration."""

from dataclasses import dataclass, field
from typing import List, Optional
import re


@dataclass
class BatchConfig:
    """All parameters needed for a batch generation run."""

    # Basic config
    batch_name: str = ""
    themes: List[str] = field(default_factory=list)
    negative_prompt: str = ""

    # Iteration settings
    variations_per_theme: int = 3
    images_per_combo: int = 1
    starred_reroll_count: int = 1

    # Style settings
    styles: List[str] = field(default_factory=lambda: ["None"])

    # Generation settings
    aspect_ratio: str = "1:1 Square (1024)"
    quality: str = "Standard"
    guidance_scale: float = 5.0
    random_seeds: bool = True

    # Ollama enhancement
    enhance: bool = False
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_length: str = "medium"
    ollama_complexity: str = "detailed"
    max_prompt_length: int = 0

    # Bot task / think mode (instruct/distil only)
    bot_task: str = "image"  # "image", "think", "think_recaption", "recaption"
    drop_think: bool = False  # If True, discard chain-of-thought output

    # Mode
    generate_prompts_only: bool = False

    def has_starred_wildcards(self) -> bool:
        """Check if any theme contains [*wildcard] syntax."""
        pattern = re.compile(r'\[\*\w+\]')
        return any(pattern.search(theme) for theme in self.themes)

    def total_images(self) -> int:
        """Calculate total number of images this batch will generate."""
        n_themes = len(self.themes)
        n_styles = max(len(self.styles), 1)
        base = n_themes * self.variations_per_theme * n_styles * self.images_per_combo

        # If starred wildcards exist, multiply by reroll count
        if self.has_starred_wildcards():
            base *= max(self.starred_reroll_count, 1)

        return base

    def preview_text(self) -> str:
        """Generate a human-readable preview of the batch configuration."""
        n_themes = len(self.themes)
        n_styles = len(self.styles)
        total = self.total_images()

        parts = [f"{n_themes} theme{'s' if n_themes != 1 else ''}"]
        parts.append(f"{n_styles} style{'s' if n_styles != 1 else ''}")
        parts.append(f"{self.variations_per_theme} variation{'s' if self.variations_per_theme != 1 else ''}")
        parts.append(f"{self.images_per_combo} img/combo")

        if self.has_starred_wildcards():
            parts.append(f"{self.starred_reroll_count} starred reroll{'s' if self.starred_reroll_count != 1 else ''}")

        formula = " x ".join(parts)
        return f"{total} images total ({formula})"

    def to_dict(self) -> dict:
        """Convert to dict for saving via batch_manager."""
        return {
            "batch_name": self.batch_name,
            "themes": "\n".join(self.themes),
            "variations_per_theme": self.variations_per_theme,
            "styles": self.styles,
            "images_per_combo": self.images_per_combo,
            "aspect_ratio": self.aspect_ratio,
            "quality_preset": self.quality,
            "guidance_scale": self.guidance_scale,
            "enhance_prompts": self.enhance,
            "ollama_model": self.ollama_model,
            "ollama_length": self.ollama_length,
            "ollama_complexity": self.ollama_complexity,
            "max_prompt_length": self.max_prompt_length,
            "random_seeds": self.random_seeds,
            "negative_prompt": self.negative_prompt,
            "starred_reroll_count": self.starred_reroll_count,
            "bot_task": self.bot_task,
            "drop_think": self.drop_think,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BatchConfig":
        """Create from a saved config dict."""
        themes_raw = data.get("themes", "")
        if isinstance(themes_raw, str):
            themes = [t.strip() for t in themes_raw.split("\n") if t.strip()]
        else:
            themes = themes_raw

        return cls(
            batch_name=data.get("batch_name", ""),
            themes=themes,
            negative_prompt=data.get("negative_prompt", ""),
            variations_per_theme=data.get("variations_per_theme", 3),
            images_per_combo=data.get("images_per_combo", 1),
            starred_reroll_count=data.get("starred_reroll_count", 1),
            styles=data.get("styles", ["None"]),
            aspect_ratio=data.get("aspect_ratio", "1:1 Square (1024)"),
            quality=data.get("quality_preset", data.get("quality", "Standard")),
            guidance_scale=data.get("guidance_scale", 5.0),
            random_seeds=data.get("random_seeds", True),
            enhance=data.get("enhance_prompts", False),
            ollama_model=data.get("ollama_model", "qwen2.5:7b-instruct"),
            ollama_length=data.get("ollama_length", "medium"),
            ollama_complexity=data.get("ollama_complexity", "detailed"),
            max_prompt_length=data.get("max_prompt_length", 0),
            bot_task=data.get("bot_task", "image"),
            drop_think=data.get("drop_think", False),
        )
