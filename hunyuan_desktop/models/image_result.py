"""Data model for image generation results."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class ImageResult:
    """Result from a single image generation."""

    image_path: str = ""
    seed: int = 0
    generation_time: float = 0.0
    prompt: str = ""
    full_prompt: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    cot_text: Optional[str] = None  # Chain-of-thought from instruct/distil

    @property
    def success(self) -> bool:
        return self.error is None and self.image_path != ""

    @property
    def filename(self) -> str:
        return Path(self.image_path).name if self.image_path else ""

    @property
    def json_path(self) -> str:
        """Path to the JSON sidecar file."""
        if not self.image_path:
            return ""
        p = Path(self.image_path)
        return str(p.with_suffix(".json"))
