"""Data model for dataset preparation configuration.

Defines pass presets and configuration for generating diverse training
datasets from character reference images using I2I generation.
"""

from dataclasses import dataclass, field
from typing import List, Dict


# Built-in pass presets for dataset preparation
DATASET_PREP_PASSES = {
    "Simplify Background": {
        "prompt": (
            "Keep the person exactly the same, do not change their appearance at all. "
            "Replace the background with a [simple white studio, minimalist indoor room, "
            "plain outdoor park, neutral gray backdrop, simple wooden desk setting]."
        ),
        "description": "Replaces busy backgrounds with clean, simple ones",
    },
    "Add Clothing": {
        "prompt": (
            "Keep this person's face, hair, and body exactly the same. Dress them in "
            "[casual jeans and t-shirt, formal business suit, athletic sportswear, "
            "a summer dress, a winter coat and scarf]. Same pose, same background."
        ),
        "description": "Adds varied clothing while preserving identity",
    },
    "Face Close-up": {
        "prompt": (
            "Create a tight close-up portrait of this person showing just their face "
            "and shoulders. Keep every facial feature exactly identical. "
            "[Neutral expression, slight smile, serious expression, looking slightly "
            "to the side]. Clean simple background."
        ),
        "description": "Generates head/shoulder crops for face detail training",
    },
    "Different Angle": {
        "prompt": (
            "Show this same person from a [three-quarter view, side profile, "
            "slight low angle, slight high angle looking down] perspective. "
            "Keep their appearance completely identical. Same clothing, same setting."
        ),
        "description": "Creates different viewing angles of the character",
    },
    "Change Pose": {
        "prompt": (
            "Show this same person [standing upright with arms at sides, "
            "walking forward naturally, sitting on a chair, leaning against a wall, "
            "reaching for something]. Keep their face and clothing identical."
        ),
        "description": "Varies the character's pose and body position",
    },
    "Lighting Variation": {
        "prompt": (
            "Keep this person completely identical. Change only the lighting to "
            "[soft natural daylight, warm golden hour sunlight, cool blue overcast light, "
            "dramatic side lighting, bright studio flash]. "
            "Same pose, same clothing, same background."
        ),
        "description": "Varies lighting while preserving everything else",
    },
}

# Ordered list of pass names for consistent UI ordering
DATASET_PREP_PASS_ORDER = [
    "Simplify Background",
    "Add Clothing",
    "Face Close-up",
    "Different Angle",
    "Change Pose",
    "Lighting Variation",
]


def _pass_name_to_folder(name: str) -> str:
    """Convert a pass display name to a filesystem-safe folder name."""
    return name.lower().replace(" ", "_").replace("-", "_")


@dataclass
class DatasetPrepConfig:
    """All parameters needed for a dataset preparation run."""

    # Folder paths
    input_folder: str = ""
    output_folder: str = ""

    # Enabled passes: {pass_name: prompt_text}
    enabled_passes: Dict[str, str] = field(default_factory=dict)

    # Generation settings
    images_per_pass: int = 1
    quality: str = "Standard"  # Quality preset name (maps to steps via constants)
    guidance_scale: float = 5.0
    random_seeds: bool = True

    # Bot task / think mode (read from global state at runtime)
    bot_task: str = "image"
    drop_think: bool = False

    # Prompt enhancement (LM Studio / Ollama)
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
        from ui.constants import MODEL_INFO
        return MODEL_INFO.get(model_type, {}).get("default_steps", 50)

    def total_images(self) -> int:
        """Calculate total images: source_count * enabled_passes * images_per_pass."""
        # Source count isn't known until runtime; this is per-source total
        n_passes = len(self.enabled_passes)
        return n_passes * self.images_per_pass

    def total_images_for_sources(self, source_count: int) -> int:
        """Calculate total with a known source count."""
        return source_count * len(self.enabled_passes) * self.images_per_pass

    def preview_text(self, source_count: int = 0) -> str:
        """Generate a human-readable preview."""
        n_passes = len(self.enabled_passes)
        pass_names = ", ".join(self.enabled_passes.keys())

        if source_count > 0:
            total = self.total_images_for_sources(source_count)
            return (
                f"{total} images total "
                f"({source_count} sources x {n_passes} passes x "
                f"{self.images_per_pass} per pass) | "
                f"Passes: {pass_names}"
            )
        return (
            f"{n_passes} passes x {self.images_per_pass} per pass | "
            f"Passes: {pass_names}"
        )

    def to_dict(self) -> dict:
        """Convert to dict for saving."""
        return {
            "input_folder": self.input_folder,
            "output_folder": self.output_folder,
            "enabled_passes": self.enabled_passes,
            "images_per_pass": self.images_per_pass,
            "quality": self.quality,
            "guidance_scale": self.guidance_scale,
            "random_seeds": self.random_seeds,
            "bot_task": self.bot_task,
            "drop_think": self.drop_think,
            "enhance": self.enhance,
            "ollama_model": self.ollama_model,
            "ollama_length": self.ollama_length,
            "ollama_complexity": self.ollama_complexity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetPrepConfig":
        """Create from a saved config dict."""
        return cls(
            input_folder=data.get("input_folder", ""),
            output_folder=data.get("output_folder", ""),
            enabled_passes=data.get("enabled_passes", {}),
            images_per_pass=data.get("images_per_pass", 1),
            quality=data.get("quality", "Standard"),
            guidance_scale=data.get("guidance_scale", 5.0),
            random_seeds=data.get("random_seeds", True),
            bot_task=data.get("bot_task", "image"),
            drop_think=data.get("drop_think", False),
            enhance=data.get("enhance", False),
            ollama_model=data.get("ollama_model", "qwen2.5:7b-instruct"),
            ollama_length=data.get("ollama_length", "medium"),
            ollama_complexity=data.get("ollama_complexity", "detailed"),
        )
