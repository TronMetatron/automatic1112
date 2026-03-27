"""
Generation Settings Component for HunyuanImage-3.0 UI.
Contains aspect ratio, quality, seed, and other generation parameters.
"""

import gradio as gr
from dataclasses import dataclass
from typing import List

from ui.constants import (
    ASPECT_RATIOS,
    QUALITY_PRESETS,
    DEFAULT_SEED,
    DEFAULT_ASPECT_RATIO,
    DEFAULT_QUALITY,
    OUTPUT_DIR,
)
import subprocess


@dataclass
class GenSettingsComponents:
    """Container for generation settings UI components."""
    aspect_ratio: gr.Dropdown
    quality: gr.Dropdown
    seed: gr.Number
    random_seed_btn: gr.Button
    batch_count: gr.Slider
    output_dir: gr.Textbox
    output_browse_btn: gr.Button
    # I2I settings
    i2i_image: gr.Image
    i2i_strength: gr.Slider


def create_gen_settings() -> GenSettingsComponents:
    """Create the generation settings UI component.

    Returns:
        GenSettingsComponents with references to all widgets.
    """
    gr.Markdown("### Generation Settings")

    with gr.Row():
        aspect_ratio = gr.Dropdown(
            label="Aspect Ratio",
            choices=list(ASPECT_RATIOS.keys()),
            value=DEFAULT_ASPECT_RATIO,
            scale=1
        )
        quality = gr.Dropdown(
            label="Quality",
            choices=list(QUALITY_PRESETS.keys()),
            value=DEFAULT_QUALITY,
            scale=1,
            info="Higher = better but slower"
        )

    with gr.Row():
        seed = gr.Number(
            label="Seed (-1 = random)",
            value=DEFAULT_SEED,
            precision=0,
            scale=2
        )
        random_seed_btn = gr.Button(
            "Random",
            size="sm",
            scale=1
        )

    # Batch count for single generation mode
    batch_count = gr.Slider(
        label="Images to Generate",
        minimum=1,
        maximum=10,
        value=1,
        step=1,
        info="Generate multiple images with same settings"
    )

    # I2I (Image-to-Image) settings
    with gr.Accordion("🖼️ Image-to-Image (I2I)", open=False):
        gr.Markdown("*Upload a reference image for I2I editing. Requires Instruct-Distil or Instruct model.*")
        i2i_image = gr.Image(
            label="Reference Image",
            type="pil",
            height=200,
        )
        i2i_strength = gr.Slider(
            label="Strength",
            minimum=0.1,
            maximum=1.0,
            value=0.8,
            step=0.05,
            info="How much to transform the image (higher = more change)"
        )

    # Output directory with browse button
    with gr.Accordion("Output Directory", open=False):
        with gr.Row():
            output_dir = gr.Textbox(
                label="Output Directory",
                value=str(OUTPUT_DIR),
                scale=4
            )
            output_browse_btn = gr.Button("Browse", size="sm", scale=1)

    return GenSettingsComponents(
        aspect_ratio=aspect_ratio,
        quality=quality,
        seed=seed,
        random_seed_btn=random_seed_btn,
        batch_count=batch_count,
        output_dir=output_dir,
        output_browse_btn=output_browse_btn,
        i2i_image=i2i_image,
        i2i_strength=i2i_strength,
    )


def get_steps_from_quality(quality_name: str) -> int:
    """Get number of steps from quality preset name."""
    if quality_name in QUALITY_PRESETS:
        return QUALITY_PRESETS[quality_name]["steps"]
    return 20  # Default


def get_size_from_aspect(aspect_name: str) -> str:
    """Get image size from aspect ratio name."""
    if aspect_name in ASPECT_RATIOS:
        return ASPECT_RATIOS[aspect_name]
    return "1024x1024"  # Default


def generate_random_seed() -> int:
    """Generate a random seed value."""
    import random
    return random.randint(0, 2**31 - 1)


def open_directory_dialog(initial_dir: str = None) -> str:
    """Open a native directory selection dialog."""
    try:
        result = subprocess.run(
            ["zenity", "--file-selection", "--directory",
             "--title=Select Output Folder",
             "--filename=" + (initial_dir if initial_dir else str(OUTPUT_DIR))],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder = filedialog.askdirectory(
            initialdir=initial_dir if initial_dir else str(OUTPUT_DIR),
            title="Select Output Folder"
        )
        root.destroy()
        if folder:
            return folder
    except Exception:
        pass

    return initial_dir or ""


def wire_gen_settings_events(components: GenSettingsComponents) -> None:
    """Wire up event handlers for generation settings.

    Args:
        components: GenSettingsComponents instance to wire up
    """
    # Random seed button
    components.random_seed_btn.click(
        fn=generate_random_seed,
        inputs=[],
        outputs=[components.seed]
    )

    # Output directory browse button
    def handle_browse_output(current_path):
        new_path = open_directory_dialog(current_path if current_path else str(OUTPUT_DIR))
        return new_path if new_path else current_path

    components.output_browse_btn.click(
        fn=handle_browse_output,
        inputs=[components.output_dir],
        outputs=[components.output_dir]
    )
