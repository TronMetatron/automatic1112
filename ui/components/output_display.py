"""
Output Display Component for HunyuanImage-3.0 UI.
Shows generated image, status, seed info, and action buttons.
"""

import gradio as gr
from dataclasses import dataclass


@dataclass
class OutputDisplayComponents:
    """Container for output display UI components."""
    output_image: gr.Image
    status_text: gr.Textbox
    info_text: gr.Textbox
    last_seed: gr.Number
    copy_seed_btn: gr.Button
    generate_btn: gr.Button
    stop_btn: gr.Button
    queue_status: gr.Textbox
    clear_queue_btn: gr.Button
    refresh_btn: gr.Button


def create_output_display() -> OutputDisplayComponents:
    """Create the output display UI component.

    Returns:
        OutputDisplayComponents with references to all widgets.
    """
    # Generated image
    output_image = gr.Image(
        label="Generated Image",
        type="filepath",
        height=512,
    )

    # Status and info
    with gr.Row():
        status_text = gr.Textbox(
            label="Status",
            value="Ready - Load model to start generating",
            interactive=False,
            scale=2
        )

    with gr.Row():
        info_text = gr.Textbox(
            label="Generation Info",
            value="",
            interactive=False,
            lines=2,
            scale=2
        )

    # Seed display and copy
    with gr.Row():
        last_seed = gr.Number(
            label="Last Seed",
            value=0,
            precision=0,
            interactive=False,
            scale=2
        )
        copy_seed_btn = gr.Button(
            "Copy Seed",
            size="sm",
            scale=1
        )

    # Generate and Stop buttons
    with gr.Row():
        generate_btn = gr.Button(
            "Generate (+ Queue)",
            variant="primary",
            scale=2
        )
        stop_btn = gr.Button(
            "Stop",
            variant="stop",
            scale=1
        )

    # Queue status
    with gr.Row():
        queue_status = gr.Textbox(
            label="Queue",
            value="Queue: 0 pending",
            interactive=False,
            scale=2
        )
        clear_queue_btn = gr.Button(
            "Clear Queue",
            variant="secondary",
            size="sm",
            scale=1
        )
        refresh_btn = gr.Button(
            "Refresh",
            variant="secondary",
            size="sm",
            scale=1
        )

    return OutputDisplayComponents(
        output_image=output_image,
        status_text=status_text,
        info_text=info_text,
        last_seed=last_seed,
        copy_seed_btn=copy_seed_btn,
        generate_btn=generate_btn,
        stop_btn=stop_btn,
        queue_status=queue_status,
        clear_queue_btn=clear_queue_btn,
        refresh_btn=refresh_btn,
    )


def wire_output_display_events(components: OutputDisplayComponents) -> None:
    """Wire up event handlers for output display.

    Args:
        components: OutputDisplayComponents instance to wire up
    """
    # Copy seed to clipboard (returns the seed as text for JS to copy)
    def copy_seed(seed: int) -> str:
        return str(int(seed))

    # Note: Actual clipboard copy would need JavaScript
    # For now this just shows the seed is ready to copy
    components.copy_seed_btn.click(
        fn=copy_seed,
        inputs=[components.last_seed],
        outputs=[components.status_text]
    )
