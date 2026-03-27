"""
Prompt Input Component for HunyuanImage-3.0 UI.
Groups prompt entry, Ollama enhancement, style selection, and wildcard browser together.
"""

import gradio as gr
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path

from ui.state import get_state
from ui.constants import (
    DEFAULT_STYLE_PRESETS,
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_LENGTH_OPTIONS,
    OLLAMA_COMPLEXITY_OPTIONS,
    OLLAMA_MODELS,
)

# Import wildcard utilities
try:
    from wildcard_utils import WildcardManager
    WILDCARD_AVAILABLE = True
except ImportError:
    WILDCARD_AVAILABLE = False
    WildcardManager = None


@dataclass
class PromptInputComponents:
    """Container for prompt input UI components."""
    # Main prompt
    prompt: gr.Textbox
    negative_prompt: gr.Textbox

    # Ollama enhancement (directly under prompt)
    use_ollama: gr.Checkbox
    ollama_model: gr.Dropdown
    ollama_length: gr.Dropdown
    ollama_complexity: gr.Dropdown

    # Style selection
    style_dropdown: gr.Dropdown

    # Settings loader
    settings_file: gr.File
    load_settings_btn: gr.Button

    # Wildcard browser (optional)
    wc_search: Optional[gr.Textbox] = None
    wc_category: Optional[gr.Radio] = None
    wc_list: Optional[gr.Dropdown] = None
    wc_preview: Optional[gr.Markdown] = None
    wc_insert_btn: Optional[gr.Button] = None

    # Prompt expansion preview (optional)
    expand_count: Optional[gr.Slider] = None
    expand_btn: Optional[gr.Button] = None
    expand_output: Optional[gr.Markdown] = None


def get_ollama_models_list() -> List[str]:
    """Get list of available Ollama models.

    Includes "Auto (use loaded)" as first option which auto-discovers
    whatever model the LLM Manager has loaded.
    """
    models = ["Auto (use loaded)"]  # Auto-discover option first

    state = get_state()
    if state.ollama_available and state.ollama_manager:
        try:
            ollama_models = state.ollama_manager.list_models()
            if ollama_models:
                for m in ollama_models:
                    if m not in models:
                        models.append(m)
                return models
        except Exception:
            pass

    # Fallback: add default models
    for m in OLLAMA_MODELS:
        if m not in models:
            models.append(m)
    return models


def get_style_presets() -> dict:
    """Get current style presets."""
    state = get_state()
    if state.style_presets:
        return state.style_presets
    return DEFAULT_STYLE_PRESETS


def get_wildcard_categories(wm) -> List[str]:
    """Extract unique category prefixes from wildcards."""
    if not wm:
        return []
    wildcards = wm.get_available_wildcards()
    categories = set()
    for w in wildcards:
        if '-' in w:
            categories.add(w.split('-')[0])
        else:
            categories.add("other")
    return sorted(categories)


def filter_wildcards_by_category(wm, category: str, search: str = "") -> List[str]:
    """Filter wildcards by category and search term."""
    if not wm:
        return []
    wildcards = wm.get_available_wildcards()

    # Filter by category
    if category and category != "all":
        wildcards = [w for w in wildcards if w.startswith(category + "-") or
                     (category == "other" and '-' not in w)]

    # Filter by search
    if search:
        search_lower = search.lower()
        wildcards = [w for w in wildcards if search_lower in w.lower()]

    return wildcards


def create_prompt_input(
    ollama_available: bool = False,
    wildcard_available: bool = False
) -> PromptInputComponents:
    """Create the prompt input UI component with wildcard browser and expansion preview.

    Args:
        ollama_available: Whether Ollama is available for enhancement
        wildcard_available: Whether wildcard system is available

    Returns:
        PromptInputComponents with references to all widgets.
    """
    # Get wildcard manager if available
    state = get_state()
    wm = state.wildcard_manager if wildcard_available and hasattr(state, 'wildcard_manager') else None

    # Initialize optional components
    wc_search = None
    wc_category = None
    wc_list = None
    wc_preview = None
    wc_insert_btn = None
    expand_count = None
    expand_btn = None
    expand_output = None

    # Main prompt section header
    gr.Markdown("### Prompt")

    # Two-column layout: Prompt on left, Wildcard browser on right
    with gr.Row():
        # Left: Main prompt area
        with gr.Column(scale=3):
            prompt = gr.Textbox(
                label="Describe your image",
                placeholder="A stunning landscape with mountains... Use [wildcard] syntax for random values",
                lines=4,
                max_lines=8,
            )

            # Prompt Expansion Preview (collapsible)
            if wildcard_available and wm:
                with gr.Accordion("Expansion Preview", open=False):
                    with gr.Row():
                        expand_count = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Variations",
                            scale=2
                        )
                        expand_btn = gr.Button("Preview", size="sm", scale=1)

                    expand_output = gr.Markdown(
                        value="*Enter a prompt with [wildcards] and click Preview*"
                    )

        # Right: Wildcard Browser sidebar
        if wildcard_available and wm:
            with gr.Column(scale=1, min_width=200):
                gr.Markdown("**Wildcards**")

                wc_search = gr.Textbox(
                    placeholder="Search...",
                    show_label=False,
                    container=False,
                )

                categories = ["all"] + get_wildcard_categories(wm)
                wc_category = gr.Radio(
                    choices=categories,
                    value="all",
                    label="Category",
                    interactive=True,
                )

                all_wildcards = wm.get_available_wildcards()
                wc_list = gr.Dropdown(
                    label="Select",
                    choices=all_wildcards,
                    interactive=True,
                    filterable=True,
                )

                wc_preview = gr.Markdown(
                    value=f"*{len(all_wildcards)} wildcards*"
                )

                wc_insert_btn = gr.Button("Insert [wildcard]", variant="primary", size="sm")

    # Ollama Enhancement - directly under prompt (the key UX change!)
    with gr.Group():
        with gr.Row():
            use_ollama = gr.Checkbox(
                label="Enhance with Ollama",
                value=False,
                scale=1,
                info="Use local LLM to expand your prompt"
            )
            models_list = get_ollama_models_list() if ollama_available else OLLAMA_MODELS
            # Ensure default model is in the list
            if DEFAULT_OLLAMA_MODEL not in models_list:
                models_list = [DEFAULT_OLLAMA_MODEL] + models_list
            ollama_model = gr.Dropdown(
                label="Model",
                choices=models_list,
                value=DEFAULT_OLLAMA_MODEL,
                scale=2,
                interactive=ollama_available,
                allow_custom_value=True
            )
        with gr.Row():
            ollama_length = gr.Dropdown(
                label="Length",
                choices=OLLAMA_LENGTH_OPTIONS,
                value="medium",
                scale=1,
                info="Output length"
            )
            ollama_complexity = gr.Dropdown(
                label="Complexity",
                choices=OLLAMA_COMPLEXITY_OPTIONS,
                value="detailed",
                scale=1,
                info="Detail level"
            )

    # Style Selection (part of "Prompt Modifiers" group)
    gr.Markdown("### Style & Modifiers")
    with gr.Row():
        style_presets = get_style_presets()
        style_dropdown = gr.Dropdown(
            label="Style Preset",
            choices=list(style_presets.keys()),
            value="None",
            scale=2,
            info="Appends style suffix to your prompt"
        )

    # Negative prompt
    negative_prompt = gr.Textbox(
        label="Negative Prompt (what to avoid)",
        placeholder="blurry, low quality, distorted...",
        lines=1,
    )

    # Settings loader (drag & drop JSON)
    gr.Markdown("### Load Settings")
    with gr.Row():
        settings_file = gr.File(
            label="Drop settings JSON here",
            file_types=[".json"],
            file_count="single",
            scale=2,
        )
        load_settings_btn = gr.Button(
            "Load Settings",
            variant="secondary",
            size="sm",
            scale=1,
        )

    return PromptInputComponents(
        prompt=prompt,
        negative_prompt=negative_prompt,
        use_ollama=use_ollama,
        ollama_model=ollama_model,
        ollama_length=ollama_length,
        ollama_complexity=ollama_complexity,
        style_dropdown=style_dropdown,
        settings_file=settings_file,
        load_settings_btn=load_settings_btn,
        # Wildcard components
        wc_search=wc_search,
        wc_category=wc_category,
        wc_list=wc_list,
        wc_preview=wc_preview,
        wc_insert_btn=wc_insert_btn,
        # Expansion preview
        expand_count=expand_count,
        expand_btn=expand_btn,
        expand_output=expand_output,
    )


def enhance_prompt_with_ollama(
    prompt: str,
    model: str,
    length: str,
    complexity: str
) -> Tuple[str, str]:
    """Enhance a prompt using Ollama.

    Args:
        prompt: Original prompt
        model: Ollama model name
        length: Desired output length
        complexity: Desired complexity level

    Returns:
        Tuple of (enhanced_prompt, status_message)
    """
    state = get_state()

    if not state.ollama_available or not state.ollama_enhancer:
        return prompt, "Ollama not available"

    try:
        enhanced = state.ollama_enhancer.enhance(
            prompt,
            length=length,
            complexity=complexity
        )
        return enhanced, f"Enhanced ({length}/{complexity})"
    except Exception as e:
        return prompt, f"Enhancement failed: {e}"


def wire_prompt_input_events(
    components: PromptInputComponents,
    refresh_models_callback=None
) -> None:
    """Wire up event handlers for prompt input components.

    Args:
        components: PromptInputComponents instance to wire up
        refresh_models_callback: Optional callback to refresh model list
    """
    state = get_state()
    wm = state.wildcard_manager if hasattr(state, 'wildcard_manager') else None

    # Enable/disable Ollama controls based on checkbox
    def toggle_ollama_controls(use_ollama: bool):
        return [
            gr.update(interactive=use_ollama),
            gr.update(interactive=use_ollama),
            gr.update(interactive=use_ollama),
        ]

    components.use_ollama.change(
        fn=toggle_ollama_controls,
        inputs=[components.use_ollama],
        outputs=[
            components.ollama_model,
            components.ollama_length,
            components.ollama_complexity
        ]
    )

    # Wildcard browser events (if available)
    if components.wc_search is not None and wm:
        all_wildcards = wm.get_available_wildcards()

        def update_wildcard_list(category, search):
            filtered = filter_wildcards_by_category(wm, category, search)
            return gr.update(choices=filtered, value=None)

        def update_wildcard_preview(wildcard):
            if not wildcard:
                return f"*{len(all_wildcards)} wildcards available*"
            preview = wm.get_wildcard_preview(wildcard, count=5)
            return f"**{wildcard}**\n\n{preview}"

        def insert_wildcard(current_text, wildcard):
            if not wildcard:
                return current_text
            tag = f"[{wildcard}]"
            if current_text:
                if not current_text.endswith(" "):
                    return current_text + " " + tag
                return current_text + tag
            return tag

        # Wire search and category filters
        components.wc_search.change(
            fn=update_wildcard_list,
            inputs=[components.wc_category, components.wc_search],
            outputs=[components.wc_list]
        )
        components.wc_category.change(
            fn=update_wildcard_list,
            inputs=[components.wc_category, components.wc_search],
            outputs=[components.wc_list]
        )

        # Wire preview update
        components.wc_list.change(
            fn=update_wildcard_preview,
            inputs=[components.wc_list],
            outputs=[components.wc_preview]
        )

        # Wire insert button
        components.wc_insert_btn.click(
            fn=insert_wildcard,
            inputs=[components.prompt, components.wc_list],
            outputs=[components.prompt]
        )

    # Prompt expansion preview events (if available)
    if components.expand_btn is not None and wm:
        import random

        def expand_prompt_previews(prompt, count):
            if not prompt:
                return "Enter a prompt to see expansion previews"
            if not wm and "[" in prompt:
                return "Wildcard manager not available"

            results = []
            count = int(count)
            for i in range(count):
                seed = random.randint(0, 2**32 - 1)
                expanded = wm.process_prompt(prompt, seed=seed) if wm else prompt
                results.append(f"**[{i+1}]** {expanded}")

            return "\n\n".join(results)

        components.expand_btn.click(
            fn=expand_prompt_previews,
            inputs=[components.prompt, components.expand_count],
            outputs=[components.expand_output]
        )
