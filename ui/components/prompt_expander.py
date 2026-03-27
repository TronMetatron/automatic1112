#!/usr/bin/env python3
"""
Prompt Expander and Wildcard Sidebar UI Components

Provides:
1. Prompt expansion preview with multiple variations
2. Interactive wildcard browser/inserter sidebar
3. Combined prompt builder with drag-and-drop wildcards
"""

import gradio as gr
from typing import Callable, Optional, List, Tuple
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from wildcard_utils import WildcardManager
    WILDCARD_AVAILABLE = True
except ImportError:
    WILDCARD_AVAILABLE = False
    WildcardManager = None

try:
    from ollama_prompts import PromptEnhancer
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    PromptEnhancer = None


def get_wildcard_categories(wm: WildcardManager) -> List[str]:
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


def filter_wildcards_by_category(wm: WildcardManager, category: str, search: str = "") -> List[str]:
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


def expand_prompt_preview(prompt: str, wm: WildcardManager, count: int = 5,
                          enhance: bool = False, ollama_model: str = None) -> str:
    """Generate multiple expansion previews of a prompt."""
    if not prompt:
        return "Enter a prompt to see expansion previews"

    if not wm and "[" in prompt:
        return "Wildcard manager not available"

    results = []
    import random

    for i in range(count):
        seed = random.randint(0, 2**32 - 1)
        expanded = wm.process_prompt(prompt, seed=seed) if wm else prompt

        # Optional enhancement
        if enhance and OLLAMA_AVAILABLE and ollama_model:
            try:
                enhancer = PromptEnhancer(model=ollama_model)
                expanded = enhancer.enhance(expanded, temperature=0.7)
            except Exception as e:
                expanded = f"{expanded} [enhancement failed: {e}]"

        results.append(f"**[{i+1}]** {expanded}")

    return "\n\n".join(results)


def insert_wildcard_at_cursor(current_text: str, wildcard: str) -> str:
    """Insert a wildcard tag at the end of the text (cursor position not available in Gradio)."""
    if not wildcard:
        return current_text

    tag = f"[{wildcard}]"
    if current_text:
        # Add space if needed
        if not current_text.endswith(" "):
            return current_text + " " + tag
        return current_text + tag
    return tag


def create_wildcard_browser(wm: WildcardManager = None) -> Tuple:
    """Create the wildcard browser sidebar component.

    Returns tuple of (component_dict, setup_function)
    """
    if not WILDCARD_AVAILABLE:
        return None, None

    if wm is None:
        wm = WildcardManager(json_path=Path(__file__).parent.parent.parent / "wildcards.json")

    categories = ["all"] + get_wildcard_categories(wm)
    all_wildcards = wm.get_available_wildcards()

    with gr.Column(elem_classes=["wildcard-sidebar"]) as sidebar:
        gr.Markdown("### Wildcard Browser")

        with gr.Row():
            wc_search = gr.Textbox(
                placeholder="Search wildcards...",
                show_label=False,
                scale=3
            )
            wc_clear = gr.Button("Clear", size="sm", scale=1)

        wc_category = gr.Radio(
            choices=categories,
            value="all",
            label="Category",
            interactive=True
        )

        wc_list = gr.Dropdown(
            label="Available Wildcards",
            choices=all_wildcards,
            multiselect=False,
            filterable=True,
            interactive=True,
            max_choices=1
        )

        wc_preview = gr.Markdown(
            value=f"*Select a wildcard to see preview*\n\n**{len(all_wildcards)}** wildcards available"
        )

        with gr.Row():
            wc_insert = gr.Button("Insert [wildcard]", variant="primary", size="sm")
            wc_combined = gr.Button("+ (Combine)", size="sm")

        # Combined wildcard builder
        with gr.Accordion("Combine Wildcards", open=False):
            gr.Markdown("Build combined wildcards like `[color+animal]`")
            combined_parts = gr.Textbox(
                label="Combined Parts",
                placeholder="color+animal+action",
                interactive=True
            )
            with gr.Row():
                combined_preview = gr.Textbox(
                    label="Preview",
                    interactive=False,
                    scale=3
                )
                combined_insert = gr.Button("Insert", size="sm", scale=1)

    # Event handlers
    def update_wildcard_list(category, search):
        filtered = filter_wildcards_by_category(wm, category, search)
        return gr.update(choices=filtered, value=None)

    def update_preview(wildcard):
        if not wildcard:
            return f"*Select a wildcard to see preview*\n\n**{len(all_wildcards)}** wildcards available"
        preview = wm.get_wildcard_preview(wildcard, count=5)
        return f"**{wildcard}**\n\n{preview}"

    def clear_search():
        return "", gr.update(choices=all_wildcards, value=None)

    def add_to_combined(current, wildcard):
        if not wildcard:
            return current
        if current:
            return f"{current}+{wildcard}"
        return wildcard

    def preview_combined(parts):
        if not parts:
            return ""
        result = wm.get_combined_value(parts)
        return result or f"Invalid: some keys not found"

    # Wire up events
    wc_search.change(
        fn=update_wildcard_list,
        inputs=[wc_category, wc_search],
        outputs=[wc_list]
    )
    wc_category.change(
        fn=update_wildcard_list,
        inputs=[wc_category, wc_search],
        outputs=[wc_list]
    )
    wc_list.change(
        fn=update_preview,
        inputs=[wc_list],
        outputs=[wc_preview]
    )
    wc_clear.click(
        fn=clear_search,
        outputs=[wc_search, wc_list]
    )
    wc_combined.click(
        fn=add_to_combined,
        inputs=[combined_parts, wc_list],
        outputs=[combined_parts]
    )
    combined_parts.change(
        fn=preview_combined,
        inputs=[combined_parts],
        outputs=[combined_preview]
    )

    return {
        "sidebar": sidebar,
        "search": wc_search,
        "category": wc_category,
        "list": wc_list,
        "preview": wc_preview,
        "insert_btn": wc_insert,
        "combined_parts": combined_parts,
        "combined_insert": combined_insert,
    }, wm


def create_prompt_expander(wm: WildcardManager = None) -> Tuple:
    """Create the prompt expansion preview component.

    Returns tuple of (components_dict, wm)
    """
    if wm is None and WILDCARD_AVAILABLE:
        wm = WildcardManager(json_path=Path(__file__).parent.parent.parent / "wildcards.json")

    with gr.Accordion("Prompt Expansion Preview", open=False) as accordion:
        gr.Markdown("""
        Preview how your prompt expands with wildcards and optional LLM enhancement.
        Shows multiple random variations.
        """)

        with gr.Row():
            expand_count = gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="Number of variations",
                scale=2
            )
            expand_enhance = gr.Checkbox(
                label="Enhance with LLM",
                value=False,
                scale=1
            )
            expand_btn = gr.Button("Generate Previews", variant="primary", scale=1)

        expand_output = gr.Markdown(
            value="*Click 'Generate Previews' to see expansion variations*",
            elem_classes=["expansion-preview"]
        )

    return {
        "accordion": accordion,
        "count": expand_count,
        "enhance": expand_enhance,
        "button": expand_btn,
        "output": expand_output,
    }, wm


def create_combined_prompt_panel(prompt_input: gr.Textbox = None) -> dict:
    """Create a combined panel with prompt input, expander, and wildcard browser side by side.

    If prompt_input is provided, uses that instead of creating a new one.
    """
    if not WILDCARD_AVAILABLE:
        wm = None
    else:
        wm = WildcardManager(json_path=Path(__file__).parent.parent.parent / "wildcards.json")

    components = {}

    with gr.Row():
        # Main prompt area (left)
        with gr.Column(scale=3):
            if prompt_input is None:
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="A [animal] in a [landscape] setting...",
                    lines=4,
                    max_lines=8,
                )
            components["prompt"] = prompt_input

            # Expansion preview
            with gr.Accordion("Expansion Preview", open=True):
                with gr.Row():
                    expand_count = gr.Slider(1, 10, 3, step=1, label="Variations", scale=2)
                    expand_btn = gr.Button("Preview", size="sm", scale=1)

                expand_output = gr.Markdown("*Enter a prompt with [wildcards] and click Preview*")
                components["expand_count"] = expand_count
                components["expand_btn"] = expand_btn
                components["expand_output"] = expand_output

        # Wildcard sidebar (right)
        if wm:
            with gr.Column(scale=1, min_width=250):
                gr.Markdown("### Insert Wildcards")

                wc_search = gr.Textbox(placeholder="Search...", show_label=False)
                wc_category = gr.Radio(
                    choices=["all"] + get_wildcard_categories(wm),
                    value="all",
                    label="Category",
                )
                wc_list = gr.Dropdown(
                    label="Wildcards",
                    choices=wm.get_available_wildcards(),
                    interactive=True,
                )
                wc_preview = gr.Textbox(label="Preview", interactive=False, lines=2)

                with gr.Row():
                    wc_insert = gr.Button("Insert", variant="primary", size="sm")
                    wc_refresh = gr.Button("↻", size="sm")

                components["wc_search"] = wc_search
                components["wc_category"] = wc_category
                components["wc_list"] = wc_list
                components["wc_preview"] = wc_preview
                components["wc_insert"] = wc_insert
                components["wc_refresh"] = wc_refresh

                # Wire up wildcard events
                def update_list(cat, search):
                    filtered = filter_wildcards_by_category(wm, cat, search)
                    return gr.update(choices=filtered)

                def show_preview(w):
                    if not w:
                        return ""
                    return wm.get_wildcard_preview(w, count=3)

                wc_search.change(update_list, [wc_category, wc_search], [wc_list])
                wc_category.change(update_list, [wc_category, wc_search], [wc_list])
                wc_list.change(show_preview, [wc_list], [wc_preview])
                wc_refresh.click(
                    lambda: (gr.update(choices=wm.get_available_wildcards()), ""),
                    outputs=[wc_list, wc_search]
                )

    # Wire up expansion preview
    def do_expand(prompt, count):
        return expand_prompt_preview(prompt, wm, count)

    if "expand_btn" in components:
        components["expand_btn"].click(
            do_expand,
            inputs=[components["prompt"], components["expand_count"]],
            outputs=[components["expand_output"]]
        )

    # Wire up insert button
    if "wc_insert" in components:
        components["wc_insert"].click(
            insert_wildcard_at_cursor,
            inputs=[components["prompt"], components["wc_list"]],
            outputs=[components["prompt"]]
        )

    components["wm"] = wm
    return components


# Standalone test
if __name__ == "__main__":
    with gr.Blocks(title="Prompt Expander Test") as demo:
        gr.Markdown("# Prompt Expander Test")

        components = create_combined_prompt_panel()

    demo.launch()
