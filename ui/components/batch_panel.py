"""
Batch Panel Component for HunyuanImage-3.0 UI.
Contains all batch generation tabs: Configure, Prompt Runs, Combine, Browse, Examples, Wildcards.
Opens by default for batch-heavy workflows.
"""

import gradio as gr
from dataclasses import dataclass
from typing import List, Optional

from ui.constants import (
    ASPECT_RATIOS,
    QUALITY_PRESETS,
    DEFAULT_STYLE_PRESETS,
    OLLAMA_MODELS,
    OLLAMA_LENGTH_OPTIONS,
    OLLAMA_COMPLEXITY_OPTIONS,
)


@dataclass
class BatchConfigComponents:
    """Configure Batch tab components."""
    batch_name: gr.Textbox
    batch_themes: gr.Textbox
    batch_negative_prompt: gr.Textbox
    batch_variations: gr.Slider
    batch_images_per: gr.Slider
    starred_reroll_count: gr.Slider  # For [*wildcard] reroll feature
    batch_styles: gr.CheckboxGroup
    batch_aspect: gr.Dropdown
    batch_quality: gr.Dropdown
    batch_guidance_scale: gr.Slider
    batch_llm_backend: gr.Dropdown
    batch_enhance: gr.Checkbox
    batch_ollama_model: gr.Dropdown
    batch_ollama_length: gr.Dropdown
    batch_ollama_complexity: gr.Dropdown
    batch_max_prompt_length: gr.Slider  # Max characters for enhanced prompts (0=no limit)
    batch_random_seeds: gr.Checkbox
    # Controls
    batch_calculate_btn: gr.Button
    batch_start_btn: gr.Button
    batch_stop_btn: gr.Button
    generate_prompts_only_btn: gr.Button
    # Status
    batch_preview: gr.Markdown
    batch_status: gr.Textbox
    batch_output_dir: gr.Textbox
    batch_gallery: gr.Gallery
    # Config management
    config_name_input: gr.Textbox
    batch_save_btn: gr.Button
    batch_config_dropdown: gr.Dropdown
    batch_load_btn: gr.Button
    refresh_configs_btn: gr.Button
    batch_config_status: gr.Textbox
    # Import from directory
    import_dir_path: gr.Textbox
    import_dir_browse_btn: gr.Button
    import_dir_btn: gr.Button
    # Expanded prompt editor
    expand_themes_btn: gr.Button
    themes_expanded: gr.Textbox
    themes_expanded_accordion: gr.Accordion
    apply_expanded_btn: gr.Button
    themes_line_count: gr.Markdown


@dataclass
class PromptRunComponents:
    """Saved Prompt Runs tab components."""
    prompt_run_dropdown: gr.Dropdown
    refresh_prompt_runs_btn: gr.Button
    interleave_checkbox: gr.Checkbox
    start_at_prompt: gr.Number
    loop_back_checkbox: gr.Checkbox
    run_from_saved_btn: gr.Button
    stop_prompt_run_btn: gr.Button
    delete_prompt_run_btn: gr.Button
    prompt_run_status: gr.Textbox
    prompt_run_output_dir: gr.Textbox
    prompt_run_preview: gr.Markdown
    prompt_run_gallery: gr.Gallery


@dataclass
class BrowseBatchComponents:
    """Browse Batches tab components."""
    batch_dir_dropdown: gr.Dropdown
    batch_dir_custom: gr.Textbox
    batch_dir_browse_btn: gr.Button
    browse_refresh_btn: gr.Button
    browse_gallery: gr.Gallery
    browse_status: gr.Textbox


@dataclass
class WildcardEditorComponents:
    """Wildcard editor tab components."""
    wildcard_search: gr.Textbox
    wildcard_list: gr.Dropdown
    wildcard_preview: gr.Textbox
    wildcard_items: gr.Textbox
    add_item_input: gr.Textbox
    add_item_btn: gr.Button
    remove_item_btn: gr.Button
    new_category_name: gr.Textbox
    create_category_btn: gr.Button
    delete_category_btn: gr.Button
    save_wildcards_btn: gr.Button
    reload_wildcards_btn: gr.Button
    import_wildcards_btn: gr.Button
    wildcard_status: gr.Textbox
    # Ollama generation
    ollama_prompt: gr.Textbox
    ollama_count: gr.Slider
    ollama_complexity: gr.Dropdown
    generate_from_ollama_btn: gr.Button
    generate_and_save_btn: gr.Button


@dataclass
class StyleEditorComponents:
    """Style/keyword editor tab components."""
    style_list: gr.Dropdown
    style_name_input: gr.Textbox
    style_suffix_input: gr.Textbox
    save_style_btn: gr.Button
    delete_style_btn: gr.Button
    new_style_name: gr.Textbox
    new_style_suffix: gr.Textbox
    create_style_btn: gr.Button
    reload_styles_btn: gr.Button
    style_status: gr.Textbox
    style_preview: gr.Markdown


@dataclass
class BatchPanelComponents:
    """Container for all batch panel components."""
    config: BatchConfigComponents
    prompt_runs: PromptRunComponents
    browse: BrowseBatchComponents
    wildcards: WildcardEditorComponents
    styles: StyleEditorComponents
    accordion: gr.Accordion


def get_style_choices() -> List[str]:
    """Get list of style preset names."""
    return list(DEFAULT_STYLE_PRESETS.keys())


def get_ollama_models() -> List[str]:
    """Get available Ollama models as string names.

    Includes "Auto (use loaded)" as first option which auto-discovers
    whatever model the LLM Manager has loaded.
    """
    models = ["Auto (use loaded)"]  # Auto-discover option first

    try:
        from ui.state import get_state
        state = get_state()
        if state.ollama_manager:
            ollama_models = state.ollama_manager.list_models()
            if ollama_models:
                # Extract model names from dicts (ollama_manager returns dicts with 'name' key)
                for m in ollama_models:
                    if isinstance(m, dict):
                        name = m.get('name', '')
                    else:
                        name = str(m)
                    if name and name not in models:
                        models.append(name)
                return models
    except Exception:
        pass

    # Fallback: add default models
    for m in OLLAMA_MODELS:
        if m not in models:
            models.append(m)
    return models


def get_saved_configs() -> List[str]:
    """Get list of saved batch configs."""
    try:
        import batch_manager as bm
        return bm.get_config_choices()
    except Exception:
        return []


def get_saved_prompt_runs() -> List[str]:
    """Get list of saved prompt runs (returns full filenames with .json extension)."""
    from ui.constants import OUTPUT_DIR
    prompt_runs_dir = OUTPUT_DIR / "prompt_runs"
    if not prompt_runs_dir.exists():
        return []
    runs = sorted(prompt_runs_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    return [r.name for r in runs]  # Return full filenames for consistency with handlers


def create_batch_config_tab() -> BatchConfigComponents:
    """Create the Configure Batch tab."""
    with gr.Row():
        # Left column - Configuration
        with gr.Column(scale=1):
            batch_name = gr.Textbox(
                label="Batch Name",
                value="my_batch",
                placeholder="Name for this batch run"
            )

            with gr.Row():
                batch_themes = gr.Textbox(
                    label="Themes/Prompts (one per line)",
                    placeholder="cyberpunk city at night\nunderwater ancient temple\nfantasy forest",
                    lines=5,
                    max_lines=15,
                    scale=10
                )
                expand_themes_btn = gr.Button("Edit", size="sm", scale=1, min_width=60)

            # Hidden placeholder components for compatibility (native popup is used instead)
            themes_expanded_accordion = gr.Accordion("", open=False, visible=False)
            themes_line_count = gr.Markdown("", visible=False)
            themes_expanded = gr.Textbox(visible=False)
            apply_expanded_btn = gr.Button(visible=False)

            batch_negative_prompt = gr.Textbox(
                label="Negative Prompt (all images)",
                placeholder="blurry, low quality, distorted",
                lines=1
            )

            with gr.Row():
                batch_variations = gr.Slider(
                    label="Iterations (cycles through all themes)",
                    minimum=1, maximum=500, value=30, step=1
                )
                batch_images_per = gr.Slider(
                    label="Images per combo",
                    minimum=1, maximum=50, value=1, step=1
                )

            with gr.Row():
                starred_reroll_count = gr.Slider(
                    label="Starred Wildcard Rerolls",
                    minimum=1, maximum=20, value=1, step=1,
                    info="Use [*wildcard] for same seed, different values"
                )
                gr.Markdown(
                    "**Tip:** `[*pose]` keeps scene, varies pose. "
                    "Files: `_base`, `_varA`, `_varB`...",
                    elem_classes=["info-text"]
                )

            batch_styles = gr.CheckboxGroup(
                label="Styles (select multiple)",
                choices=get_style_choices(),
                value=["Photorealistic", "Cinematic"]
            )

            with gr.Row():
                batch_aspect = gr.Dropdown(
                    label="Aspect", choices=list(ASPECT_RATIOS.keys()),
                    value="1:1 (Square)"
                )
                batch_quality = gr.Dropdown(
                    label="Quality", choices=list(QUALITY_PRESETS.keys()),
                    value="Standard"
                )
                batch_guidance_scale = gr.Slider(
                    label="Guidance", minimum=1.0, maximum=15.0,
                    value=5.0, step=0.5
                )

            with gr.Row():
                batch_llm_backend = gr.Dropdown(
                    label="LLM Backend",
                    choices=["Ollama", "LM Studio"],
                    value="Ollama"
                )
                batch_enhance = gr.Checkbox(
                    label="Enhance prompts", value=True
                )

            with gr.Row():
                batch_ollama_model = gr.Dropdown(
                    label="Model",
                    choices=get_ollama_models(),
                    value="qwen2.5:7b-instruct",
                    allow_custom_value=True
                )

            with gr.Row():
                batch_ollama_length = gr.Dropdown(
                    label="Length",
                    choices=OLLAMA_LENGTH_OPTIONS,
                    value="medium"
                )
                batch_ollama_complexity = gr.Dropdown(
                    label="Complexity",
                    choices=OLLAMA_COMPLEXITY_OPTIONS,
                    value="detailed"
                )

            batch_max_prompt_length = gr.Slider(
                label="Max Prompt Length (0=no limit)",
                minimum=0, maximum=4000, value=0, step=100,
                info="Truncate or retry if enhanced prompt exceeds this character count"
            )

            batch_random_seeds = gr.Checkbox(
                label="Random seeds", value=True
            )

        # Right column - Preview and controls
        with gr.Column(scale=1):
            batch_preview = gr.Markdown("Enter themes to see preview...")

            with gr.Row():
                batch_calculate_btn = gr.Button("Calculate", size="sm")
                generate_prompts_only_btn = gr.Button(
                    "Generate Prompts Only", variant="secondary"
                )

            with gr.Row():
                batch_start_btn = gr.Button("Start Batch", variant="primary", size="lg")
                batch_stop_btn = gr.Button("Stop", variant="stop")

            batch_status = gr.Textbox(
                label="Status", value="Ready", interactive=False, lines=2
            )
            batch_output_dir = gr.Textbox(
                label="Output Directory", interactive=False
            )

            gr.Markdown("---\n**Config Management**")
            with gr.Row():
                config_name_input = gr.Textbox(
                    label="Config Name", placeholder="my_config", scale=2
                )
                batch_save_btn = gr.Button("Save", variant="primary", size="sm")

            with gr.Row():
                batch_config_dropdown = gr.Dropdown(
                    label="Saved Configs",
                    choices=get_saved_configs(),
                    scale=3
                )
                batch_load_btn = gr.Button("Load", size="sm")
                refresh_configs_btn = gr.Button("↻", size="sm")

            gr.Markdown("---\n**Import from Directory**")
            with gr.Row():
                import_dir_path = gr.Textbox(
                    label="Image Directory",
                    placeholder="/path/to/images/folder",
                    scale=3
                )
                import_dir_browse_btn = gr.Button("📁", size="sm", min_width=40)
                import_dir_btn = gr.Button("Import", variant="secondary", size="sm")

            batch_config_status = gr.Textbox(
                label="Config Status", interactive=False
            )

    gr.Markdown("---\n**Batch Progress**")
    batch_gallery = gr.Gallery(
        label="Generated Images", columns=6, rows=2, height=180
    )

    return BatchConfigComponents(
        batch_name=batch_name,
        batch_themes=batch_themes,
        batch_negative_prompt=batch_negative_prompt,
        batch_variations=batch_variations,
        batch_images_per=batch_images_per,
        starred_reroll_count=starred_reroll_count,
        batch_styles=batch_styles,
        batch_aspect=batch_aspect,
        batch_quality=batch_quality,
        batch_guidance_scale=batch_guidance_scale,
        batch_llm_backend=batch_llm_backend,
        batch_enhance=batch_enhance,
        batch_ollama_model=batch_ollama_model,
        batch_ollama_length=batch_ollama_length,
        batch_ollama_complexity=batch_ollama_complexity,
        batch_max_prompt_length=batch_max_prompt_length,
        batch_random_seeds=batch_random_seeds,
        batch_calculate_btn=batch_calculate_btn,
        batch_start_btn=batch_start_btn,
        batch_stop_btn=batch_stop_btn,
        generate_prompts_only_btn=generate_prompts_only_btn,
        batch_preview=batch_preview,
        batch_status=batch_status,
        batch_output_dir=batch_output_dir,
        batch_gallery=batch_gallery,
        config_name_input=config_name_input,
        batch_save_btn=batch_save_btn,
        batch_config_dropdown=batch_config_dropdown,
        batch_load_btn=batch_load_btn,
        refresh_configs_btn=refresh_configs_btn,
        batch_config_status=batch_config_status,
        import_dir_path=import_dir_path,
        import_dir_browse_btn=import_dir_browse_btn,
        import_dir_btn=import_dir_btn,
        # Expanded editor
        expand_themes_btn=expand_themes_btn,
        themes_expanded=themes_expanded,
        themes_expanded_accordion=themes_expanded_accordion,
        apply_expanded_btn=apply_expanded_btn,
        themes_line_count=themes_line_count,
    )


def create_prompt_runs_tab() -> PromptRunComponents:
    """Create the Saved Prompt Runs tab."""
    gr.Markdown("""
    **Load and run saved prompt lists**
    Seeds are generated fresh each run for variety.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                prompt_run_dropdown = gr.Dropdown(
                    label="Select Prompt Run",
                    choices=get_saved_prompt_runs(),
                    scale=3
                )
                refresh_prompt_runs_btn = gr.Button("↻", size="sm")

            with gr.Row():
                interleave_checkbox = gr.Checkbox(
                    label="Interleave prompts", value=True,
                    info="Alternate themes instead of sequential"
                )

            with gr.Row():
                start_at_prompt = gr.Number(
                    label="Start at #", value=1, minimum=1
                )
                loop_back_checkbox = gr.Checkbox(
                    label="Loop", value=False,
                    info="Continue indefinitely"
                )

            with gr.Row():
                run_from_saved_btn = gr.Button(
                    "Run Selected", variant="primary", size="lg"
                )
                stop_prompt_run_btn = gr.Button("Stop", variant="stop")

            delete_prompt_run_btn = gr.Button("Delete Run", variant="stop", size="sm")

            prompt_run_status = gr.Textbox(
                label="Status", interactive=False, lines=2
            )
            prompt_run_output_dir = gr.Textbox(
                label="Output Directory", interactive=False
            )

        with gr.Column(scale=1):
            prompt_run_preview = gr.Markdown("Select a run to preview...")

    gr.Markdown("---\n**Progress**")
    prompt_run_gallery = gr.Gallery(columns=6, rows=2, height=180)

    return PromptRunComponents(
        prompt_run_dropdown=prompt_run_dropdown,
        refresh_prompt_runs_btn=refresh_prompt_runs_btn,
        interleave_checkbox=interleave_checkbox,
        start_at_prompt=start_at_prompt,
        loop_back_checkbox=loop_back_checkbox,
        run_from_saved_btn=run_from_saved_btn,
        stop_prompt_run_btn=stop_prompt_run_btn,
        delete_prompt_run_btn=delete_prompt_run_btn,
        prompt_run_status=prompt_run_status,
        prompt_run_output_dir=prompt_run_output_dir,
        prompt_run_preview=prompt_run_preview,
        prompt_run_gallery=prompt_run_gallery,
    )


def create_browse_tab() -> BrowseBatchComponents:
    """Create the Browse Batches tab."""
    gr.Markdown("**Browse generated batches**")

    with gr.Row():
        batch_dir_dropdown = gr.Dropdown(
            label="Batch Directory",
            choices=[],
            scale=2
        )
        batch_dir_custom = gr.Textbox(
            label="Custom Path",
            placeholder="/path/to/batch",
            scale=2
        )
        batch_dir_browse_btn = gr.Button("📁", size="sm", min_width=40)
        browse_refresh_btn = gr.Button("↻", size="sm")

    browse_gallery = gr.Gallery(columns=6, rows=3, height=250)

    browse_status = gr.Textbox(
        label="Status", interactive=False
    )

    return BrowseBatchComponents(
        batch_dir_dropdown=batch_dir_dropdown,
        batch_dir_custom=batch_dir_custom,
        batch_dir_browse_btn=batch_dir_browse_btn,
        browse_refresh_btn=browse_refresh_btn,
        browse_gallery=browse_gallery,
        browse_status=browse_status,
    )


def get_initial_wildcard_categories() -> list:
    """Get initial wildcard categories for dropdown."""
    try:
        from ui.state import get_state
        state = get_state()
        if state.wildcard_manager:
            return state.wildcard_manager.get_available_wildcards()
    except Exception:
        pass
    return []


def create_wildcard_editor_tab() -> WildcardEditorComponents:
    """Create the Wildcards editor tab."""
    gr.Markdown("""
    **Edit wildcard categories and items**
    Use `[category]` in prompts for random substitution.
    """)

    # Get initial categories
    initial_categories = get_initial_wildcard_categories()

    with gr.Row():
        with gr.Column(scale=1):
            wildcard_search = gr.Textbox(
                label="Search Categories",
                placeholder="Type to filter...",
            )
            wildcard_list = gr.Dropdown(
                label="Category",
                choices=initial_categories,
                value=initial_categories[0] if initial_categories else None,
                scale=2,
            )
            wildcard_preview = gr.Textbox(
                label="Category Info",
                interactive=False,
                lines=2,
            )

        with gr.Column(scale=2):
            wildcard_items = gr.Textbox(
                label="Items (one per line)",
                lines=12,
                max_lines=20,
            )

    with gr.Row():
        add_item_input = gr.Textbox(
            label="Add Item",
            placeholder="New item to add",
            scale=3,
        )
        add_item_btn = gr.Button("Add", size="sm", scale=1)
        remove_item_btn = gr.Button("Remove Selected", size="sm", variant="stop", scale=1)

    gr.Markdown("---\n**Create / Delete Category**")
    with gr.Row():
        new_category_name = gr.Textbox(
            label="New Category Name",
            placeholder="my-category",
            scale=2,
        )
        create_category_btn = gr.Button("Create", variant="primary", size="sm", scale=1)
        delete_category_btn = gr.Button("Delete Category", variant="stop", size="sm", scale=1)

    with gr.Row():
        save_wildcards_btn = gr.Button("Save All Changes", variant="primary", scale=2)
        reload_wildcards_btn = gr.Button("Reload from File", scale=1)
        import_wildcards_btn = gr.Button("Import from sd-wildcards/", variant="secondary", scale=1)

    wildcard_status = gr.Textbox(label="Status", interactive=False)

    gr.Markdown("---\n### Generate from Ollama LLM")
    gr.Markdown("*Ask the LLM to generate wildcard lists or detailed scene descriptions*")

    with gr.Row():
        ollama_prompt = gr.Textbox(
            label="Category / What to generate",
            placeholder="dungeon environments, fantasy weapons, epic battle scenes...",
            scale=3,
        )
        ollama_count = gr.Slider(
            label="Count",
            minimum=5,
            maximum=50,
            value=20,
            step=5,
            scale=1,
        )
        ollama_complexity = gr.Dropdown(
            label="Detail Level",
            choices=["keywords", "short", "medium", "long", "cinematic"],
            value="medium",
            scale=1,
        )

    with gr.Row():
        generate_from_ollama_btn = gr.Button(
            "Generate (Preview)",
            variant="secondary",
            scale=1,
        )
        generate_and_save_btn = gr.Button(
            "Generate & Save Category",
            variant="primary",
            scale=2,
        )

    return WildcardEditorComponents(
        wildcard_search=wildcard_search,
        wildcard_list=wildcard_list,
        wildcard_preview=wildcard_preview,
        wildcard_items=wildcard_items,
        add_item_input=add_item_input,
        add_item_btn=add_item_btn,
        remove_item_btn=remove_item_btn,
        new_category_name=new_category_name,
        create_category_btn=create_category_btn,
        delete_category_btn=delete_category_btn,
        save_wildcards_btn=save_wildcards_btn,
        reload_wildcards_btn=reload_wildcards_btn,
        import_wildcards_btn=import_wildcards_btn,
        wildcard_status=wildcard_status,
        ollama_prompt=ollama_prompt,
        ollama_count=ollama_count,
        ollama_complexity=ollama_complexity,
        generate_from_ollama_btn=generate_from_ollama_btn,
        generate_and_save_btn=generate_and_save_btn,
    )


def get_initial_style_names() -> list:
    """Get initial style names for dropdown."""
    try:
        from ui.constants import DEFAULT_STYLE_PRESETS
        return list(DEFAULT_STYLE_PRESETS.keys())
    except Exception:
        pass
    return ["None"]


def create_style_editor_tab() -> StyleEditorComponents:
    """Create the Styles/Keywords editor tab."""
    gr.Markdown("""
    **Edit style presets**
    Styles append text to your prompts automatically.
    """)

    # Get initial styles
    initial_styles = get_initial_style_names()

    with gr.Row():
        with gr.Column(scale=1):
            style_list = gr.Dropdown(
                label="Select Style",
                choices=initial_styles,
                value=initial_styles[0] if initial_styles else None,
            )
            style_name_input = gr.Textbox(
                label="Style Name",
                interactive=False,
            )

        with gr.Column(scale=2):
            style_suffix_input = gr.Textbox(
                label="Style Suffix (appended to prompts)",
                lines=4,
                placeholder=", cinematic lighting, 8k resolution, hyperdetailed",
            )

    with gr.Row():
        save_style_btn = gr.Button("Save Changes", variant="primary", scale=2)
        delete_style_btn = gr.Button("Delete Style", variant="stop", scale=1)

    gr.Markdown("---\n**Create New Style**")
    with gr.Row():
        new_style_name = gr.Textbox(
            label="New Style Name",
            placeholder="My Custom Style",
            scale=1,
        )
        new_style_suffix = gr.Textbox(
            label="Suffix",
            placeholder=", dramatic lighting, award winning",
            scale=2,
        )
        create_style_btn = gr.Button("Create", variant="primary", size="sm", scale=1)

    with gr.Row():
        reload_styles_btn = gr.Button("Reload Styles", scale=1)

    style_status = gr.Textbox(label="Status", interactive=False)

    style_preview = gr.Markdown("**Preview:** Select a style to see its suffix")

    return StyleEditorComponents(
        style_list=style_list,
        style_name_input=style_name_input,
        style_suffix_input=style_suffix_input,
        save_style_btn=save_style_btn,
        delete_style_btn=delete_style_btn,
        new_style_name=new_style_name,
        new_style_suffix=new_style_suffix,
        create_style_btn=create_style_btn,
        reload_styles_btn=reload_styles_btn,
        style_status=style_status,
        style_preview=style_preview,
    )


def create_batch_panel(open_by_default: bool = True) -> BatchPanelComponents:
    """Create the complete batch panel with all tabs.

    Args:
        open_by_default: Whether accordion starts open (True for batch-heavy workflows)

    Returns:
        BatchPanelComponents with references to all widgets.
    """
    with gr.Accordion("Batch Generation", open=open_by_default) as accordion:
        gr.Markdown("""
        **Mass image generation with themes, styles, and LLM enhancement**
        """)

        with gr.Tabs():
            with gr.Tab("Configure"):
                config = create_batch_config_tab()

            with gr.Tab("Prompt Runs"):
                prompt_runs = create_prompt_runs_tab()

            with gr.Tab("Browse"):
                browse = create_browse_tab()

            with gr.Tab("Wildcards"):
                wildcards = create_wildcard_editor_tab()

            with gr.Tab("Styles"):
                styles = create_style_editor_tab()

            with gr.Tab("Examples"):
                gr.Markdown("""
                **Example themes to try:**

                ```
                cyberpunk city at night with neon signs
                underwater ancient temple with bioluminescent creatures
                fantasy forest with magical creatures and floating lights
                steampunk airship flying over Victorian London
                alien landscape with multiple moons and strange vegetation
                ```

                **Wildcard examples:**
                ```
                A [animal] in a [landscape] with [weather] weather
                Portrait of a [occupation] wearing [color] clothes
                [art-movement] painting of [subject]
                ```

                **Starred Wildcard examples (for video AI):**
                ```
                A woman [*pose] in a [setting]
                A warrior [*pose] holding a [weapon] in [landscape]
                [race] character [*pose] wearing [clothing]
                ```
                Use `[*wildcard]` to mark wildcards for re-rolling.
                Same seed = same scene, but different poses!
                Files named: `_base`, `_varA`, `_varB`, `_varC`...

                **Tips:**
                - Use specific, descriptive themes
                - Let Ollama add the creative details
                - Mix different style presets for variety
                - Use wildcards for random variations
                - Use starred wildcards for video frame variations
                """)

    return BatchPanelComponents(
        config=config,
        prompt_runs=prompt_runs,
        browse=browse,
        wildcards=wildcards,
        styles=styles,
        accordion=accordion,
    )
