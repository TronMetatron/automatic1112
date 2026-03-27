"""
HunyuanImage-3.0 UI Components Package
Individual UI components that make up the interface.
"""

from ui.components.system_bar import (
    SystemBarComponents,
    create_system_bar,
    wire_system_bar_events,
)

from ui.components.prompt_input import (
    PromptInputComponents,
    create_prompt_input,
    wire_prompt_input_events,
    enhance_prompt_with_ollama,
)

from ui.components.gen_settings import (
    GenSettingsComponents,
    create_gen_settings,
    wire_gen_settings_events,
    get_steps_from_quality,
    get_size_from_aspect,
)

from ui.components.output_display import (
    OutputDisplayComponents,
    create_output_display,
    wire_output_display_events,
)

from ui.components.batch_panel import (
    BatchPanelComponents,
    BatchConfigComponents,
    PromptRunComponents,
    create_batch_panel,
)

from ui.components.prompt_expander import (
    create_wildcard_browser,
    create_prompt_expander,
    create_combined_prompt_panel,
    expand_prompt_preview,
    insert_wildcard_at_cursor,
)

__all__ = [
    # System bar
    'SystemBarComponents',
    'create_system_bar',
    'wire_system_bar_events',
    # Prompt input
    'PromptInputComponents',
    'create_prompt_input',
    'wire_prompt_input_events',
    'enhance_prompt_with_ollama',
    # Generation settings
    'GenSettingsComponents',
    'create_gen_settings',
    'wire_gen_settings_events',
    'get_steps_from_quality',
    'get_size_from_aspect',
    # Output display
    'OutputDisplayComponents',
    'create_output_display',
    'wire_output_display_events',
    # Batch panel
    'BatchPanelComponents',
    'BatchConfigComponents',
    'PromptRunComponents',
    'create_batch_panel',
    # Prompt expander and wildcard browser
    'create_wildcard_browser',
    'create_prompt_expander',
    'create_combined_prompt_panel',
    'expand_prompt_preview',
    'insert_wildcard_at_cursor',
]
