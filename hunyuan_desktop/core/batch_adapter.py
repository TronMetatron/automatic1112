"""Adapter for batch_manager.py that strips Gradio dependencies.

The existing batch_manager.py returns gr.update() objects. This module
wraps those functions and extracts raw values, so the desktop app
can use batch config save/load without importing Gradio.
"""

import json
from pathlib import Path


# Batch configs directory (same location the Gradio UI uses)
def _get_configs_dir() -> Path:
    from ui.constants import OUTPUT_DIR
    configs_dir = OUTPUT_DIR / "batch_configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    return configs_dir


def save_batch_config(name: str, config_dict: dict) -> str:
    """Save a batch configuration to disk.

    Args:
        name: Config name (used as filename)
        config_dict: Dict from BatchConfig.to_dict()

    Returns:
        Path to saved config file
    """
    configs_dir = _get_configs_dir()
    safe_name = "".join(
        c for c in name if c.isalnum() or c in " -_"
    ).strip().replace(" ", "_")

    if not safe_name:
        safe_name = "unnamed"

    filepath = configs_dir / f"{safe_name}.json"
    with open(filepath, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    return str(filepath)


def load_batch_config(name: str) -> dict:
    """Load a batch configuration from disk.

    Args:
        name: Config name (without .json extension)

    Returns:
        Config dict, or empty dict if not found
    """
    configs_dir = _get_configs_dir()

    # Try exact match first
    filepath = configs_dir / f"{name}.json"
    if not filepath.exists():
        # Try with safe name conversion
        safe_name = "".join(
            c for c in name if c.isalnum() or c in " -_"
        ).strip().replace(" ", "_")
        filepath = configs_dir / f"{safe_name}.json"

    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)

    return {}


def get_saved_configs() -> list:
    """Get list of saved config names.

    Returns:
        List of config names (without .json extension)
    """
    configs_dir = _get_configs_dir()
    configs = []
    for f in sorted(configs_dir.glob("*.json")):
        configs.append(f.stem)
    return configs


def delete_batch_config(name: str) -> bool:
    """Delete a saved batch configuration.

    Args:
        name: Config name to delete

    Returns:
        True if deleted, False if not found
    """
    configs_dir = _get_configs_dir()
    filepath = configs_dir / f"{name}.json"
    if filepath.exists():
        filepath.unlink()
        return True
    return False


def import_batch_from_directory(directory: str) -> dict:
    """Import batch settings from a batch output directory.

    Reads the batch_manifest.json from a completed batch run.

    Args:
        directory: Path to batch output directory

    Returns:
        Config dict from the batch manifest, or empty dict
    """
    dir_path = Path(directory)
    manifest = dir_path / "batch_manifest.json"
    if manifest.exists():
        with open(manifest) as f:
            data = json.load(f)
        return data.get("settings", {})
    return {}
