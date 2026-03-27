"""Project save/load for complete UI state across all tabs.

Saves and restores the entire application state (all 4 tabs) as a named project.
"""

import json
from pathlib import Path


def _get_projects_dir() -> Path:
    from ui.constants import OUTPUT_DIR
    projects_dir = OUTPUT_DIR / "projects"
    projects_dir.mkdir(parents=True, exist_ok=True)
    return projects_dir


def save_project(name: str, project_dict: dict) -> str:
    """Save a complete project state to disk.

    Args:
        name: Project name (used as filename)
        project_dict: Dict containing all tab states

    Returns:
        Path to saved project file
    """
    projects_dir = _get_projects_dir()
    safe_name = "".join(
        c for c in name if c.isalnum() or c in " -_"
    ).strip().replace(" ", "_")

    if not safe_name:
        safe_name = "untitled_project"

    filepath = projects_dir / f"{safe_name}.json"
    with open(filepath, "w") as f:
        json.dump(project_dict, f, indent=2, default=str)

    return str(filepath)


def load_project(name: str) -> dict:
    """Load a project from disk.

    Args:
        name: Project name (without .json extension)

    Returns:
        Project dict, or empty dict if not found
    """
    projects_dir = _get_projects_dir()

    # Try exact match first
    filepath = projects_dir / f"{name}.json"
    if not filepath.exists():
        # Try with safe name conversion
        safe_name = "".join(
            c for c in name if c.isalnum() or c in " -_"
        ).strip().replace(" ", "_")
        filepath = projects_dir / f"{safe_name}.json"

    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)

    return {}


def get_saved_projects() -> list:
    """Get list of saved project names.

    Returns:
        List of project names (without .json extension)
    """
    projects_dir = _get_projects_dir()
    projects = []
    for f in sorted(projects_dir.glob("*.json")):
        projects.append(f.stem)
    return projects


def delete_project(name: str) -> bool:
    """Delete a saved project.

    Args:
        name: Project name to delete

    Returns:
        True if deleted, False if not found
    """
    projects_dir = _get_projects_dir()
    filepath = projects_dir / f"{name}.json"
    if filepath.exists():
        filepath.unlink()
        return True
    return False
