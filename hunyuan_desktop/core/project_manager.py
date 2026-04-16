"""Project save/load for complete UI state across all tabs.

Saves and restores the entire application state (all 5 tabs) as a named project.
"""

import json
import time
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

    # Embed save timestamp + display name into the file
    project_dict = dict(project_dict)
    project_dict["_saved_at"] = time.time()
    project_dict["_display_name"] = name

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
    """Get list of saved project names (legacy: returns names only)."""
    return [p["name"] for p in get_saved_projects_with_meta()]


def get_saved_projects_with_meta() -> list:
    """Get saved projects with metadata, sorted by save time (newest first).

    Returns:
        List of dicts: {name, display_name, saved_at, path}
        - name: stem (used as id for load/delete)
        - display_name: human-readable name (from _display_name or stem)
        - saved_at: epoch float (from embedded _saved_at, falling back to file mtime)
        - path: full filepath as string
    """
    projects_dir = _get_projects_dir()
    items = []
    for f in projects_dir.glob("*.json"):
        try:
            saved_at = None
            display_name = None
            try:
                with open(f) as fh:
                    data = json.load(fh)
                saved_at = data.get("_saved_at")
                display_name = data.get("_display_name")
            except Exception:
                pass
            if saved_at is None:
                saved_at = f.stat().st_mtime
            items.append({
                "name": f.stem,
                "display_name": display_name or f.stem,
                "saved_at": float(saved_at),
                "path": str(f),
            })
        except Exception:
            continue
    items.sort(key=lambda d: d["saved_at"], reverse=True)
    return items


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
