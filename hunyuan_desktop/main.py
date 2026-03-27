#!/usr/bin/env python3
"""
HunyuanImage Desktop - PySide6 Native Application
Entry point: sets up sys.path, environment variables, and launches the app.
"""

import sys
import os

# Add parent directory for importing existing backend modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Add hunyuan_desktop package directory for bare imports
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PACKAGE_DIR)

# Add HunyuanImage-3.0 for sdnq module
HUNYUAN_DIR = os.path.join(PROJECT_ROOT, "HunyuanImage-3.0")
if os.path.isdir(HUNYUAN_DIR):
    sys.path.insert(0, HUNYUAN_DIR)

# Set environment variables before any CUDA imports
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# Allow user environment to dictate cache directories.
# Only default if absolutely necessary (e.g. system default locations).


def main():
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt

    # High DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("HunyuanImage Desktop")
    app.setOrganizationName("HunyuanImage")
    app.setApplicationVersion("1.0.0")

    # Apply dark theme
    from theme import apply_dark_theme
    apply_dark_theme(app)

    # Initialize core state
    from core.app_state import DesktopState
    desktop_state = DesktopState()
    desktop_state.initialize()

    # Create and show main window
    from widgets.main_window import MainWindow
    window = MainWindow(desktop_state)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
