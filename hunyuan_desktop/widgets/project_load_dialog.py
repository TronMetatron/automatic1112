"""Project load dialog with save dates, sorting, and delete support."""

import time
from datetime import datetime

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QMessageBox,
)
from PySide6.QtCore import Qt


def _format_age(saved_at: float) -> str:
    """Format a timestamp as a relative + absolute string."""
    now = time.time()
    delta = now - saved_at
    if delta < 60:
        rel = "just now"
    elif delta < 3600:
        rel = f"{int(delta / 60)}m ago"
    elif delta < 86400:
        rel = f"{int(delta / 3600)}h ago"
    elif delta < 86400 * 7:
        rel = f"{int(delta / 86400)}d ago"
    else:
        rel = datetime.fromtimestamp(saved_at).strftime("%b %-d, %Y")
    abs_str = datetime.fromtimestamp(saved_at).strftime("%Y-%m-%d %H:%M")
    return f"{rel}  ·  {abs_str}"


class ProjectLoadDialog(QDialog):
    """Dialog showing saved projects with save dates, sorted newest-first."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Project")
        self.setMinimumSize(560, 420)
        self._selected_name = None
        self._setup_ui()
        self._populate()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Saved projects (newest first):"))

        self.list_widget = QListWidget()
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.itemDoubleClicked.connect(self._on_load)
        layout.addWidget(self.list_widget, stretch=1)

        self.empty_label = QLabel("No saved projects.")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("color: #808080; padding: 20px;")
        self.empty_label.setVisible(False)
        layout.addWidget(self.empty_label)

        btn_row = QHBoxLayout()
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._on_delete)
        btn_row.addWidget(self.delete_btn)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._populate)
        btn_row.addWidget(self.refresh_btn)

        btn_row.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.cancel_btn)

        self.load_btn = QPushButton("Load")
        self.load_btn.setDefault(True)
        self.load_btn.clicked.connect(self._on_load)
        btn_row.addWidget(self.load_btn)

        layout.addLayout(btn_row)

    def _populate(self):
        from core.project_manager import get_saved_projects_with_meta
        self.list_widget.clear()
        items = get_saved_projects_with_meta()
        if not items:
            self.empty_label.setVisible(True)
            self.list_widget.setVisible(False)
            self.delete_btn.setEnabled(False)
            self.load_btn.setEnabled(False)
            return
        self.empty_label.setVisible(False)
        self.list_widget.setVisible(True)
        self.delete_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        for it in items:
            label = f"{it['display_name']}\n    {_format_age(it['saved_at'])}"
            li = QListWidgetItem(label)
            li.setData(Qt.ItemDataRole.UserRole, it["name"])
            self.list_widget.addItem(li)
        self.list_widget.setCurrentRow(0)

    def _current_name(self) -> str:
        item = self.list_widget.currentItem()
        if not item:
            return ""
        return item.data(Qt.ItemDataRole.UserRole) or ""

    def _on_load(self, *_args):
        name = self._current_name()
        if not name:
            return
        self._selected_name = name
        self.accept()

    def _on_delete(self):
        name = self._current_name()
        if not name:
            return
        item = self.list_widget.currentItem()
        display = item.text().split("\n")[0] if item else name
        reply = QMessageBox.question(
            self, "Delete Project",
            f"Delete project '{display}'? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        from core.project_manager import delete_project
        delete_project(name)
        self._populate()

    def selected_name(self) -> str:
        return self._selected_name or ""
