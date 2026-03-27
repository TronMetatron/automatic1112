"""Style preset editor dialog: create, edit, and delete style presets."""

import json

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QTextEdit, QListWidget, QListWidgetItem, QPushButton,
    QMessageBox, QSplitter, QWidget
)
from PySide6.QtCore import Qt, Signal, Slot


class StyleEditorDialog(QDialog):
    """Dialog for creating, editing, and deleting style presets."""

    styles_changed = Signal()

    def __init__(self, desktop_state, parent=None):
        super().__init__(parent)
        self.state = desktop_state
        self.setWindowTitle("Style Preset Editor")
        self.setMinimumSize(700, 500)

        self._setup_ui()
        self._connect_signals()
        self._load_styles()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Style list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(QLabel("Style Presets:"))
        self.style_list = QListWidget()
        self.style_list.setAlternatingRowColors(True)
        left_layout.addWidget(self.style_list)

        btn_row = QHBoxLayout()
        self.new_btn = QPushButton("New")
        btn_row.addWidget(self.new_btn)
        self.delete_btn = QPushButton("Delete")
        btn_row.addWidget(self.delete_btn)
        left_layout.addLayout(btn_row)

        splitter.addWidget(left_widget)

        # Right: Editor
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.name_input = QLineEdit()
        name_row.addWidget(self.name_input, stretch=1)
        right_layout.addLayout(name_row)

        right_layout.addWidget(QLabel("Suffix (appended to prompt):"))
        self.suffix_editor = QTextEdit()
        self.suffix_editor.setPlaceholderText(
            "e.g., , cinematic lighting, dramatic composition, "
            "professional photography"
        )
        right_layout.addWidget(self.suffix_editor)

        right_layout.addWidget(QLabel(
            "Preview: The suffix text is appended to the user's prompt. "
            "Include a leading comma."
        ))

        self.save_btn = QPushButton("Save Style")
        right_layout.addWidget(self.save_btn)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

        # Bottom
        bottom_row = QHBoxLayout()
        bottom_row.addStretch()
        self.close_btn = QPushButton("Close")
        bottom_row.addWidget(self.close_btn)
        layout.addLayout(bottom_row)

    def _connect_signals(self):
        self.style_list.currentItemChanged.connect(self._on_style_selected)
        self.new_btn.clicked.connect(self._on_new)
        self.delete_btn.clicked.connect(self._on_delete)
        self.save_btn.clicked.connect(self._on_save)
        self.close_btn.clicked.connect(self.close)

    def _load_styles(self):
        self.style_list.clear()
        presets = self.state.get_style_presets()
        for name in sorted(presets.keys()):
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, presets[name])
            self.style_list.addItem(item)

    @Slot()
    def _on_style_selected(self):
        item = self.style_list.currentItem()
        if item:
            self.name_input.setText(item.text())
            self.suffix_editor.setPlainText(
                item.data(Qt.ItemDataRole.UserRole) or ""
            )

    @Slot()
    def _on_new(self):
        self.name_input.clear()
        self.suffix_editor.clear()
        self.style_list.clearSelection()
        self.name_input.setFocus()

    @Slot()
    def _on_delete(self):
        item = self.style_list.currentItem()
        if not item:
            return
        name = item.text()
        if name == "None":
            QMessageBox.information(self, "Delete", "Cannot delete the 'None' preset")
            return

        reply = QMessageBox.question(
            self, "Delete Style",
            f"Delete style preset '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            presets = self.state.get_style_presets()
            if name in presets:
                del presets[name]
                self._save_presets(presets)
                self._load_styles()

    @Slot()
    def _on_save(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Save", "Enter a style name")
            return

        suffix = self.suffix_editor.toPlainText().strip()
        presets = self.state.get_style_presets()
        presets[name] = suffix
        self._save_presets(presets)
        self._load_styles()
        self.styles_changed.emit()

    def _save_presets(self, presets: dict):
        """Save presets to state and disk."""
        self.state.state.style_presets = presets

        from ui.constants import STYLE_PRESETS_FILE, DEFAULT_STYLE_PRESETS
        # Only save custom presets (not defaults)
        custom = {k: v for k, v in presets.items() if k not in DEFAULT_STYLE_PRESETS}
        try:
            with open(STYLE_PRESETS_FILE, "w") as f:
                json.dump(custom, f, indent=2)
        except Exception:
            pass
