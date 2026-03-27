"""Import batch settings from a completed batch directory."""

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QTextEdit, QMessageBox
)
from PySide6.QtCore import Signal, Slot

from core.batch_adapter import import_batch_from_directory


class ImportDialog(QDialog):
    """Dialog for importing batch configuration from a batch output directory."""

    config_imported = Signal(dict)  # config_dict

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Batch Configuration")
        self.setMinimumSize(500, 350)
        self._config = {}

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            "Import settings from a previous batch run.\n"
            "Select the batch output directory containing batch_manifest.json."
        ))

        # Directory selection
        dir_row = QHBoxLayout()
        self.dir_input = QLineEdit()
        self.dir_input.setPlaceholderText("Select batch output directory...")
        self.dir_input.setReadOnly(True)
        dir_row.addWidget(self.dir_input, stretch=1)
        self.browse_btn = QPushButton("Browse...")
        dir_row.addWidget(self.browse_btn)
        layout.addLayout(dir_row)

        # Preview
        layout.addWidget(QLabel("Configuration Preview:"))
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(200)
        layout.addWidget(self.preview_text)

        # Status
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.import_btn = QPushButton("Import")
        self.import_btn.setEnabled(False)
        btn_row.addWidget(self.import_btn)
        self.cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

    def _connect_signals(self):
        self.browse_btn.clicked.connect(self._on_browse)
        self.import_btn.clicked.connect(self._on_import)
        self.cancel_btn.clicked.connect(self.close)

    @Slot()
    def _on_browse(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Batch Output Directory"
        )
        if not directory:
            return

        self.dir_input.setText(directory)
        self._config = import_batch_from_directory(directory)

        if self._config:
            import json
            self.preview_text.setPlainText(
                json.dumps(self._config, indent=2, default=str)
            )
            themes = self._config.get("themes", [])
            self.status_label.setText(
                f"Found config with {len(themes)} themes"
            )
            self.import_btn.setEnabled(True)
        else:
            self.preview_text.clear()
            self.status_label.setText(
                "No batch_manifest.json found in this directory"
            )
            self.import_btn.setEnabled(False)

    @Slot()
    def _on_import(self):
        if self._config:
            self.config_imported.emit(self._config)
            self.close()
