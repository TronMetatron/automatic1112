"""Batch configuration save/load/manage dialog."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QPushButton, QMessageBox
)
from PySide6.QtCore import Qt, Signal, Slot

from core.batch_adapter import save_batch_config, load_batch_config, get_saved_configs, delete_batch_config


class BatchConfigDialog(QDialog):
    """Dialog for managing saved batch configurations."""

    config_loaded = Signal(dict)  # config_dict

    def __init__(self, current_config: dict = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Configurations")
        self.setMinimumSize(500, 400)
        self._current_config = current_config or {}

        self._setup_ui()
        self._connect_signals()
        self._refresh_list()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Save section
        save_row = QHBoxLayout()
        save_row.addWidget(QLabel("Name:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter config name...")
        save_row.addWidget(self.name_input, stretch=1)
        self.save_btn = QPushButton("Save Current")
        save_row.addWidget(self.save_btn)
        layout.addLayout(save_row)

        # List of saved configs
        layout.addWidget(QLabel("Saved Configurations:"))
        self.config_list = QListWidget()
        self.config_list.setAlternatingRowColors(True)
        layout.addWidget(self.config_list)

        # Actions
        btn_row = QHBoxLayout()
        self.load_btn = QPushButton("Load Selected")
        btn_row.addWidget(self.load_btn)
        self.delete_btn = QPushButton("Delete")
        btn_row.addWidget(self.delete_btn)
        btn_row.addStretch()
        self.close_btn = QPushButton("Close")
        btn_row.addWidget(self.close_btn)
        layout.addLayout(btn_row)

    def _connect_signals(self):
        self.save_btn.clicked.connect(self._on_save)
        self.load_btn.clicked.connect(self._on_load)
        self.delete_btn.clicked.connect(self._on_delete)
        self.close_btn.clicked.connect(self.close)
        self.config_list.itemDoubleClicked.connect(self._on_load)

    def _refresh_list(self):
        self.config_list.clear()
        for name in get_saved_configs():
            self.config_list.addItem(name)

    @Slot()
    def _on_save(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Save", "Enter a config name")
            return
        save_batch_config(name, self._current_config)
        self._refresh_list()
        QMessageBox.information(self, "Saved", f"Configuration '{name}' saved")

    @Slot()
    def _on_load(self):
        item = self.config_list.currentItem()
        if not item:
            return
        name = item.text()
        config = load_batch_config(name)
        if config:
            self.config_loaded.emit(config)
            self.close()
        else:
            QMessageBox.warning(self, "Load", f"Could not load '{name}'")

    @Slot()
    def _on_delete(self):
        item = self.config_list.currentItem()
        if not item:
            return
        name = item.text()
        reply = QMessageBox.question(
            self, "Delete",
            f"Delete configuration '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            delete_batch_config(name)
            self._refresh_list()
