"""Full wildcard manager dialog: CRUD, AI generation, import."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QTextEdit, QPushButton,
    QGroupBox, QSpinBox, QComboBox, QSplitter, QMessageBox,
    QFileDialog, QWidget
)
from PySide6.QtCore import Qt, Slot


class WildcardDialog(QDialog):
    """Full wildcard manager with CRUD, AI generation, and import capabilities."""

    def __init__(self, desktop_state, parent=None):
        super().__init__(parent)
        self.state = desktop_state
        self.setWindowTitle("Wildcard Manager")
        self.setMinimumSize(800, 600)
        self.resize(1000, 700)

        self._setup_ui()
        self._connect_signals()
        self._load_categories()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Category list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Search
        search_row = QHBoxLayout()
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search categories...")
        self.search_box.setClearButtonEnabled(True)
        search_row.addWidget(self.search_box)
        left_layout.addLayout(search_row)

        # Category list
        self.category_list = QListWidget()
        self.category_list.setAlternatingRowColors(True)
        left_layout.addWidget(self.category_list)

        # Category buttons
        cat_btn_row = QHBoxLayout()
        self.add_category_btn = QPushButton("New Category")
        cat_btn_row.addWidget(self.add_category_btn)
        self.delete_category_btn = QPushButton("Delete")
        cat_btn_row.addWidget(self.delete_category_btn)
        self.rename_category_btn = QPushButton("Rename")
        cat_btn_row.addWidget(self.rename_category_btn)
        left_layout.addLayout(cat_btn_row)

        # Stats
        self.stats_label = QLabel("")
        left_layout.addWidget(self.stats_label)

        splitter.addWidget(left_widget)

        # Right: Items editor
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.category_name_label = QLabel("Select a category")
        self.category_name_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        right_layout.addWidget(self.category_name_label)

        # Items text area (one per line)
        right_layout.addWidget(QLabel("Items (one per line):"))
        self.items_editor = QTextEdit()
        self.items_editor.setPlaceholderText("Enter items, one per line...")
        right_layout.addWidget(self.items_editor)

        # Item actions
        item_btn_row = QHBoxLayout()
        self.save_items_btn = QPushButton("Save Items")
        item_btn_row.addWidget(self.save_items_btn)
        self.sort_items_btn = QPushButton("Sort A-Z")
        item_btn_row.addWidget(self.sort_items_btn)
        self.dedup_items_btn = QPushButton("Remove Duplicates")
        item_btn_row.addWidget(self.dedup_items_btn)
        right_layout.addLayout(item_btn_row)

        # AI Generation group
        ai_group = QGroupBox("AI Generate Items")
        ai_layout = QVBoxLayout()

        ai_row1 = QHBoxLayout()
        ai_row1.addWidget(QLabel("Prompt:"))
        self.ai_prompt = QLineEdit()
        self.ai_prompt.setPlaceholderText("e.g., types of mythical creatures")
        ai_row1.addWidget(self.ai_prompt, stretch=1)
        ai_layout.addLayout(ai_row1)

        ai_row2 = QHBoxLayout()
        ai_row2.addWidget(QLabel("Count:"))
        self.ai_count = QSpinBox()
        self.ai_count.setRange(5, 200)
        self.ai_count.setValue(20)
        ai_row2.addWidget(self.ai_count)
        ai_row2.addWidget(QLabel("Style:"))
        self.ai_style = QComboBox()
        self.ai_style.addItems(["simple", "descriptive", "poetic", "technical"])
        ai_row2.addWidget(self.ai_style)
        self.ai_generate_btn = QPushButton("Generate with Ollama")
        ai_row2.addWidget(self.ai_generate_btn)
        ai_layout.addLayout(ai_row2)

        ai_group.setLayout(ai_layout)
        right_layout.addWidget(ai_group)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

        # Bottom buttons
        bottom_row = QHBoxLayout()
        self.import_btn = QPushButton("Import from Files...")
        bottom_row.addWidget(self.import_btn)
        self.export_btn = QPushButton("Export All")
        bottom_row.addWidget(self.export_btn)
        bottom_row.addStretch()
        self.reload_btn = QPushButton("Reload from Disk")
        bottom_row.addWidget(self.reload_btn)
        self.close_btn = QPushButton("Close")
        bottom_row.addWidget(self.close_btn)
        layout.addLayout(bottom_row)

    def _connect_signals(self):
        self.search_box.textChanged.connect(self._filter_categories)
        self.category_list.currentItemChanged.connect(self._on_category_selected)
        self.add_category_btn.clicked.connect(self._on_add_category)
        self.delete_category_btn.clicked.connect(self._on_delete_category)
        self.rename_category_btn.clicked.connect(self._on_rename_category)
        self.save_items_btn.clicked.connect(self._on_save_items)
        self.sort_items_btn.clicked.connect(self._on_sort_items)
        self.dedup_items_btn.clicked.connect(self._on_dedup_items)
        self.ai_generate_btn.clicked.connect(self._on_ai_generate)
        self.import_btn.clicked.connect(self._on_import)
        self.export_btn.clicked.connect(self._on_export)
        self.reload_btn.clicked.connect(self._on_reload)
        self.close_btn.clicked.connect(self.close)

    def _get_wm(self):
        """Get the wildcard manager."""
        return self.state.get_wildcard_manager()

    def _load_categories(self):
        """Load all categories into the list."""
        self.category_list.clear()
        wm = self._get_wm()
        if not wm:
            self.stats_label.setText("No wildcard manager available")
            return

        keys = sorted(wm.get_available_wildcards()) if hasattr(wm, 'get_available_wildcards') else []
        if not keys and hasattr(wm, 'data'):
            keys = sorted(wm.data.keys())

        for key in keys:
            count = len(wm.data.get(key, [])) if hasattr(wm, 'data') else 0
            item = QListWidgetItem(f"{key} ({count})")
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.category_list.addItem(item)

        self.stats_label.setText(f"{len(keys)} categories")

    @Slot(str)
    def _filter_categories(self, text):
        search = text.lower()
        for i in range(self.category_list.count()):
            item = self.category_list.item(i)
            key = item.data(Qt.ItemDataRole.UserRole)
            item.setHidden(search not in key.lower())

    @Slot()
    def _on_category_selected(self):
        item = self.category_list.currentItem()
        if not item:
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        self.category_name_label.setText(f"Category: {key}")

        wm = self._get_wm()
        if wm and hasattr(wm, 'data') and key in wm.data:
            items = wm.data[key]
            self.items_editor.setPlainText("\n".join(items))
        else:
            self.items_editor.clear()

    @Slot()
    def _on_add_category(self):
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "New Category", "Category name:")
        if ok and name.strip():
            wm = self._get_wm()
            if wm and hasattr(wm, 'data'):
                name = name.strip().lower().replace(" ", "-")
                if name not in wm.data:
                    wm.data[name] = []
                    wm.save()
                    self._load_categories()
                else:
                    QMessageBox.warning(self, "Exists", f"Category '{name}' already exists")

    @Slot()
    def _on_delete_category(self):
        item = self.category_list.currentItem()
        if not item:
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        reply = QMessageBox.question(
            self, "Delete Category",
            f"Delete category '{key}' and all its items?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            wm = self._get_wm()
            if wm and hasattr(wm, 'data') and key in wm.data:
                del wm.data[key]
                wm.save()
                self._load_categories()
                self.items_editor.clear()

    @Slot()
    def _on_rename_category(self):
        item = self.category_list.currentItem()
        if not item:
            return
        old_key = item.data(Qt.ItemDataRole.UserRole)

        from PySide6.QtWidgets import QInputDialog
        new_name, ok = QInputDialog.getText(
            self, "Rename Category", "New name:", text=old_key
        )
        if ok and new_name.strip() and new_name.strip() != old_key:
            wm = self._get_wm()
            if wm and hasattr(wm, 'data'):
                new_name = new_name.strip().lower().replace(" ", "-")
                wm.data[new_name] = wm.data.pop(old_key, [])
                wm.save()
                self._load_categories()

    @Slot()
    def _on_save_items(self):
        item = self.category_list.currentItem()
        if not item:
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        text = self.items_editor.toPlainText()
        items = [line.strip() for line in text.split("\n") if line.strip()]

        wm = self._get_wm()
        if wm and hasattr(wm, 'data'):
            wm.data[key] = items
            wm.save()
            self._load_categories()
            # Reselect the item
            for i in range(self.category_list.count()):
                if self.category_list.item(i).data(Qt.ItemDataRole.UserRole) == key:
                    self.category_list.setCurrentRow(i)
                    break

    @Slot()
    def _on_sort_items(self):
        text = self.items_editor.toPlainText()
        items = sorted(
            [line.strip() for line in text.split("\n") if line.strip()],
            key=str.lower
        )
        self.items_editor.setPlainText("\n".join(items))

    @Slot()
    def _on_dedup_items(self):
        text = self.items_editor.toPlainText()
        items = [line.strip() for line in text.split("\n") if line.strip()]
        seen = set()
        unique = []
        for item in items:
            lower = item.lower()
            if lower not in seen:
                seen.add(lower)
                unique.append(item)
        self.items_editor.setPlainText("\n".join(unique))

    @Slot()
    def _on_ai_generate(self):
        """Generate wildcard items using Ollama."""
        prompt = self.ai_prompt.text().strip()
        if not prompt:
            QMessageBox.information(self, "AI Generate", "Enter a prompt first")
            return

        if not self.state.state.ollama_available or not self.state.state.ollama_enhancer:
            QMessageBox.warning(self, "AI Generate", "Ollama is not available")
            return

        count = self.ai_count.value()
        style = self.ai_style.currentText()

        self.ai_generate_btn.setEnabled(False)
        self.ai_generate_btn.setText("Generating...")

        try:
            full_prompt = (
                f"Generate exactly {count} {style} items for the category: {prompt}. "
                f"Return ONLY the items, one per line, no numbering, no explanations. "
                f"Each item should be a concise descriptive phrase suitable for "
                f"image generation prompts."
            )

            result = self.state.state.ollama_enhancer.enhance(
                full_prompt,
                length="detailed",
                complexity="moderate",
            )

            # Parse result into individual items
            new_items = [
                line.strip().lstrip("0123456789.-) ")
                for line in result.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]

            if new_items:
                current_text = self.items_editor.toPlainText().strip()
                if current_text:
                    current_text += "\n"
                self.items_editor.setPlainText(current_text + "\n".join(new_items))

        except Exception as e:
            QMessageBox.warning(self, "AI Generate Error", str(e))
        finally:
            self.ai_generate_btn.setEnabled(True)
            self.ai_generate_btn.setText("Generate with Ollama")

    @Slot()
    def _on_import(self):
        """Import wildcards from text files."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Import Wildcard Files", "",
            "Text Files (*.txt);;All Files (*)"
        )
        if not files:
            return

        wm = self._get_wm()
        if not wm or not hasattr(wm, 'data'):
            return

        imported = 0
        for filepath in files:
            from pathlib import Path
            path = Path(filepath)
            category = path.stem.lower().replace(" ", "-")

            try:
                with open(filepath) as f:
                    items = [line.strip() for line in f if line.strip()]
                if items:
                    if category in wm.data:
                        wm.data[category].extend(items)
                        # Deduplicate
                        wm.data[category] = list(dict.fromkeys(wm.data[category]))
                    else:
                        wm.data[category] = items
                    imported += 1
            except Exception:
                pass

        if imported > 0:
            wm.save()
            self._load_categories()
            QMessageBox.information(
                self, "Import", f"Imported {imported} category file(s)"
            )

    @Slot()
    def _on_export(self):
        """Export all wildcards to a directory."""
        directory = QFileDialog.getExistingDirectory(self, "Export Directory")
        if not directory:
            return

        wm = self._get_wm()
        if not wm or not hasattr(wm, 'data'):
            return

        from pathlib import Path
        export_dir = Path(directory)
        exported = 0

        for key, items in wm.data.items():
            filepath = export_dir / f"{key}.txt"
            with open(filepath, "w") as f:
                f.write("\n".join(items))
            exported += 1

        QMessageBox.information(
            self, "Export", f"Exported {exported} categories to {directory}"
        )

    @Slot()
    def _on_reload(self):
        """Reload wildcards from disk."""
        self.state._init_wildcards()
        self._load_categories()
