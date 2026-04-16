"""Prompt input panel: prompt editor, negative prompt, style, and enhancement controls."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QCheckBox, QPushButton, QGroupBox, QTextEdit,
    QDialog, QDialogButtonBox
)
from PySide6.QtCore import Signal, Slot

from widgets.prompt_editor import PromptEditor


class PromptPanel(QWidget):
    """Panel containing prompt input, style selection, and Ollama enhancement controls."""

    generate_requested = Signal()  # Emitted when user wants to generate
    enhance_requested = Signal(str, str, str, str)  # prompt, model, length, complexity

    def __init__(self, desktop_state, parent=None):
        super().__init__(parent)
        self.state = desktop_state
        self._original_prompt = ""  # For undo
        self._setup_ui()
        self._connect_signals()
        self._populate_dropdowns()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Prompt editor
        prompt_label = QLabel("Prompt:")
        layout.addWidget(prompt_label)

        self.prompt_editor = PromptEditor(
            wildcard_manager=self.state.get_wildcard_manager()
        )
        self.prompt_editor.setMinimumHeight(120)
        layout.addWidget(self.prompt_editor)

        # Negative prompt
        neg_layout = QHBoxLayout()
        neg_layout.addWidget(QLabel("Negative:"))
        self.negative_prompt = QLineEdit()
        self.negative_prompt.setPlaceholderText("no watermark, no text, blurry, deformed...")
        neg_layout.addWidget(self.negative_prompt)
        layout.addLayout(neg_layout)

        # Style selector
        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel("Style:"))
        self.style_combo = QComboBox()
        self.style_combo.setMinimumWidth(200)
        style_layout.addWidget(self.style_combo, stretch=1)
        layout.addLayout(style_layout)

        # Enhancement group
        enhance_group = QGroupBox("Prompt Enhancement (LM Studio)")
        enhance_layout = QVBoxLayout()

        # Enable checkbox + model selector
        row1 = QHBoxLayout()
        self.use_ollama = QCheckBox("Enhance with LM Studio")
        row1.addWidget(self.use_ollama)
        row1.addWidget(QLabel("Model:"))
        self.ollama_model = QComboBox()
        self.ollama_model.setMinimumWidth(180)
        row1.addWidget(self.ollama_model, stretch=1)
        enhance_layout.addLayout(row1)

        # Length + Complexity
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Length:"))
        self.ollama_length = QComboBox()
        row2.addWidget(self.ollama_length)
        row2.addWidget(QLabel("Complexity:"))
        self.ollama_complexity = QComboBox()
        row2.addWidget(self.ollama_complexity)
        enhance_layout.addLayout(row2)

        # Enhance / Undo buttons
        row3 = QHBoxLayout()
        self.enhance_btn = QPushButton("Enhance Now")
        self.enhance_btn.setToolTip("Enhance the prompt with LM Studio (Ctrl+E)")
        row3.addWidget(self.enhance_btn)
        self.undo_enhance_btn = QPushButton("Undo")
        self.undo_enhance_btn.setToolTip("Restore original prompt before enhancement")
        self.undo_enhance_btn.setEnabled(False)
        row3.addWidget(self.undo_enhance_btn)
        self.enhance_status = QLabel("")
        row3.addWidget(self.enhance_status, stretch=1)
        enhance_layout.addLayout(row3)

        # LM Studio Settings button — always visible, separate row
        settings_row = QHBoxLayout()
        self.edit_sysprompt_btn = QPushButton("LM Studio Settings (URL + System Prompt)")
        self.edit_sysprompt_btn.setToolTip(
            "Edit the LM Studio server URL and the system prompt used for enhancement"
        )
        self.edit_sysprompt_btn.setStyleSheet(
            "QPushButton { background-color: #3a3a5a; padding: 6px; }"
            "QPushButton:hover { background-color: #4a4a7a; }"
        )
        settings_row.addWidget(self.edit_sysprompt_btn)
        enhance_layout.addLayout(settings_row)

        enhance_group.setLayout(enhance_layout)
        layout.addWidget(enhance_group)

        # Preview group
        preview_group = QGroupBox("Prompt Preview")
        preview_layout = QVBoxLayout()

        preview_btn_row = QHBoxLayout()
        self.preview_wildcards_btn = QPushButton("Preview Wildcards")
        self.preview_wildcards_btn.setToolTip("Resolve wildcards and show the resulting prompt")
        preview_btn_row.addWidget(self.preview_wildcards_btn)

        self.preview_enhanced_btn = QPushButton("Preview Enhanced")
        self.preview_enhanced_btn.setToolTip("Resolve wildcards AND run LM Studio enhancement")
        preview_btn_row.addWidget(self.preview_enhanced_btn)

        self.preview_status = QLabel("")
        preview_btn_row.addWidget(self.preview_status, stretch=1)
        preview_layout.addLayout(preview_btn_row)

        self.preview_display = QTextEdit()
        self.preview_display.setReadOnly(True)
        self.preview_display.setMaximumHeight(120)
        self.preview_display.setStyleSheet(
            "font-size: 11px; color: #c0c0c0; background-color: #1e1e1e; "
            "border: 1px solid #404040; padding: 4px;"
        )
        self.preview_display.setPlaceholderText("Click a preview button to see the resolved prompt...")
        preview_layout.addWidget(self.preview_display)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # Set initial state
        self._toggle_ollama_controls(False)

    def _connect_signals(self):
        self.use_ollama.toggled.connect(self._toggle_ollama_controls)
        self.enhance_btn.clicked.connect(self._on_enhance)
        self.undo_enhance_btn.clicked.connect(self._on_undo_enhance)
        self.edit_sysprompt_btn.clicked.connect(self._on_edit_system_prompt)
        self.state.enhancement_completed.connect(self._on_enhancement_done)
        self.state.enhancement_failed.connect(self._on_enhancement_failed)
        self.preview_wildcards_btn.clicked.connect(self._on_preview_wildcards)
        self.preview_enhanced_btn.clicked.connect(self._on_preview_enhanced)

    def _populate_dropdowns(self):
        """Populate style and Ollama dropdowns."""
        # Styles
        presets = self.state.get_style_presets()
        self.style_combo.clear()
        for name in presets:
            self.style_combo.addItem(name)

        # Ollama models
        self._refresh_ollama_models()

        # Length options
        from ui.constants import OLLAMA_LENGTH_OPTIONS, OLLAMA_COMPLEXITY_OPTIONS
        self.ollama_length.clear()
        self.ollama_length.addItems(OLLAMA_LENGTH_OPTIONS)
        self.ollama_length.setCurrentText("medium")

        self.ollama_complexity.clear()
        self.ollama_complexity.addItems(OLLAMA_COMPLEXITY_OPTIONS)
        self.ollama_complexity.setCurrentText("detailed")

    def _refresh_ollama_models(self):
        """Refresh the list of available models from LM Studio or Ollama."""
        self.ollama_model.clear()
        try:
            from lmstudio_client import LMStudioClient
            from ui.constants import get_lmstudio_url
            client = LMStudioClient(base_url=get_lmstudio_url())
            if client.is_available():
                models = client.list_models()
                for m in models:
                    self.ollama_model.addItem(m)
            if self.ollama_model.count() == 0:
                self.ollama_model.addItem("lmstudio")
        except Exception:
            self.ollama_model.addItem("lmstudio")

    @Slot(bool)
    def _toggle_ollama_controls(self, enabled: bool):
        """Enable/disable Ollama controls based on checkbox."""
        self.ollama_model.setEnabled(enabled)
        self.ollama_length.setEnabled(enabled)
        self.ollama_complexity.setEnabled(enabled)
        self.enhance_btn.setEnabled(enabled)

    @Slot()
    def _on_enhance(self):
        """Enhance the current prompt with Ollama."""
        prompt = self.prompt_editor.toPlainText().strip()
        if not prompt:
            self.enhance_status.setText("Enter a prompt first")
            return

        self._original_prompt = prompt
        self.enhance_btn.setEnabled(False)
        self.enhance_status.setText("Enhancing...")

        model = self.ollama_model.currentText()
        length = self.ollama_length.currentText()
        complexity = self.ollama_complexity.currentText()
        self.enhance_requested.emit(prompt, model, length, complexity)

    @Slot()
    def _on_undo_enhance(self):
        """Restore the original prompt before enhancement."""
        if self._original_prompt:
            self.prompt_editor.setPlainText(self._original_prompt)
            self.undo_enhance_btn.setEnabled(False)
            self.enhance_status.setText("Restored original prompt")

    @Slot()
    def _on_edit_system_prompt(self):
        """Open a dialog to edit the LM Studio system prompt."""
        from core.settings import get_settings
        from ollama_prompts import get_enhance_system_prompt

        settings = get_settings()
        saved = settings.enhance_system_prompt

        # Generate the current default so the user can see/copy it
        length = self.ollama_length.currentText()
        if length == "random":
            length = "medium"
        complexity = self.ollama_complexity.currentText()
        if complexity == "random":
            complexity = "detailed"
        default_prompt = get_enhance_system_prompt(length, complexity)

        dialog = QDialog(self)
        dialog.setWindowTitle("LM Studio Settings")
        dialog.setMinimumSize(700, 550)
        dlg_layout = QVBoxLayout(dialog)

        # LM Studio URL
        url_row = QHBoxLayout()
        url_row.addWidget(QLabel("LM Studio URL:"))
        url_edit = QLineEdit()
        url_edit.setText(settings.lmstudio_url)
        url_edit.setPlaceholderText("http://localhost:1234 (leave empty for default)")
        url_row.addWidget(url_edit, stretch=1)

        discover_btn = QPushButton("Auto-Discover")
        discover_btn.setToolTip("Scan local network for LM Studio server")
        discover_status = QLabel("")

        def _do_discover():
            discover_btn.setEnabled(False)
            discover_status.setText("Scanning network...")
            import threading
            from PySide6.QtCore import QMetaObject, Qt as QtConst, Q_ARG

            def _scan():
                from lmstudio_client import discover_lmstudio, LMStudioClient
                try:
                    found_url = discover_lmstudio()
                except Exception:
                    found_url = None
                if found_url:
                    try:
                        client = LMStudioClient(base_url=found_url)
                        model = client.get_loaded_model()
                        model_info = f" -- model: {model}" if model else ""
                    except Exception:
                        model_info = ""
                    # Marshal back to main thread
                    QMetaObject.invokeMethod(
                        url_edit, "setText", QtConst.ConnectionType.QueuedConnection,
                        Q_ARG(str, found_url))
                    QMetaObject.invokeMethod(
                        discover_status, "setText", QtConst.ConnectionType.QueuedConnection,
                        Q_ARG(str, f"Found: {found_url}{model_info}"))
                else:
                    QMetaObject.invokeMethod(
                        discover_status, "setText", QtConst.ConnectionType.QueuedConnection,
                        Q_ARG(str, "No LM Studio found on network"))
                QMetaObject.invokeMethod(
                    discover_btn, "setEnabled", QtConst.ConnectionType.QueuedConnection,
                    Q_ARG(bool, True))
            threading.Thread(target=_scan, daemon=True).start()

        discover_btn.clicked.connect(_do_discover)
        url_row.addWidget(discover_btn)
        dlg_layout.addLayout(url_row)
        dlg_layout.addWidget(discover_status)

        # Length/Complexity preview — shows what the auto-generated prompt looks like
        preview_row = QHBoxLayout()
        preview_row.addWidget(QLabel("Preview default for:"))
        preview_row.addWidget(QLabel("Length:"))
        from ui.constants import OLLAMA_LENGTH_OPTIONS, OLLAMA_COMPLEXITY_OPTIONS
        dlg_length = QComboBox()
        dlg_length.addItems([x for x in OLLAMA_LENGTH_OPTIONS if x != "random"])
        dlg_length.setCurrentText(length)
        preview_row.addWidget(dlg_length)
        preview_row.addWidget(QLabel("Complexity:"))
        dlg_complexity = QComboBox()
        dlg_complexity.addItems([x for x in OLLAMA_COMPLEXITY_OPTIONS if x != "random"])
        dlg_complexity.setCurrentText(complexity)
        preview_row.addWidget(dlg_complexity)
        preview_row.addStretch()
        dlg_layout.addLayout(preview_row)

        # Show current default for reference
        default_label = QLabel("Auto-generated default (read-only — changes with length/complexity above):")
        dlg_layout.addWidget(default_label)
        default_display = QTextEdit()
        default_display.setPlainText(default_prompt)
        default_display.setReadOnly(True)
        default_display.setMaximumHeight(150)
        default_display.setStyleSheet(
            "color: #999; background-color: #1e1e1e; border: 1px solid #404040; padding: 4px;"
        )
        dlg_layout.addWidget(default_display)

        def _update_default_preview():
            p = get_enhance_system_prompt(dlg_length.currentText(), dlg_complexity.currentText())
            default_display.setPlainText(p)
        dlg_length.currentTextChanged.connect(lambda: _update_default_preview())
        dlg_complexity.currentTextChanged.connect(lambda: _update_default_preview())

        dlg_layout.addWidget(QLabel(
            "Your custom system prompt (overrides the default above — leave empty to use default):"
        ))

        dlg_layout.addWidget(QLabel("Your custom system prompt:"))
        editor = QTextEdit()
        editor.setPlainText(saved)
        editor.setPlaceholderText(
            "Enter your custom system prompt here...\n\n"
            "The user's prompt will be sent as the user message.\n"
            "This replaces the auto-generated system prompt entirely."
        )
        editor.setMinimumHeight(200)
        dlg_layout.addWidget(editor)

        # Buttons: Copy Default | Reset | Cancel | Save
        btn_row = QHBoxLayout()
        copy_default_btn = QPushButton("Copy Default to Editor")
        copy_default_btn.clicked.connect(lambda: editor.setPlainText(default_prompt))
        btn_row.addWidget(copy_default_btn)
        reset_btn = QPushButton("Clear (Use Default)")
        reset_btn.clicked.connect(lambda: editor.clear())
        btn_row.addWidget(reset_btn)
        btn_row.addStretch()

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        btn_row.addWidget(button_box)
        dlg_layout.addLayout(btn_row)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Save URL
            new_url = url_edit.text().strip()
            settings.lmstudio_url = new_url

            # Save system prompt
            new_prompt = editor.toPlainText().strip()
            settings.enhance_system_prompt = new_prompt

            parts = []
            if new_url:
                parts.append(f"URL: {new_url}")
            if new_prompt:
                parts.append("Custom system prompt saved")
            self.enhance_status.setText("; ".join(parts) if parts else "Using defaults")

            # Refresh model list with new URL
            self._refresh_ollama_models()

    @Slot(str, str)
    def _on_enhancement_done(self, original: str, enhanced: str):
        """Handle enhancement completion."""
        self.prompt_editor.setPlainText(enhanced)
        self.enhance_btn.setEnabled(True)
        self.undo_enhance_btn.setEnabled(True)
        self.enhance_status.setText("Enhanced!")

    @Slot(str)
    def _on_enhancement_failed(self, error: str):
        """Handle enhancement failure."""
        self.enhance_btn.setEnabled(True)
        self.enhance_status.setText(f"Error: {error[:50]}")

    # -- Preview --

    def _resolve_wildcards(self, prompt):
        """Resolve wildcards in a prompt, handling starred wildcards for preview."""
        import re
        wm = self.state.get_wildcard_manager()
        if not wm:
            return prompt

        # For preview: temporarily convert [*key] to [key] so they get resolved
        starred_pattern = r'\[\*([^\]]+)\]'
        preview_prompt = re.sub(starred_pattern, r'[\1]', prompt)

        try:
            return wm.process_prompt(preview_prompt, generation_index=0)
        except Exception as e:
            print(f"[PREVIEW] Wildcard resolution failed: {e}")
            return preview_prompt

    @Slot()
    def _on_preview_wildcards(self):
        """Preview the prompt with wildcards resolved."""
        prompt = self.prompt_editor.toPlainText().strip()
        if not prompt:
            self.preview_display.setText("Enter a prompt first.")
            return

        # Add style suffix
        from ui.constants import DEFAULT_STYLE_PRESETS
        style = self.style_combo.currentText()
        style_suffix = DEFAULT_STYLE_PRESETS.get(style, "")
        full_prompt = prompt + style_suffix

        # Resolve wildcards
        resolved = self._resolve_wildcards(full_prompt)

        lines = []
        lines.append("--- Wildcards Resolved ---")
        lines.append(resolved)
        if resolved != full_prompt:
            lines.append("")
            lines.append("--- Original ---")
            lines.append(full_prompt)
        self.preview_display.setText("\n".join(lines))
        self.preview_status.setText("Wildcards resolved")

    @Slot()
    def _on_preview_enhanced(self):
        """Preview the prompt with wildcards resolved AND Ollama enhancement."""
        prompt = self.prompt_editor.toPlainText().strip()
        if not prompt:
            self.preview_display.setText("Enter a prompt first.")
            return

        # Add style suffix
        from ui.constants import DEFAULT_STYLE_PRESETS
        style = self.style_combo.currentText()
        style_suffix = DEFAULT_STYLE_PRESETS.get(style, "")
        full_prompt = prompt + style_suffix

        # Resolve wildcards first
        resolved = self._resolve_wildcards(full_prompt)

        # Show wildcard-resolved version immediately
        self.preview_display.setText("--- Wildcards Resolved ---\n" + resolved + "\n\n--- Enhancing with LM Studio... ---")
        self.preview_status.setText("Enhancing...")
        self.preview_enhanced_btn.setEnabled(False)

        # Store for the callback
        self._preview_wildcard_result = resolved
        self._preview_original = full_prompt

        # Run Ollama enhancement in background
        model = self.ollama_model.currentText()
        length = self.ollama_length.currentText()
        complexity = self.ollama_complexity.currentText()

        from core.ollama_worker import OllamaEnhanceWorker
        self._preview_worker = OllamaEnhanceWorker(resolved, model, length, complexity)
        self._preview_worker.completed.connect(self._on_preview_enhance_done)
        self._preview_worker.failed.connect(self._on_preview_enhance_failed)
        self._preview_worker.start()

    @Slot(str, str)
    def _on_preview_enhance_done(self, original, enhanced):
        """Handle preview enhancement completion."""
        lines = []
        lines.append("--- Enhanced (LM Studio) ---")
        lines.append(enhanced)
        lines.append("")
        lines.append("--- Wildcards Resolved ---")
        lines.append(self._preview_wildcard_result)
        if self._preview_wildcard_result != self._preview_original:
            lines.append("")
            lines.append("--- Original ---")
            lines.append(self._preview_original)
        self.preview_display.setText("\n".join(lines))
        self.preview_enhanced_btn.setEnabled(True)
        self.preview_status.setText("Preview complete")

    @Slot(str)
    def _on_preview_enhance_failed(self, error):
        """Handle preview enhancement failure."""
        lines = []
        lines.append(f"--- Enhancement Failed: {error} ---")
        lines.append("")
        lines.append("--- Wildcards Resolved ---")
        lines.append(self._preview_wildcard_result)
        self.preview_display.setText("\n".join(lines))
        self.preview_enhanced_btn.setEnabled(True)
        self.preview_status.setText(f"Enhancement failed: {error[:40]}")

    # -- Public API --

    def get_prompt(self) -> str:
        return self.prompt_editor.toPlainText().strip()

    def set_prompt(self, text: str):
        self.prompt_editor.setPlainText(text)

    def get_negative_prompt(self) -> str:
        return self.negative_prompt.text().strip()

    def set_negative_prompt(self, text: str):
        self.negative_prompt.setText(text)

    def get_style(self) -> str:
        return self.style_combo.currentText()

    def set_style(self, name: str):
        idx = self.style_combo.findText(name)
        if idx >= 0:
            self.style_combo.setCurrentIndex(idx)

    def get_ollama_settings(self) -> dict:
        return {
            "use_ollama": self.use_ollama.isChecked(),
            "model": self.ollama_model.currentText(),
            "length": self.ollama_length.currentText(),
            "complexity": self.ollama_complexity.currentText(),
        }

    def set_ollama_settings(self, settings: dict):
        self.use_ollama.setChecked(settings.get("use_ollama", False))

        model = settings.get("model", "qwen2.5:7b-instruct")
        idx = self.ollama_model.findText(model)
        if idx >= 0:
            self.ollama_model.setCurrentIndex(idx)

        length = settings.get("length", "medium")
        idx = self.ollama_length.findText(length)
        if idx >= 0:
            self.ollama_length.setCurrentIndex(idx)

        complexity = settings.get("complexity", "detailed")
        idx = self.ollama_complexity.findText(complexity)
        if idx >= 0:
            self.ollama_complexity.setCurrentIndex(idx)

    def refresh_styles(self):
        """Refresh style dropdown from state."""
        current = self.style_combo.currentText()
        self._populate_dropdowns()
        idx = self.style_combo.findText(current)
        if idx >= 0:
            self.style_combo.setCurrentIndex(idx)
