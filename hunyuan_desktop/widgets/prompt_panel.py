"""Prompt input panel: prompt editor, negative prompt, style, and enhancement controls."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QCheckBox, QPushButton, QGroupBox, QTextEdit
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
        enhance_group = QGroupBox("Prompt Enhancement (Ollama)")
        enhance_layout = QVBoxLayout()

        # Enable checkbox + model selector
        row1 = QHBoxLayout()
        self.use_ollama = QCheckBox("Enhance with Ollama")
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
        self.enhance_btn.setToolTip("Enhance the prompt with Ollama (Ctrl+E)")
        row3.addWidget(self.enhance_btn)
        self.undo_enhance_btn = QPushButton("Undo")
        self.undo_enhance_btn.setToolTip("Restore original prompt before enhancement")
        self.undo_enhance_btn.setEnabled(False)
        row3.addWidget(self.undo_enhance_btn)
        self.enhance_status = QLabel("")
        row3.addWidget(self.enhance_status, stretch=1)
        enhance_layout.addLayout(row3)

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
        self.preview_enhanced_btn.setToolTip("Resolve wildcards AND run Ollama enhancement")
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
        """Refresh the list of available Ollama models."""
        self.ollama_model.clear()
        try:
            if self.state.state.ollama_available and self.state.state.ollama_manager:
                models = self.state.state.ollama_manager.list_models()
                for m in models:
                    self.ollama_model.addItem(m.get("name", str(m)))
                if not models:
                    self.ollama_model.addItem("qwen2.5:7b-instruct")
            else:
                self.ollama_model.addItem("qwen2.5:7b-instruct")
        except Exception:
            self.ollama_model.addItem("qwen2.5:7b-instruct")

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
        self.preview_display.setText("--- Wildcards Resolved ---\n" + resolved + "\n\n--- Enhancing with Ollama... ---")
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
        lines.append("--- Enhanced (Ollama) ---")
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
