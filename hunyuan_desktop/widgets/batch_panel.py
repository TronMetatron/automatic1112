"""Batch processing panel: configuration, execution, and gallery."""

import time

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox,
    QSpinBox, QCheckBox, QPushButton, QGroupBox, QSplitter,
    QListWidget, QListWidgetItem, QDoubleSpinBox, QSlider
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer

from widgets.prompt_editor import PromptEditor
from widgets.gallery_panel import GalleryPanel
from widgets.progress_widget import ProgressWidget
from models.batch_config import BatchConfig
from models.i2i_batch_config import BOT_TASK_OPTIONS
from core.batch_worker import BatchWorker


class BatchPanel(QWidget):
    """Full batch processing interface - the primary use case."""

    def __init__(self, desktop_state, parent=None):
        super().__init__(parent)
        self.state = desktop_state
        self._batch_worker = None
        self._start_time = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_elapsed)
        self._setup_ui()
        self._connect_signals()
        self._populate_styles()
        self._refresh_config_list()  # Load saved configs on startup

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # LEFT: Configuration
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)

        # Batch name
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Batch Name:"))
        self.batch_name = QLineEdit()
        self.batch_name.setPlaceholderText("my_batch")
        name_row.addWidget(self.batch_name, stretch=1)
        config_layout.addLayout(name_row)

        # Themes editor
        config_layout.addWidget(QLabel("Themes (one per line):"))
        self.themes_editor = PromptEditor(
            wildcard_manager=self.state.get_wildcard_manager()
        )
        self.themes_editor.setMinimumHeight(150)
        self.themes_editor.setPlaceholderText(
            "A majestic dragon soaring over [landscape]\n"
            "A [*pose] warrior in [setting]\n"
            "Portrait of a [fantasy-race] sorcerer"
        )
        config_layout.addWidget(self.themes_editor)

        # Negative prompt
        neg_row = QHBoxLayout()
        neg_row.addWidget(QLabel("Negative:"))
        self.negative_prompt = QLineEdit()
        self.negative_prompt.setPlaceholderText("no watermark, no text...")
        neg_row.addWidget(self.negative_prompt, stretch=1)
        config_layout.addLayout(neg_row)

        # Iteration settings
        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("Variations:"))
        self.variations_spin = QSpinBox()
        self.variations_spin.setRange(1, 1000)
        self.variations_spin.setValue(3)
        iter_row.addWidget(self.variations_spin)

        iter_row.addWidget(QLabel("Imgs/combo:"))
        self.images_per_spin = QSpinBox()
        self.images_per_spin.setRange(1, 50)
        self.images_per_spin.setValue(1)
        iter_row.addWidget(self.images_per_spin)

        iter_row.addWidget(QLabel("Starred rerolls:"))
        self.starred_reroll_spin = QSpinBox()
        self.starred_reroll_spin.setRange(1, 50)
        self.starred_reroll_spin.setValue(1)
        self.starred_reroll_spin.setToolTip("Number of variations for [*starred] wildcards")
        iter_row.addWidget(self.starred_reroll_spin)
        config_layout.addLayout(iter_row)

        # Styles (checkbox list)
        config_layout.addWidget(QLabel("Styles:"))
        self.style_list = QListWidget()
        self.style_list.setMaximumHeight(120)
        config_layout.addWidget(self.style_list)

        # Generation settings
        gen_row = QHBoxLayout()
        gen_row.addWidget(QLabel("Aspect:"))
        self.aspect_combo = QComboBox()
        gen_row.addWidget(self.aspect_combo)
        gen_row.addWidget(QLabel("Quality:"))
        self.quality_combo = QComboBox()
        gen_row.addWidget(self.quality_combo)
        config_layout.addLayout(gen_row)

        # Guidance scale
        guidance_row = QHBoxLayout()
        guidance_row.addWidget(QLabel("Guidance:"))
        self.guidance_slider = QSlider(Qt.Orientation.Horizontal)
        self.guidance_slider.setRange(10, 150)
        self.guidance_slider.setValue(50)
        guidance_row.addWidget(self.guidance_slider)
        self.guidance_label = QLabel("5.0")
        guidance_row.addWidget(self.guidance_label)
        config_layout.addLayout(guidance_row)
        self.guidance_slider.valueChanged.connect(
            lambda v: self.guidance_label.setText(f"{v/10:.1f}")
        )

        # NOTE: Think mode controls moved to global system bar
        # Keeping think_group for CoT display visibility toggle only
        self.think_group = QGroupBox("Think Mode (Instruct/Distil)")
        self.think_group.setVisible(False)  # Hidden - mode now in system bar
        config_layout.addWidget(self.think_group)

        # Enhancement group
        enhance_group = QGroupBox("Prompt Enhancement")
        enhance_layout = QVBoxLayout()

        self.enhance_check = QCheckBox("Enhance with LM Studio")
        enhance_layout.addWidget(self.enhance_check)

        enh_row1 = QHBoxLayout()
        enh_row1.addWidget(QLabel("Model:"))
        self.ollama_model = QComboBox()
        self.ollama_model.setMinimumWidth(150)
        enh_row1.addWidget(self.ollama_model, stretch=1)
        enhance_layout.addLayout(enh_row1)

        enh_row2 = QHBoxLayout()
        enh_row2.addWidget(QLabel("Length:"))
        self.ollama_length = QComboBox()
        enh_row2.addWidget(self.ollama_length)
        enh_row2.addWidget(QLabel("Complexity:"))
        self.ollama_complexity = QComboBox()
        enh_row2.addWidget(self.ollama_complexity)
        enhance_layout.addLayout(enh_row2)

        enh_row3 = QHBoxLayout()
        enh_row3.addWidget(QLabel("Max length:"))
        self.max_prompt_length = QSpinBox()
        self.max_prompt_length.setRange(0, 5000)
        self.max_prompt_length.setValue(0)
        self.max_prompt_length.setToolTip("0 = no limit")
        enh_row3.addWidget(self.max_prompt_length)
        self.random_seeds_check = QCheckBox("Random seeds")
        self.random_seeds_check.setChecked(True)
        enh_row3.addWidget(self.random_seeds_check)
        enhance_layout.addLayout(enh_row3)

        enhance_group.setLayout(enhance_layout)
        config_layout.addWidget(enhance_group)

        # Config management
        config_mgmt_row = QHBoxLayout()
        self.config_name = QLineEdit()
        self.config_name.setPlaceholderText("Config name...")
        config_mgmt_row.addWidget(self.config_name)
        self.save_config_btn = QPushButton("Save")
        config_mgmt_row.addWidget(self.save_config_btn)
        self.load_config_combo = QComboBox()
        self.load_config_combo.setMinimumWidth(120)
        config_mgmt_row.addWidget(self.load_config_combo)
        self.load_config_btn = QPushButton("Load")
        config_mgmt_row.addWidget(self.load_config_btn)

        self.insert_config_btn = QPushButton("Insert")
        self.insert_config_btn.setToolTip("Insert themes from a saved config (appends to existing themes)")
        config_mgmt_row.addWidget(self.insert_config_btn)
        config_layout.addLayout(config_mgmt_row)

        # Preview and action buttons
        self.preview_label = QLabel("Enter themes to see preview...")
        self.preview_label.setStyleSheet("color: #a0a0a0; padding: 4px;")
        config_layout.addWidget(self.preview_label)

        # Prompt preview section
        preview_btn_row = QHBoxLayout()
        self.preview_wildcards_btn = QPushButton("Preview Wildcards")
        self.preview_wildcards_btn.setToolTip("Resolve wildcards in themes and show results")
        self.preview_wildcards_btn.clicked.connect(self._on_preview_wildcards)
        preview_btn_row.addWidget(self.preview_wildcards_btn)

        self.preview_enhanced_btn = QPushButton("Preview Enhanced")
        self.preview_enhanced_btn.setToolTip("Resolve wildcards AND run LM Studio enhancement")
        self.preview_enhanced_btn.clicked.connect(self._on_preview_enhanced)
        preview_btn_row.addWidget(self.preview_enhanced_btn)

        self.preview_prompt_status = QLabel("")
        preview_btn_row.addWidget(self.preview_prompt_status, stretch=1)
        config_layout.addLayout(preview_btn_row)

        from PySide6.QtWidgets import QTextEdit as _QTextEdit
        self.preview_display = _QTextEdit()
        self.preview_display.setReadOnly(True)
        self.preview_display.setMaximumHeight(120)
        self.preview_display.setStyleSheet(
            "font-size: 11px; color: #c0c0c0; background-color: #1e1e1e; "
            "border: 1px solid #404040; padding: 4px;"
        )
        self.preview_display.setPlaceholderText("Click a preview button to see resolved prompts...")
        config_layout.addWidget(self.preview_display)

        btn_row = QHBoxLayout()
        self.calculate_btn = QPushButton("Calculate")
        self.calculate_btn.clicked.connect(self._on_calculate)
        btn_row.addWidget(self.calculate_btn)

        self.start_btn = QPushButton("START BATCH")
        self.start_btn.setObjectName("start_batch_btn")
        self.start_btn.setMinimumHeight(36)
        btn_row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.stop_btn)

        self.update_btn = QPushButton("UPDATE")
        self.update_btn.setToolTip("Apply prompt changes to the running batch (takes effect on next iteration)")
        self.update_btn.setEnabled(False)
        btn_row.addWidget(self.update_btn)
        config_layout.addLayout(btn_row)

        self.gen_prompts_btn = QPushButton("Generate Prompts Only")
        config_layout.addWidget(self.gen_prompts_btn)

        # RIGHT: Progress and gallery
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Progress
        self.progress = ProgressWidget()
        right_layout.addWidget(self.progress)

        # Status
        status_row = QHBoxLayout()
        self.status_label = QLabel("Ready")
        status_row.addWidget(self.status_label, stretch=1)
        self.elapsed_label = QLabel("")
        status_row.addWidget(self.elapsed_label)
        self.eta_label = QLabel("")
        status_row.addWidget(self.eta_label)
        right_layout.addLayout(status_row)

        # Chain-of-thought display (collapsible)
        self.cot_group = QGroupBox("Chain of Thought")
        self.cot_group.setCheckable(True)
        self.cot_group.setChecked(False)
        cot_layout = QVBoxLayout()
        from PySide6.QtWidgets import QTextEdit
        self.cot_display = QTextEdit()
        self.cot_display.setReadOnly(True)
        self.cot_display.setMaximumHeight(100)
        self.cot_display.setStyleSheet("font-size: 11px; color: #b0b0b0;")
        cot_layout.addWidget(self.cot_display)
        self.cot_group.setLayout(cot_layout)
        self.cot_group.setVisible(False)
        right_layout.addWidget(self.cot_group)

        # Batch gallery - default to batches output directory
        self.batch_gallery = GalleryPanel(self.state)
        right_layout.addWidget(self.batch_gallery)

        # Auto-load the batches directory so gallery shows recent batch output
        try:
            from ui.constants import OUTPUT_DIR
            batches_dir = OUTPUT_DIR / "batches"
            if batches_dir.is_dir():
                # Find the most recent batch subdirectory
                subdirs = sorted(
                    [d for d in batches_dir.iterdir() if d.is_dir()],
                    key=lambda d: d.stat().st_mtime,
                    reverse=True
                )
                if subdirs:
                    self.batch_gallery.load_directory(str(subdirs[0]))
                else:
                    self.batch_gallery.set_directory_label(str(batches_dir))
        except Exception:
            pass

        # Assemble splitter
        splitter.addWidget(config_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter)

    def _connect_signals(self):
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.update_btn.clicked.connect(self._on_update)
        self.enhance_check.toggled.connect(self._toggle_enhance)
        self.save_config_btn.clicked.connect(self._on_save_config)
        self.load_config_btn.clicked.connect(self._on_load_config)
        self.insert_config_btn.clicked.connect(self._on_insert_config)
        # Gallery insert prompt -> append to themes
        self.batch_gallery.insert_prompt_requested.connect(self._on_insert_prompt)
        # Model state -> show/hide think mode
        self.state.model_loaded.connect(self._on_model_loaded)
        self.state.model_unloaded.connect(self._on_model_unloaded)

    def _populate_styles(self):
        """Populate style checkboxes and dropdowns."""
        presets = self.state.get_style_presets()
        self.style_list.clear()
        for name in presets:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if name == "None" else Qt.CheckState.Unchecked)
            self.style_list.addItem(item)

        # Aspect ratios and quality
        from ui.constants import ASPECT_RATIOS, QUALITY_PRESETS
        self.aspect_combo.clear()
        for name in ASPECT_RATIOS:
            self.aspect_combo.addItem(name)
        self.aspect_combo.setCurrentText("1:1 Square (1024)")

        self.quality_combo.clear()
        for name in QUALITY_PRESETS:
            self.quality_combo.addItem(name)
        self.quality_combo.setCurrentText("Standard")

        # Ollama
        from ui.constants import OLLAMA_LENGTH_OPTIONS, OLLAMA_COMPLEXITY_OPTIONS
        self.ollama_length.clear()
        self.ollama_length.addItems(OLLAMA_LENGTH_OPTIONS)
        self.ollama_length.setCurrentText("medium")
        self.ollama_complexity.clear()
        self.ollama_complexity.addItems(OLLAMA_COMPLEXITY_OPTIONS)
        self.ollama_complexity.setCurrentText("detailed")

        # LM Studio models
        self.ollama_model.clear()
        try:
            from lmstudio_client import LMStudioClient
            from ui.constants import get_lmstudio_url
            client = LMStudioClient(base_url=get_lmstudio_url())
            if client.is_available():
                models = client.list_models()
                for m in models:
                    self.ollama_model.addItem(m)
        except Exception:
            pass
        if self.ollama_model.count() == 0:
            self.ollama_model.addItem("lmstudio")

        self._toggle_enhance(False)

    @Slot(bool)
    def _toggle_enhance(self, enabled):
        self.ollama_model.setEnabled(enabled)
        self.ollama_length.setEnabled(enabled)
        self.ollama_complexity.setEnabled(enabled)
        self.max_prompt_length.setEnabled(enabled)

    def _build_config(self) -> BatchConfig:
        """Build a BatchConfig from current widget values."""
        themes_text = self.themes_editor.toPlainText().strip()
        themes = [t.strip() for t in themes_text.split("\n") if t.strip()]

        selected_styles = []
        for i in range(self.style_list.count()):
            item = self.style_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_styles.append(item.text())
        if not selected_styles:
            selected_styles = ["None"]

        # Get bot_task and drop_think from global state
        from ui.state import get_state
        state = get_state()
        bot_task = state.global_bot_task
        drop_think = state.global_drop_think

        return BatchConfig(
            batch_name=self.batch_name.text().strip() or "batch",
            themes=themes,
            negative_prompt=self.negative_prompt.text().strip(),
            variations_per_theme=self.variations_spin.value(),
            images_per_combo=self.images_per_spin.value(),
            starred_reroll_count=self.starred_reroll_spin.value(),
            styles=selected_styles,
            aspect_ratio=self.aspect_combo.currentText(),
            quality=self.quality_combo.currentText(),
            guidance_scale=self.guidance_slider.value() / 10.0,
            random_seeds=self.random_seeds_check.isChecked(),
            enhance=self.enhance_check.isChecked(),
            ollama_model=self.ollama_model.currentText(),
            ollama_length=self.ollama_length.currentText(),
            ollama_complexity=self.ollama_complexity.currentText(),
            max_prompt_length=self.max_prompt_length.value(),
            bot_task=bot_task,
            drop_think=drop_think,
        )

    @Slot()
    def _on_calculate(self):
        """Preview total image count."""
        config = self._build_config()
        self.preview_label.setText(config.preview_text())

    @Slot()
    def _on_start(self):
        """Start batch generation."""
        config = self._build_config()
        if not config.themes:
            self.status_label.setText("Enter at least one theme")
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.update_btn.setEnabled(True)
        self.status_label.setText("Starting batch...")
        self.batch_gallery.clear()

        total = config.total_images()
        self.progress.start(total)

        self._start_time = time.time()
        self._timer.start(1000)

        self._batch_worker = BatchWorker(config)
        self._batch_worker.progress.connect(self._on_progress)
        self._batch_worker.image_ready.connect(self._on_image_ready)
        self._batch_worker.cot_received.connect(self._on_cot_received)
        self._batch_worker.completed.connect(self._on_completed)
        self._batch_worker.stopped.connect(self._on_stopped)
        self._batch_worker.error.connect(self._on_error)
        self._batch_worker.config_updated.connect(
            lambda: self.status_label.setText("Config update applied!")
        )
        self._batch_worker.start()

        self.state.batch_started.emit(config.batch_name)

    @Slot()
    def _on_update(self):
        """Push prompt changes to the running batch worker."""
        if not self._batch_worker or not self._batch_worker.isRunning():
            self.status_label.setText("No batch running to update")
            return

        themes_text = self.themes_editor.toPlainText().strip()
        themes = [t.strip() for t in themes_text.split("\n") if t.strip()]
        if not themes:
            self.status_label.setText("Enter at least one theme to update")
            return

        negative = self.negative_prompt.text().strip()
        self._batch_worker.update_config(themes, negative)
        self.status_label.setText("Update queued - will apply on next iteration")

    @Slot()
    def _on_stop(self):
        """Stop the current batch."""
        if self._batch_worker:
            self._batch_worker.request_stop()
            self.status_label.setText("Stopping after current image...")
            self.stop_btn.setEnabled(False)

    @Slot(int, int, str)
    def _on_progress(self, current, total, status):
        self.progress.update_progress(current, total, status)
        self.status_label.setText(status)

    @Slot(str)
    def _on_image_ready(self, image_path):
        self.batch_gallery.add_thumbnail(image_path)
        self.state.batch_image_ready.emit(image_path)

    @Slot(str, int)
    def _on_completed(self, batch_dir, total_count):
        self._finish_batch(f"Complete! {total_count} images in {batch_dir}")
        self.batch_gallery.load_directory(batch_dir)
        self.state.batch_completed.emit(batch_dir, total_count)

    @Slot(str, int)
    def _on_stopped(self, batch_dir, count):
        self._finish_batch(f"Stopped. {count} images generated in {batch_dir}")
        self.state.batch_stopped.emit(batch_dir, count)

    @Slot(str)
    def _on_error(self, error_msg):
        self._finish_batch(f"Error: {error_msg}")

    def _finish_batch(self, message):
        self._timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.update_btn.setEnabled(False)
        self.status_label.setText(message)
        self.progress.complete(message)

    def _update_elapsed(self):
        if self._start_time:
            elapsed = time.time() - self._start_time
            mins, secs = divmod(int(elapsed), 60)
            hours, mins = divmod(mins, 60)
            if hours:
                self.elapsed_label.setText(f"Elapsed: {hours}:{mins:02d}:{secs:02d}")
            else:
                self.elapsed_label.setText(f"Elapsed: {mins}:{secs:02d}")

    @Slot(str)
    def _on_cot_received(self, text):
        """Display chain-of-thought reasoning."""
        self.cot_display.setText(text)
        self.cot_group.setChecked(True)

    @Slot(str, str)
    def _on_model_loaded(self, model_type, msg):
        """Show/hide think mode based on model type and set appropriate quality."""
        is_instruct = model_type in ("instruct", "distil", "nf4", "distil_nf4", "instruct_int8", "distil_int8")
        self.think_group.setVisible(is_instruct)
        self.cot_group.setVisible(is_instruct)

        # Auto-select the appropriate quality preset for this model
        from ui.constants import MODEL_DEFAULT_QUALITY
        default_quality = MODEL_DEFAULT_QUALITY.get(model_type, "Standard")
        idx = self.quality_combo.findText(default_quality, Qt.MatchFlag.MatchContains)
        if idx >= 0:
            self.quality_combo.setCurrentIndex(idx)

    @Slot()
    def _on_model_unloaded(self):
        self.think_group.setVisible(False)
        self.cot_group.setVisible(False)

    # -- Preview Methods --

    def _resolve_wildcards(self, prompt, generation_index=0):
        """Resolve wildcards in a prompt, handling starred wildcards for preview."""
        import re
        wm = self.state.get_wildcard_manager()
        if not wm:
            return prompt

        # For preview: temporarily convert [*key] to [key] so they get resolved
        starred_pattern = r'\[\*([^\]]+)\]'
        preview_prompt = re.sub(starred_pattern, r'[\1]', prompt)

        try:
            return wm.process_prompt(preview_prompt, generation_index=generation_index)
        except Exception as e:
            print(f"[PREVIEW] Wildcard resolution failed for theme {generation_index+1}: {e}")
            return preview_prompt

    @Slot()
    def _on_preview_wildcards(self):
        """Preview themes with wildcards resolved."""
        themes_text = self.themes_editor.toPlainText().strip()
        if not themes_text:
            self.preview_display.setText("Enter themes first.")
            return

        themes = [t.strip() for t in themes_text.split("\n") if t.strip()]

        from ui.constants import DEFAULT_STYLE_PRESETS
        # Get first selected style for preview
        selected_styles = []
        for i in range(self.style_list.count()):
            item = self.style_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_styles.append(item.text())
        style = selected_styles[0] if selected_styles else "None"
        style_suffix = DEFAULT_STYLE_PRESETS.get(style, "")

        lines = []
        for idx, theme in enumerate(themes):
            full = theme + style_suffix
            resolved = self._resolve_wildcards(full, generation_index=idx)

            lines.append(f"--- Theme {idx+1} ---")
            if resolved != full:
                lines.append(f"Original: {theme}")
                lines.append(f"Resolved: {resolved}")
            else:
                lines.append(resolved)
            lines.append("")

        self.preview_display.setText("\n".join(lines))
        self.preview_prompt_status.setText(f"Resolved {len(themes)} themes")

    @Slot()
    def _on_preview_enhanced(self):
        """Preview first theme with wildcards AND Ollama enhancement."""
        themes_text = self.themes_editor.toPlainText().strip()
        if not themes_text:
            self.preview_display.setText("Enter themes first.")
            return

        themes = [t.strip() for t in themes_text.split("\n") if t.strip()]

        from ui.constants import DEFAULT_STYLE_PRESETS
        selected_styles = []
        for i in range(self.style_list.count()):
            item = self.style_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_styles.append(item.text())
        style = selected_styles[0] if selected_styles else "None"
        style_suffix = DEFAULT_STYLE_PRESETS.get(style, "")

        # Resolve wildcards for all themes
        resolved_themes = []
        for idx, theme in enumerate(themes):
            full = theme + style_suffix
            resolved = self._resolve_wildcards(full, generation_index=idx)
            resolved_themes.append((theme, resolved))

        # Show wildcard results immediately and start enhancing all themes
        lines = []
        for idx, (orig, resolved) in enumerate(resolved_themes):
            lines.append(f"--- Theme {idx+1} ---")
            lines.append(f"Resolved: {resolved}")
            lines.append("")
        lines.append(f"--- Enhancing {len(resolved_themes)} theme(s) with LM Studio... ---")
        self.preview_display.setText("\n".join(lines))
        self.preview_prompt_status.setText(f"Enhancing 1/{len(resolved_themes)}...")
        self.preview_enhanced_btn.setEnabled(False)

        # Store for sequential enhancement
        self._preview_resolved_themes = resolved_themes
        self._preview_enhanced_results = []  # list of (original, enhanced_or_None)
        self._preview_enhance_index = 0
        self._preview_model = self.ollama_model.currentText()
        self._preview_length = self.ollama_length.currentText()
        self._preview_complexity = self.ollama_complexity.currentText()

        # Start enhancing the first theme
        self._enhance_next_preview_theme()

    def _enhance_next_preview_theme(self):
        """Enhance the next theme in the queue."""
        idx = self._preview_enhance_index
        themes = self._preview_resolved_themes

        if idx >= len(themes):
            # All done — show results
            self._show_all_preview_results()
            return

        self.preview_prompt_status.setText(
            f"Enhancing {idx+1}/{len(themes)}..."
        )

        from core.ollama_worker import OllamaEnhanceWorker
        self._preview_worker = OllamaEnhanceWorker(
            themes[idx][1], self._preview_model,
            self._preview_length, self._preview_complexity
        )
        self._preview_worker.completed.connect(self._on_batch_preview_enhance_done)
        self._preview_worker.failed.connect(self._on_batch_preview_enhance_failed)
        self._preview_worker.start()

    @Slot(str, str)
    def _on_batch_preview_enhance_done(self, original, enhanced):
        """Handle one theme's enhancement completion, then start the next."""
        self._preview_enhanced_results.append((original, enhanced))
        self._preview_enhance_index += 1
        self._enhance_next_preview_theme()

    @Slot(str)
    def _on_batch_preview_enhance_failed(self, error):
        """Handle one theme's enhancement failure, continue with the rest."""
        idx = self._preview_enhance_index
        orig = self._preview_resolved_themes[idx][1]
        self._preview_enhanced_results.append((orig, f"[FAILED: {error}]"))
        self._preview_enhance_index += 1
        self._enhance_next_preview_theme()

    def _show_all_preview_results(self):
        """Display all enhanced themes in the preview."""
        lines = []
        for idx, ((orig_theme, resolved), (_, enhanced)) in enumerate(
            zip(self._preview_resolved_themes, self._preview_enhanced_results)
        ):
            lines.append(f"=== Theme {idx+1} ===")
            if orig_theme != resolved:
                lines.append(f"Original:  {orig_theme}")
                lines.append(f"Wildcards: {resolved}")
            else:
                lines.append(f"Input: {resolved}")
            lines.append(f"Enhanced:  {enhanced}")
            lines.append("")

        self.preview_display.setText("\n".join(lines))
        self.preview_enhanced_btn.setEnabled(True)
        total = len(self._preview_enhanced_results)
        failed = sum(1 for _, e in self._preview_enhanced_results if e.startswith("[FAILED"))
        if failed:
            self.preview_prompt_status.setText(f"Done: {total - failed}/{total} enhanced, {failed} failed")
        else:
            self.preview_prompt_status.setText(f"All {total} themes enhanced")

    @Slot()
    def _on_save_config(self):
        """Save current batch configuration."""
        config = self._build_config()
        name = self.config_name.text().strip()
        if not name:
            self.status_label.setText("Enter a config name first")
            return
        try:
            from core.batch_adapter import save_batch_config
            save_batch_config(name, config.to_dict())
            self.status_label.setText(f"Saved config: {name}")
            self._refresh_config_list()
        except Exception as e:
            self.status_label.setText(f"Save error: {e}")

    @Slot()
    def _on_load_config(self):
        """Load a saved batch configuration."""
        name = self.load_config_combo.currentText()
        if not name:
            return
        try:
            from core.batch_adapter import load_batch_config
            data = load_batch_config(name)
            if data:
                config = BatchConfig.from_dict(data)
                self._apply_config(config)
                self.status_label.setText(f"Loaded: {name}")
        except Exception as e:
            self.status_label.setText(f"Load error: {e}")

    @Slot(str)
    def _on_insert_prompt(self, prompt_text):
        """Insert a prompt from a gallery image into the themes editor."""
        existing = self.themes_editor.toPlainText().strip()
        if existing:
            self.themes_editor.setPlainText(existing + "\n" + prompt_text)
        else:
            self.themes_editor.setPlainText(prompt_text)
        self.status_label.setText("Prompt inserted into themes")

    @Slot()
    def _on_insert_config(self):
        """Insert themes from a saved config, appending to existing themes."""
        name = self.load_config_combo.currentText()
        if not name:
            self.status_label.setText("Select a config to insert from")
            return
        try:
            from core.batch_adapter import load_batch_config
            data = load_batch_config(name)
            if data:
                config = BatchConfig.from_dict(data)
                if config.themes:
                    existing = self.themes_editor.toPlainText().strip()
                    new_themes = "\n".join(config.themes)
                    if existing:
                        self.themes_editor.setPlainText(existing + "\n" + new_themes)
                    else:
                        self.themes_editor.setPlainText(new_themes)
                    self.status_label.setText(
                        f"Inserted {len(config.themes)} themes from '{name}'"
                    )
                else:
                    self.status_label.setText(f"Config '{name}' has no themes")
        except Exception as e:
            self.status_label.setText(f"Insert error: {e}")

    def _apply_config(self, config: BatchConfig):
        """Apply a BatchConfig to all widgets."""
        self.batch_name.setText(config.batch_name)
        self.themes_editor.setPlainText("\n".join(config.themes))
        self.negative_prompt.setText(config.negative_prompt)
        self.variations_spin.setValue(config.variations_per_theme)
        self.images_per_spin.setValue(config.images_per_combo)
        self.starred_reroll_spin.setValue(config.starred_reroll_count)
        self.aspect_combo.setCurrentText(config.aspect_ratio)
        self.quality_combo.setCurrentText(config.quality)
        self.guidance_slider.setValue(int(config.guidance_scale * 10))
        self.random_seeds_check.setChecked(config.random_seeds)
        self.enhance_check.setChecked(config.enhance)
        self.ollama_model.setCurrentText(config.ollama_model)
        self.ollama_length.setCurrentText(config.ollama_length)
        self.ollama_complexity.setCurrentText(config.ollama_complexity)
        self.max_prompt_length.setValue(config.max_prompt_length)

        # Update styles
        for i in range(self.style_list.count()):
            item = self.style_list.item(i)
            item.setCheckState(
                Qt.CheckState.Checked if item.text() in config.styles
                else Qt.CheckState.Unchecked
            )

        # Note: bot_task and drop_think are now global settings in system bar
        # When loading a config, we update the global state instead of local controls
        from ui.state import get_state
        state = get_state()
        state.global_bot_task = config.bot_task
        state.global_drop_think = config.drop_think

    def _refresh_config_list(self):
        """Refresh the list of saved configs."""
        try:
            from core.batch_adapter import get_saved_configs
            configs = get_saved_configs()
            self.load_config_combo.clear()
            self.load_config_combo.addItems(configs)
        except Exception:
            pass

    def get_state_dict(self) -> dict:
        """Get current batch settings as a dict for project save."""
        config = self._build_config()
        return config.to_dict()

    def set_state_dict(self, data: dict):
        """Restore batch settings from a dict (project load)."""
        from models.batch_config import BatchConfig
        config = BatchConfig.from_dict(data)
        self._apply_config(config)
