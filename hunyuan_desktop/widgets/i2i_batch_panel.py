"""Image-to-Image batch processing panel.

Provides UI for batch I2I generation with:
- Global reference images (1-2 images applied to all prompts)
- Per-prompt image overrides via [img:/path] syntax
- Bot task / think mode selection
- Style, wildcard, and Ollama enhancement support
"""

import json
import time
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox,
    QSpinBox, QCheckBox, QPushButton, QGroupBox, QSplitter,
    QListWidget, QListWidgetItem, QSlider, QFileDialog, QTextEdit
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer

from widgets.prompt_editor import PromptEditor
from widgets.gallery_panel import GalleryPanel
from widgets.progress_widget import ProgressWidget
from widgets.gen_settings_panel import ImageDropZone
from models.i2i_batch_config import I2IBatchConfig, BOT_TASK_OPTIONS
from core.i2i_batch_worker import I2IBatchWorker


class I2IBatchPanel(QWidget):
    """Image-to-image batch processing interface."""

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
        self.batch_name.setPlaceholderText("i2i_batch")
        name_row.addWidget(self.batch_name, stretch=1)
        config_layout.addLayout(name_row)

        # ── Global Reference Images ──
        images_group = QGroupBox("Global Reference Images")
        images_layout = QVBoxLayout()

        # Image 1
        img1_row = QHBoxLayout()
        img1_row.addWidget(QLabel("Image 1:"))
        self.global_image1 = ImageDropZone()
        self.global_image1.setMinimumHeight(120)
        self.global_image1.setMaximumHeight(150)
        img1_row.addWidget(self.global_image1)
        self.clear_img1_btn = QPushButton("Clear")
        self.clear_img1_btn.setMaximumWidth(50)
        self.clear_img1_btn.clicked.connect(lambda: self.global_image1.clear_image())
        img1_row.addWidget(self.clear_img1_btn)
        images_layout.addLayout(img1_row)

        # Image 2
        img2_row = QHBoxLayout()
        img2_row.addWidget(QLabel("Image 2:"))
        self.global_image2 = ImageDropZone()
        self.global_image2.setMinimumHeight(120)
        self.global_image2.setMaximumHeight(150)
        img2_row.addWidget(self.global_image2)
        self.clear_img2_btn = QPushButton("Clear")
        self.clear_img2_btn.setMaximumWidth(50)
        self.clear_img2_btn.clicked.connect(lambda: self.global_image2.clear_image())
        img2_row.addWidget(self.clear_img2_btn)
        images_layout.addLayout(img2_row)

        info_label = QLabel(
            "These images apply to all prompts. "
            "Use [img:/path] in a prompt line to override per-prompt."
        )
        info_label.setStyleSheet("color: #808080; font-size: 11px;")
        info_label.setWordWrap(True)
        images_layout.addWidget(info_label)

        images_group.setLayout(images_layout)
        config_layout.addWidget(images_group)

        # ── Prompts editor ──
        config_layout.addWidget(QLabel("Prompts (one per line):"))
        self.prompts_editor = PromptEditor(
            wildcard_manager=self.state.get_wildcard_manager()
        )
        self.prompts_editor.setMinimumHeight(120)
        self.prompts_editor.setPlaceholderText(
            "Transform this into a cyberpunk scene\n"
            "Apply watercolor painting style\n"
            "[img:/path/to/other.png] Use this specific reference\n"
            "[img1:/path/a.png] [img2:/path/b.png] Blend these two"
        )
        config_layout.addWidget(self.prompts_editor)

        # NOTE: Bot Task / Think Mode controls moved to global system bar
        # Mode is now controlled globally across all tabs

        # ── Iteration settings ──
        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("Variations:"))
        self.variations_spin = QSpinBox()
        self.variations_spin.setRange(1, 1000)
        self.variations_spin.setValue(1)
        iter_row.addWidget(self.variations_spin)

        iter_row.addWidget(QLabel("Imgs/combo:"))
        self.images_per_spin = QSpinBox()
        self.images_per_spin.setRange(1, 50)
        self.images_per_spin.setValue(1)
        iter_row.addWidget(self.images_per_spin)
        config_layout.addLayout(iter_row)

        # ── Styles ──
        config_layout.addWidget(QLabel("Styles:"))
        self.style_list = QListWidget()
        self.style_list.setMaximumHeight(100)
        config_layout.addWidget(self.style_list)

        # ── Guidance ──
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

        # ── Enhancement group ──
        enhance_group = QGroupBox("Prompt Enhancement")
        enhance_layout = QVBoxLayout()

        self.enhance_check = QCheckBox("Enhance with Ollama")
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

        self.random_seeds_check = QCheckBox("Random seeds")
        self.random_seeds_check.setChecked(True)
        enhance_layout.addWidget(self.random_seeds_check)

        enhance_group.setLayout(enhance_layout)
        config_layout.addWidget(enhance_group)

        # ── Config management ──
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
        config_layout.addLayout(config_mgmt_row)

        # ── Preview and action buttons ──
        self.preview_label = QLabel("Add prompts and images to see preview...")
        self.preview_label.setStyleSheet("color: #a0a0a0; padding: 4px;")
        self.preview_label.setWordWrap(True)
        config_layout.addWidget(self.preview_label)

        btn_row = QHBoxLayout()
        self.calculate_btn = QPushButton("Calculate")
        self.calculate_btn.clicked.connect(self._on_calculate)
        btn_row.addWidget(self.calculate_btn)

        self.start_btn = QPushButton("START I2I BATCH")
        self.start_btn.setObjectName("start_batch_btn")
        self.start_btn.setMinimumHeight(36)
        btn_row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.stop_btn)
        config_layout.addLayout(btn_row)

        # ── RIGHT: Progress and gallery ──
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.progress = ProgressWidget()
        right_layout.addWidget(self.progress)

        status_row = QHBoxLayout()
        self.status_label = QLabel("Ready")
        status_row.addWidget(self.status_label, stretch=1)
        self.elapsed_label = QLabel("")
        status_row.addWidget(self.elapsed_label)
        self.eta_label = QLabel("")
        status_row.addWidget(self.eta_label)
        right_layout.addLayout(status_row)

        # CoT display (collapsible)
        self.cot_group = QGroupBox("Chain of Thought")
        self.cot_group.setCheckable(True)
        self.cot_group.setChecked(True)  # Expanded by default so CoT is visible
        cot_layout = QVBoxLayout()
        self.cot_display = QTextEdit()
        self.cot_display.setReadOnly(True)
        self.cot_display.setMaximumHeight(100)
        self.cot_display.setStyleSheet("font-size: 11px; color: #b0b0b0;")
        cot_layout.addWidget(self.cot_display)
        self.cot_group.setLayout(cot_layout)
        right_layout.addWidget(self.cot_group)

        # Batch gallery
        self.batch_gallery = GalleryPanel(self.state)
        right_layout.addWidget(self.batch_gallery)

        # Assemble
        splitter.addWidget(config_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter)

    def _connect_signals(self):
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.enhance_check.toggled.connect(self._toggle_enhance)
        self.save_config_btn.clicked.connect(self._on_save_config)
        self.load_config_btn.clicked.connect(self._on_load_config)

    def _populate_styles(self):
        """Populate style checkboxes and dropdowns."""
        presets = self.state.get_style_presets()
        self.style_list.clear()
        for name in presets:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if name == "None"
                else Qt.CheckState.Unchecked
            )
            self.style_list.addItem(item)

        # Ollama dropdowns
        from ui.constants import OLLAMA_LENGTH_OPTIONS, OLLAMA_COMPLEXITY_OPTIONS
        self.ollama_length.clear()
        for opt in OLLAMA_LENGTH_OPTIONS:
            self.ollama_length.addItem(opt)
        self.ollama_length.setCurrentText("medium")

        self.ollama_complexity.clear()
        for opt in OLLAMA_COMPLEXITY_OPTIONS:
            self.ollama_complexity.addItem(opt)
        self.ollama_complexity.setCurrentText("detailed")

        # Ollama models
        self._refresh_ollama_models()

    def _refresh_ollama_models(self):
        """Refresh Ollama model list."""
        self.ollama_model.clear()
        self.ollama_model.addItem("qwen2.5:7b-instruct")
        try:
            from ui.state import get_state
            state = get_state()
            if state.ollama_available and state.ollama_manager:
                models = state.ollama_manager.list_models()
                for m in models:
                    name = m.get("name", "")
                    if name and name != "qwen2.5:7b-instruct":
                        self.ollama_model.addItem(name)
        except Exception:
            pass

    def _toggle_enhance(self, checked):
        self.ollama_model.setEnabled(checked)
        self.ollama_length.setEnabled(checked)
        self.ollama_complexity.setEnabled(checked)

    def _get_selected_styles(self):
        """Get list of checked style names."""
        styles = []
        for i in range(self.style_list.count()):
            item = self.style_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                styles.append(item.text())
        return styles or ["None"]

    def _get_global_images(self):
        """Get list of global image paths."""
        images = []
        path1 = self.global_image1.get_image_path()
        if path1:
            images.append(path1)
        path2 = self.global_image2.get_image_path()
        if path2:
            images.append(path2)
        return images

    def _build_config(self) -> I2IBatchConfig:
        """Build I2IBatchConfig from current UI state."""
        raw_text = self.prompts_editor.toPlainText()
        global_images = self._get_global_images()

        prompts, overrides = I2IBatchConfig.parse_prompt_lines(
            raw_text, global_images
        )

        # Get bot_task and drop_think from global state
        from ui.state import get_state
        state = get_state()
        bot_task = state.global_bot_task
        drop_think = state.global_drop_think

        return I2IBatchConfig(
            batch_name=self.batch_name.text().strip() or "i2i_batch",
            prompts=prompts,
            global_images=global_images,
            prompt_image_overrides=overrides,
            bot_task=bot_task,
            drop_think=drop_think,
            variations_per_prompt=self.variations_spin.value(),
            images_per_combo=self.images_per_spin.value(),
            styles=self._get_selected_styles(),
            guidance_scale=self.guidance_slider.value() / 10.0,
            random_seeds=self.random_seeds_check.isChecked(),
            enhance=self.enhance_check.isChecked(),
            ollama_model=self.ollama_model.currentText(),
            ollama_length=self.ollama_length.currentText(),
            ollama_complexity=self.ollama_complexity.currentText(),
        )

    @Slot()
    def _on_calculate(self):
        """Update the preview label."""
        config = self._build_config()
        self.preview_label.setText(config.preview_text())

    @Slot()
    def _on_start(self):
        """Start I2I batch generation."""
        if self._batch_worker and self._batch_worker.isRunning():
            return

        config = self._build_config()
        print(f"[I2I BATCH UI] Starting batch with config:")
        print(f"  enhance={config.enhance}")
        print(f"  ollama_model={config.ollama_model}")
        print(f"  bot_task={config.bot_task}")
        print(f"  drop_think={config.drop_think}")

        if not config.prompts:
            self.status_label.setText("No prompts entered")
            return

        if not config.global_images and not config.prompt_image_overrides:
            self.status_label.setText("No reference images provided")
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.batch_gallery.clear()
        self.cot_display.clear()

        total = config.total_images()
        self.progress.start(total)
        self.status_label.setText("Starting I2I batch...")
        self._start_time = time.time()
        self._timer.start(1000)

        self._batch_worker = I2IBatchWorker(config)
        self._batch_worker.progress.connect(self._on_progress)
        self._batch_worker.image_ready.connect(self._on_image_ready)
        self._batch_worker.cot_received.connect(self._on_cot)
        self._batch_worker.completed.connect(self._on_completed)
        self._batch_worker.stopped.connect(self._on_stopped)
        self._batch_worker.error.connect(self._on_error)
        self._batch_worker.start()

    @Slot()
    def _on_stop(self):
        """Stop the running batch."""
        if self._batch_worker and self._batch_worker.isRunning():
            self._batch_worker.request_stop()
            self.status_label.setText("Stopping...")
            self.stop_btn.setEnabled(False)

    @Slot(int, int, str)
    def _on_progress(self, current, total, status):
        self.progress.update_progress(current, total, status)
        self.status_label.setText(status)

        if self._start_time and current > 0:
            elapsed = time.time() - self._start_time
            per_image = elapsed / current
            remaining = (total - current) * per_image
            mins, secs = divmod(int(remaining), 60)
            self.eta_label.setText(f"ETA: {mins}m {secs}s")

    @Slot(str)
    def _on_image_ready(self, path):
        self.batch_gallery.add_thumbnail(path)

    @Slot(str)
    def _on_cot(self, text):
        self.cot_display.setText(text[:500])

    @Slot(str, int)
    def _on_completed(self, batch_dir, count):
        self._finish(f"Batch complete! {count} images in {batch_dir}")

    @Slot(str, int)
    def _on_stopped(self, batch_dir, count):
        self._finish(f"Batch stopped. {count} images in {batch_dir}")

    @Slot(str)
    def _on_error(self, error_msg):
        self._finish(f"Error: {error_msg[:100]}")

    def _finish(self, message):
        self._timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.complete(message)
        self.status_label.setText(message)

    def _update_elapsed(self):
        if self._start_time:
            elapsed = int(time.time() - self._start_time)
            mins, secs = divmod(elapsed, 60)
            self.elapsed_label.setText(f"Elapsed: {mins}m {secs}s")

    # ── Config save/load ──

    @Slot()
    def _on_save_config(self):
        """Save current config to JSON file."""
        name = self.config_name.text().strip()
        if not name:
            self.status_label.setText("Enter a config name first")
            return

        from ui.constants import OUTPUT_DIR
        config_dir = OUTPUT_DIR / "configs" / "i2i_batch"
        config_dir.mkdir(parents=True, exist_ok=True)

        config = self._build_config()
        filepath = config_dir / f"{name}.json"
        with open(filepath, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        self.status_label.setText(f"Config saved: {name}")
        self._refresh_config_list()

    @Slot()
    def _on_load_config(self):
        """Load a saved config."""
        name = self.load_config_combo.currentText()
        if not name:
            return

        from ui.constants import OUTPUT_DIR
        filepath = OUTPUT_DIR / "configs" / "i2i_batch" / f"{name}.json"
        if not filepath.exists():
            self.status_label.setText(f"Config not found: {name}")
            return

        with open(filepath) as f:
            data = json.load(f)

        config = I2IBatchConfig.from_dict(data)
        self._apply_config(config)
        self.status_label.setText(f"Config loaded: {name}")

    def _refresh_config_list(self):
        """Refresh the list of saved configs."""
        from ui.constants import OUTPUT_DIR
        config_dir = OUTPUT_DIR / "configs" / "i2i_batch"
        self.load_config_combo.clear()
        if config_dir.exists():
            for f in sorted(config_dir.glob("*.json")):
                self.load_config_combo.addItem(f.stem)

    def _apply_config(self, config: I2IBatchConfig):
        """Apply a config to the UI."""
        self.batch_name.setText(config.batch_name)
        self.prompts_editor.setPlainText("\n".join(config.prompts))
        self.variations_spin.setValue(config.variations_per_prompt)
        self.images_per_spin.setValue(config.images_per_combo)
        self.guidance_slider.setValue(int(config.guidance_scale * 10))
        self.random_seeds_check.setChecked(config.random_seeds)
        self.enhance_check.setChecked(config.enhance)

        # Note: bot_task and drop_think are now global settings in system bar
        # When loading a config, we update the global state instead of local controls
        from ui.state import get_state
        state = get_state()
        state.global_bot_task = config.bot_task
        state.global_drop_think = config.drop_think

        # Styles
        for i in range(self.style_list.count()):
            item = self.style_list.item(i)
            item.setCheckState(
                Qt.CheckState.Checked if item.text() in config.styles
                else Qt.CheckState.Unchecked
            )

        # Restore global images
        from pathlib import Path
        if config.global_images:
            if len(config.global_images) > 0 and config.global_images[0]:
                if Path(config.global_images[0]).exists():
                    self.global_image1.set_image(config.global_images[0])
                else:
                    self.global_image1.clear_image()
            if len(config.global_images) > 1 and config.global_images[1]:
                if Path(config.global_images[1]).exists():
                    self.global_image2.set_image(config.global_images[1])
                else:
                    self.global_image2.clear_image()

        self._on_calculate()

    def get_state_dict(self) -> dict:
        """Get current I2I batch settings as a dict for project save."""
        config = self._build_config()
        return config.to_dict()

    def set_state_dict(self, data: dict):
        """Restore I2I batch settings from a dict (project load)."""
        from models.i2i_batch_config import I2IBatchConfig
        config = I2IBatchConfig.from_dict(data)
        self._apply_config(config)
