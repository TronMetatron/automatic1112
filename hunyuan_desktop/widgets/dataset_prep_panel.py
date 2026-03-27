"""Dataset preparation panel for generating diverse training datasets.

Provides UI for batch I2I generation through multiple passes
(background simplification, clothing variation, pose changes, etc.)
to create well-rounded character LoRA training datasets.
"""

import json
import time
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox,
    QSpinBox, QPushButton, QGroupBox, QSplitter,
    QSlider, QFileDialog, QTextEdit, QCheckBox, QScrollArea,
    QFrame,
)
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QPixmap

from widgets.gallery_panel import GalleryPanel
from widgets.progress_widget import ProgressWidget
from models.dataset_prep_config import (
    DatasetPrepConfig, DATASET_PREP_PASSES, DATASET_PREP_PASS_ORDER,
)
from core.dataset_prep_worker import DatasetPrepWorker, IMAGE_EXTENSIONS


class DatasetPrepPanel(QWidget):
    """Dataset preparation interface for character LoRA training."""

    def __init__(self, desktop_state, parent=None):
        super().__init__(parent)
        self.state = desktop_state
        self._batch_worker = None
        self._start_time = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_elapsed)
        self._source_images = []
        self._pass_widgets = {}  # {pass_name: (checkbox, text_edit)}
        self._setup_ui()
        self._connect_signals()
        self._refresh_config_list()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # LEFT: Configuration (scrollable)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)

        # ── Input folder ──
        input_group = QGroupBox("Source Images")
        input_layout = QVBoxLayout()

        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("Input Folder:"))
        self.input_folder = QLineEdit()
        self.input_folder.setPlaceholderText("/path/to/character/images")
        input_row.addWidget(self.input_folder, stretch=1)
        self.browse_input_btn = QPushButton("Browse...")
        self.browse_input_btn.setMaximumWidth(80)
        input_row.addWidget(self.browse_input_btn)
        input_layout.addLayout(input_row)

        # Thumbnail preview area
        self.thumb_frame = QFrame()
        self.thumb_frame.setStyleSheet(
            "QFrame { background: #1a1a2e; border: 1px solid #333; "
            "border-radius: 4px; }"
        )
        self.thumb_frame.setMinimumHeight(90)
        self.thumb_frame.setMaximumHeight(110)
        self.thumb_layout = QHBoxLayout(self.thumb_frame)
        self.thumb_layout.setContentsMargins(4, 4, 4, 4)
        self.thumb_layout.setSpacing(4)
        self.source_count_label = QLabel("No images loaded")
        self.source_count_label.setStyleSheet("color: #808080; font-size: 11px;")
        self.thumb_layout.addWidget(self.source_count_label)
        self.thumb_layout.addStretch()
        input_layout.addWidget(self.thumb_frame)

        input_group.setLayout(input_layout)
        config_layout.addWidget(input_group)

        # ── Output folder ──
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output Folder:"))
        self.output_folder = QLineEdit()
        self.output_folder.setPlaceholderText("/path/to/output/dataset")
        output_row.addWidget(self.output_folder, stretch=1)
        self.browse_output_btn = QPushButton("Browse...")
        self.browse_output_btn.setMaximumWidth(80)
        output_row.addWidget(self.browse_output_btn)
        config_layout.addLayout(output_row)

        # ── Pass presets ──
        passes_group = QGroupBox("Generation Passes")
        passes_layout = QVBoxLayout()

        info_label = QLabel(
            "Each enabled pass generates variations of every source image. "
            "Edit prompts to customize. [options] are randomly selected."
        )
        info_label.setStyleSheet("color: #808080; font-size: 11px;")
        info_label.setWordWrap(True)
        passes_layout.addWidget(info_label)

        for pass_name in DATASET_PREP_PASS_ORDER:
            preset = DATASET_PREP_PASSES[pass_name]

            # Pass header: checkbox + description
            header_row = QHBoxLayout()
            checkbox = QCheckBox(pass_name)
            checkbox.setChecked(False)
            checkbox.setStyleSheet("font-weight: bold;")
            header_row.addWidget(checkbox)
            desc_label = QLabel(f"- {preset['description']}")
            desc_label.setStyleSheet("color: #909090; font-size: 11px;")
            header_row.addWidget(desc_label, stretch=1)
            passes_layout.addLayout(header_row)

            # Editable prompt
            text_edit = QTextEdit()
            text_edit.setPlainText(preset["prompt"])
            text_edit.setMaximumHeight(60)
            text_edit.setStyleSheet("font-size: 11px;")
            text_edit.setEnabled(False)
            passes_layout.addWidget(text_edit)

            # Wire checkbox to enable/disable text edit
            checkbox.toggled.connect(text_edit.setEnabled)

            self._pass_widgets[pass_name] = (checkbox, text_edit)

        passes_group.setLayout(passes_layout)
        config_layout.addWidget(passes_group)

        # ── Generation settings ──
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout()

        ipp_row = QHBoxLayout()
        ipp_row.addWidget(QLabel("Images per pass:"))
        self.images_per_pass = QSpinBox()
        self.images_per_pass.setRange(1, 50)
        self.images_per_pass.setValue(1)
        self.images_per_pass.setToolTip(
            "How many variations to generate per source image per pass"
        )
        ipp_row.addWidget(self.images_per_pass)
        ipp_row.addStretch()
        settings_layout.addLayout(ipp_row)

        guidance_row = QHBoxLayout()
        guidance_row.addWidget(QLabel("Guidance:"))
        self.guidance_slider = QSlider(Qt.Orientation.Horizontal)
        self.guidance_slider.setRange(10, 150)
        self.guidance_slider.setValue(50)
        guidance_row.addWidget(self.guidance_slider)
        self.guidance_label = QLabel("5.0")
        guidance_row.addWidget(self.guidance_label)
        settings_layout.addLayout(guidance_row)
        self.guidance_slider.valueChanged.connect(
            lambda v: self.guidance_label.setText(f"{v/10:.1f}")
        )

        self.random_seeds_check = QCheckBox("Random seeds")
        self.random_seeds_check.setChecked(True)
        settings_layout.addWidget(self.random_seeds_check)

        settings_group.setLayout(settings_layout)
        config_layout.addWidget(settings_group)

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
        self.preview_label = QLabel(
            "Select input folder and enable passes to see preview..."
        )
        self.preview_label.setStyleSheet("color: #a0a0a0; padding: 4px;")
        self.preview_label.setWordWrap(True)
        config_layout.addWidget(self.preview_label)

        btn_row = QHBoxLayout()
        self.calculate_btn = QPushButton("Calculate")
        self.calculate_btn.clicked.connect(self._on_calculate)
        btn_row.addWidget(self.calculate_btn)

        self.start_btn = QPushButton("START DATASET PREP")
        self.start_btn.setObjectName("start_batch_btn")
        self.start_btn.setMinimumHeight(36)
        btn_row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.stop_btn)
        config_layout.addLayout(btn_row)

        config_layout.addStretch()

        scroll.setWidget(config_widget)

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
        self.cot_group.setChecked(True)
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
        splitter.addWidget(scroll)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter)

    def _connect_signals(self):
        self.browse_input_btn.clicked.connect(self._on_browse_input)
        self.browse_output_btn.clicked.connect(self._on_browse_output)
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.save_config_btn.clicked.connect(self._on_save_config)
        self.load_config_btn.clicked.connect(self._on_load_config)
        self.input_folder.textChanged.connect(self._on_input_changed)

    # ── Folder browsing ──

    @Slot()
    def _on_browse_input(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Source Image Folder",
            self.input_folder.text() or ""
        )
        if folder:
            self.input_folder.setText(folder)

    @Slot()
    def _on_browse_output(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder",
            self.output_folder.text() or ""
        )
        if folder:
            self.output_folder.setText(folder)

    @Slot(str)
    def _on_input_changed(self, path):
        """Scan input folder and show thumbnails."""
        self._source_images = []
        # Clear existing thumbnails
        while self.thumb_layout.count() > 0:
            item = self.thumb_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        input_path = Path(path)
        if not input_path.is_dir():
            self.source_count_label = QLabel("Folder not found")
            self.source_count_label.setStyleSheet(
                "color: #808080; font-size: 11px;"
            )
            self.thumb_layout.addWidget(self.source_count_label)
            self.thumb_layout.addStretch()
            return

        # Scan for images
        for f in sorted(input_path.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                self._source_images.append(str(f))

        if not self._source_images:
            label = QLabel("No images found")
            label.setStyleSheet("color: #808080; font-size: 11px;")
            self.thumb_layout.addWidget(label)
            self.thumb_layout.addStretch()
            return

        # Show count
        count_label = QLabel(f"{len(self._source_images)} images")
        count_label.setStyleSheet("color: #a0d0ff; font-size: 11px;")
        self.thumb_layout.addWidget(count_label)

        # Show up to 8 thumbnails
        for img_path in self._source_images[:8]:
            thumb_label = QLabel()
            pixmap = QPixmap(img_path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    70, 70,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                thumb_label.setPixmap(scaled)
                thumb_label.setToolTip(Path(img_path).name)
            self.thumb_layout.addWidget(thumb_label)

        if len(self._source_images) > 8:
            more_label = QLabel(f"+{len(self._source_images) - 8}")
            more_label.setStyleSheet("color: #808080; font-size: 11px;")
            self.thumb_layout.addWidget(more_label)

        self.thumb_layout.addStretch()

    # ── Config building ──

    def _get_enabled_passes(self) -> dict:
        """Get dict of {pass_name: prompt_text} for enabled passes."""
        passes = {}
        for pass_name in DATASET_PREP_PASS_ORDER:
            checkbox, text_edit = self._pass_widgets[pass_name]
            if checkbox.isChecked():
                passes[pass_name] = text_edit.toPlainText().strip()
        return passes

    def _build_config(self) -> DatasetPrepConfig:
        """Build DatasetPrepConfig from current UI state."""
        from ui.state import get_state
        state = get_state()

        return DatasetPrepConfig(
            input_folder=self.input_folder.text().strip(),
            output_folder=self.output_folder.text().strip(),
            enabled_passes=self._get_enabled_passes(),
            images_per_pass=self.images_per_pass.value(),
            guidance_scale=self.guidance_slider.value() / 10.0,
            random_seeds=self.random_seeds_check.isChecked(),
            bot_task=state.global_bot_task,
            drop_think=state.global_drop_think,
        )

    # ── Actions ──

    @Slot()
    def _on_calculate(self):
        """Update the preview label."""
        config = self._build_config()
        source_count = len(self._source_images)
        self.preview_label.setText(config.preview_text(source_count))

    @Slot()
    def _on_start(self):
        """Start dataset preparation generation."""
        if self._batch_worker and self._batch_worker.isRunning():
            return

        config = self._build_config()

        if not config.input_folder or not Path(config.input_folder).is_dir():
            self.status_label.setText("Select a valid input folder")
            return

        if not config.output_folder:
            self.status_label.setText("Select an output folder")
            return

        if not config.enabled_passes:
            self.status_label.setText("Enable at least one pass")
            return

        if not self._source_images:
            self.status_label.setText("No source images found in input folder")
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.batch_gallery.clear()
        self.cot_display.clear()

        total = config.total_images_for_sources(len(self._source_images))
        self.progress.start(total)
        self.status_label.setText("Starting dataset prep...")
        self._start_time = time.time()
        self._timer.start(1000)

        self._batch_worker = DatasetPrepWorker(config)
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
    def _on_completed(self, output_dir, count):
        self._finish(f"Dataset prep complete! {count} images in {output_dir}")

    @Slot(str, int)
    def _on_stopped(self, output_dir, count):
        self._finish(f"Dataset prep stopped. {count} images in {output_dir}")

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
        config_dir = OUTPUT_DIR / "configs" / "dataset_prep"
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
        filepath = OUTPUT_DIR / "configs" / "dataset_prep" / f"{name}.json"
        if not filepath.exists():
            self.status_label.setText(f"Config not found: {name}")
            return

        with open(filepath) as f:
            data = json.load(f)

        config = DatasetPrepConfig.from_dict(data)
        self._apply_config(config)
        self.status_label.setText(f"Config loaded: {name}")

    def _refresh_config_list(self):
        """Refresh the list of saved configs."""
        from ui.constants import OUTPUT_DIR
        config_dir = OUTPUT_DIR / "configs" / "dataset_prep"
        self.load_config_combo.clear()
        if config_dir.exists():
            for f in sorted(config_dir.glob("*.json")):
                self.load_config_combo.addItem(f.stem)

    def _apply_config(self, config: DatasetPrepConfig):
        """Apply a config to the UI."""
        self.input_folder.setText(config.input_folder)
        self.output_folder.setText(config.output_folder)
        self.images_per_pass.setValue(config.images_per_pass)
        self.guidance_slider.setValue(int(config.guidance_scale * 10))
        self.random_seeds_check.setChecked(config.random_seeds)

        # Update global state
        from ui.state import get_state
        state = get_state()
        state.global_bot_task = config.bot_task
        state.global_drop_think = config.drop_think

        # Restore pass states
        for pass_name in DATASET_PREP_PASS_ORDER:
            checkbox, text_edit = self._pass_widgets[pass_name]
            if pass_name in config.enabled_passes:
                checkbox.setChecked(True)
                text_edit.setPlainText(config.enabled_passes[pass_name])
            else:
                checkbox.setChecked(False)
                # Keep the default prompt text

        self._on_calculate()

    def get_state_dict(self) -> dict:
        """Get current settings as a dict for project save."""
        config = self._build_config()
        return config.to_dict()

    def set_state_dict(self, data: dict):
        """Restore settings from a dict (project load)."""
        config = DatasetPrepConfig.from_dict(data)
        self._apply_config(config)
