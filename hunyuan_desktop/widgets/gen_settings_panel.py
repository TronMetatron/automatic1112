"""Generation settings panel: aspect ratio, quality, seed, I2I controls."""

import random

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSpinBox, QPushButton, QGroupBox, QSlider, QFileDialog, QCheckBox
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QPixmap

from models.i2i_batch_config import BOT_TASK_OPTIONS


class ImageDropZone(QLabel):
    """Image drop zone that accepts drag-and-drop images."""

    image_dropped = Signal(str)  # file path

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.setMaximumHeight(150)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAcceptDrops(True)
        self.setStyleSheet(
            "border: 2px dashed #3c3c3c; border-radius: 6px; "
            "background-color: #1e1e1e; color: #646464;"
        )
        self.setText("Drop image here for I2I\nor click to browse")
        self._image_path = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Image", "",
                "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
            )
            if path:
                self.set_image(path)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                self.set_image(path)

    def set_image(self, path: str):
        """Display the dropped image as a thumbnail."""
        self._image_path = path
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                self.width() - 4, self.height() - 4,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled)
            self.setStyleSheet(
                "border: 2px solid #4a90d9; border-radius: 6px; background-color: #1e1e1e;"
            )
        self.image_dropped.emit(path)

    def clear_image(self):
        """Clear the current image."""
        self._image_path = None
        self.clear()
        self.setText("Drop image here for I2I\nor click to browse")
        self.setStyleSheet(
            "border: 2px dashed #3c3c3c; border-radius: 6px; "
            "background-color: #1e1e1e; color: #646464;"
        )

    def get_image_path(self) -> str:
        return self._image_path or ""


class GenSettingsPanel(QWidget):
    """Generation settings: aspect ratio, quality, seed, I2I controls."""

    def __init__(self, desktop_state, parent=None):
        super().__init__(parent)
        self.state = desktop_state
        self._setup_ui()
        self._connect_signals()
        self._populate_presets()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Aspect ratio
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Aspect:"))
        self.aspect_combo = QComboBox()
        self.aspect_combo.setMinimumWidth(180)
        row1.addWidget(self.aspect_combo, stretch=1)
        layout.addLayout(row1)

        # Quality preset
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Quality:"))
        self.quality_combo = QComboBox()
        self.quality_combo.setMinimumWidth(180)
        row2.addWidget(self.quality_combo, stretch=1)
        layout.addLayout(row2)

        # Seed
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Seed:"))
        self.seed_input = QSpinBox()
        self.seed_input.setRange(-1, 2**31 - 1)
        self.seed_input.setValue(-1)
        self.seed_input.setToolTip("-1 = random seed")
        self.seed_input.setMinimumWidth(120)
        row3.addWidget(self.seed_input)
        self.random_seed_btn = QPushButton("Random")
        self.random_seed_btn.setMaximumWidth(70)
        row3.addWidget(self.random_seed_btn)
        layout.addLayout(row3)

        # Batch count
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Count:"))
        self.batch_count = QSpinBox()
        self.batch_count.setRange(1, 100)
        self.batch_count.setValue(1)
        self.batch_count.setToolTip("Number of images to generate")
        row4.addWidget(self.batch_count)
        row4.addStretch()
        layout.addLayout(row4)

        # I2I group (collapsible)
        self.i2i_group = QGroupBox("Image-to-Image")
        self.i2i_group.setCheckable(True)
        self.i2i_group.setChecked(False)
        i2i_layout = QVBoxLayout()

        # Reference image slots (up to 3 for FireRed)
        self.image_drops = []
        for i in range(3):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"Img {i+1}:"))
            drop = ImageDropZone()
            drop.setMinimumHeight(80)
            drop.setMaximumHeight(120)
            row.addWidget(drop)
            clear_btn = QPushButton("X")
            clear_btn.setMaximumWidth(30)
            clear_btn.clicked.connect(drop.clear_image)
            row.addWidget(clear_btn)
            i2i_layout.addLayout(row)
            self.image_drops.append(drop)

        # Backward-compat alias
        self.image_drop = self.image_drops[0]

        # Guidance scale
        guidance_row = QHBoxLayout()
        guidance_row.addWidget(QLabel("Guidance:"))
        self.guidance_slider = QSlider(Qt.Orientation.Horizontal)
        self.guidance_slider.setRange(10, 150)  # 1.0 - 15.0
        self.guidance_slider.setValue(50)  # 5.0
        guidance_row.addWidget(self.guidance_slider)
        self.guidance_label = QLabel("5.0")
        self.guidance_label.setMinimumWidth(30)
        guidance_row.addWidget(self.guidance_label)
        i2i_layout.addLayout(guidance_row)

        self.i2i_group.setLayout(i2i_layout)
        layout.addWidget(self.i2i_group)

        # Think mode group (instruct/distil only)
        self.think_group = QGroupBox("Think Mode (Instruct/Distil)")
        think_layout = QVBoxLayout()
        task_row = QHBoxLayout()
        task_row.addWidget(QLabel("Mode:"))
        self.bot_task_combo = QComboBox()
        for display_name in BOT_TASK_OPTIONS:
            self.bot_task_combo.addItem(display_name)
        task_row.addWidget(self.bot_task_combo)
        think_layout.addLayout(task_row)
        self.show_think_check = QCheckBox("Show thinking output")
        self.show_think_check.setChecked(True)
        self.show_think_check.setToolTip(
            "When checked, chain-of-thought reasoning is displayed"
        )
        think_layout.addWidget(self.show_think_check)
        self.think_group.setLayout(think_layout)
        self.think_group.setVisible(False)  # Hidden until instruct/distil loaded
        layout.addWidget(self.think_group)

        # Generate button
        self.generate_btn = QPushButton("GENERATE")
        self.generate_btn.setObjectName("generate_btn")
        self.generate_btn.setMinimumHeight(40)
        layout.addWidget(self.generate_btn)

        # Progress area
        from widgets.progress_widget import ProgressWidget
        self.progress = ProgressWidget()
        layout.addWidget(self.progress)

        layout.addStretch()

    def _connect_signals(self):
        self.random_seed_btn.clicked.connect(self._randomize_seed)
        self.guidance_slider.valueChanged.connect(self._on_guidance_changed)

        # Show/hide I2I based on model type
        self.state.model_loaded.connect(self._on_model_loaded)
        self.state.model_unloaded.connect(self._on_model_unloaded)

    def _populate_presets(self, model_type=None):
        """Populate aspect ratio and quality dropdowns for the given model type."""
        from ui.constants import get_aspect_ratios, get_quality_presets

        ratios = get_aspect_ratios(model_type)
        presets = get_quality_presets(model_type)

        self.aspect_combo.clear()
        for name in ratios:
            self.aspect_combo.addItem(name)
        # Default to first item (1:1 square for both model types)
        self.aspect_combo.setCurrentIndex(0)

        self.quality_combo.clear()
        for name, info in presets.items():
            self.quality_combo.addItem(f"{name} ({info['steps']} steps)", name)
        self.quality_combo.setCurrentIndex(1)  # Standard

    @Slot()
    def _randomize_seed(self):
        self.seed_input.setValue(random.randint(0, 2**31 - 1))

    @Slot(int)
    def _on_guidance_changed(self, value: int):
        self.guidance_label.setText(f"{value / 10:.1f}")

    @Slot(str, str)
    def _on_model_loaded(self, model_type: str, msg: str):
        """Show/hide I2I and think mode based on model capabilities."""
        from ui.constants import MODEL_INFO, MODEL_DEFAULT_QUALITY, get_default_guidance
        info = MODEL_INFO.get(model_type, {})
        supports_i2i = info.get("supports_img2img", False)
        self.i2i_group.setVisible(supports_i2i)
        # Think mode is available for instruct/distil models
        is_instruct = model_type in ("instruct", "distil", "nf4", "distil_nf4", "instruct_int8", "distil_int8")
        self.think_group.setVisible(is_instruct)

        # Rebuild dropdowns with model-appropriate presets
        self._populate_presets(model_type)

        # Auto-select the appropriate quality preset for this model
        default_quality = MODEL_DEFAULT_QUALITY.get(model_type, "Standard")
        self.set_quality(default_quality)

        # Set guidance scale to model default
        default_guidance = get_default_guidance(model_type)
        self.guidance_slider.setValue(int(default_guidance * 10))

    @Slot()
    def _on_model_unloaded(self):
        self.i2i_group.setVisible(True)  # Show by default
        self.think_group.setVisible(False)
        # Reset to HunyuanImage defaults
        self._populate_presets()
        self.guidance_slider.setValue(50)  # 5.0

    # -- Public API --

    def get_aspect_ratio(self) -> str:
        return self.aspect_combo.currentText()

    def get_quality(self) -> str:
        data = self.quality_combo.currentData()
        return data if data else self.quality_combo.currentText().split(" (")[0]

    def get_seed(self) -> int:
        return self.seed_input.value()

    def set_seed(self, seed: int):
        self.seed_input.setValue(seed)

    def get_batch_count(self) -> int:
        return self.batch_count.value()

    def get_guidance_scale(self) -> float:
        return self.guidance_slider.value() / 10.0

    def get_i2i_image_path(self) -> str:
        """Legacy: return first image path."""
        if self.i2i_group.isChecked():
            return self.image_drops[0].get_image_path()
        return ""

    def get_i2i_image_paths(self) -> list:
        """Return all set reference image paths."""
        if not self.i2i_group.isChecked():
            return []
        return [d.get_image_path() for d in self.image_drops if d.get_image_path()]

    def get_bot_task(self) -> str:
        """Get the bot_task model value (e.g. 'image', 'think')."""
        display = self.bot_task_combo.currentText()
        return BOT_TASK_OPTIONS.get(display, "image")

    def get_drop_think(self) -> bool:
        """True if thinking output should be discarded."""
        return not self.show_think_check.isChecked()

    def is_think_mode_visible(self) -> bool:
        """Whether think mode controls are currently shown."""
        return self.think_group.isVisible()

    def set_aspect_ratio(self, name: str):
        idx = self.aspect_combo.findText(name)
        if idx >= 0:
            self.aspect_combo.setCurrentIndex(idx)

    def set_quality(self, name: str):
        # Try to find by display text first
        for i in range(self.quality_combo.count()):
            if self.quality_combo.itemData(i) == name or name in self.quality_combo.itemText(i):
                self.quality_combo.setCurrentIndex(i)
                return

    def set_batch_count(self, count: int):
        self.batch_count.setValue(count)

    def set_bot_task(self, value: str):
        from models.i2i_batch_config import BOT_TASK_REVERSE
        display = BOT_TASK_REVERSE.get(value, "Direct (image)")
        idx = self.bot_task_combo.findText(display)
        if idx >= 0:
            self.bot_task_combo.setCurrentIndex(idx)

    def set_drop_think(self, value: bool):
        self.show_think_check.setChecked(not value)

    def get_state_dict(self) -> dict:
        """Get all settings as a dict for project save."""
        return {
            "aspect_ratio": self.get_aspect_ratio(),
            "quality": self.get_quality(),
            "seed": self.get_seed(),
            "batch_count": self.get_batch_count(),
            "guidance_scale": self.get_guidance_scale(),
            "i2i_image_path": self.get_i2i_image_path(),
            "i2i_image_paths": self.get_i2i_image_paths(),
            "bot_task": self.get_bot_task(),
            "drop_think": self.get_drop_think(),
        }

    def set_state_dict(self, data: dict):
        """Restore settings from a dict (project load)."""
        if "aspect_ratio" in data:
            self.set_aspect_ratio(data["aspect_ratio"])
        if "quality" in data:
            self.set_quality(data["quality"])
        if "seed" in data:
            self.set_seed(data["seed"])
        if "batch_count" in data:
            self.set_batch_count(data["batch_count"])
        if "guidance_scale" in data:
            self.guidance_slider.setValue(int(data["guidance_scale"] * 10))
        if "i2i_image_paths" in data and data["i2i_image_paths"]:
            from pathlib import Path
            for i, p in enumerate(data["i2i_image_paths"][:3]):
                if p and Path(p).exists():
                    self.image_drops[i].set_image(p)
            self.i2i_group.setChecked(True)
        elif "i2i_image_path" in data and data["i2i_image_path"]:
            from pathlib import Path
            if Path(data["i2i_image_path"]).exists():
                self.image_drops[0].set_image(data["i2i_image_path"])
                self.i2i_group.setChecked(True)
        if "bot_task" in data:
            self.set_bot_task(data["bot_task"])
        if "drop_think" in data:
            self.set_drop_think(data["drop_think"])

    def set_generating(self, is_generating: bool):
        """Toggle UI state during generation."""
        self.generate_btn.setEnabled(not is_generating)
        if is_generating:
            self.generate_btn.setText("Generating...")
        else:
            self.generate_btn.setText("GENERATE")
