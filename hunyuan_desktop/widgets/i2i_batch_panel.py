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
        # Folder paths for the 3 slots; "" if none. Folder takes precedence
        # over the slot's single image and cycles per generation.
        self._slot_folders = ["", "", ""]
        self._setup_ui()
        self._connect_signals()
        self._populate_quality()    # Populate with default presets
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

        # 3 slots — each with a single image OR a folder (cycles per generation)
        self.global_image_slots = []   # ImageDropZone widgets
        self.folder_status_labels = []  # QLabel showing "Folder: name (N)"
        for slot_idx in range(3):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"Image {slot_idx+1}:"))

            drop = ImageDropZone()
            drop.setMinimumHeight(120)
            drop.setMaximumHeight(150)
            row.addWidget(drop)
            self.global_image_slots.append(drop)

            btn_col = QVBoxLayout()
            clear_btn = QPushButton("Clear")
            clear_btn.setMaximumWidth(60)
            clear_btn.clicked.connect(
                lambda _=False, i=slot_idx: self._on_clear_slot(i)
            )
            btn_col.addWidget(clear_btn)

            folder_btn = QPushButton("📁 Folder")
            folder_btn.setMaximumWidth(80)
            folder_btn.setToolTip(
                "Pick a folder of images. The batch will cycle through them, "
                "one per generation, instead of using a single image."
            )
            folder_btn.clicked.connect(
                lambda _=False, i=slot_idx: self._on_pick_folder(i)
            )
            btn_col.addWidget(folder_btn)
            row.addLayout(btn_col)

            images_layout.addLayout(row)

            status = QLabel("")
            status.setStyleSheet("color: #66bb6a; font-size: 10px; padding-left: 60px;")
            status.setWordWrap(True)
            images_layout.addWidget(status)
            self.folder_status_labels.append(status)

        # Back-compat aliases (older code paths reference these names)
        self.global_image1, self.global_image2, self.global_image3 = self.global_image_slots

        info_label = QLabel(
            "Up to 3 reference image slots. Each slot can hold a single image "
            "OR a folder of images that cycles per generation. "
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

        # ── Quality / Steps ──
        quality_row = QHBoxLayout()
        quality_row.addWidget(QLabel("Quality:"))
        self.quality_combo = QComboBox()
        self.quality_combo.setMinimumWidth(180)
        quality_row.addWidget(self.quality_combo, stretch=1)
        config_layout.addLayout(quality_row)

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

        # ── Prompt preview (wildcards + enhancement) ──
        preview_btn_row = QHBoxLayout()
        self.preview_wildcards_btn = QPushButton("Preview Wildcards")
        self.preview_wildcards_btn.setToolTip("Resolve wildcards in prompts and show results")
        self.preview_wildcards_btn.clicked.connect(self._on_preview_wildcards)
        preview_btn_row.addWidget(self.preview_wildcards_btn)

        self.preview_enhanced_btn = QPushButton("Preview Enhanced")
        self.preview_enhanced_btn.setToolTip("Resolve wildcards AND run LM Studio enhancement")
        self.preview_enhanced_btn.clicked.connect(self._on_preview_enhanced)
        preview_btn_row.addWidget(self.preview_enhanced_btn)

        self.preview_prompt_status = QLabel("")
        preview_btn_row.addWidget(self.preview_prompt_status, stretch=1)
        config_layout.addLayout(preview_btn_row)

        self.preview_display = QTextEdit()
        self.preview_display.setReadOnly(True)
        self.preview_display.setMaximumHeight(120)
        self.preview_display.setStyleSheet(
            "font-size: 11px; color: #c0c0c0; background-color: #1e1e1e; "
            "border: 1px solid #404040; padding: 4px;"
        )
        self.preview_display.setPlaceholderText("Click a preview button to see resolved prompts...")
        config_layout.addWidget(self.preview_display)

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

        # Update quality/guidance when model loads
        self.state.model_loaded.connect(self._on_model_loaded)
        self.state.model_unloaded.connect(self._on_model_unloaded)

    def _populate_quality(self, model_type=None):
        """Populate quality dropdown with model-appropriate presets."""
        from ui.constants import get_quality_presets
        presets = get_quality_presets(model_type)
        self.quality_combo.clear()
        for name, info in presets.items():
            self.quality_combo.addItem(f"{name} ({info['steps']} steps)", name)
        # Default to second item (usually "Standard")
        if self.quality_combo.count() > 1:
            self.quality_combo.setCurrentIndex(1)

    @Slot(str, str)
    def _on_model_loaded(self, model_type: str, msg: str):
        """Update quality presets and guidance when model loads."""
        self._populate_quality(model_type)
        # Set model-appropriate default quality
        from ui.constants import MODEL_DEFAULT_QUALITY, get_default_guidance
        default_quality = MODEL_DEFAULT_QUALITY.get(model_type, "Standard")
        for i in range(self.quality_combo.count()):
            if self.quality_combo.itemData(i) == default_quality or default_quality in self.quality_combo.itemText(i):
                self.quality_combo.setCurrentIndex(i)
                break
        # Set model-appropriate guidance default
        default_guidance = get_default_guidance(model_type)
        self.guidance_slider.setValue(int(default_guidance * 10))

    @Slot()
    def _on_model_unloaded(self):
        """Reset to default presets."""
        self._populate_quality()
        self.guidance_slider.setValue(50)

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

    # ── Image / Folder slot helpers ──

    IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

    def _scan_folder_images(self, folder_path: str) -> list:
        """Return sorted image paths inside a folder (non-recursive)."""
        if not folder_path:
            return []
        p = Path(folder_path)
        if not p.is_dir():
            return []
        return sorted(
            str(f) for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in self.IMAGE_EXTS
        )

    @Slot()
    def _on_pick_folder(self, slot_idx: int):
        """Open a folder picker for the given slot."""
        folder = QFileDialog.getExistingDirectory(
            self, f"Pick folder for slot {slot_idx+1}", ""
        )
        if not folder:
            return
        images = self._scan_folder_images(folder)
        if not images:
            self.folder_status_labels[slot_idx].setText(
                f"⚠ No images found in {Path(folder).name}"
            )
            return
        self._slot_folders[slot_idx] = folder
        # Clear any single image so the folder is the source of truth
        self.global_image_slots[slot_idx].clear_image()
        # Show first image as preview
        self.global_image_slots[slot_idx].set_image(images[0])
        # Drop the path tracking the slot lost during set_image — we want
        # the folder to take precedence, so the slot's _image_path is just
        # used as a thumbnail. Worker uses _slot_folders[i] instead.
        self.folder_status_labels[slot_idx].setText(
            f"📁 {Path(folder).name} — cycling {len(images)} image(s)"
        )

    @Slot()
    def _on_clear_slot(self, slot_idx: int):
        """Clear both the single image and any folder assigned to this slot."""
        self._slot_folders[slot_idx] = ""
        self.global_image_slots[slot_idx].clear_image()
        self.folder_status_labels[slot_idx].setText("")

    def _get_global_images(self):
        """Get list of global image paths (single-image slots only, no folders)."""
        images = []
        for idx, slot in enumerate(self.global_image_slots):
            if self._slot_folders[idx]:
                continue  # folder mode — handled separately
            path = slot.get_image_path()
            if path:
                images.append(path)
        return images

    def _get_slot_arrays(self):
        """Return (slots, folders) — both length-3 with "" for empty entries."""
        slots = ["", "", ""]
        folders = list(self._slot_folders)
        for idx, drop in enumerate(self.global_image_slots):
            if folders[idx]:
                continue  # folder takes precedence
            slots[idx] = drop.get_image_path() or ""
        return slots, folders

    # ── Prompt Preview (wildcards + enhancement) ──

    def _resolve_wildcards(self, prompt, generation_index=0):
        """Resolve wildcards in a prompt, handling starred wildcards for preview."""
        import re
        wm = self.state.get_wildcard_manager()
        if not wm:
            return prompt
        # Convert [*key] -> [key] so they get resolved during preview
        starred_pattern = r'\[\*([^\]]+)\]'
        preview_prompt = re.sub(starred_pattern, r'[\1]', prompt)
        try:
            return wm.process_prompt(preview_prompt, generation_index=generation_index)
        except Exception as e:
            print(f"[I2I PREVIEW] Wildcard resolution failed for prompt {generation_index+1}: {e}")
            return preview_prompt

    def _get_preview_prompts(self):
        """Parse prompt editor and return clean prompts (with [img:] tags stripped)."""
        raw_text = self.prompts_editor.toPlainText().strip()
        if not raw_text:
            return []
        global_images = self._get_global_images()
        prompts, _overrides = I2IBatchConfig.parse_prompt_lines(raw_text, global_images)
        return prompts

    @Slot()
    def _on_preview_wildcards(self):
        """Preview prompts with wildcards resolved."""
        prompts = self._get_preview_prompts()
        if not prompts:
            self.preview_display.setText("Enter prompts first.")
            return

        from ui.constants import DEFAULT_STYLE_PRESETS
        styles = self._get_selected_styles()
        style = styles[0] if styles else "None"
        style_suffix = DEFAULT_STYLE_PRESETS.get(style, "")

        lines = []
        for idx, prompt in enumerate(prompts):
            full = prompt + style_suffix
            resolved = self._resolve_wildcards(full, generation_index=idx)
            lines.append(f"--- Prompt {idx+1} ---")
            if resolved != full:
                lines.append(f"Original: {prompt}")
                lines.append(f"Resolved: {resolved}")
            else:
                lines.append(resolved)
            lines.append("")

        self.preview_display.setText("\n".join(lines))
        self.preview_prompt_status.setText(f"Resolved {len(prompts)} prompts")

    @Slot()
    def _on_preview_enhanced(self):
        """Preview all prompts with wildcards AND LM Studio enhancement."""
        prompts = self._get_preview_prompts()
        if not prompts:
            self.preview_display.setText("Enter prompts first.")
            return

        from ui.constants import DEFAULT_STYLE_PRESETS
        styles = self._get_selected_styles()
        style = styles[0] if styles else "None"
        style_suffix = DEFAULT_STYLE_PRESETS.get(style, "")

        # Resolve wildcards for all prompts
        resolved_prompts = []
        for idx, prompt in enumerate(prompts):
            full = prompt + style_suffix
            resolved = self._resolve_wildcards(full, generation_index=idx)
            resolved_prompts.append((prompt, resolved))

        # Show wildcard results immediately
        lines = []
        for idx, (orig, resolved) in enumerate(resolved_prompts):
            lines.append(f"--- Prompt {idx+1} ---")
            lines.append(f"Resolved: {resolved}")
            lines.append("")
        lines.append(f"--- Enhancing {len(resolved_prompts)} prompt(s) with LM Studio... ---")
        self.preview_display.setText("\n".join(lines))
        self.preview_prompt_status.setText(f"Enhancing 1/{len(resolved_prompts)}...")
        self.preview_enhanced_btn.setEnabled(False)

        # Store state for sequential enhancement
        self._preview_resolved_prompts = resolved_prompts
        self._preview_enhanced_results = []
        self._preview_enhance_index = 0
        self._preview_model = self.ollama_model.currentText()
        self._preview_length = self.ollama_length.currentText()
        self._preview_complexity = self.ollama_complexity.currentText()

        self._enhance_next_preview_prompt()

    def _enhance_next_preview_prompt(self):
        """Enhance the next prompt in the queue."""
        idx = self._preview_enhance_index
        prompts = self._preview_resolved_prompts

        if idx >= len(prompts):
            self._show_all_preview_results()
            return

        self.preview_prompt_status.setText(
            f"Enhancing {idx+1}/{len(prompts)}..."
        )

        from core.ollama_worker import OllamaEnhanceWorker
        self._preview_worker = OllamaEnhanceWorker(
            prompts[idx][1], self._preview_model,
            self._preview_length, self._preview_complexity
        )
        self._preview_worker.completed.connect(self._on_i2i_preview_enhance_done)
        self._preview_worker.failed.connect(self._on_i2i_preview_enhance_failed)
        self._preview_worker.start()

    @Slot(str, str)
    def _on_i2i_preview_enhance_done(self, original, enhanced):
        self._preview_enhanced_results.append((original, enhanced))
        self._preview_enhance_index += 1
        self._enhance_next_preview_prompt()

    @Slot(str)
    def _on_i2i_preview_enhance_failed(self, error):
        idx = self._preview_enhance_index
        orig = self._preview_resolved_prompts[idx][1]
        self._preview_enhanced_results.append((orig, f"[FAILED: {error}]"))
        self._preview_enhance_index += 1
        self._enhance_next_preview_prompt()

    def _show_all_preview_results(self):
        """Display all enhanced prompts in the preview area."""
        lines = []
        for idx, ((orig_prompt, resolved), (_, enhanced)) in enumerate(
            zip(self._preview_resolved_prompts, self._preview_enhanced_results)
        ):
            lines.append(f"=== Prompt {idx+1} ===")
            if orig_prompt != resolved:
                lines.append(f"Original:  {orig_prompt}")
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
            self.preview_prompt_status.setText(f"All {total} prompts enhanced")

    def _build_config(self) -> I2IBatchConfig:
        """Build I2IBatchConfig from current UI state."""
        raw_text = self.prompts_editor.toPlainText()
        global_images = self._get_global_images()
        slot_paths, slot_folders = self._get_slot_arrays()

        prompts, overrides = I2IBatchConfig.parse_prompt_lines(
            raw_text, global_images
        )

        # Get bot_task and drop_think from global state
        from ui.state import get_state
        state = get_state()
        bot_task = state.global_bot_task
        drop_think = state.global_drop_think

        # Get quality preset name (strip the steps suffix)
        quality_data = self.quality_combo.currentData()
        quality = quality_data if quality_data else self.quality_combo.currentText().split(" (")[0]

        return I2IBatchConfig(
            batch_name=self.batch_name.text().strip() or "i2i_batch",
            prompts=prompts,
            global_images=global_images,
            global_image_slots=slot_paths,
            global_image_folders=slot_folders,
            prompt_image_overrides=overrides,
            bot_task=bot_task,
            drop_think=drop_think,
            variations_per_prompt=self.variations_spin.value(),
            images_per_combo=self.images_per_spin.value(),
            styles=self._get_selected_styles(),
            quality=quality,
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

        has_folders = any(config.global_image_folders)
        if (not config.global_images
                and not config.prompt_image_overrides
                and not has_folders):
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

        # Restore quality preset
        for i in range(self.quality_combo.count()):
            if self.quality_combo.itemData(i) == config.quality or config.quality in self.quality_combo.itemText(i):
                self.quality_combo.setCurrentIndex(i)
                break

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

        # Restore slot images & folders
        for i in range(3):
            self._on_clear_slot(i)
        slots = list(config.global_image_slots) + ["", "", ""]
        folders = list(config.global_image_folders) + ["", "", ""]
        for i in range(3):
            folder = folders[i]
            if folder and Path(folder).is_dir():
                imgs = self._scan_folder_images(folder)
                if imgs:
                    self._slot_folders[i] = folder
                    self.global_image_slots[i].set_image(imgs[0])
                    self.folder_status_labels[i].setText(
                        f"📁 {Path(folder).name} — cycling {len(imgs)} image(s)"
                    )
                continue
            single = slots[i]
            if single and Path(single).exists():
                self.global_image_slots[i].set_image(single)
        # Legacy fallback: if no slot data was provided, fall back to flat list
        if not any(config.global_image_slots) and not any(config.global_image_folders):
            for i, p in enumerate(config.global_images[:3]):
                if p and Path(p).exists():
                    self.global_image_slots[i].set_image(p)

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
