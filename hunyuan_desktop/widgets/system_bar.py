"""System toolbar: model selector, GPU selector, VRAM display, Ollama status."""

from PySide6.QtWidgets import (
    QToolBar, QLabel, QComboBox, QPushButton, QProgressBar, QWidget, QHBoxLayout, QMessageBox, QMenu, QWidgetAction, QCheckBox, QVBoxLayout, QSpinBox
)
from PySide6.QtGui import QStandardItemModel, QStandardItem, QColor, QBrush, QFont
from PySide6.QtCore import Qt, QThread, Slot

from core.model_worker import ModelLoadWorker, ModelUnloadWorker
from core.settings import get_settings
from ui.state import get_state


class SystemBar(QToolBar):
    """Top toolbar with model controls, GPU selection, and VRAM status."""

    def __init__(self, desktop_state):
        super().__init__("System")
        self.state = desktop_state
        self.setMovable(False)
        self.setFloatable(False)

        self._load_worker = None
        self._unload_worker = None
        self._ollama_gpu_checkboxes = {}
        self._current_vram_gb = 0.0  # Track VRAM for enabling unload button

        self._setup_ui()
        self._connect_signals()
        self._populate_gpus()

    def _setup_ui(self):
        # Model selector — grouped by quantization class with colored rows
        self.addWidget(QLabel("Model:"))
        self.model_selector = QComboBox()
        self._populate_model_selector()
        self.model_selector.setMinimumWidth(340)
        self.addWidget(self.model_selector)

        self.addSeparator()

        # GPU selector
        self.addWidget(QLabel("GPU:"))
        self.gpu_selector = QComboBox()
        self.gpu_selector.setMinimumWidth(250)
        self.addWidget(self.gpu_selector)

        self.addSeparator()

        # RAM Offload toggle
        self.offload_check = QCheckBox("RAM Offload")
        self.offload_check.setToolTip(
            "Offload model layers to system RAM when VRAM is full.\n"
            "Slower but allows running larger models."
        )
        settings = get_settings()
        self.offload_check.setChecked(settings.cpu_offload_enabled)
        self.addWidget(self.offload_check)

        # RAM Offload settings button
        self.offload_settings_btn = QPushButton("⚙")
        self.offload_settings_btn.setMaximumWidth(30)
        self.offload_settings_btn.setToolTip("RAM offload settings")
        self._setup_offload_menu()
        self.addWidget(self.offload_settings_btn)

        self.addSeparator()

        # Global Generation Mode
        self.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Direct", "image")
        self.mode_combo.addItem("Think", "think")
        self.mode_combo.addItem("Think+Rewrite", "think_recaption")
        self.mode_combo.addItem("Rewrite", "recaption")
        self.mode_combo.setToolTip(
            "Generation mode (applies to all tabs):\n"
            "• Direct: Generate image directly\n"
            "• Think: Show reasoning before generating\n"
            "• Think+Rewrite: Think, then rewrite prompt, then generate\n"
            "• Rewrite: Rewrite prompt silently, then generate"
        )
        self.mode_combo.setMinimumWidth(100)
        # Load saved setting
        settings = get_settings()
        saved_mode = settings.global_bot_task
        for i in range(self.mode_combo.count()):
            if self.mode_combo.itemData(i) == saved_mode:
                self.mode_combo.setCurrentIndex(i)
                break
        self.addWidget(self.mode_combo)

        self.show_think_check = QCheckBox("Show CoT")
        self.show_think_check.setToolTip("Show Chain-of-Thought reasoning output")
        self.show_think_check.setChecked(not settings.global_drop_think)
        self.addWidget(self.show_think_check)

        self.addSeparator()

        # Load/Unload buttons
        self.load_btn = QPushButton("Load Model")
        self.load_btn.setMinimumWidth(100)
        self.addWidget(self.load_btn)

        self.unload_btn = QPushButton("Unload")
        self.unload_btn.setMinimumWidth(80)
        self.unload_btn.setEnabled(True)  # Always enabled - user can always try to free memory
        self.unload_btn.setToolTip("Unload model / Free GPU memory. Right-click for Force Cleanup.")
        self.unload_btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.unload_btn.customContextMenuRequested.connect(self._show_unload_menu)
        self.addWidget(self.unload_btn)

        self.addSeparator()

        # VRAM display
        self.vram_bar = QProgressBar()
        self.vram_bar.setMinimumWidth(150)
        self.vram_bar.setMaximumWidth(200)
        self.vram_bar.setFormat("VRAM: %v / %m GB")
        self.vram_bar.setTextVisible(True)
        self.addWidget(self.vram_bar)

        # Model status label
        self.status_label = QLabel("Not loaded")
        self.status_label.setMinimumWidth(120)
        self.addWidget(self.status_label)

        self.addSeparator()

        # Ollama GPU selector (multi-select dropdown)
        self.addWidget(QLabel("LLM GPUs:"))
        self.ollama_gpu_btn = QPushButton("0, 2")
        self.ollama_gpu_btn.setMinimumWidth(80)
        self.ollama_gpu_btn.setToolTip("Select GPUs for Ollama LLM (GPU 1/Blackwell excluded)")
        # Menu is set up in _populate_gpus() after GPU list is available
        self.addWidget(self.ollama_gpu_btn)

        # Ollama status
        self.ollama_label = QLabel("LLM: Not initialized")
        self.ollama_label.setMinimumWidth(150)
        self.addWidget(self.ollama_label)

    def _populate_model_selector(self):
        """Build the model dropdown with grouped sections and row colors.

        Groups:
          • Full Quality (SDNQ) — green tint
          • 4-bit NF4 (single-GPU friendly) — amber tint
          • Other Pipelines — blue tint
        """
        model = QStandardItemModel(self.model_selector)

        # color palette: (fg, bg) per group
        green_fg = QColor("#a5d6a7")
        amber_fg = QColor("#ffcc80")
        blue_fg = QColor("#90caf9")
        header_fg = QColor("#9e9e9e")

        def _header(text):
            item = QStandardItem(text)
            item.setFlags(Qt.ItemFlag.NoItemFlags)  # not selectable
            font = QFont()
            font.setBold(True)
            font.setPointSize(font.pointSize() - 1)
            item.setFont(font)
            item.setForeground(QBrush(header_fg))
            return item

        def _entry(text, data, fg):
            item = QStandardItem(text)
            item.setData(data, Qt.ItemDataRole.UserRole)
            item.setForeground(QBrush(fg))
            return item

        # ── Full Quality (SDNQ) ──
        model.appendRow(_header("── Full Quality (SDNQ) ──"))
        model.appendRow(_entry(
            "● Base — Create Mode (T2I, 20 steps)", "base", green_fg))
        model.appendRow(_entry(
            "● Instruct — Quality Mode (T2I+I2I, 50 steps)", "instruct", green_fg))
        model.appendRow(_entry(
            "● Distil — Edit Mode (T2I+I2I, 8 steps)", "distil", green_fg))

        # ── 4-bit NF4 ──
        model.appendRow(_header("── 4-bit NF4 (~48GB single GPU) ──"))
        model.appendRow(_entry(
            "◆ Instruct NF4 — 4-bit Quality (T2I+I2I, 50 steps)",
            "nf4", amber_fg))
        model.appendRow(_entry(
            "◆ Distil NF4 — 4-bit Fast (T2I+I2I, 8 steps)",
            "distil_nf4", amber_fg))

        # ── Other Pipelines ──
        model.appendRow(_header("── Other Pipelines ──"))
        model.appendRow(_entry(
            "▲ FireRed 1.1 — Image Edit (T2I+I2I, 40 steps)",
            "firered", blue_fg))

        self.model_selector.setModel(model)

        # Skip the first header row when auto-selecting
        for i in range(model.rowCount()):
            if model.item(i).flags() & Qt.ItemFlag.ItemIsSelectable:
                self.model_selector.setCurrentIndex(i)
                break

    def _connect_signals(self):
        self.load_btn.clicked.connect(self._on_load)
        self.unload_btn.clicked.connect(self._on_unload)
        self.gpu_selector.currentIndexChanged.connect(self._on_gpu_changed)
        self.offload_check.stateChanged.connect(self._on_offload_changed)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.show_think_check.stateChanged.connect(self._on_show_think_changed)

        # State signals
        self.state.vram_updated.connect(self._on_vram_updated)
        self.state.ollama_status_changed.connect(self._on_ollama_status)
        self.state.model_loaded.connect(self._on_model_loaded)
        self.state.model_unloaded.connect(self._on_model_unloaded)

    def _populate_gpus(self):
        """Populate GPU dropdown from detected GPUs."""
        self.gpu_selector.blockSignals(True)
        self.gpu_selector.clear()
        gpus = self.state.get_gpu_list()
        for gpu in gpus:
            self.gpu_selector.addItem(gpu["display"], gpu["index"])

        # Set default selection to state's selected GPU
        for i in range(self.gpu_selector.count()):
            if self.gpu_selector.itemData(i) == self.state.state.selected_gpu:
                self.gpu_selector.setCurrentIndex(i)
                break
        self.gpu_selector.blockSignals(False)

        # Also update Ollama GPU menu
        self._setup_ollama_gpu_menu()

    def _setup_offload_menu(self):
        """Setup the RAM offload settings menu."""
        menu = QMenu(self)
        settings = get_settings()
        app_state = get_state()

        # Container widget for the menu
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)

        # Max GPU Memory
        gpu_row = QHBoxLayout()
        gpu_row.addWidget(QLabel("Max GPU (GB):"))
        self.gpu_mem_spin = QSpinBox()
        self.gpu_mem_spin.setRange(16, 96)
        self.gpu_mem_spin.setValue(settings.max_gpu_memory_gb)
        self.gpu_mem_spin.setToolTip("Maximum VRAM to use before offloading to RAM")
        gpu_row.addWidget(self.gpu_mem_spin)
        layout.addLayout(gpu_row)

        # Max CPU/RAM Memory
        cpu_row = QHBoxLayout()
        cpu_row.addWidget(QLabel("Max RAM (GB):"))
        self.cpu_mem_spin = QSpinBox()
        self.cpu_mem_spin.setRange(8, 256)
        self.cpu_mem_spin.setValue(settings.max_cpu_memory_gb)
        self.cpu_mem_spin.setToolTip("Maximum system RAM for offloaded layers")
        cpu_row.addWidget(self.cpu_mem_spin)
        layout.addLayout(cpu_row)

        # NF4 dual-GPU split
        self.nf4_dual_check = QCheckBox("NF4 dual-GPU split (VAE/vision → 2nd GPU)")
        self.nf4_dual_check.setToolTip(
            "For NF4 models: offload VAE + vision encoder to the secondary GPU\n"
            "to free VRAM on the primary GPU for multi-image I2I conditioning.\n"
            "Requires reload to take effect."
        )
        self.nf4_dual_check.setChecked(settings.nf4_dual_gpu)
        self.nf4_dual_check.stateChanged.connect(self._on_nf4_dual_changed)
        layout.addWidget(self.nf4_dual_check)

        # Apply button
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._on_apply_offload_settings)
        layout.addWidget(apply_btn)

        # Info label
        info = QLabel("⚠️ Unload & reload model to apply changes")
        info.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(info)

        widget_action = QWidgetAction(menu)
        widget_action.setDefaultWidget(container)
        menu.addAction(widget_action)

        self.offload_settings_btn.setMenu(menu)

    @Slot()
    def _on_apply_offload_settings(self):
        """Save offload memory settings."""
        settings = get_settings()
        app_state = get_state()

        gpu_mem = self.gpu_mem_spin.value()
        cpu_mem = self.cpu_mem_spin.value()

        # Save to persistent settings
        settings.max_gpu_memory_gb = gpu_mem
        settings.max_cpu_memory_gb = cpu_mem

        # Update runtime state
        app_state.max_gpu_memory_gb = gpu_mem
        app_state.max_cpu_memory_gb = cpu_mem

        self.status_label.setText(f"Offload: GPU≤{gpu_mem}GB, RAM≤{cpu_mem}GB")

    @Slot(int)
    def _on_nf4_dual_changed(self, state: int):
        """Persist NF4 dual-GPU split toggle."""
        enabled = state == Qt.CheckState.Checked.value
        get_settings().nf4_dual_gpu = enabled
        self.status_label.setText(
            "NF4 dual-GPU ON (reload model)" if enabled else "NF4 dual-GPU OFF"
        )

    @Slot(int)
    def _on_offload_changed(self, state: int):
        """Handle RAM offload checkbox change."""
        enabled = state == Qt.CheckState.Checked.value
        settings = get_settings()
        app_state = get_state()

        # Save to persistent settings
        settings.cpu_offload_enabled = enabled

        # Update runtime state
        app_state.cpu_offload_enabled = enabled

        if enabled:
            self.status_label.setText("RAM offload ON (reload model)")
        else:
            self.status_label.setText("RAM offload OFF")

    @Slot(int)
    def _on_mode_changed(self, index: int):
        """Handle global generation mode change."""
        bot_task = self.mode_combo.currentData()
        settings = get_settings()
        app_state = get_state()

        # Save to persistent settings
        settings.global_bot_task = bot_task

        # Update runtime state
        app_state.global_bot_task = bot_task

        mode_names = {"image": "Direct", "think": "Think", "think_recaption": "Think+Rewrite", "recaption": "Rewrite"}
        self.status_label.setText(f"Mode: {mode_names.get(bot_task, bot_task)}")

    @Slot(int)
    def _on_show_think_changed(self, state: int):
        """Handle show thinking checkbox change."""
        show_think = state == Qt.CheckState.Checked.value
        drop_think = not show_think
        settings = get_settings()
        app_state = get_state()

        # Save to persistent settings
        settings.global_drop_think = drop_think

        # Update runtime state
        app_state.global_drop_think = drop_think

    def get_global_bot_task(self) -> str:
        """Get current global bot task mode."""
        return self.mode_combo.currentData() or "image"

    def get_global_drop_think(self) -> bool:
        """Get whether to drop thinking output."""
        return not self.show_think_check.isChecked()

    def _setup_ollama_gpu_menu(self):
        """Setup the Ollama GPU multi-select dropdown menu."""
        menu = QMenu(self)
        gpus = self.state.get_gpu_list()
        settings = get_settings()
        selected_indices = settings.ollama_gpu_indices

        self._ollama_gpu_checkboxes = {}

        for gpu in gpus:
            idx = gpu["index"]
            name = gpu["name"]
            mem = gpu.get("memory_gb", 0)

            # Skip GPU 1 (Blackwell) - always excluded
            if idx == 1:
                # Add disabled item to show it's excluded
                action = menu.addAction(f"GPU {idx}: {name} ({mem:.0f}GB) [Reserved for Image Gen]")
                action.setEnabled(False)
                continue

            # Create checkable action
            checkbox = QCheckBox(f"GPU {idx}: {name} ({mem:.0f}GB)")
            checkbox.setChecked(idx in selected_indices)
            checkbox.stateChanged.connect(lambda state, i=idx: self._on_ollama_gpu_changed(i, state))

            widget_action = QWidgetAction(menu)
            widget_action.setDefaultWidget(checkbox)
            menu.addAction(widget_action)
            self._ollama_gpu_checkboxes[idx] = checkbox

        menu.addSeparator()

        # Add restart Ollama button
        restart_action = menu.addAction("Restart Ollama with selected GPUs")
        restart_action.triggered.connect(self._on_restart_ollama)

        self.ollama_gpu_btn.setMenu(menu)
        self._update_ollama_gpu_button_text()

    def _update_ollama_gpu_button_text(self):
        """Update the Ollama GPU button text to show selected GPUs."""
        settings = get_settings()
        indices = settings.ollama_gpu_indices
        # Filter out GPU 1
        safe_indices = [i for i in indices if i != 1]
        if safe_indices:
            self.ollama_gpu_btn.setText(", ".join(map(str, safe_indices)))
        else:
            self.ollama_gpu_btn.setText("0")

    def _on_ollama_gpu_changed(self, gpu_idx: int, state: int):
        """Handle Ollama GPU checkbox state change."""
        settings = get_settings()
        current = settings.ollama_gpu_indices

        if state == Qt.CheckState.Checked.value:
            if gpu_idx not in current:
                current.append(gpu_idx)
        else:
            if gpu_idx in current:
                current.remove(gpu_idx)

        # Ensure at least one GPU is selected (not GPU 1)
        safe_gpus = [i for i in current if i != 1]
        if not safe_gpus:
            safe_gpus = [0]
            if 0 in self._ollama_gpu_checkboxes:
                self._ollama_gpu_checkboxes[0].blockSignals(True)
                self._ollama_gpu_checkboxes[0].setChecked(True)
                self._ollama_gpu_checkboxes[0].blockSignals(False)

        settings.ollama_gpu_indices = safe_gpus
        self._update_ollama_gpu_button_text()

    @Slot()
    def _on_restart_ollama(self):
        """Restart Ollama with the currently selected GPUs."""
        settings = get_settings()
        gpu_indices = settings.ollama_gpu_indices

        # Filter out GPU 1 (Blackwell)
        safe_gpus = [i for i in gpu_indices if i != 1]
        if not safe_gpus:
            safe_gpus = [0]

        self.ollama_label.setText("LLM: Restarting...")

        # Use the state's refresh method
        self.state.refresh_ollama(safe_gpus)

    @Slot()
    def _on_load(self):
        """Start model loading on a background thread."""
        model_type = self.model_selector.currentData()
        if not model_type:
            return

        self.load_btn.setEnabled(False)
        self.load_btn.setText("Loading...")
        self.status_label.setText("Loading...")

        # Notify state
        self.state.model_loading_started.emit(model_type)

        self._load_worker = ModelLoadWorker(model_type)
        self._load_worker.progress.connect(self._on_load_progress)
        self._load_worker.finished.connect(self._on_load_finished)
        self._load_worker.start(QThread.LowPriority)

    @Slot()
    def _on_unload(self):
        """Start model unloading."""
        self.unload_btn.setEnabled(False)
        self.status_label.setText("Unloading...")

        self._unload_worker = ModelUnloadWorker()
        self._unload_worker.finished.connect(self._on_unload_finished)
        self._unload_worker.start()

    @Slot(str)
    def _on_load_progress(self, message: str):
        """Update status during model loading."""
        self.status_label.setText(message[:60] + "..." if len(message) > 60 else message)
        self.state.model_loading_progress.emit(message)

    @Slot(bool, str)
    def _on_load_finished(self, success: bool, message: str):
        """Handle model load completion."""
        self.load_btn.setEnabled(True)
        self.load_btn.setText("Load Model")

        if success:
            model_type = self.model_selector.currentData()
            self.status_label.setText(f"Loaded: {model_type}")
            self.unload_btn.setEnabled(True)
            self.state.model_loaded.emit(model_type, message)
        else:
            self.status_label.setText("Load failed")
            self.state.model_load_failed.emit(message)

            # Show error details in a message box
            error_dialog = QMessageBox(self)
            error_dialog.setIcon(QMessageBox.Icon.Critical)
            error_dialog.setWindowTitle("Model Load Failed")
            error_dialog.setText("Failed to load the model.")
            error_dialog.setDetailedText(message)
            error_dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
            error_dialog.exec()

    @Slot(str)
    def _on_unload_finished(self, message: str):
        """Handle model unload completion."""
        self.status_label.setText(message[:40] if len(message) > 40 else message)
        self.unload_btn.setEnabled(True)  # Keep enabled for subsequent cleanup attempts
        self.state.model_unloaded.emit()

    @Slot()
    def _on_gpu_changed(self):
        """Handle GPU selection change."""
        gpu_index = self.gpu_selector.currentData()
        if gpu_index is not None:
            from ui.state import set_gpu
            set_gpu(gpu_index)

    @Slot(int, float, float)
    def _on_vram_updated(self, gpu_idx: int, used_gb: float, total_gb: float):
        """Update the VRAM progress bar."""
        self.vram_bar.setMaximum(int(total_gb))
        self.vram_bar.setValue(int(used_gb))
        self.vram_bar.setFormat(f"VRAM: {used_gb:.1f} / {total_gb:.0f} GB")

        # Track VRAM for the context menu display
        self._current_vram_gb = used_gb

        # Update tooltip with current VRAM
        if used_gb > 1:
            self.unload_btn.setToolTip(f"Free GPU memory ({used_gb:.1f}GB in use). Right-click for Force Cleanup.")

    def _show_unload_menu(self, pos):
        """Show context menu for unload button with force cleanup option."""
        menu = QMenu(self)

        unload_action = menu.addAction("Unload Model")
        unload_action.triggered.connect(self._on_unload)

        menu.addSeparator()

        force_action = menu.addAction(f"Force Cleanup GPU ({self._current_vram_gb:.1f}GB)")
        force_action.triggered.connect(self._on_force_cleanup)
        force_action.setToolTip("Use if normal unload doesn't free memory")

        menu.exec(self.unload_btn.mapToGlobal(pos))

    @Slot()
    def _on_force_cleanup(self):
        """Force cleanup GPU memory."""
        self.unload_btn.setEnabled(False)
        self.status_label.setText("Force cleanup...")

        self._unload_worker = ModelUnloadWorker(force=True)
        self._unload_worker.finished.connect(self._on_unload_finished)
        self._unload_worker.start()

    @Slot(str)
    def _on_ollama_status(self, status: str):
        """Update Ollama status label."""
        self.ollama_label.setText(status)

    @Slot(str, str)
    def _on_model_loaded(self, model_type: str, message: str):
        """Handle model loaded signal."""
        self.status_label.setText(f"Loaded: {model_type}")
        self.unload_btn.setEnabled(True)

    @Slot()
    def _on_model_unloaded(self):
        """Handle model unloaded signal."""
        self.status_label.setText("Not loaded")
        # Keep unload button enabled for cleanup attempts
        self.unload_btn.setEnabled(True)

    def get_selected_model_type(self) -> str:
        """Get the currently selected model type."""
        return self.model_selector.currentData() or "base"
