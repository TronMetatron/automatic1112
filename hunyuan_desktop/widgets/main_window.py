"""Main application window: tabs, docks, menus, shortcuts, signal wiring."""

from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QDockWidget, QMenuBar, QMenu, QStatusBar,
    QMessageBox, QLabel, QGroupBox, QPushButton, QSlider, QSpinBox,
    QComboBox, QTextEdit, QCheckBox, QDialog, QScrollArea
)
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QAction, QKeySequence

from core.settings import get_settings
from core.generation_worker import GenerationWorker
from core.ollama_worker import OllamaEnhanceWorker
from models.generation_params import GenerationParams


class MainWindow(QMainWindow):
    """Primary application window with tabbed interface and dockable panels."""

    def __init__(self, desktop_state):
        super().__init__()
        self.state = desktop_state
        self.settings = get_settings()
        self._gen_worker = None
        self._ollama_worker = None
        self._current_project_name = None  # For quick re-save

        self.setWindowTitle("HunyuanImage Desktop")
        self.setMinimumSize(1200, 800)

        self._setup_ui()
        self._setup_menus()
        self._setup_shortcuts()
        self._connect_signals()
        self._restore_state()

    def _setup_ui(self):
        """Build the complete UI layout."""
        # System toolbar
        from widgets.system_bar import SystemBar
        self.system_bar = SystemBar(self.state)
        self.system_bar.setObjectName("system_bar")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.system_bar)

        # Central tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self.tabs)

        # Tab 1: Generate
        self._setup_generate_tab()

        # Tab 2: Batch
        self._setup_batch_tab()

        # Tab 3: Edit (I2I) - rebuilt with multi-image + think mode
        self._setup_edit_tab()

        # Tab 4: I2I Batch
        self._setup_i2i_batch_tab()

        # Tab 5: Dataset Prep
        self._setup_dataset_prep_tab()

        # Tab 6: Log
        self._setup_log_tab()

        # Dock: Wildcard sidebar (right)
        self._setup_wildcard_dock()

        # Dock: Gallery (bottom)
        self._setup_gallery_dock()

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _setup_generate_tab(self):
        """Create the single image generation tab."""
        from widgets.prompt_panel import PromptPanel
        from widgets.gen_settings_panel import GenSettingsPanel
        from widgets.output_panel import OutputPanel

        gen_widget = QWidget()
        gen_layout = QHBoxLayout(gen_widget)
        gen_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Prompt + Settings
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.prompt_panel = PromptPanel(self.state)
        left_layout.addWidget(self.prompt_panel, stretch=3)

        self.gen_settings = GenSettingsPanel(self.state)
        left_layout.addWidget(self.gen_settings, stretch=2)

        splitter.addWidget(left_widget)

        # Right: Output + CoT display
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.output_panel = OutputPanel(self.state)
        right_layout.addWidget(self.output_panel)

        # Chain-of-thought display for Generate tab (instruct/distil)
        self.gen_cot_group = QGroupBox("Chain of Thought")
        self.gen_cot_group.setCheckable(True)
        self.gen_cot_group.setChecked(False)
        gen_cot_layout = QVBoxLayout()
        self.gen_cot_display = QTextEdit()
        self.gen_cot_display.setReadOnly(True)
        self.gen_cot_display.setMaximumHeight(120)
        self.gen_cot_display.setStyleSheet(
            "font-size: 11px; color: #b0b0b0;"
        )
        gen_cot_layout.addWidget(self.gen_cot_display)
        self.gen_cot_group.setLayout(gen_cot_layout)
        self.gen_cot_group.setVisible(False)  # Hidden until instruct/distil loaded
        right_layout.addWidget(self.gen_cot_group)

        splitter.addWidget(right_widget)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        gen_layout.addWidget(splitter)
        self.tabs.addTab(gen_widget, "Generate")

    def _setup_batch_tab(self):
        """Create the batch processing tab."""
        from widgets.batch_panel import BatchPanel
        self.batch_panel = BatchPanel(self.state)
        self.tabs.addTab(self.batch_panel, "Batch")

    def _setup_edit_tab(self):
        """Create the image editing (I2I) tab with multi-image + think mode."""
        from widgets.prompt_editor import PromptEditor
        from widgets.output_panel import OutputPanel
        from widgets.gen_settings_panel import ImageDropZone
        from widgets.progress_widget import ProgressWidget

        edit_widget = QWidget()
        edit_layout = QHBoxLayout(edit_widget)
        edit_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ─── Left: Input controls ───
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Reference Image 1
        left_layout.addWidget(QLabel("Reference Image 1:"))
        self.edit_image1_drop = ImageDropZone()
        self.edit_image1_drop.setMinimumHeight(160)
        left_layout.addWidget(self.edit_image1_drop)

        # Reference Image 2 (optional)
        img2_row = QHBoxLayout()
        img2_row.addWidget(QLabel("Reference Image 2 (optional):"))
        self.edit_clear_img2_btn = QPushButton("Clear")
        self.edit_clear_img2_btn.setMaximumWidth(50)
        img2_row.addWidget(self.edit_clear_img2_btn)
        left_layout.addLayout(img2_row)
        self.edit_image2_drop = ImageDropZone()
        self.edit_image2_drop.setMinimumHeight(120)
        left_layout.addWidget(self.edit_image2_drop)
        self.edit_clear_img2_btn.clicked.connect(
            lambda: self.edit_image2_drop.clear_image()
        )

        # Reference Image 3 (optional - for FireRed)
        img3_row = QHBoxLayout()
        img3_row.addWidget(QLabel("Reference Image 3 (optional):"))
        self.edit_clear_img3_btn = QPushButton("Clear")
        self.edit_clear_img3_btn.setMaximumWidth(50)
        img3_row.addWidget(self.edit_clear_img3_btn)
        left_layout.addLayout(img3_row)
        self.edit_image3_drop = ImageDropZone()
        self.edit_image3_drop.setMinimumHeight(120)
        left_layout.addWidget(self.edit_image3_drop)
        self.edit_clear_img3_btn.clicked.connect(
            lambda: self.edit_image3_drop.clear_image()
        )

        # Edit instruction
        left_layout.addWidget(QLabel("Edit Instruction:"))
        self.edit_prompt = PromptEditor(
            wildcard_manager=self.state.get_wildcard_manager()
        )
        self.edit_prompt.setMinimumHeight(80)
        self.edit_prompt.setPlaceholderText(
            "Replace the cat with a dragon breathing fire"
        )
        left_layout.addWidget(self.edit_prompt)

        # NOTE: Bot Task / Think Mode controls moved to global system bar
        # Mode is now controlled globally across all tabs

        # Quick presets
        presets_group = QGroupBox("Quick Presets")
        presets_layout = QHBoxLayout()
        preset_buttons = [
            ("Style Transfer", "Apply the style of a watercolor painting"),
            ("Object Replace", "Replace the [object] with a [new object]"),
            ("Scene Modify", "Change the background to a sunset beach"),
            ("Camera Angle", "Change to a dramatic low angle shot"),
        ]
        for label, template in preset_buttons:
            btn = QPushButton(label)
            btn.clicked.connect(
                lambda checked, t=template: self.edit_prompt.setPlainText(t)
            )
            btn.setToolTip(template)
            presets_layout.addWidget(btn)
        presets_group.setLayout(presets_layout)
        left_layout.addWidget(presets_group)

        # Edit settings
        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("Guidance:"))
        self.edit_guidance = QSlider(Qt.Orientation.Horizontal)
        self.edit_guidance.setRange(10, 150)
        self.edit_guidance.setValue(50)
        settings_layout.addWidget(self.edit_guidance)
        self.edit_guidance_label = QLabel("5.0")
        self.edit_guidance_label.setMinimumWidth(30)
        settings_layout.addWidget(self.edit_guidance_label)
        self.edit_guidance.valueChanged.connect(
            lambda v: self.edit_guidance_label.setText(f"{v/10:.1f}")
        )
        left_layout.addLayout(settings_layout)

        seed_layout = QHBoxLayout()
        seed_layout.addWidget(QLabel("Seed:"))
        self.edit_seed = QSpinBox()
        self.edit_seed.setRange(-1, 2**31 - 1)
        self.edit_seed.setValue(-1)
        seed_layout.addWidget(self.edit_seed)
        seed_layout.addStretch()
        left_layout.addLayout(seed_layout)

        # Prompt enhancement group
        from ui.constants import OLLAMA_LENGTH_OPTIONS, OLLAMA_COMPLEXITY_OPTIONS
        enhance_group = QGroupBox("Prompt Enhancement")
        enhance_layout = QVBoxLayout()
        self.edit_enhance_check = QCheckBox("Enhance with LM Studio / Ollama")
        enhance_layout.addWidget(self.edit_enhance_check)
        enh_row = QHBoxLayout()
        enh_row.addWidget(QLabel("Length:"))
        self.edit_ollama_length = QComboBox()
        for opt in OLLAMA_LENGTH_OPTIONS:
            self.edit_ollama_length.addItem(opt)
        self.edit_ollama_length.setCurrentText("medium")
        enh_row.addWidget(self.edit_ollama_length)
        enh_row.addWidget(QLabel("Complexity:"))
        self.edit_ollama_complexity = QComboBox()
        for opt in OLLAMA_COMPLEXITY_OPTIONS:
            self.edit_ollama_complexity.addItem(opt)
        self.edit_ollama_complexity.setCurrentText("detailed")
        enh_row.addWidget(self.edit_ollama_complexity)
        enhance_layout.addLayout(enh_row)
        enhance_group.setLayout(enhance_layout)
        left_layout.addWidget(enhance_group)

        # Edit button
        self.edit_generate_btn = QPushButton("EDIT IMAGE")
        self.edit_generate_btn.setObjectName("generate_btn")
        self.edit_generate_btn.setMinimumHeight(40)
        self.edit_generate_btn.clicked.connect(self._on_edit_generate)
        left_layout.addWidget(self.edit_generate_btn)

        self.edit_progress = ProgressWidget()
        left_layout.addWidget(self.edit_progress)

        left_layout.addStretch()
        splitter.addWidget(left_widget)

        # ─── Right: Output with CoT + compare ───
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.edit_output = OutputPanel(self.state)
        right_layout.addWidget(self.edit_output)

        # Chain-of-thought display (collapsible)
        self.edit_cot_group = QGroupBox("Chain of Thought")
        self.edit_cot_group.setCheckable(True)
        self.edit_cot_group.setChecked(False)
        cot_layout = QVBoxLayout()
        self.edit_cot_display = QTextEdit()
        self.edit_cot_display.setReadOnly(True)
        self.edit_cot_display.setMaximumHeight(120)
        self.edit_cot_display.setStyleSheet(
            "font-size: 11px; color: #b0b0b0;"
        )
        cot_layout.addWidget(self.edit_cot_display)
        self.edit_cot_group.setLayout(cot_layout)
        right_layout.addWidget(self.edit_cot_group)

        # Compare button
        compare_row = QHBoxLayout()
        self.compare_btn = QPushButton("Compare Before/After")
        self.compare_btn.setEnabled(False)
        self.compare_btn.clicked.connect(self._on_compare)
        compare_row.addWidget(self.compare_btn)
        compare_row.addStretch()
        right_layout.addLayout(compare_row)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        edit_layout.addWidget(splitter)
        self.tabs.addTab(edit_widget, "Edit")

        # Notice for incompatible models
        self._edit_notice = QLabel(
            "Load an Instruct or Distil model to use image editing.\n"
            "Base model only supports text-to-image generation."
        )
        self._edit_notice.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._edit_notice.setStyleSheet(
            "color: #808080; font-size: 14px; padding: 20px;"
        )

    def _setup_i2i_batch_tab(self):
        """Create the image-to-image batch processing tab."""
        from widgets.i2i_batch_panel import I2IBatchPanel
        self.i2i_batch_panel = I2IBatchPanel(self.state)
        self.tabs.addTab(self.i2i_batch_panel, "I2I Batch")

    def _setup_dataset_prep_tab(self):
        """Create the dataset preparation tab for LoRA training datasets."""
        from widgets.dataset_prep_panel import DatasetPrepPanel
        self.dataset_prep_panel = DatasetPrepPanel(self.state)
        self.tabs.addTab(self.dataset_prep_panel, "Dataset Prep")

    def _setup_log_tab(self):
        """Create the log/console tab for viewing output and errors."""
        from widgets.log_panel import LogPanel
        self.log_panel = LogPanel()
        self.tabs.addTab(self.log_panel, "Log")

    def _setup_wildcard_dock(self):
        """Create the wildcard sidebar as a dockable panel."""
        from widgets.wildcard_sidebar import WildcardSidebar

        self.wildcard_dock = QDockWidget("Wildcards", self)
        self.wildcard_dock.setObjectName("wildcard_dock")
        self.wildcard_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.wildcard_sidebar = WildcardSidebar(self.state)
        self.wildcard_dock.setWidget(self.wildcard_sidebar)
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.wildcard_dock
        )
        self.wildcard_dock.setMinimumWidth(220)

    def _setup_gallery_dock(self):
        """Create the gallery as a dockable bottom panel."""
        from widgets.gallery_panel import GalleryPanel

        self.gallery_dock = QDockWidget("Gallery", self)
        self.gallery_dock.setObjectName("gallery_dock")
        self.gallery_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.TopDockWidgetArea
        )
        self.gallery_panel = GalleryPanel(self.state)
        self.gallery_dock.setWidget(self.gallery_panel)
        self.addDockWidget(
            Qt.DockWidgetArea.BottomDockWidgetArea, self.gallery_dock
        )
        self.gallery_dock.setMaximumHeight(250)

        # Auto-load the main output directory so gallery shows recent images
        try:
            from pathlib import Path
            from ui.constants import OUTPUT_DIR
            output_path = str(OUTPUT_DIR)
            if Path(output_path).is_dir():
                self.gallery_panel.load_directory(output_path)
        except Exception:
            pass

    def _setup_menus(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        save_project_action = QAction("&Save Project", self)
        save_project_action.setShortcut(QKeySequence("Ctrl+S"))
        save_project_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_project_action)

        save_project_as_action = QAction("Save Project &As...", self)
        save_project_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_project_as_action.triggered.connect(self._on_save_project_as)
        file_menu.addAction(save_project_as_action)

        load_project_action = QAction("&Load Project...", self)
        load_project_action.setShortcut(QKeySequence("Ctrl+Shift+L"))
        load_project_action.triggered.connect(self._on_load_project)
        file_menu.addAction(load_project_action)

        file_menu.addSeparator()

        load_model_action = QAction("Load &Model", self)
        load_model_action.setShortcut(QKeySequence("Ctrl+L"))
        load_model_action.triggered.connect(self.system_bar._on_load)
        file_menu.addAction(load_model_action)

        unload_model_action = QAction("&Unload Model", self)
        unload_model_action.triggered.connect(self.system_bar._on_unload)
        file_menu.addAction(unload_model_action)

        force_cleanup_action = QAction("&Force GPU Cleanup", self)
        force_cleanup_action.setToolTip("Force cleanup GPU memory if unload doesn't work")
        force_cleanup_action.triggered.connect(self.system_bar._on_force_cleanup)
        file_menu.addAction(force_cleanup_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        toggle_wildcards = self.wildcard_dock.toggleViewAction()
        toggle_wildcards.setShortcut(QKeySequence("Ctrl+W"))
        view_menu.addAction(toggle_wildcards)

        toggle_gallery = self.gallery_dock.toggleViewAction()
        toggle_gallery.setShortcut(QKeySequence("Ctrl+G"))
        view_menu.addAction(toggle_gallery)

        view_menu.addSeparator()

        gen_tab_action = QAction("&Generate Tab", self)
        gen_tab_action.setShortcut(QKeySequence("Ctrl+1"))
        gen_tab_action.triggered.connect(lambda: self.tabs.setCurrentIndex(0))
        view_menu.addAction(gen_tab_action)

        batch_tab_action = QAction("&Batch Tab", self)
        batch_tab_action.setShortcut(QKeySequence("Ctrl+2"))
        batch_tab_action.triggered.connect(lambda: self.tabs.setCurrentIndex(1))
        view_menu.addAction(batch_tab_action)

        edit_tab_action = QAction("&Edit Tab", self)
        edit_tab_action.setShortcut(QKeySequence("Ctrl+3"))
        edit_tab_action.triggered.connect(lambda: self.tabs.setCurrentIndex(2))
        view_menu.addAction(edit_tab_action)

        i2i_batch_tab_action = QAction("&I2I Batch Tab", self)
        i2i_batch_tab_action.setShortcut(QKeySequence("Ctrl+4"))
        i2i_batch_tab_action.triggered.connect(
            lambda: self.tabs.setCurrentIndex(3)
        )
        view_menu.addAction(i2i_batch_tab_action)

        dataset_prep_tab_action = QAction("&Dataset Prep Tab", self)
        dataset_prep_tab_action.setShortcut(QKeySequence("Ctrl+5"))
        dataset_prep_tab_action.triggered.connect(
            lambda: self.tabs.setCurrentIndex(4)
        )
        view_menu.addAction(dataset_prep_tab_action)

        log_tab_action = QAction("&Log Tab", self)
        log_tab_action.setShortcut(QKeySequence("Ctrl+6"))
        log_tab_action.triggered.connect(lambda: self.tabs.setCurrentIndex(5))
        view_menu.addAction(log_tab_action)

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")

        refresh_ollama_action = QAction("Refresh &Ollama", self)
        refresh_ollama_action.triggered.connect(self.state.refresh_ollama)
        tools_menu.addAction(refresh_ollama_action)

        refresh_wildcards_action = QAction("Refresh &Wildcards", self)
        refresh_wildcards_action.triggered.connect(self._refresh_wildcards)
        tools_menu.addAction(refresh_wildcards_action)

        tools_menu.addSeparator()

        lmstudio_settings_action = QAction("&LM Studio Settings...", self)
        lmstudio_settings_action.triggered.connect(
            lambda: self.prompt_panel._on_edit_system_prompt()
        )
        tools_menu.addAction(lmstudio_settings_action)

        tools_menu.addSeparator()

        open_output_action = QAction("Open Output &Directory", self)
        open_output_action.triggered.connect(self._open_output_dir)
        tools_menu.addAction(open_output_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        firered_help_action = QAction("FireRed &Editing Guide", self)
        firered_help_action.triggered.connect(self._show_firered_guide)
        help_menu.addAction(firered_help_action)

        prompting_help_action = QAction("&Prompting Guide", self)
        prompting_help_action.triggered.connect(self._show_prompting_guide)
        help_menu.addAction(prompting_help_action)

        help_menu.addSeparator()

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_shortcuts(self):
        """Set up keyboard shortcuts."""
        from PySide6.QtGui import QShortcut

        # Ctrl+Enter - Generate
        gen_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
        gen_shortcut.activated.connect(self._on_generate_shortcut)

        # Ctrl+E - Enhance prompt
        enhance_shortcut = QShortcut(QKeySequence("Ctrl+E"), self)
        enhance_shortcut.activated.connect(self._on_enhance_shortcut)

        # Ctrl+B - Switch to Batch tab
        batch_shortcut = QShortcut(QKeySequence("Ctrl+B"), self)
        batch_shortcut.activated.connect(lambda: self.tabs.setCurrentIndex(1))

        # Escape - Stop current generation
        stop_shortcut = QShortcut(QKeySequence("Escape"), self)
        stop_shortcut.activated.connect(self._on_stop)

        # Ctrl+Z in prompt - Undo enhancement
        undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_shortcut.activated.connect(self._on_undo_enhance)

    def _connect_signals(self):
        """Wire up all cross-widget signal connections."""
        # Generate button -> start generation
        self.gen_settings.generate_btn.clicked.connect(self._on_generate)

        # Prompt panel enhance -> Ollama worker
        self.prompt_panel.enhance_requested.connect(self._on_enhance)

        # Output panel load settings -> apply to prompt/settings panels
        self.output_panel.load_settings_requested.connect(
            self._apply_loaded_settings
        )

        # Wildcard sidebar insert -> prompt editor
        self.wildcard_sidebar.insert_requested.connect(self._insert_wildcard)

        # Gallery image selected -> show in output
        self.gallery_panel.image_selected.connect(
            self._on_gallery_image_selected
        )

        # Main gallery insert prompt -> insert into active tab
        self.gallery_panel.insert_prompt_requested.connect(
            self._on_gallery_insert_prompt
        )

        # Batch gallery image -> show in main gallery too
        self.batch_panel.batch_gallery.image_selected.connect(
            self._on_gallery_image_selected
        )

        # I2I Batch gallery -> show in main gallery
        self.i2i_batch_panel.batch_gallery.image_selected.connect(
            self._on_gallery_image_selected
        )

        # Dataset Prep gallery -> show in main gallery
        self.dataset_prep_panel.batch_gallery.image_selected.connect(
            self._on_gallery_image_selected
        )

        # Batch completed -> update main gallery dock to show batch output
        self.state.batch_completed.connect(self._on_batch_completed_gallery)

        # Model loaded/unloaded -> update tab visibility
        self.state.model_loaded.connect(self._on_model_state_changed)
        self.state.model_unloaded.connect(self._on_model_unloaded)

        # Tab changed -> save preference
        self.tabs.currentChanged.connect(self._on_tab_changed)

    # ─── Generation ───────────────────────────────────────────

    @Slot()
    def _on_generate(self):
        """Start single image generation."""
        if self._gen_worker and self._gen_worker.isRunning():
            self.status_bar.showMessage("Generation already in progress")
            return

        prompt = self.prompt_panel.get_prompt()
        if not prompt:
            self.status_bar.showMessage("Enter a prompt first")
            return

        # Use global mode from system bar
        params = GenerationParams(
            prompt=prompt,
            negative_prompt=self.prompt_panel.get_negative_prompt(),
            style=self.prompt_panel.get_style(),
            aspect_ratio=self.gen_settings.get_aspect_ratio(),
            quality=self.gen_settings.get_quality(),
            seed=self.gen_settings.get_seed(),
            batch_count=self.gen_settings.get_batch_count(),
            use_ollama=self.prompt_panel.get_ollama_settings()["use_ollama"],
            ollama_model=self.prompt_panel.get_ollama_settings()["model"],
            ollama_length=self.prompt_panel.get_ollama_settings()["length"],
            ollama_complexity=self.prompt_panel.get_ollama_settings()["complexity"],
            guidance_scale=self.gen_settings.get_guidance_scale(),
            input_image_paths=self.gen_settings.get_i2i_image_paths(),
            bot_task=self.system_bar.get_global_bot_task(),
            drop_think=self.system_bar.get_global_drop_think(),
        )

        self.gen_settings.set_generating(True)
        self.gen_settings.progress.start(params.batch_count)
        self.status_bar.showMessage("Generating...")

        self.gen_cot_display.clear()

        self._gen_worker = GenerationWorker(params)
        self._gen_worker.progress.connect(self._on_gen_progress)
        self._gen_worker.image_generated.connect(self._on_gen_image)
        self._gen_worker.cot_received.connect(self._on_gen_cot)
        self._gen_worker.completed.connect(self._on_gen_completed)
        self._gen_worker.error.connect(self._on_gen_error)
        self._gen_worker.start()

    @Slot(int, int, str)
    def _on_gen_progress(self, current, total, status):
        self.gen_settings.progress.update_progress(current, total, status)
        self.status_bar.showMessage(status)

    @Slot(str, int, float)
    def _on_gen_image(self, filepath, seed, gen_time):
        """Handle a generated image."""
        self.output_panel.display_image(filepath, seed, gen_time)
        self.gallery_panel.add_thumbnail(filepath)
        # Point gallery at the session directory so Browse/Refresh works
        from pathlib import Path
        session_dir = str(Path(filepath).parent)
        self.gallery_panel.set_directory_label(session_dir)
        self.gen_settings.set_seed(seed)
        self.status_bar.showMessage(
            f"Generated: {filepath} (seed: {seed}, {gen_time:.1f}s)"
        )

    @Slot(int)
    def _on_gen_completed(self, total_count):
        self.gen_settings.set_generating(False)
        self.gen_settings.progress.complete(
            f"Done! {total_count} images generated"
        )
        self.status_bar.showMessage(
            f"Generation complete: {total_count} images"
        )

    @Slot(str)
    def _on_gen_error(self, error_msg):
        self.gen_settings.set_generating(False)
        self.gen_settings.progress.reset()
        self.status_bar.showMessage(f"Error: {error_msg[:100]}")
        QMessageBox.warning(self, "Generation Error", error_msg)

    @Slot(str)
    def _on_gen_cot(self, text):
        """Display chain-of-thought reasoning in Generate tab."""
        self.gen_cot_display.setText(text)
        self.gen_cot_group.setChecked(True)

    # ─── Edit Tab Generation ──────────────────────────────────

    @Slot()
    def _on_edit_generate(self):
        """Start I2I generation from Edit tab."""
        if self._gen_worker and self._gen_worker.isRunning():
            self.status_bar.showMessage("Generation already in progress")
            return

        # Collect images (up to 3 for FireRed)
        image_paths = []
        for drop in (self.edit_image1_drop, self.edit_image2_drop, self.edit_image3_drop):
            p = drop.get_image_path()
            if p:
                image_paths.append(p)

        if not image_paths:
            self.status_bar.showMessage("Drop or select at least one reference image")
            return

        prompt = self.edit_prompt.toPlainText().strip()
        if not prompt:
            self.status_bar.showMessage("Enter an edit instruction")
            return

        import random
        seed = self.edit_seed.value()
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)

        # Use global mode from system bar
        from ui.state import get_state
        edit_state = get_state()
        params = GenerationParams(
            prompt=prompt,
            seed=seed,
            batch_count=1,
            aspect_ratio="Auto (Model decides)",
            guidance_scale=self.edit_guidance.value() / 10.0,
            input_image_paths=image_paths,
            bot_task=self.system_bar.get_global_bot_task(),
            drop_think=self.system_bar.get_global_drop_think(),
            use_ollama=self.edit_enhance_check.isChecked(),
            ollama_length=self.edit_ollama_length.currentText(),
            ollama_complexity=self.edit_ollama_complexity.currentText(),
        )

        self.edit_generate_btn.setEnabled(False)
        self.edit_generate_btn.setText("Editing...")
        self.edit_progress.start(1)
        self.edit_cot_display.clear()
        self.status_bar.showMessage("Editing image...")

        self._gen_worker = GenerationWorker(params)
        self._gen_worker.progress.connect(
            lambda c, t, s: self.edit_progress.update_progress(c, t, s)
        )
        self._gen_worker.image_generated.connect(self._on_edit_image_ready)
        self._gen_worker.cot_received.connect(self._on_edit_cot)
        self._gen_worker.completed.connect(self._on_edit_completed)
        self._gen_worker.error.connect(self._on_edit_error)
        self._gen_worker.start()

    @Slot(str, int, float)
    def _on_edit_image_ready(self, filepath, seed, gen_time):
        self.edit_output.display_image(filepath, seed, gen_time)
        self.gallery_panel.add_thumbnail(filepath)
        self.compare_btn.setEnabled(True)
        self.status_bar.showMessage(
            f"Edit complete: {filepath} ({gen_time:.1f}s)"
        )

    @Slot(str)
    def _on_edit_cot(self, text):
        """Display chain-of-thought reasoning."""
        self.edit_cot_display.setText(text)
        self.edit_cot_group.setChecked(True)

    @Slot(int)
    def _on_edit_completed(self, total):
        self.edit_generate_btn.setEnabled(True)
        self.edit_generate_btn.setText("EDIT IMAGE")
        self.edit_progress.complete("Done!")

    @Slot(str)
    def _on_edit_error(self, error_msg):
        self.edit_generate_btn.setEnabled(True)
        self.edit_generate_btn.setText("EDIT IMAGE")
        self.edit_progress.reset()
        self.status_bar.showMessage(f"Edit error: {error_msg[:100]}")
        QMessageBox.warning(self, "Edit Error", error_msg)

    @Slot()
    def _on_compare(self):
        """Open before/after comparison."""
        source = self.edit_image1_drop.get_image_path()
        result = self.edit_output.get_current_path()
        if source and result:
            from widgets.image_compare import ImageCompareDialog
            dialog = ImageCompareDialog(source, result, self)
            dialog.exec()

    # ─── Ollama Enhancement ───────────────────────────────────

    @Slot(str, str, str, str)
    def _on_enhance(self, prompt, model, length, complexity):
        """Start Ollama enhancement in background thread."""
        if self._ollama_worker and self._ollama_worker.isRunning():
            self.status_bar.showMessage("Enhancement already in progress")
            return

        self.status_bar.showMessage("Enhancing prompt...")
        self._ollama_worker = OllamaEnhanceWorker(
            prompt, model, length, complexity
        )
        self._ollama_worker.completed.connect(self._on_enhance_done)
        self._ollama_worker.failed.connect(self._on_enhance_failed)
        self._ollama_worker.start()

    @Slot(str, str)
    def _on_enhance_done(self, original, enhanced):
        self.state.enhancement_completed.emit(original, enhanced)
        self.status_bar.showMessage("Prompt enhanced!")

    @Slot(str)
    def _on_enhance_failed(self, error):
        self.state.enhancement_failed.emit(error)
        self.status_bar.showMessage(f"Enhancement failed: {error[:80]}")

    # ─── Wildcard Insertion ───────────────────────────────────

    @Slot(str)
    def _insert_wildcard(self, tag_text):
        """Insert a wildcard tag into the active prompt editor."""
        current_tab = self.tabs.currentIndex()
        if current_tab == 0:
            self.prompt_panel.prompt_editor.insertPlainText(tag_text)
        elif current_tab == 1:
            self.batch_panel.themes_editor.insertPlainText(tag_text)
        elif current_tab == 2:
            self.edit_prompt.insertPlainText(tag_text)
        elif current_tab == 3:
            self.i2i_batch_panel.prompts_editor.insertPlainText(tag_text)
        # Tab 4 (Dataset Prep) has no single prompt editor for wildcard insertion

    # ─── Gallery ──────────────────────────────────────────────

    @Slot(str, int)
    def _on_batch_completed_gallery(self, batch_dir, total_count):
        """Update the main gallery dock when a batch completes."""
        self.gallery_panel.load_directory(batch_dir)

    @Slot(str)
    def _on_gallery_insert_prompt(self, prompt_text):
        """Insert a prompt from gallery into the active editor."""
        current_tab = self.tabs.currentIndex()
        if current_tab == 0:
            # Generate tab: set prompt
            self.prompt_panel.set_prompt(prompt_text)
            self.status_bar.showMessage("Prompt inserted")
        elif current_tab == 1:
            # Batch tab: append to themes
            existing = self.batch_panel.themes_editor.toPlainText().strip()
            if existing:
                self.batch_panel.themes_editor.setPlainText(
                    existing + "\n" + prompt_text
                )
            else:
                self.batch_panel.themes_editor.setPlainText(prompt_text)
            self.status_bar.showMessage("Prompt appended to batch themes")
        elif current_tab == 2:
            # Edit tab: set edit prompt
            self.edit_prompt.setPlainText(prompt_text)
            self.status_bar.showMessage("Prompt inserted")

    @Slot(str)
    def _on_gallery_image_selected(self, image_path):
        """Show selected gallery image in the output panel."""
        # Pass the gallery's full image list so the output panel's full-size
        # viewer can navigate with arrow keys
        sender = self.sender()
        if hasattr(sender, '_thumbnails'):
            self.output_panel.image_view._image_list = [
                t.image_path for t in sender._thumbnails
            ]
        self.output_panel.display_image(image_path)

    # ─── Keyboard Shortcuts ───────────────────────────────────

    @Slot()
    def _on_generate_shortcut(self):
        """Ctrl+Enter: generate based on current tab."""
        current = self.tabs.currentIndex()
        if current == 0:
            self._on_generate()
        elif current == 1:
            self.batch_panel._on_start()
        elif current == 2:
            self._on_edit_generate()
        elif current == 3:
            self.i2i_batch_panel._on_start()
        elif current == 4:
            self.dataset_prep_panel._on_start()

    @Slot()
    def _on_enhance_shortcut(self):
        """Ctrl+E: enhance the current prompt."""
        if self.tabs.currentIndex() == 0:
            self.prompt_panel._on_enhance()

    @Slot()
    def _on_stop(self):
        """Escape: stop current operation."""
        if self._gen_worker and self._gen_worker.isRunning():
            self._gen_worker.stop()
            self.status_bar.showMessage("Stopping generation...")
        current = self.tabs.currentIndex()
        if current == 1:
            self.batch_panel._on_stop()
        elif current == 3:
            self.i2i_batch_panel._on_stop()
        elif current == 4:
            self.dataset_prep_panel._on_stop()

    @Slot()
    def _on_undo_enhance(self):
        """Ctrl+Z: undo prompt enhancement."""
        if self.tabs.currentIndex() == 0:
            self.prompt_panel._on_undo_enhance()

    # ─── Model State ──────────────────────────────────────────

    @Slot(str, str)
    def _on_model_state_changed(self, model_type, message):
        """Update UI when model loads."""
        from ui.constants import MODEL_INFO, get_default_guidance
        info = MODEL_INFO.get(model_type, {})
        supports_i2i = info.get("supports_img2img", False)
        is_instruct = model_type in ("instruct", "distil", "nf4", "distil_nf4", "instruct_int8", "distil_int8")

        # Show/hide CoT display in Generate tab
        self.gen_cot_group.setVisible(is_instruct)

        # Update Edit tab guidance default to match model
        default_guidance = get_default_guidance(model_type)
        self.edit_guidance.setValue(int(default_guidance * 10))

        # Update Edit tab (index 2), I2I Batch tab (index 3), Dataset Prep tab (index 4)
        if supports_i2i:
            self.tabs.setTabEnabled(2, True)
            self.tabs.setTabToolTip(2, f"Edit - {model_type} mode")
            self.tabs.setTabEnabled(3, True)
            self.tabs.setTabToolTip(3, f"I2I Batch - {model_type} mode")
            self.tabs.setTabEnabled(4, True)
            self.tabs.setTabToolTip(4, f"Dataset Prep - {model_type} mode")
        else:
            self.tabs.setTabEnabled(2, False)
            self.tabs.setTabToolTip(
                2, "Load Instruct, Distil, DeepGen, or FireRed model for editing"
            )
            self.tabs.setTabEnabled(3, False)
            self.tabs.setTabToolTip(
                3, "Load Instruct, Distil, DeepGen, or FireRed model for I2I batch"
            )
            self.tabs.setTabEnabled(4, False)
            self.tabs.setTabToolTip(
                4, "Load Instruct, Distil, DeepGen, or FireRed model for dataset prep"
            )

        self.status_bar.showMessage(f"Model loaded: {model_type}")

    @Slot()
    def _on_model_unloaded(self):
        """Update UI when model unloads."""
        self.tabs.setTabEnabled(2, True)  # Re-enable so user can see notice
        self.tabs.setTabEnabled(3, True)
        self.tabs.setTabEnabled(4, True)
        self.gen_cot_group.setVisible(False)
        self.status_bar.showMessage("Model unloaded")

    # ─── Settings Application ─────────────────────────────────

    @Slot(dict)
    def _apply_loaded_settings(self, metadata):
        """Apply generation settings loaded from an image's JSON sidecar."""
        if "prompt" in metadata:
            self.prompt_panel.set_prompt(metadata["prompt"])
        if "negative_prompt" in metadata:
            self.prompt_panel.negative_prompt.setText(
                metadata["negative_prompt"]
            )
        if "style" in metadata:
            idx = self.prompt_panel.style_combo.findText(metadata["style"])
            if idx >= 0:
                self.prompt_panel.style_combo.setCurrentIndex(idx)
        if "aspect_ratio" in metadata:
            idx = self.gen_settings.aspect_combo.findText(
                metadata["aspect_ratio"]
            )
            if idx >= 0:
                self.gen_settings.aspect_combo.setCurrentIndex(idx)
        if "seed" in metadata:
            self.gen_settings.set_seed(metadata["seed"])
        self.status_bar.showMessage("Settings loaded from image metadata")

    @Slot(int)
    def _on_tab_changed(self, index):
        """Save the current tab index."""
        self.settings.last_tab_index = index

    # ─── Utility Actions ──────────────────────────────────────

    def _refresh_wildcards(self):
        """Reload wildcards from disk."""
        self.state._init_wildcards()
        self.wildcard_sidebar.refresh()
        self.status_bar.showMessage("Wildcards reloaded")

    def _open_output_dir(self):
        """Open the output directory in the system file manager."""
        from ui.constants import OUTPUT_DIR
        import subprocess
        output_path = str(OUTPUT_DIR)
        try:
            subprocess.Popen(["xdg-open", output_path])
        except Exception:
            self.status_bar.showMessage(f"Output dir: {output_path}")

    def _show_about(self):
        QMessageBox.about(
            self,
            "HunyuanImage Desktop",
            "<h3>HunyuanImage Desktop</h3>"
            "<p>PySide6 desktop application for HunyuanImage-3.0</p>"
            "<p>80B MoE model with SDNQ quantization</p>"
            "<p>Features: T2I, I2I, Batch, I2I Batch, Wildcards, "
            "Ollama Enhancement, FireRed Image Edit</p>"
        )

    def _show_guide_dialog(self, title, html):
        """Show a scrollable help dialog with the given HTML content."""
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(820, 700)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QLabel(html)
        content.setWordWrap(True)
        content.setTextFormat(Qt.TextFormat.RichText)
        content.setOpenExternalLinks(True)
        content.setContentsMargins(18, 14, 18, 14)
        content.setStyleSheet("QLabel { font-size: 13px; line-height: 1.5; }")
        scroll.setWidget(content)
        layout.addWidget(scroll)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        btn_layout.addStretch()
        btn_layout.setContentsMargins(8, 4, 8, 8)
        layout.addLayout(btn_layout)

        dialog.exec()

    def _show_firered_guide(self):
        html = (
            "<h2>FireRed Image Edit 1.1 Guide</h2>"

            "<h3>What Is FireRed?</h3>"
            "<p>FireRed is a dedicated image editing model built on the Qwen-Image backbone. "
            "Unlike HunyuanImage's instruct mode which does generation + editing in one model, "
            "FireRed is purpose-built for editing and excels at identity-preserving edits, "
            "multi-image element transfer, makeup, and text overlays.</p>"

            "<h3>How to Use It</h3>"
            "<ol>"
            "<li>Select <b>FireRed 1.1 - Image Edit</b> from the model dropdown</li>"
            "<li>Click <b>Load Model</b> (~30GB+ VRAM required)</li>"
            "<li>Go to the <b>Edit</b> tab</li>"
            "<li>Drop your base image into <b>Reference Image 1</b></li>"
            "<li>Optionally drop a source image into <b>Reference Image 2</b> (for element transfer)</li>"
            "<li>Write an edit instruction and click <b>EDIT IMAGE</b></li>"
            "</ol>"

            "<h3>Prompt Style</h3>"
            "<p>Describe <b>the edit operation</b>, not the final image. Be explicit about "
            "what to keep and what to change.</p>"
            "<ul>"
            "<li>For multi-image: reference as <b>image 1</b> / <b>image 2</b> (or 图1 / 图2)</li>"
            "<li>Bilingual - Chinese and English both work well</li>"
            "<li>Longer, more detailed prompts produce better results</li>"
            "</ul>"

            "<h3>Parameters</h3>"
            "<table cellpadding='4' cellspacing='0' border='1' style='border-collapse:collapse;'>"
            "<tr><th>Parameter</th><th>Default</th><th>Notes</th></tr>"
            "<tr><td>Guidance (true_cfg)</td><td>4.0</td><td>Sweet spot. 1.0-10.0 range</td></tr>"
            "<tr><td>Steps</td><td>40</td><td>30-40 optimal. Below 20 loses quality</td></tr>"
            "<tr><td>Aspect Ratio</td><td>Auto</td><td>Auto matches input image dimensions</td></tr>"
            "<tr><td>Max Input Images</td><td>3</td><td>UI supports 2 slots (Edit tab)</td></tr>"
            "</table>"

            "<hr>"
            "<h3>Single-Image Editing Examples</h3>"

            "<p><b>Portrait Modification:</b></p>"
            "<pre>Change the background to light blue with natural lighting\n"
            "effects. Wear a light beige lace-trimmed blouse. Adjust\n"
            "the hairstyle to have a delicate pearl hairpin on the\n"
            "right side. The face shows a smile.</pre>"

            "<p><b>Full Scene Change:</b></p>"
            "<pre>The character crouching down, wearing a black suit,\n"
            "with hair tied in a high ponytail. She has a blue shoulder\n"
            "bag slung over one shoulder. The background is a scene of\n"
            "blooming cherry trees. Bright colors, vibrant lighting.</pre>"

            "<p><b>Outfit + Props + Background:</b></p>"
            "<pre>Replace the background with an indoor setting featuring\n"
            "white walls, mirrors, and wooden decorations. Change the\n"
            "outfit to a light blue shirt and striped pants. The\n"
            "character holds a bouquet of roses in one hand and a\n"
            "smartphone in the other. Zoom out from the camera.</pre>"

            "<p><b>Camera / Framing:</b></p>"
            "<pre>Adjust the camera to a close-up focusing on the upper\n"
            "body. Change posture to standing sideways. Adjust the\n"
            "overall color tone to cool colors.</pre>"

            "<p><b>Cultural Costume:</b></p>"
            "<pre>Replace the headdress with a hairpin flower, change\n"
            "the clothing to Hanfu, change the background to a lit\n"
            "Chinese clock tower at night, and remove the people\n"
            "in the background.</pre>"

            "<p><b>Quick / Simple Edits:</b></p>"
            "<pre>Transform into anime style\n"
            "Change the background to a sunset beach\n"
            "Replace the cat with a golden retriever\n"
            "Make the person look 20 years older with grey hair\n"
            "Add dramatic rain and wet reflections on the ground</pre>"

            "<hr>"
            "<h3>Multi-Image Editing (Element Transfer)</h3>"
            "<p>Drop a base image in <b>Image 1</b> and a source/reference in <b>Image 2</b>, "
            "then describe what to transfer between them.</p>"

            "<p><b>Virtual Try-On:</b></p>"
            "<pre>Put the outfit from image 2 onto the model in image 1,\n"
            "keeping the original pose, accessories, and background\n"
            "unchanged.</pre>"

            "<p><b>Detailed Outfit Swap:</b></p>"
            "<pre>Replace the white shirt and brown skirt in image 1\n"
            "with the grey-brown hoodie, black striped pants, khaki\n"
            "boots, and matching cloud bag from image 2. Keep the\n"
            "model's pose and background unchanged.</pre>"

            "<p><b>Style / Element Fusion:</b></p>"
            "<pre>Apply the color palette and artistic style from\n"
            "image 2 to the scene in image 1, preserving the\n"
            "composition and subjects.</pre>"

            "<p><b>Object Transfer:</b></p>"
            "<pre>Place the furniture from image 2 into the room\n"
            "shown in image 1, matching the lighting and perspective.</pre>"

            "<hr>"
            "<h3>Makeup Editing</h3>"
            "<p>Detailed cosmetic instructions work best - specify each step:</p>"
            "<pre>Apply makeup: Use ivory matte foundation to even skin\n"
            "tone. Draw thin willow-leaf brows in light brown. Blend\n"
            "light brown eyeshadow deepening at outer corners. Natural\n"
            "black eyeliner. Thick curled false lashes. Bean-paste\n"
            "matte lipstick with defined lip line. Light pink blush\n"
            "on both cheeks. Highlight nose bridge and cheekbones.\n"
            "Contour along the jawline.</pre>"

            "<hr>"
            "<h3>Text / Typography</h3>"
            "<p>Use two images: image 1 = target photo, image 2 = font style reference.</p>"
            "<pre>Add the title text \"ADVENTURE AWAITS\" to image 1,\n"
            "using the font style from image 2. Place horizontally\n"
            "in the lower-left corner with staggered multi-line\n"
            "layout. Avoid covering the main subjects.</pre>"

            "<hr>"
            "<h3>Tips</h3>"
            "<ol>"
            "<li><b>Be surgical</b> - State what to keep (\"maintain the original pose\") "
            "and what to change. Vague prompts cause unwanted edits.</li>"
            "<li><b>Stack edits</b> - Describe all changes in one prompt rather than "
            "editing one thing at a time.</li>"
            "<li><b>Camera language works</b> - \"Zoom in\", \"close-up\", \"low angle\" "
            "are all understood.</li>"
            "<li><b>Lighting changes</b> - \"Adjust lighting to a dimmer indoor atmosphere\" "
            "or \"bright, vibrant lighting\".</li>"
            "<li><b>Use Auto resolution</b> - Avoids unwanted cropping or stretching.</li>"
            "<li><b>Guidance 4.0</b> is the sweet spot. Higher = stricter but may artifact.</li>"
            "<li><b>30-40 steps</b> optimal. Below 20 loses quality, above 40 diminishing returns.</li>"
            "<li><b>Identity preservation</b> - Faces stay consistent even with full outfit, "
            "pose, and background changes.</li>"
            "<li><b>Photo restoration</b> - Try: \"Restore this photo: repair scratches, "
            "enhance colors, sharpen details, remove grain.\"</li>"
            "<li><b>Multi-image: always specify which</b> - Say \"image 1\" and \"image 2\" "
            "to avoid ambiguity.</li>"
            "</ol>"

            "<hr>"
            "<h3>Available LoRAs</h3>"
            "<table cellpadding='4' cellspacing='0' border='1' style='border-collapse:collapse;'>"
            "<tr><th>LoRA</th><th>Purpose</th><th>Steps</th><th>CFG</th></tr>"
            "<tr><td>None (base)</td><td>General editing</td><td>30-40</td><td>4.0</td></tr>"
            "<tr><td>Makeup</td><td>Beauty / cosmetic edits</td><td>30-40</td><td>4.0</td></tr>"
            "<tr><td>CoverCraft</td><td>Typography / text overlays</td><td>30-40</td><td>4.0</td></tr>"
            "<tr><td>Lightning</td><td>Fast inference (distilled)</td><td>8</td><td>1.0</td></tr>"
            "</table>"
            "<p><i>Note: LoRA loading is not yet integrated into the desktop UI.</i></p>"
        )
        self._show_guide_dialog("FireRed Image Edit 1.1 - Editing Guide", html)

    def _show_prompting_guide(self):
        html = (
            "<h2>HunyuanImage 3.0 Prompting Guide</h2>"

            "<h3>Prose, Not Keywords</h3>"
            "<p>HunyuanImage uses an mT5 encoder (512+ tokens). Write <b>full sentences</b>, "
            "not comma-separated keyword lists. No bracket/weight syntax like SD.</p>"

            "<p><b>Good:</b></p>"
            "<pre>A weathered lighthouse keeper in his 60s with a\n"
            "salt-and-pepper beard, wearing a thick navy wool sweater,\n"
            "standing at the top of a spiral staircase inside a\n"
            "Victorian lighthouse. Late afternoon golden hour light\n"
            "streaming through the lantern room windows.</pre>"

            "<p><b>Bad:</b></p>"
            "<pre>lighthouse keeper, old man, beard, sweater, lighthouse,\n"
            "beautiful, amazing, stunning, dramatic, 8k, masterpiece</pre>"

            "<h3>Prompt Structure</h3>"
            "<p><b>Subject</b> + <b>Details</b> + <b>Environment</b> + "
            "<b>Composition</b> + <b>Lighting</b> + <b>Style</b> + <b>Mood</b></p>"

            "<h3>Text Rendering</h3>"
            "<p>Use quotation marks for exact text:</p>"
            "<pre>A coffee shop with the text \"ARTISAN COFFEE\" in\n"
            "bold serif font on a hand-painted wooden sign</pre>"

            "<h3>Negative Concepts</h3>"
            "<p>Embed \"no\" keywords in your prompt (works better than the negative field):</p>"
            "<pre>no watermark, no text, no logo, no extra limbs</pre>"

            "<h3>Useful Vocabulary</h3>"
            "<p><b>Camera:</b> 35mm, 85mm, f/1.4, f/8, low angle, Dutch angle, close-up</p>"
            "<p><b>Lighting:</b> Rembrandt, rim lighting, golden hour, chiaroscuro</p>"
            "<p><b>Materials:</b> brushed aluminum, frosted glass, weathered oak</p>"

            "<h3>Key Settings</h3>"
            "<table cellpadding='4' cellspacing='0' border='1' style='border-collapse:collapse;'>"
            "<tr><th>Setting</th><th>Default</th><th>Notes</th></tr>"
            "<tr><td>Guidance (CFG)</td><td>5.0</td><td>1-3 creative, 4-7 balanced, 8+ strict</td></tr>"
            "<tr><td>Steps</td><td>20</td><td>20 standard, 30-50 max quality</td></tr>"
            "</table>"

            "<p>See <b>HUNYUAN_GUIDE.md</b> for the complete reference.</p>"
        )
        self._show_guide_dialog("HunyuanImage 3.0 - Prompting Guide", html)

    # ─── Project Save/Load ────────────────────────────────────

    def _collect_project_state(self) -> dict:
        """Collect the complete state from all tabs."""
        # Generate tab
        generate_state = {
            "prompt": self.prompt_panel.get_prompt(),
            "negative_prompt": self.prompt_panel.get_negative_prompt(),
            "style": self.prompt_panel.get_style(),
            "ollama_settings": self.prompt_panel.get_ollama_settings(),
            "gen_settings": self.gen_settings.get_state_dict(),
        }

        # Batch tab
        batch_state = self.batch_panel.get_state_dict()

        # Edit tab
        from ui.state import get_state
        state = get_state()
        edit_state = {
            "prompt": self.edit_prompt.toPlainText(),
            "guidance": self.edit_guidance.value() / 10.0,
            "seed": self.edit_seed.value(),
            "bot_task": state.global_bot_task,  # Now uses global mode
            "drop_think": state.global_drop_think,  # Now uses global mode
            "image1": self.edit_image1_drop.get_image_path(),
            "image2": self.edit_image2_drop.get_image_path(),
        }

        # I2I Batch tab
        i2i_batch_state = self.i2i_batch_panel.get_state_dict()

        # Dataset Prep tab
        dataset_prep_state = self.dataset_prep_panel.get_state_dict()

        # LM Studio settings (per-project)
        from core.settings import get_settings
        settings = get_settings()
        lmstudio_state = {
            "url": settings.lmstudio_url,
            "system_prompt": settings.enhance_system_prompt,
        }

        return {
            "version": "1.0",
            "generate": generate_state,
            "batch": batch_state,
            "edit": edit_state,
            "i2i_batch": i2i_batch_state,
            "dataset_prep": dataset_prep_state,
            "lmstudio": lmstudio_state,
        }

    def _apply_project_state(self, data: dict):
        """Apply a complete project state to all tabs."""
        # Generate tab
        if "generate" in data:
            gen = data["generate"]
            if "prompt" in gen:
                self.prompt_panel.set_prompt(gen["prompt"])
            if "negative_prompt" in gen:
                self.prompt_panel.set_negative_prompt(gen["negative_prompt"])
            if "style" in gen:
                self.prompt_panel.set_style(gen["style"])
            if "ollama_settings" in gen:
                self.prompt_panel.set_ollama_settings(gen["ollama_settings"])
            if "gen_settings" in gen:
                self.gen_settings.set_state_dict(gen["gen_settings"])

        # Batch tab
        if "batch" in data:
            self.batch_panel.set_state_dict(data["batch"])

        # Edit tab
        if "edit" in data:
            edit = data["edit"]
            if "prompt" in edit:
                self.edit_prompt.setPlainText(edit["prompt"])
            if "guidance" in edit:
                self.edit_guidance.setValue(int(edit["guidance"] * 10))
            if "seed" in edit:
                self.edit_seed.setValue(edit["seed"])
            # Restore global mode settings from project
            from ui.state import get_state
            state = get_state()
            if "bot_task" in edit:
                state.global_bot_task = edit["bot_task"]
            if "drop_think" in edit:
                state.global_drop_think = edit["drop_think"]
            if "image1" in edit and edit["image1"]:
                from pathlib import Path
                if Path(edit["image1"]).exists():
                    self.edit_image1_drop.set_image(edit["image1"])
            if "image2" in edit and edit["image2"]:
                from pathlib import Path
                if Path(edit["image2"]).exists():
                    self.edit_image2_drop.set_image(edit["image2"])

        # I2I Batch tab
        if "i2i_batch" in data:
            self.i2i_batch_panel.set_state_dict(data["i2i_batch"])

        # Dataset Prep tab
        if "dataset_prep" in data:
            self.dataset_prep_panel.set_state_dict(data["dataset_prep"])

        # LM Studio settings (per-project)
        if "lmstudio" in data:
            from core.settings import get_settings
            settings = get_settings()
            lm = data["lmstudio"]
            if "url" in lm:
                settings.lmstudio_url = lm["url"]
            if "system_prompt" in lm:
                settings.enhance_system_prompt = lm["system_prompt"]

    @Slot()
    def _on_save_project(self):
        """Save current project (quick save if name exists)."""
        if self._current_project_name:
            self._save_project_with_name(self._current_project_name)
        else:
            self._on_save_project_as()

    @Slot()
    def _on_save_project_as(self):
        """Save project with a new name."""
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(
            self, "Save Project As", "Project name:",
            text=self._current_project_name or ""
        )
        if ok and name.strip():
            self._save_project_with_name(name.strip())

    def _save_project_with_name(self, name: str):
        """Internal method to save project with given name."""
        from core.project_manager import save_project
        try:
            state = self._collect_project_state()
            filepath = save_project(name, state)
            self._current_project_name = name
            self.status_bar.showMessage(f"Project saved: {name}")
            self.setWindowTitle(f"HunyuanImage Desktop - {name}")
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Failed to save project: {e}")

    @Slot()
    def _on_load_project(self):
        """Load a saved project (custom dialog with dates, sorting, delete)."""
        from widgets.project_load_dialog import ProjectLoadDialog
        from core.project_manager import load_project

        dlg = ProjectLoadDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        name = dlg.selected_name()
        if not name:
            return
        try:
            data = load_project(name)
            if data:
                self._apply_project_state(data)
                self._current_project_name = name
                display = data.get("_display_name", name)
                self.status_bar.showMessage(f"Project loaded: {display}")
                self.setWindowTitle(f"HunyuanImage Desktop - {display}")
            else:
                QMessageBox.warning(
                    self, "Load Error", f"Project '{name}' not found."
                )
        except Exception as e:
            QMessageBox.warning(
                self, "Load Error", f"Failed to load project: {e}"
            )

    # ─── State Persistence ────────────────────────────────────

    def _restore_state(self):
        """Restore window geometry and UI state from settings."""
        geometry = self.settings.load_geometry()
        if geometry:
            self.restoreGeometry(geometry)

        state = self.settings.load_window_state()
        if state:
            self.restoreState(state)

        # Restore tab
        tab_idx = self.settings.last_tab_index
        if 0 <= tab_idx < self.tabs.count():
            self.tabs.setCurrentIndex(tab_idx)

        # Restore dock visibility
        self.wildcard_dock.setVisible(self.settings.wildcard_sidebar_visible)
        self.gallery_dock.setVisible(self.settings.gallery_visible)

    def closeEvent(self, event):
        """Save state on close, optionally keep model loaded."""
        self.settings.save_geometry(self.saveGeometry())
        self.settings.save_window_state(self.saveState())
        self.settings.last_tab_index = self.tabs.currentIndex()
        self.settings.wildcard_sidebar_visible = self.wildcard_dock.isVisible()
        self.settings.gallery_visible = self.gallery_dock.isVisible()

        # Stop any running workers
        if self._gen_worker and self._gen_worker.isRunning():
            self._gen_worker.stop()
            self._gen_worker.wait(3000)

        if self._ollama_worker and self._ollama_worker.isRunning():
            self._ollama_worker.wait(3000)

        # Silently keep the model in VRAM on quit — reload from the OS page
        # cache is near-instant and avoids interrupting the user with a dialog.
        if self.state.is_model_loaded():
            self.settings.keep_model_loaded = True
            self.settings.last_model_type = self.state.get_model_type()

        # Clean up log panel (restore stdout/stderr)
        if hasattr(self, 'log_panel'):
            self.log_panel.cleanup()

        event.accept()
