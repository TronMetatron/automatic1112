"""Dark theme for the HunyuanImage Desktop application."""

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import Qt


# Color constants
BG_DARK = QColor(26, 26, 26)         # #1a1a1a
BG_MID = QColor(35, 35, 35)          # #232323
BG_LIGHT = QColor(45, 45, 45)        # #2d2d2d
BG_HOVER = QColor(55, 55, 55)        # #373737
TEXT_PRIMARY = QColor(224, 224, 224)  # #e0e0e0
TEXT_SECONDARY = QColor(160, 160, 160)  # #a0a0a0
TEXT_DISABLED = QColor(100, 100, 100)  # #646464
ACCENT = QColor(76, 175, 80)         # #4caf50 - Green
ACCENT_HOVER = QColor(102, 187, 106) # #66bb6a - Light green
SUCCESS = QColor(102, 187, 106)      # #66bb6a
ERROR = QColor(239, 83, 80)          # #ef5350
WARNING = QColor(255, 183, 77)       # #ffb74d
BORDER = QColor(60, 60, 60)          # #3c3c3c


def apply_dark_theme(app: QApplication):
    """Apply a modern dark theme to the application."""
    palette = QPalette()

    # Window backgrounds
    palette.setColor(QPalette.ColorRole.Window, BG_DARK)
    palette.setColor(QPalette.ColorRole.WindowText, TEXT_PRIMARY)
    palette.setColor(QPalette.ColorRole.Base, BG_MID)
    palette.setColor(QPalette.ColorRole.AlternateBase, BG_LIGHT)

    # Text
    palette.setColor(QPalette.ColorRole.Text, TEXT_PRIMARY)
    palette.setColor(QPalette.ColorRole.PlaceholderText, TEXT_DISABLED)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.white)

    # Buttons
    palette.setColor(QPalette.ColorRole.Button, BG_LIGHT)
    palette.setColor(QPalette.ColorRole.ButtonText, TEXT_PRIMARY)

    # Highlights / selections
    palette.setColor(QPalette.ColorRole.Highlight, ACCENT)
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)

    # Tooltips
    palette.setColor(QPalette.ColorRole.ToolTipBase, BG_LIGHT)
    palette.setColor(QPalette.ColorRole.ToolTipText, TEXT_PRIMARY)

    # Links
    palette.setColor(QPalette.ColorRole.Link, ACCENT)
    palette.setColor(QPalette.ColorRole.LinkVisited, QColor(130, 130, 200))

    # Disabled state
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, TEXT_DISABLED)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, TEXT_DISABLED)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, TEXT_DISABLED)

    app.setPalette(palette)
    app.setStyleSheet(get_stylesheet())


def get_stylesheet() -> str:
    """Return the QSS stylesheet for fine-grained widget styling."""
    return """
    /* Global */
    QWidget {
        font-size: 13px;
    }

    /* Tool tips */
    QToolTip {
        background-color: #2d2d2d;
        color: #e0e0e0;
        border: 1px solid #3c3c3c;
        padding: 4px;
        border-radius: 3px;
    }

    /* Push buttons */
    QPushButton {
        background-color: #2d2d2d;
        color: #e0e0e0;
        border: 1px solid #3c3c3c;
        border-radius: 4px;
        padding: 6px 16px;
        min-height: 24px;
    }
    QPushButton:hover {
        background-color: #373737;
        border-color: #4caf50;
    }
    QPushButton:pressed {
        background-color: #4caf50;
    }
    QPushButton:disabled {
        background-color: #1e1e1e;
        color: #646464;
        border-color: #2a2a2a;
    }
    QPushButton#generate_btn, QPushButton#start_batch_btn, QPushButton#edit_btn {
        background-color: #2a5a2a;
        border-color: #3a7a3a;
        font-weight: bold;
        font-size: 14px;
        min-height: 36px;
    }
    QPushButton#generate_btn:hover, QPushButton#start_batch_btn:hover, QPushButton#edit_btn:hover {
        background-color: #3a7a3a;
    }
    QPushButton#stop_btn {
        background-color: #5a2a2a;
        border-color: #7a3a3a;
    }
    QPushButton#stop_btn:hover {
        background-color: #7a3a3a;
    }

    /* Text inputs */
    QLineEdit, QSpinBox, QDoubleSpinBox {
        background-color: #232323;
        color: #e0e0e0;
        border: 1px solid #3c3c3c;
        border-radius: 3px;
        padding: 4px 8px;
        min-height: 22px;
        selection-background-color: #4caf50;
    }
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
        border-color: #4caf50;
    }

    /* Plain text edit (prompt editors) */
    QPlainTextEdit, QTextEdit {
        background-color: #1e1e1e;
        color: #e0e0e0;
        border: 1px solid #3c3c3c;
        border-radius: 3px;
        selection-background-color: #4caf50;
    }
    QPlainTextEdit:focus, QTextEdit:focus {
        border-color: #4caf50;
    }

    /* Combo boxes */
    QComboBox {
        background-color: #2d2d2d;
        color: #e0e0e0;
        border: 1px solid #3c3c3c;
        border-radius: 3px;
        padding: 4px 8px;
        min-height: 22px;
    }
    QComboBox:hover {
        border-color: #4caf50;
    }
    QComboBox::drop-down {
        border: none;
        width: 20px;
    }
    QComboBox QAbstractItemView {
        background-color: #2d2d2d;
        color: #e0e0e0;
        border: 1px solid #3c3c3c;
        selection-background-color: #4caf50;
    }

    /* Check boxes */
    QCheckBox {
        spacing: 8px;
        color: #e0e0e0;
    }
    QCheckBox::indicator {
        width: 16px;
        height: 16px;
        border: 1px solid #3c3c3c;
        border-radius: 3px;
        background-color: #232323;
    }
    QCheckBox::indicator:checked {
        background-color: #4caf50;
        border-color: #4caf50;
    }

    /* Sliders */
    QSlider::groove:horizontal {
        background: #2d2d2d;
        height: 6px;
        border-radius: 3px;
    }
    QSlider::handle:horizontal {
        background: #4caf50;
        width: 16px;
        height: 16px;
        margin: -5px 0;
        border-radius: 8px;
    }
    QSlider::handle:horizontal:hover {
        background: #66bb6a;
    }
    QSlider::sub-page:horizontal {
        background: #4caf50;
        border-radius: 3px;
    }

    /* Progress bars */
    QProgressBar {
        background-color: #232323;
        border: 1px solid #3c3c3c;
        border-radius: 4px;
        text-align: center;
        color: #e0e0e0;
        min-height: 20px;
    }
    QProgressBar::chunk {
        background-color: #4caf50;
        border-radius: 3px;
    }

    /* Tab widget */
    QTabWidget::pane {
        border: 1px solid #3c3c3c;
        background-color: #1a1a1a;
    }
    QTabBar::tab {
        background-color: #232323;
        color: #a0a0a0;
        padding: 8px 20px;
        border: 1px solid #3c3c3c;
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        margin-right: 2px;
    }
    QTabBar::tab:selected {
        background-color: #1a1a1a;
        color: #e0e0e0;
        border-bottom: 2px solid #4caf50;
    }
    QTabBar::tab:hover:!selected {
        background-color: #2d2d2d;
        color: #e0e0e0;
    }

    /* Dock widgets */
    QDockWidget {
        titlebar-close-icon: none;
        color: #e0e0e0;
    }
    QDockWidget::title {
        background-color: #232323;
        padding: 6px;
        border: 1px solid #3c3c3c;
    }

    /* Main Window Title Bar styling */
    QMainWindow {
        background-color: #1a1a1a;
    }

    /* Toolbar - Blue accent to stand out */
    QToolBar {
        background-color: #1a3a5c;
        border-bottom: 2px solid #2a5a8c;
        spacing: 6px;
        padding: 6px;
    }
    QToolBar#system_bar {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #1e4a7a, stop:1 #1a3a5c);
        border-bottom: 2px solid #3a7acc;
    }
    QToolBar QLabel {
        color: #e0e0e0;
        margin-right: 4px;
        font-weight: bold;
    }

    /* Scroll bars */
    QScrollBar:vertical {
        background: #1a1a1a;
        width: 10px;
        margin: 0;
    }
    QScrollBar::handle:vertical {
        background: #3c3c3c;
        border-radius: 5px;
        min-height: 30px;
    }
    QScrollBar::handle:vertical:hover {
        background: #4caf50;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0;
    }
    QScrollBar:horizontal {
        background: #1a1a1a;
        height: 10px;
        margin: 0;
    }
    QScrollBar::handle:horizontal {
        background: #3c3c3c;
        border-radius: 5px;
        min-width: 30px;
    }
    QScrollBar::handle:horizontal:hover {
        background: #4caf50;
    }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0;
    }

    /* Tree widget (wildcard sidebar) */
    QTreeWidget {
        background-color: #1e1e1e;
        color: #e0e0e0;
        border: 1px solid #3c3c3c;
        border-radius: 3px;
    }
    QTreeWidget::item {
        padding: 3px 0;
    }
    QTreeWidget::item:hover {
        background-color: #2d2d2d;
    }
    QTreeWidget::item:selected {
        background-color: #4caf50;
    }

    /* List widget (style checkboxes, etc.) */
    QListWidget {
        background-color: #1e1e1e;
        color: #e0e0e0;
        border: 1px solid #3c3c3c;
        border-radius: 3px;
    }
    QListWidget::item {
        padding: 4px;
    }
    QListWidget::item:hover {
        background-color: #2d2d2d;
    }
    QListWidget::item:selected {
        background-color: #4caf50;
    }

    /* Group boxes */
    QGroupBox {
        border: 1px solid #3c3c3c;
        border-radius: 4px;
        margin-top: 12px;
        padding-top: 12px;
        font-weight: bold;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
        color: #4caf50;
    }

    /* Splitter */
    QSplitter::handle {
        background-color: #3c3c3c;
    }
    QSplitter::handle:horizontal {
        width: 2px;
    }
    QSplitter::handle:vertical {
        height: 2px;
    }

    /* Menu bar - Blue to match toolbar */
    QMenuBar {
        background-color: #1a3a5c;
        color: #e0e0e0;
        border-bottom: 1px solid #2a5a8c;
        font-weight: bold;
    }
    QMenuBar::item {
        padding: 6px 12px;
    }
    QMenuBar::item:selected {
        background-color: #2a6aac;
    }
    QMenu {
        background-color: #2d2d2d;
        color: #e0e0e0;
        border: 1px solid #3c3c3c;
    }
    QMenu::item:selected {
        background-color: #4caf50;
    }
    QMenu::separator {
        height: 1px;
        background: #3c3c3c;
        margin: 4px 8px;
    }

    /* Status bar */
    QStatusBar {
        background-color: #232323;
        color: #a0a0a0;
        border-top: 1px solid #3c3c3c;
    }
    """
