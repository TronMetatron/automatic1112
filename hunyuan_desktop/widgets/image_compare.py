"""Side-by-side image comparison dialog for before/after editing."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QSplitter, QWidget
)
from PySide6.QtGui import QPixmap, QPainter
from PySide6.QtCore import Qt, Signal


class ImageCompareDialog(QDialog):
    """Modal dialog showing before/after image comparison with slider."""

    def __init__(self, before_path: str, after_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Compare Before / After")
        self.setMinimumSize(900, 600)
        self.resize(1200, 800)

        self._before_path = before_path
        self._after_path = after_path
        self._before_pixmap = QPixmap(before_path)
        self._after_pixmap = QPixmap(after_path)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Side-by-side view
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Before
        before_widget = QWidget()
        before_layout = QVBoxLayout(before_widget)
        before_layout.setContentsMargins(0, 0, 0, 0)
        before_layout.addWidget(QLabel("BEFORE"))
        self._before_label = QLabel()
        self._before_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._before_label.setStyleSheet("background-color: #1a1a1a;")
        before_layout.addWidget(self._before_label, stretch=1)
        splitter.addWidget(before_widget)

        # After
        after_widget = QWidget()
        after_layout = QVBoxLayout(after_widget)
        after_layout.setContentsMargins(0, 0, 0, 0)
        after_layout.addWidget(QLabel("AFTER"))
        self._after_label = QLabel()
        self._after_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._after_label.setStyleSheet("background-color: #1a1a1a;")
        after_layout.addWidget(self._after_label, stretch=1)
        splitter.addWidget(after_widget)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, stretch=1)

        # Opacity slider for overlay comparison
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Blend:"))
        self._blend_slider = QSlider(Qt.Orientation.Horizontal)
        self._blend_slider.setRange(0, 100)
        self._blend_slider.setValue(50)
        self._blend_slider.setToolTip("Slide to blend between before and after")
        slider_row.addWidget(self._blend_slider)
        self._blend_label = QLabel("50%")
        self._blend_label.setMinimumWidth(35)
        slider_row.addWidget(self._blend_label)
        layout.addLayout(slider_row)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        # Connect slider
        self._blend_slider.valueChanged.connect(self._on_blend_changed)

        # Initial display
        self._update_images()

    def _update_images(self):
        """Scale and display both images."""
        max_w = max(300, self.width() // 2 - 20)
        max_h = max(300, self.height() - 120)

        if not self._before_pixmap.isNull():
            scaled = self._before_pixmap.scaled(
                max_w, max_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self._before_label.setPixmap(scaled)

        if not self._after_pixmap.isNull():
            scaled = self._after_pixmap.scaled(
                max_w, max_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self._after_label.setPixmap(scaled)

    def _on_blend_changed(self, value):
        """Update blend display."""
        self._blend_label.setText(f"{value}%")
        # Could implement pixel-level blending here if needed

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_images()
