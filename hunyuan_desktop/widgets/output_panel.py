"""Output panel: zoomable image viewer with metadata display."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QTextEdit, QGroupBox,
    QSplitter, QPushButton
)
from PySide6.QtGui import QPixmap, QWheelEvent
from PySide6.QtCore import Qt, Signal, Slot

import json
from pathlib import Path


class ZoomableImageView(QGraphicsView):
    """QGraphicsView with mouse wheel zoom and click-drag pan."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = None
        self._zoom_level = 1.0

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setBackgroundBrush(Qt.GlobalColor.black)
        self.setMinimumSize(300, 300)

    def wheelEvent(self, event: QWheelEvent):
        """Zoom in/out with mouse wheel."""
        factor = 1.15
        if event.angleDelta().y() < 0:
            factor = 1.0 / factor

        self._zoom_level *= factor
        self.scale(factor, factor)

    def set_image(self, image_path: str):
        """Load and display an image."""
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return

        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._zoom_level = 1.0
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def set_pixmap(self, pixmap: QPixmap):
        """Display a QPixmap directly."""
        if pixmap.isNull():
            return
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._zoom_level = 1.0
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def clear(self):
        self._scene.clear()
        self._pixmap_item = None

    def fit_to_view(self):
        """Reset zoom to fit image in view."""
        if self._pixmap_item:
            self._zoom_level = 1.0
            self.resetTransform()
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)


class MetadataViewer(QGroupBox):
    """Collapsible JSON metadata display."""

    def __init__(self, parent=None):
        super().__init__("Metadata", parent)
        self.setCheckable(True)
        self.setChecked(False)

        layout = QVBoxLayout()
        self.text_view = QTextEdit()
        self.text_view.setReadOnly(True)
        self.text_view.setMaximumHeight(200)
        self.text_view.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(self.text_view)
        self.setLayout(layout)

        self.toggled.connect(self._on_toggle)

    def _on_toggle(self, checked):
        self.text_view.setVisible(checked)

    def set_metadata(self, metadata: dict):
        """Display metadata as formatted JSON."""
        self.text_view.setPlainText(json.dumps(metadata, indent=2, default=str))

    def clear(self):
        self.text_view.clear()

    def load_from_image(self, image_path: str):
        """Load metadata from the JSON sidecar of an image."""
        json_path = Path(image_path).with_suffix(".json")
        if json_path.exists():
            try:
                with open(json_path) as f:
                    data = json.load(f)
                self.set_metadata(data)
                return data
            except Exception:
                pass
        self.clear()
        return {}


class OutputPanel(QWidget):
    """Image display with zoom/pan, info bar, and metadata viewer."""

    image_loaded = Signal(str)  # image_path
    load_settings_requested = Signal(dict)  # metadata dict

    def __init__(self, desktop_state, parent=None):
        super().__init__(parent)
        self.state = desktop_state
        self._current_path = ""
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Image viewer
        self.image_view = ZoomableImageView()
        layout.addWidget(self.image_view, stretch=5)

        # Info bar
        info_layout = QHBoxLayout()
        self.seed_label = QLabel("Seed: --")
        info_layout.addWidget(self.seed_label)
        self.size_label = QLabel("Size: --")
        info_layout.addWidget(self.size_label)
        self.time_label = QLabel("Time: --")
        info_layout.addWidget(self.time_label)
        self.steps_label = QLabel("Steps: --")
        info_layout.addWidget(self.steps_label)
        info_layout.addStretch()

        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setMaximumWidth(40)
        self.fit_btn.clicked.connect(self.image_view.fit_to_view)
        info_layout.addWidget(self.fit_btn)

        self.load_settings_btn = QPushButton("Load Settings")
        self.load_settings_btn.setToolTip("Load generation settings from this image")
        self.load_settings_btn.clicked.connect(self._on_load_settings)
        self.load_settings_btn.setEnabled(False)
        info_layout.addWidget(self.load_settings_btn)

        layout.addLayout(info_layout)

        # Metadata viewer (collapsible)
        self.metadata_viewer = MetadataViewer()
        layout.addWidget(self.metadata_viewer)

    def display_image(self, image_path: str, seed: int = 0, gen_time: float = 0.0):
        """Display a generated image with its info."""
        self._current_path = image_path
        self.image_view.set_image(image_path)

        # Update info labels
        self.seed_label.setText(f"Seed: {seed}")
        self.time_label.setText(f"Time: {gen_time:.1f}s" if gen_time > 0 else "Time: --")

        # Try to load full metadata from sidecar
        metadata = self.metadata_viewer.load_from_image(image_path)
        if metadata:
            self.size_label.setText(f"Size: {metadata.get('image_size', '--')}")
            self.steps_label.setText(f"Steps: {metadata.get('steps', '--')}")
            self.load_settings_btn.setEnabled(True)
        else:
            self.size_label.setText("Size: --")
            self.steps_label.setText("Steps: --")
            self.load_settings_btn.setEnabled(False)

        self.image_loaded.emit(image_path)

    def display_pixmap(self, pixmap: QPixmap):
        """Display a QPixmap directly."""
        self.image_view.set_pixmap(pixmap)

    def clear(self):
        """Clear the display."""
        self.image_view.clear()
        self.seed_label.setText("Seed: --")
        self.size_label.setText("Size: --")
        self.time_label.setText("Time: --")
        self.steps_label.setText("Steps: --")
        self.metadata_viewer.clear()
        self.load_settings_btn.setEnabled(False)
        self._current_path = ""

    @Slot()
    def _on_load_settings(self):
        """Load settings from current image's metadata."""
        if self._current_path:
            metadata = self.metadata_viewer.load_from_image(self._current_path)
            if metadata:
                self.load_settings_requested.emit(metadata)

    def get_current_path(self) -> str:
        return self._current_path
