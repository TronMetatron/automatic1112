"""Output panel: zoomable image viewer with metadata display."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QTextEdit, QGroupBox,
    QSplitter, QPushButton, QDialog
)
from PySide6.QtGui import QPixmap, QWheelEvent, QKeySequence, QShortcut
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
        self._current_path = ""

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setBackgroundBrush(Qt.GlobalColor.black)
        self.setMinimumSize(300, 300)
        self._image_list = []  # Set by gallery for navigation in full-size viewer

    def wheelEvent(self, event: QWheelEvent):
        """Zoom in/out with mouse wheel."""
        factor = 1.15
        if event.angleDelta().y() < 0:
            factor = 1.0 / factor

        self._zoom_level *= factor
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event):
        """Open full-size image viewer on double-click with navigation."""
        if self._current_path:
            from widgets.gallery_panel import FullImageDialog
            dialog = FullImageDialog(
                self._current_path, self, image_list=self._image_list or None
            )
            dialog.exec()
        else:
            super().mouseDoubleClickEvent(event)

    def set_image(self, image_path: str):
        """Load and display an image."""
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return

        self._current_path = image_path
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
        self._current_path = ""

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


class CompareDialog(QDialog):
    """Side-by-side image comparison dialog."""

    def __init__(self, path_a: str, path_b: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Compare Images")
        self.setMinimumSize(1000, 600)
        self.resize(1400, 800)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Side-by-side views
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._view_a = self._make_view(path_a)
        self._view_b = self._make_view(path_b)
        splitter.addWidget(self._view_a["widget"])
        splitter.addWidget(self._view_b["widget"])
        splitter.setSizes([700, 700])

        layout.addWidget(splitter, stretch=1)

        # Info bar
        info = QHBoxLayout()
        info.addWidget(QLabel(f"A: {Path(path_a).name}"))
        info.addStretch()

        swap_btn = QPushButton("Swap")
        swap_btn.setMaximumWidth(60)
        swap_btn.clicked.connect(lambda: self._swap(path_a, path_b))
        info.addWidget(swap_btn)

        info.addStretch()
        info.addWidget(QLabel(f"B: {Path(path_b).name}"))
        layout.addLayout(info)

        # Metadata comparison
        meta_a = self._load_metadata(path_a)
        meta_b = self._load_metadata(path_b)
        if meta_a or meta_b:
            diff_label = QLabel(self._format_diff(meta_a, meta_b))
            diff_label.setStyleSheet("font-size: 11px; color: #c0c0c0; padding: 4px;")
            diff_label.setWordWrap(True)
            layout.addWidget(diff_label)

        close_shortcut = QShortcut(QKeySequence("Escape"), self)
        close_shortcut.activated.connect(self.close)

    def _make_view(self, path):
        container = QWidget()
        vlayout = QVBoxLayout(container)
        vlayout.setContentsMargins(0, 0, 0, 0)

        scene = QGraphicsScene()
        view = QGraphicsView(scene)
        view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        view.setBackgroundBrush(Qt.GlobalColor.black)

        pixmap = QPixmap(path)
        item = None
        if not pixmap.isNull():
            item = scene.addPixmap(pixmap)

        # Override wheel for zoom
        original_wheel = view.wheelEvent
        def zoom_wheel(event):
            factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
            view.scale(factor, factor)
        view.wheelEvent = zoom_wheel

        vlayout.addWidget(view)
        return {"widget": container, "view": view, "scene": scene, "item": item}

    def showEvent(self, event):
        super().showEvent(event)
        for v in (self._view_a, self._view_b):
            if v["item"]:
                v["view"].fitInView(v["item"], Qt.AspectRatioMode.KeepAspectRatio)

    def _swap(self, path_a, path_b):
        """Reload with swapped images (quick visual swap)."""
        # Just swap the pixmaps in the existing views
        pix_a = QPixmap(path_b)
        pix_b = QPixmap(path_a)
        for v, pix in ((self._view_a, pix_a), (self._view_b, pix_b)):
            v["scene"].clear()
            if not pix.isNull():
                v["item"] = v["scene"].addPixmap(pix)
                v["view"].fitInView(v["item"], Qt.AspectRatioMode.KeepAspectRatio)

    def _load_metadata(self, path):
        json_path = Path(path).with_suffix(".json")
        if json_path.exists():
            try:
                with open(json_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _format_diff(self, meta_a, meta_b):
        """Show key differences between two images."""
        keys = ["seed", "steps", "quality", "aspect_ratio", "style",
                "model_type", "bot_task", "generation_time"]
        parts = []
        for k in keys:
            va = meta_a.get(k, "--")
            vb = meta_b.get(k, "--")
            if isinstance(va, float):
                va = f"{va:.1f}"
            if isinstance(vb, float):
                vb = f"{vb:.1f}"
            if va != vb:
                parts.append(f"{k}: {va} vs {vb}")
            else:
                parts.append(f"{k}: {va}")
        return "  |  ".join(parts)


class OutputPanel(QWidget):
    """Image display with zoom/pan, info bar, compare, and metadata viewer."""

    image_loaded = Signal(str)  # image_path
    load_settings_requested = Signal(dict)  # metadata dict

    def __init__(self, desktop_state, parent=None):
        super().__init__(parent)
        self.state = desktop_state
        self._current_path = ""
        self._compare_slot = ""  # path stored for comparison
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

        # Compare button
        self.compare_btn = QPushButton("Compare")
        self.compare_btn.setToolTip(
            "Click to mark this image for comparison, then view another and click again"
        )
        self.compare_btn.clicked.connect(self._on_compare)
        info_layout.addWidget(self.compare_btn)

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
    def _on_compare(self):
        """Two-click compare: first click stores image A, second opens side-by-side."""
        if not self._current_path:
            return

        if not self._compare_slot:
            # First click: store this image
            self._compare_slot = self._current_path
            self.compare_btn.setText(f"Compare with: {Path(self._current_path).name[:20]}...")
            self.compare_btn.setToolTip(
                "Now navigate to another image and click Compare again to view side-by-side"
            )
        else:
            # Second click: open comparison
            if self._compare_slot != self._current_path:
                dialog = CompareDialog(self._compare_slot, self._current_path, self)
                dialog.exec()
            # Reset
            self._compare_slot = ""
            self.compare_btn.setText("Compare")
            self.compare_btn.setToolTip(
                "Click to mark this image for comparison, then view another and click again"
            )

    @Slot()
    def _on_load_settings(self):
        """Load settings from current image's metadata."""
        if self._current_path:
            metadata = self.metadata_viewer.load_from_image(self._current_path)
            if metadata:
                self.load_settings_requested.emit(metadata)

    def get_current_path(self) -> str:
        return self._current_path
