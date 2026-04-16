"""Thumbnail gallery panel with flow layout."""

from pathlib import Path
from typing import List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QScrollArea, QLayout, QFileDialog, QSizePolicy,
    QMenu, QDialog, QGraphicsView, QGraphicsScene
)
from PySide6.QtGui import QPixmap, QWheelEvent, QKeySequence, QShortcut, QDrag
from PySide6.QtCore import Qt, Signal, Slot, QRect, QSize, QPoint, QUrl, QMimeData


class FlowLayout(QLayout):
    """Flow layout that wraps items like a word processor wraps text."""

    def __init__(self, parent=None, margin=4, spacing=4):
        super().__init__(parent)
        self._items = []
        self._margin = margin
        self._spacing = spacing

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self._margin
        size += QSize(2 * m, 2 * m)
        return size

    def _do_layout(self, rect, test_only=False):
        x = rect.x() + self._margin
        y = rect.y() + self._margin
        line_height = 0

        for item in self._items:
            item_size = item.sizeHint()
            next_x = x + item_size.width() + self._spacing

            if next_x - self._spacing > rect.right() and line_height > 0:
                x = rect.x() + self._margin
                y = y + line_height + self._spacing
                next_x = x + item_size.width() + self._spacing
                line_height = 0

            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item_size))

            x = next_x
            line_height = max(line_height, item_size.height())

        return y + line_height - rect.y() + self._margin

    def clear_all(self):
        """Remove all items from the layout."""
        while self.count():
            item = self.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


class FullImageDialog(QDialog):
    """Full-size image viewer dialog with zoom, pan, and arrow-key navigation."""

    def __init__(self, image_path: str, parent=None, image_list: list = None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.resize(1200, 900)

        # Image list for navigation
        self._image_list = image_list or [image_path]
        self._current_index = 0
        if image_path in self._image_list:
            self._current_index = self._image_list.index(image_path)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._scene = QGraphicsScene(self)
        self._view = QGraphicsView(self._scene)
        self._view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._view.setBackgroundBrush(Qt.GlobalColor.black)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._view.wheelEvent = self._wheel_zoom
        self._pixmap_item = None

        layout.addWidget(self._view)

        # Info / nav bar
        info = QHBoxLayout()
        self._prev_btn = QPushButton("<")
        self._prev_btn.setMaximumWidth(30)
        self._prev_btn.clicked.connect(self._prev_image)
        info.addWidget(self._prev_btn)

        self._info_label = QLabel()
        info.addWidget(self._info_label, stretch=1)

        self._next_btn = QPushButton(">")
        self._next_btn.setMaximumWidth(30)
        self._next_btn.clicked.connect(self._next_image)
        info.addWidget(self._next_btn)

        fit_btn = QPushButton("Fit")
        fit_btn.setMaximumWidth(40)
        fit_btn.clicked.connect(self._fit)
        info.addWidget(fit_btn)
        actual_btn = QPushButton("1:1")
        actual_btn.setMaximumWidth(40)
        actual_btn.clicked.connect(self._actual_size)
        info.addWidget(actual_btn)
        layout.addLayout(info)

        # Load initial image
        self._load_image(self._current_index)

        # Keyboard shortcuts
        close_shortcut = QShortcut(QKeySequence("Escape"), self)
        close_shortcut.activated.connect(self.close)

    def _load_image(self, index: int):
        """Load and display the image at the given index."""
        if not self._image_list or index < 0 or index >= len(self._image_list):
            return
        self._current_index = index
        image_path = self._image_list[index]

        self._scene.clear()
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self._pixmap_item = self._scene.addPixmap(pixmap)
            self._view.resetTransform()
            self._view.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            dims = f"{pixmap.width()}x{pixmap.height()}"
        else:
            self._pixmap_item = None
            dims = ""

        name = Path(image_path).name
        total = len(self._image_list)
        self.setWindowTitle(f"{name} ({index + 1}/{total})")
        self._info_label.setText(f"  {name}  {dims}  [{index + 1}/{total}]")
        self._prev_btn.setEnabled(index > 0)
        self._next_btn.setEnabled(index < total - 1)

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Left, Qt.Key.Key_A):
            self._prev_image()
        elif key in (Qt.Key.Key_Right, Qt.Key.Key_D):
            self._next_image()
        else:
            super().keyPressEvent(event)

    def _prev_image(self):
        if self._current_index > 0:
            self._load_image(self._current_index - 1)

    def _next_image(self):
        if self._current_index < len(self._image_list) - 1:
            self._load_image(self._current_index + 1)

    def showEvent(self, event):
        super().showEvent(event)
        if self._pixmap_item:
            self._view.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def _wheel_zoom(self, event):
        factor = 1.15
        if event.angleDelta().y() < 0:
            factor = 1.0 / factor
        self._view.scale(factor, factor)

    def _fit(self):
        if self._pixmap_item:
            self._view.resetTransform()
            self._view.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def _actual_size(self):
        if self._pixmap_item:
            self._view.resetTransform()


class ThumbnailWidget(QWidget):
    """Clickable thumbnail image with hover effect."""

    clicked = Signal(str)  # image_path
    double_clicked = Signal(str)  # image_path
    context_menu_requested = Signal(str, object)  # image_path, QPoint (global pos)

    def __init__(self, image_path: str, size: int = 150, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self._size = size

        self.setFixedSize(size, size)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(Path(image_path).name)

        self._pixmap = QPixmap(image_path).scaled(
            size - 4, size - 4,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self._hover = False

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QColor

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        if self._hover:
            painter.fillRect(self.rect(), QColor(74, 144, 217, 40))
            painter.setPen(QColor(74, 144, 217))
            painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        else:
            painter.fillRect(self.rect(), QColor(30, 30, 30))

        # Center the pixmap
        if not self._pixmap.isNull():
            x = (self.width() - self._pixmap.width()) // 2
            y = (self.height() - self._pixmap.height()) // 2
            painter.drawPixmap(x, y, self._pixmap)

        painter.end()

    def enterEvent(self, event):
        self._hover = True
        self.update()

    def leaveEvent(self, event):
        self._hover = False
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.pos()
            self.clicked.emit(self.image_path)
        elif event.button() == Qt.MouseButton.RightButton:
            self.context_menu_requested.emit(self.image_path, self.mapToGlobal(event.pos()))

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        if not hasattr(self, '_drag_start_pos'):
            return
        if (event.pos() - self._drag_start_pos).manhattanLength() < 20:
            return

        # Build file list: image + JSON sidecar
        urls = [QUrl.fromLocalFile(self.image_path)]
        json_path = Path(self.image_path).with_suffix(".json")
        if json_path.exists():
            urls.append(QUrl.fromLocalFile(str(json_path)))

        mime = QMimeData()
        mime.setUrls(urls)

        drag = QDrag(self)
        drag.setMimeData(mime)
        # Use thumbnail as drag pixmap
        if not self._pixmap.isNull():
            drag.setPixmap(self._pixmap.scaled(
                80, 80, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
        drag.exec(Qt.DropAction.CopyAction)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.double_clicked.emit(self.image_path)


class GalleryPanel(QWidget):
    """Scrollable thumbnail grid of generated images."""

    image_selected = Signal(str)  # image_path
    insert_prompt_requested = Signal(str)  # prompt text from image metadata
    _compare_slot = ""  # class-level so it persists across context menu calls

    def __init__(self, desktop_state=None, parent=None):
        super().__init__(parent)
        self.state = desktop_state
        self._thumbnails: List[ThumbnailWidget] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Directory bar
        dir_bar = QHBoxLayout()
        self.dir_path = QLineEdit()
        self.dir_path.setPlaceholderText("Output directory...")
        self.dir_path.setReadOnly(True)
        dir_bar.addWidget(self.dir_path, stretch=1)

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self._on_browse)
        dir_bar.addWidget(self.browse_btn)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._on_refresh)
        dir_bar.addWidget(self.refresh_btn)

        self.count_label = QLabel("0 images")
        dir_bar.addWidget(self.count_label)

        layout.addLayout(dir_bar)

        # Scroll area with flow layout
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(100)

        self._container = QWidget()
        self._flow_layout = FlowLayout(self._container)
        self._container.setLayout(self._flow_layout)
        self.scroll_area.setWidget(self._container)

        layout.addWidget(self.scroll_area)

    @Slot()
    def _on_browse(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.load_directory(dir_path)

    @Slot()
    def _on_refresh(self):
        if self.dir_path.text():
            self.load_directory(self.dir_path.text())

    def load_directory(self, directory: str):
        """Load all images from a directory into the gallery."""
        self.dir_path.setText(directory)
        self.clear()

        path = Path(directory)
        if not path.is_dir():
            return

        images = sorted(
            [f for f in path.iterdir()
             if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")],
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )

        for img_path in images:
            self.add_thumbnail(str(img_path))

        self.count_label.setText(f"{len(images)} images")

    def add_thumbnail(self, image_path: str):
        """Add a single thumbnail to the gallery."""
        thumb = ThumbnailWidget(image_path)
        thumb.clicked.connect(self.image_selected.emit)
        thumb.double_clicked.connect(self._on_thumbnail_double_click)
        thumb.context_menu_requested.connect(self._on_thumbnail_context_menu)
        self._thumbnails.append(thumb)
        self._flow_layout.addWidget(thumb)
        self.count_label.setText(f"{len(self._thumbnails)} images")

    def clear(self):
        """Remove all thumbnails."""
        self._flow_layout.clear_all()
        self._thumbnails.clear()
        self.count_label.setText("0 images")

    def set_directory_label(self, directory: str):
        """Set the directory label without loading."""
        self.dir_path.setText(directory)

    def _on_thumbnail_double_click(self, image_path):
        """Open full-size image viewer on double-click with navigation."""
        image_list = [t.image_path for t in self._thumbnails]
        dialog = FullImageDialog(image_path, self, image_list=image_list)
        dialog.exec()

    def _on_thumbnail_context_menu(self, image_path, global_pos):
        """Show context menu for a thumbnail with insert prompt and compare options."""
        import json

        menu = QMenu(self)

        # View full size
        view_action = menu.addAction("View Full Size")
        view_action.triggered.connect(lambda: self._on_thumbnail_double_click(image_path))

        # Compare
        if not GalleryPanel._compare_slot:
            compare_action = menu.addAction("Mark for Compare")
            compare_action.triggered.connect(lambda: self._mark_compare(image_path))
        else:
            if image_path != GalleryPanel._compare_slot:
                compare_action = menu.addAction(
                    f"Compare with {Path(GalleryPanel._compare_slot).name[:25]}..."
                )
                compare_action.triggered.connect(lambda: self._do_compare(image_path))
            clear_compare = menu.addAction("Clear Compare Selection")
            clear_compare.triggered.connect(self._clear_compare)

        menu.addSeparator()

        # Try to load the JSON sidecar
        json_path = Path(image_path).with_suffix(".json")
        prompt_text = None
        full_prompt_text = None
        if json_path.exists():
            try:
                with open(json_path) as f:
                    metadata = json.load(f)
                prompt_text = metadata.get("prompt", "")
                full_prompt_text = metadata.get("full_prompt", "")
            except Exception:
                pass

        if prompt_text:
            insert_action = menu.addAction(f"Insert Prompt: {prompt_text[:50]}...")
            insert_action.triggered.connect(
                lambda checked, p=prompt_text: self.insert_prompt_requested.emit(p)
            )
        if full_prompt_text and full_prompt_text != prompt_text:
            insert_full_action = menu.addAction(f"Insert Enhanced Prompt: {full_prompt_text[:50]}...")
            insert_full_action.triggered.connect(
                lambda checked, p=full_prompt_text: self.insert_prompt_requested.emit(p)
            )
        if not prompt_text and not full_prompt_text:
            no_data = menu.addAction("No prompt metadata found")
            no_data.setEnabled(False)

        menu.exec(global_pos)

    def _mark_compare(self, image_path):
        GalleryPanel._compare_slot = image_path

    def _do_compare(self, image_path):
        from widgets.output_panel import CompareDialog
        dialog = CompareDialog(GalleryPanel._compare_slot, image_path, self)
        GalleryPanel._compare_slot = ""
        dialog.exec()

    def _clear_compare(self):
        GalleryPanel._compare_slot = ""
