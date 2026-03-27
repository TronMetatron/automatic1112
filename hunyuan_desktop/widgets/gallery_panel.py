"""Thumbnail gallery panel with flow layout."""

from pathlib import Path
from typing import List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QScrollArea, QLayout, QFileDialog, QSizePolicy,
    QMenu
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, Signal, Slot, QRect, QSize, QPoint


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


class ThumbnailWidget(QWidget):
    """Clickable thumbnail image with hover effect."""

    clicked = Signal(str)  # image_path
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
            self.clicked.emit(self.image_path)
        elif event.button() == Qt.MouseButton.RightButton:
            self.context_menu_requested.emit(self.image_path, self.mapToGlobal(event.pos()))


class GalleryPanel(QWidget):
    """Scrollable thumbnail grid of generated images."""

    image_selected = Signal(str)  # image_path
    insert_prompt_requested = Signal(str)  # prompt text from image metadata

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

    def _on_thumbnail_context_menu(self, image_path, global_pos):
        """Show context menu for a thumbnail with insert prompt option."""
        import json

        menu = QMenu(self)

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
                lambda: self.insert_prompt_requested.emit(prompt_text)
            )
        if full_prompt_text and full_prompt_text != prompt_text:
            insert_full_action = menu.addAction(f"Insert Full Prompt: {full_prompt_text[:50]}...")
            insert_full_action.triggered.connect(
                lambda: self.insert_prompt_requested.emit(full_prompt_text)
            )
        if not prompt_text and not full_prompt_text:
            no_data = menu.addAction("No prompt metadata found")
            no_data.setEnabled(False)

        menu.exec(global_pos)
