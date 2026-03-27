"""Image utility functions for PIL <-> QPixmap conversion and thumbnails."""

from pathlib import Path
from typing import Optional

from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt


def pil_to_qpixmap(pil_image) -> QPixmap:
    """Convert a PIL Image to a QPixmap."""
    # Convert to RGB if necessary
    if pil_image.mode == "RGBA":
        data = pil_image.tobytes("raw", "RGBA")
        qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format.Format_RGBA8888)
    else:
        pil_image = pil_image.convert("RGB")
        data = pil_image.tobytes("raw", "RGB")
        qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format.Format_RGB888)

    return QPixmap.fromImage(qimage)


def load_thumbnail(image_path: str, size: int = 150) -> Optional[QPixmap]:
    """Load an image as a thumbnail QPixmap."""
    try:
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return None
        return pixmap.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
    except Exception:
        return None


def load_pixmap(image_path: str) -> Optional[QPixmap]:
    """Load a full-resolution QPixmap from a file path."""
    try:
        pixmap = QPixmap(image_path)
        return pixmap if not pixmap.isNull() else None
    except Exception:
        return None


def get_image_dimensions(image_path: str) -> tuple:
    """Get image dimensions without loading full pixmap."""
    try:
        from PySide6.QtGui import QImageReader
        reader = QImageReader(image_path)
        size = reader.size()
        return (size.width(), size.height()) if size.isValid() else (0, 0)
    except Exception:
        return (0, 0)


def find_latest_image(directory: str, extensions: tuple = (".png", ".jpg", ".jpeg")) -> Optional[str]:
    """Find the most recently modified image in a directory."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return None

    images = []
    for ext in extensions:
        images.extend(dir_path.glob(f"*{ext}"))

    if not images:
        return None

    latest = max(images, key=lambda p: p.stat().st_mtime)
    return str(latest)


def get_session_dir(base_dir: str, batch_name: str = "") -> Path:
    """Create a timestamped session directory for outputs."""
    from datetime import datetime

    base = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if batch_name:
        # Sanitize batch name for filesystem
        safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in batch_name)
        safe_name = safe_name.strip()[:50]
        dir_name = f"{safe_name}_{timestamp}"
    else:
        dir_name = f"session_{timestamp}"

    session_dir = base / dir_name
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir
