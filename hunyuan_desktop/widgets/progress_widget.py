"""Progress display widget for generation status."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar
from PySide6.QtCore import Slot


class ProgressWidget(QWidget):
    """Shows generation progress with status text and progress bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.hide()  # Hidden by default

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)

        info_row = QHBoxLayout()
        self.status_label = QLabel("")
        info_row.addWidget(self.status_label, stretch=1)
        self.time_label = QLabel("")
        info_row.addWidget(self.time_label)
        layout.addLayout(info_row)

    def start(self, total: int = 0):
        """Show and initialize progress."""
        self.progress_bar.setMaximum(total if total > 0 else 0)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting...")
        self.time_label.setText("")
        self.show()

    @Slot(int, int, str)
    def update_progress(self, current: int, total: int, status: str):
        """Update progress bar and status text."""
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
        self.status_label.setText(status)

    def set_time(self, elapsed: str, eta: str = ""):
        """Update time labels."""
        time_text = f"Elapsed: {elapsed}"
        if eta:
            time_text += f" | ETA: {eta}"
        self.time_label.setText(time_text)

    def complete(self, message: str = "Complete"):
        """Mark as complete."""
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.status_label.setText(message)

    def reset(self):
        """Reset and hide."""
        self.progress_bar.setValue(0)
        self.status_label.setText("")
        self.time_label.setText("")
        self.hide()
