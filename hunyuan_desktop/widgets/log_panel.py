"""Log panel widget for displaying application logs and errors.

Captures stdout/stderr and displays them in a scrollable, searchable text area.
"""

import sys
from io import StringIO
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QLineEdit, QLabel, QCheckBox, QComboBox
)
from PySide6.QtCore import Qt, Signal, Slot, QObject
from PySide6.QtGui import QTextCursor, QColor, QTextCharFormat


class OutputRedirector(QObject):
    """Redirects stdout/stderr to a signal for Qt display."""

    text_written = Signal(str, str)  # text, stream_type ('stdout' or 'stderr')

    def __init__(self, stream_type: str, original_stream):
        super().__init__()
        self.stream_type = stream_type
        self.original_stream = original_stream

    def write(self, text):
        if text:
            # Write to original stream first
            if self.original_stream:
                self.original_stream.write(text)
                self.original_stream.flush()
            # Emit signal for Qt display
            self.text_written.emit(text, self.stream_type)

    def flush(self):
        if self.original_stream:
            self.original_stream.flush()


class LogPanel(QWidget):
    """Panel for displaying application logs and errors."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._setup_redirectors()
        self._log_count = 0
        self._error_count = 0

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Top controls
        controls = QHBoxLayout()

        # Filter dropdown
        controls.addWidget(QLabel("Show:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All", "all")
        self.filter_combo.addItem("Prompt Pipeline", "prompt")
        self.filter_combo.addItem("Stdout Only", "stdout")
        self.filter_combo.addItem("Stderr Only", "stderr")
        self.filter_combo.currentIndexChanged.connect(self._apply_filter)
        controls.addWidget(self.filter_combo)

        # Search
        controls.addWidget(QLabel("Search:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter logs...")
        self.search_input.textChanged.connect(self._on_search)
        controls.addWidget(self.search_input, stretch=1)

        # Auto-scroll toggle
        self.auto_scroll_check = QCheckBox("Auto-scroll")
        self.auto_scroll_check.setChecked(True)
        controls.addWidget(self.auto_scroll_check)

        # Wrap text toggle
        self.wrap_check = QCheckBox("Wrap")
        self.wrap_check.setChecked(True)
        self.wrap_check.toggled.connect(self._toggle_wrap)
        controls.addWidget(self.wrap_check)

        # Stats
        self.stats_label = QLabel("Lines: 0 | Errors: 0")
        controls.addWidget(self.stats_label)

        # Clear button
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_logs)
        controls.addWidget(self.clear_btn)

        layout.addLayout(controls)

        # Log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 12px;
                border: 1px solid #3c3c3c;
            }
        """)
        self.log_display.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        layout.addWidget(self.log_display)

        # Store all logs for filtering
        self._all_logs = []  # List of (text, stream_type) tuples

    def _setup_redirectors(self):
        """Set up stdout/stderr redirection."""
        # Store original streams
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

        # Create redirectors
        self._stdout_redirector = OutputRedirector('stdout', self._original_stdout)
        self._stderr_redirector = OutputRedirector('stderr', self._original_stderr)

        # Connect signals
        self._stdout_redirector.text_written.connect(self._on_text_received)
        self._stderr_redirector.text_written.connect(self._on_text_received)

        # Redirect streams
        sys.stdout = self._stdout_redirector
        sys.stderr = self._stderr_redirector

    def _restore_streams(self):
        """Restore original stdout/stderr."""
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

    @Slot(str, str)
    def _on_text_received(self, text: str, stream_type: str):
        """Handle incoming text from redirected streams."""
        self._all_logs.append((text, stream_type))

        # Update counts
        self._log_count += text.count('\n') or 1
        if stream_type == 'stderr':
            self._error_count += text.count('\n') or 1

        # Apply current filter
        current_filter = self.filter_combo.currentData()
        show = False
        if current_filter == 'all':
            show = True
        elif current_filter == 'prompt':
            # Show only prompt pipeline tags
            show = any(tag in text for tag in ('[ORIGINAL]', '[WILDCARD]', '[LLM]', '[FINAL PROMPT]'))
        elif current_filter == stream_type:
            show = True

        if show:
            search_text = self.search_input.text().lower()
            if not search_text or search_text in text.lower():
                self._append_text(text, stream_type)

        self._update_stats()

    def _append_text(self, text: str, stream_type: str):
        """Append text to the display with appropriate formatting."""
        cursor = self.log_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        # Set color based on stream type and content tags
        fmt = QTextCharFormat()
        if stream_type == 'stderr':
            fmt.setForeground(QColor("#ff6b6b"))  # Red for errors
        elif '[ERROR]' in text or 'Error' in text or 'error' in text:
            fmt.setForeground(QColor("#ff9f43"))  # Orange for error mentions
        elif '[FINAL PROMPT]' in text:
            fmt.setForeground(QColor("#00e676"))  # Bright green — what the AI renderer sees
            fmt.setFontWeight(700)
        elif '[WILDCARD]' in text:
            fmt.setForeground(QColor("#ff6ac1"))  # Hot pink — wildcard resolution
        elif '[LLM]' in text or '[LLM ENHANCE]' in text or '[BATCH ENHANCE]' in text:
            fmt.setForeground(QColor("#42a5f5"))  # Bright blue — LLM enhancement
        elif '[ORIGINAL]' in text:
            fmt.setForeground(QColor("#b0b0b0"))  # Dim gray — original input
        elif '[WARN]' in text or 'Warning' in text:
            fmt.setForeground(QColor("#feca57"))  # Yellow for warnings
        elif '[INFO]' in text:
            fmt.setForeground(QColor("#54a0ff"))  # Blue for info
        elif '[INIT]' in text:
            fmt.setForeground(QColor("#5f27cd"))  # Purple for init
        elif '[GPU]' in text:
            fmt.setForeground(QColor("#00d2d3"))  # Cyan for GPU
        elif '[GEN]' in text or '[BATCH]' in text:
            fmt.setForeground(QColor("#aaaaaa"))  # Light gray for gen/batch status
        else:
            fmt.setForeground(QColor("#d4d4d4"))  # Default gray

        cursor.setCharFormat(fmt)
        cursor.insertText(text)

        # Auto-scroll if enabled
        if self.auto_scroll_check.isChecked():
            self.log_display.setTextCursor(cursor)
            self.log_display.ensureCursorVisible()

    def _update_stats(self):
        """Update the stats label."""
        self.stats_label.setText(f"Lines: {self._log_count} | Errors: {self._error_count}")

    @Slot()
    def _clear_logs(self):
        """Clear all logs."""
        self.log_display.clear()
        self._all_logs.clear()
        self._log_count = 0
        self._error_count = 0
        self._update_stats()

    @Slot(int)
    def _apply_filter(self, index: int):
        """Re-render logs with current filter."""
        self.log_display.clear()
        current_filter = self.filter_combo.currentData()
        search_text = self.search_input.text().lower()

        prompt_tags = ('[ORIGINAL]', '[WILDCARD]', '[LLM]', '[FINAL PROMPT]')
        for text, stream_type in self._all_logs:
            show = False
            if current_filter == 'all':
                show = True
            elif current_filter == 'prompt':
                show = any(tag in text for tag in prompt_tags)
            elif current_filter == stream_type:
                show = True
            if show:
                if not search_text or search_text in text.lower():
                    self._append_text(text, stream_type)

    @Slot(str)
    def _on_search(self, text: str):
        """Filter logs by search text."""
        self._apply_filter(0)

    @Slot(bool)
    def _toggle_wrap(self, checked: bool):
        """Toggle text wrapping."""
        if checked:
            self.log_display.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        else:
            self.log_display.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)

    def closeEvent(self, event):
        """Restore streams when widget is closed."""
        self._restore_streams()
        super().closeEvent(event)

    def cleanup(self):
        """Clean up redirectors. Call this before application exit."""
        self._restore_streams()
