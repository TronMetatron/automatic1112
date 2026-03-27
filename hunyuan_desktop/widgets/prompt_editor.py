"""Custom prompt text editor with line numbers and wildcard autocomplete."""

from PySide6.QtWidgets import (
    QPlainTextEdit, QWidget, QCompleter, QTextEdit
)
from PySide6.QtGui import (
    QPainter, QColor, QTextFormat, QFont, QTextCursor
)
from PySide6.QtCore import Qt, Signal, QRect, QSize, QStringListModel, Slot


class LineNumberArea(QWidget):
    """Line number gutter widget for the prompt editor."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self):
        return QSize(self.editor.line_number_area_width(), 0)

    def paintEvent(self, event):
        self.editor.line_number_area_paint_event(event)


class PromptEditor(QPlainTextEdit):
    """Rich prompt editor with line numbers and wildcard autocomplete.

    Features:
    - Line number gutter
    - Wildcard autocomplete triggered by '['
    - Starred wildcard support '[*'
    - Drag-and-drop wildcard tags
    """

    wildcard_inserted = Signal(str)  # Emitted when a wildcard tag is inserted

    def __init__(self, wildcard_manager=None, parent=None):
        super().__init__(parent)
        self.wm = wildcard_manager
        self._categories = []

        # Setup font
        font = QFont("Consolas, Monaco, monospace", 12)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)

        # Line numbers
        self._line_number_area = LineNumberArea(self)
        self.blockCountChanged.connect(self._update_line_number_area_width)
        self.updateRequest.connect(self._update_line_number_area)
        self._update_line_number_area_width()

        # Completer for wildcards
        self._completer = QCompleter(self)
        self._completer.setWidget(self)
        self._completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.activated.connect(self._insert_completion)
        self._completer_model = QStringListModel()
        self._completer.setModel(self._completer_model)

        # State
        self._bracket_start = -1
        self._is_starred = False

        # Placeholder
        self.setPlaceholderText("Enter your prompt here...\nType [ to insert wildcards")

        # Accept drops
        self.setAcceptDrops(True)

    def set_wildcard_manager(self, wm):
        """Set or update the wildcard manager."""
        self.wm = wm
        self._refresh_categories()

    def _refresh_categories(self):
        """Refresh the list of available wildcard categories."""
        if self.wm:
            try:
                self._categories = sorted(self.wm.get_available_wildcards())
            except Exception:
                self._categories = []

    # -- Line numbers --

    def line_number_area_width(self) -> int:
        digits = max(1, len(str(self.blockCount())))
        return 8 + self.fontMetrics().horizontalAdvance("9") * digits

    def _update_line_number_area_width(self):
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def _update_line_number_area(self, rect, dy):
        if dy:
            self._line_number_area.scroll(0, dy)
        else:
            self._line_number_area.update(0, rect.y(),
                                          self._line_number_area.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_line_number_area_width()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self._line_number_area.setGeometry(
            QRect(cr.left(), cr.top(), self.line_number_area_width(), cr.height())
        )

    def line_number_area_paint_event(self, event):
        painter = QPainter(self._line_number_area)
        painter.fillRect(event.rect(), QColor(35, 35, 35))
        painter.setPen(QColor(100, 100, 100))

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                painter.drawText(0, top, self._line_number_area.width() - 4,
                                self.fontMetrics().height(),
                                Qt.AlignmentFlag.AlignRight, number)
            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            block_number += 1

        painter.end()

    # -- Wildcard autocomplete --

    def keyPressEvent(self, event):
        # If completer popup is visible, let it handle certain keys
        if self._completer.popup().isVisible():
            if event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return,
                              Qt.Key.Key_Tab, Qt.Key.Key_Escape):
                event.ignore()
                return

        super().keyPressEvent(event)

        # Check if we should trigger autocomplete
        text = event.text()
        if text == "[":
            self._bracket_start = self.textCursor().position()
            self._is_starred = False
            self._show_autocomplete("")
        elif text == "*" and self._bracket_start >= 0:
            # Check if previous char was '['
            cursor = self.textCursor()
            pos = cursor.position()
            if pos >= 2:
                cursor.setPosition(pos - 2)
                cursor.setPosition(pos, QTextCursor.MoveMode.KeepAnchor)
                if cursor.selectedText() == "[*":
                    self._is_starred = True
                    self._show_autocomplete("")
        elif self._bracket_start >= 0:
            # We're inside a bracket - update autocomplete
            if text == "]" or event.key() == Qt.Key.Key_Escape:
                self._bracket_start = -1
                self._completer.popup().hide()
            else:
                prefix = self._get_bracket_text()
                if prefix is not None:
                    self._show_autocomplete(prefix)
                else:
                    self._bracket_start = -1
                    self._completer.popup().hide()

    def _get_bracket_text(self) -> str:
        """Get the text typed after '[' or '[*'."""
        cursor = self.textCursor()
        pos = cursor.position()
        if pos <= self._bracket_start:
            return None

        cursor.setPosition(self._bracket_start)
        cursor.setPosition(pos, QTextCursor.MoveMode.KeepAnchor)
        text = cursor.selectedText()

        # Strip the opening bracket and optional star
        if text.startswith("[*"):
            return text[2:]
        elif text.startswith("["):
            return text[1:]
        return None

    def _show_autocomplete(self, prefix: str):
        """Show the wildcard autocomplete popup."""
        if not self._categories:
            self._refresh_categories()

        if not self._categories:
            return

        # Filter categories by prefix
        if prefix:
            filtered = [c for c in self._categories if prefix.lower() in c.lower()]
        else:
            filtered = self._categories

        if not filtered:
            self._completer.popup().hide()
            return

        self._completer_model.setStringList(filtered)
        self._completer.setCompletionPrefix(prefix)

        # Position popup
        cr = self.cursorRect()
        cr.setWidth(min(300, self.viewport().width()))
        self._completer.complete(cr)

    @Slot(str)
    def _insert_completion(self, completion: str):
        """Insert the selected wildcard category as a tag."""
        cursor = self.textCursor()

        # Remove everything from bracket start to cursor
        cursor.setPosition(self._bracket_start)
        cursor.setPosition(self.textCursor().position(), QTextCursor.MoveMode.KeepAnchor)

        # Insert the complete tag
        if self._is_starred:
            tag = f"[*{completion}]"
        else:
            tag = f"[{completion}]"

        cursor.insertText(tag)
        self.setTextCursor(cursor)

        self._bracket_start = -1
        self._is_starred = False
        self.wildcard_inserted.emit(completion)

    # -- Drag and drop --

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasText():
            text = event.mimeData().text()
            # If it looks like a wildcard tag, insert it
            cursor = self.cursorForPosition(event.position().toPoint())
            cursor.insertText(text)
            self.setTextCursor(cursor)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)
