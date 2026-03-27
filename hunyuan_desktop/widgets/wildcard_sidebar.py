"""Wildcard sidebar: searchable tree view with drag-to-prompt support."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QTreeWidget, QTreeWidgetItem, QPushButton, QMenu
)
from PySide6.QtGui import QDrag
from PySide6.QtCore import Qt, Signal, Slot, QMimeData


class WildcardSidebar(QWidget):
    """Searchable tree view of wildcard categories with drag support.

    Categories are grouped by prefix (animal-, color-, landscape-).
    Items can be dragged into the prompt editor as [category] tags.
    """

    insert_requested = Signal(str)  # [category] tag text

    def __init__(self, desktop_state, parent=None):
        super().__init__(parent)
        self.state = desktop_state
        self._setup_ui()
        self._connect_signals()
        self._populate_tree()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Search bar
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search wildcards...")
        self.search_box.setClearButtonEnabled(True)
        layout.addWidget(self.search_box)

        # Tree view
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Category", "Items"])
        self.tree.setColumnWidth(0, 180)
        self.tree.setDragEnabled(True)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.setAlternatingRowColors(True)
        layout.addWidget(self.tree)

        # Buttons
        btn_row = QHBoxLayout()
        self.insert_btn = QPushButton("Insert [wildcard]")
        self.insert_btn.setToolTip("Insert selected wildcard into prompt editor")
        btn_row.addWidget(self.insert_btn)

        self.insert_starred_btn = QPushButton("Insert [*starred]")
        self.insert_starred_btn.setToolTip("Insert as starred wildcard (re-rolls per variation)")
        btn_row.addWidget(self.insert_starred_btn)

        self.manage_btn = QPushButton("Manage...")
        self.manage_btn.setToolTip("Open full wildcard manager")
        btn_row.addWidget(self.manage_btn)

        layout.addLayout(btn_row)

        # Status
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

    def _connect_signals(self):
        self.search_box.textChanged.connect(self._filter_tree)
        self.insert_btn.clicked.connect(self._on_insert)
        self.insert_starred_btn.clicked.connect(self._on_insert_starred)
        self.manage_btn.clicked.connect(self._on_manage)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        self.tree.itemDoubleClicked.connect(self._on_double_click)

    def _populate_tree(self):
        """Build tree from WildcardManager categories."""
        self.tree.clear()
        wm = self.state.get_wildcard_manager()
        if not wm:
            self.status_label.setText("No wildcards loaded")
            return

        try:
            categories = wm.get_categories() if hasattr(wm, 'get_categories') else {}

            # If get_categories returns a dict of prefix -> list of keys
            if isinstance(categories, dict):
                for group_name, keys in sorted(categories.items()):
                    group_item = QTreeWidgetItem([group_name, str(len(keys))])
                    group_item.setFlags(group_item.flags() | Qt.ItemFlag.ItemIsDragEnabled)
                    for key in sorted(keys):
                        items = wm.data.get(key, [])
                        child = QTreeWidgetItem([key, str(len(items))])
                        child.setFlags(child.flags() | Qt.ItemFlag.ItemIsDragEnabled)
                        child.setData(0, Qt.ItemDataRole.UserRole, key)
                        group_item.addChild(child)
                    self.tree.addTopLevelItem(group_item)
            else:
                # Flat list of wildcard keys
                all_keys = wm.get_available_wildcards() if hasattr(wm, 'get_available_wildcards') else []
                # Group by prefix
                groups = {}
                for key in sorted(all_keys):
                    parts = key.split("-", 1)
                    prefix = parts[0] if len(parts) > 1 else "other"
                    if prefix not in groups:
                        groups[prefix] = []
                    groups[prefix].append(key)

                for prefix, keys in sorted(groups.items()):
                    group_item = QTreeWidgetItem([prefix, str(len(keys))])
                    for key in keys:
                        items = wm.data.get(key, []) if hasattr(wm, 'data') else []
                        child = QTreeWidgetItem([key, str(len(items))])
                        child.setFlags(child.flags() | Qt.ItemFlag.ItemIsDragEnabled)
                        child.setData(0, Qt.ItemDataRole.UserRole, key)
                        group_item.addChild(child)
                    self.tree.addTopLevelItem(group_item)

            total = sum(1 for i in range(self.tree.topLevelItemCount())
                       for j in range(self.tree.topLevelItem(i).childCount()))
            self.status_label.setText(f"{total} categories loaded")

        except Exception as e:
            self.status_label.setText(f"Error: {e}")

    @Slot(str)
    def _filter_tree(self, text: str):
        """Filter tree items by search text."""
        search = text.lower()
        for i in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(i)
            group_visible = False
            for j in range(group.childCount()):
                child = group.child(j)
                matches = search in child.text(0).lower()
                child.setHidden(not matches)
                if matches:
                    group_visible = True

            # Show group if any child matches, or if search matches group name
            if search in group.text(0).lower():
                group_visible = True
                for j in range(group.childCount()):
                    group.child(j).setHidden(False)

            group.setHidden(not group_visible)
            if group_visible:
                group.setExpanded(bool(search))

    def _get_selected_key(self) -> str:
        """Get the wildcard key of the selected item."""
        item = self.tree.currentItem()
        if not item:
            return ""
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if key:
            return key
        # If a group is selected, return the group name
        return item.text(0)

    @Slot()
    def _on_insert(self):
        key = self._get_selected_key()
        if key:
            self.insert_requested.emit(f"[{key}]")

    @Slot()
    def _on_insert_starred(self):
        key = self._get_selected_key()
        if key:
            self.insert_requested.emit(f"[*{key}]")

    @Slot(QTreeWidgetItem, int)
    def _on_double_click(self, item, column):
        """Double-click inserts the wildcard."""
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if key:
            self.insert_requested.emit(f"[{key}]")

    def _show_context_menu(self, pos):
        """Right-click context menu."""
        item = self.tree.itemAt(pos)
        if not item:
            return

        key = item.data(0, Qt.ItemDataRole.UserRole)
        menu = QMenu(self)

        if key:
            menu.addAction(f"Insert [{key}]", lambda: self.insert_requested.emit(f"[{key}]"))
            menu.addAction(f"Insert [*{key}]", lambda: self.insert_requested.emit(f"[*{key}]"))
            menu.addSeparator()

            # Show sample items
            wm = self.state.get_wildcard_manager()
            if wm and hasattr(wm, 'data') and key in wm.data:
                items = wm.data[key][:5]
                if items:
                    menu.addSection("Sample items:")
                    for val in items:
                        menu.addAction(f"  {val}").setEnabled(False)
                    if len(wm.data[key]) > 5:
                        menu.addAction(f"  ... ({len(wm.data[key])} total)").setEnabled(False)

        menu.exec(self.tree.mapToGlobal(pos))

    @Slot()
    def _on_manage(self):
        """Open the wildcard manager dialog."""
        from dialogs.wildcard_dialog import WildcardDialog
        dialog = WildcardDialog(self.state, parent=self)
        dialog.exec()
        self._populate_tree()  # Refresh after dialog closes

    def refresh(self):
        """Reload wildcards from manager."""
        self._populate_tree()
