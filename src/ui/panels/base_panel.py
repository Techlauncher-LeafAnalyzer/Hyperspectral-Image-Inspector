from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QLineEdit, QPushButton, QWidget

if TYPE_CHECKING:
    from core.hsi_data import HSIData


class FeaturePanel(QWidget):
    """Contract for all swappable feature panels.

    Each subclass must:
      1. Call ``uic.loadUi(self._UI_PATH, self)`` in its ``__init__`` to
         inflate its ``.ui`` file directly onto ``self``.
      2. Implement ``on_image_loaded`` to enable/update controls after an
         image is loaded.
      3. Implement ``reset`` to return the panel to its default (no-image)
         state.

    ``_hsi_data`` is injected at construction and provides read-only access
    to the currently loaded hyperspectral image data.
    """

    def __init__(self, hsi_data: "HSIData", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._hsi_data = hsi_data

    def polish_controls(self, primary_buttons: set[str] | None = None) -> None:
        """Apply common interaction affordances after a .ui file is loaded."""
        primary_buttons = primary_buttons or set()

        for button in self.findChildren(QPushButton):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setMinimumHeight(34)
            if button.objectName() in primary_buttons:
                button.setProperty("primaryButton", True)
            self.refresh_widget_style(button)

        for edit in self.findChildren(QLineEdit):
            edit.setMinimumHeight(34)

        for combo_box in self.findChildren(QComboBox):
            combo_box.setMinimumHeight(34)

    def refresh_widget_style(self, widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def on_image_loaded(self) -> None:
        """Called by MainWindowController after a new HSI file is loaded."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement on_image_loaded()"
        )

    def reset(self) -> None:
        """Return panel to its default (no-image) state."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement reset()"
        )
