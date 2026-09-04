"""Photoshop-style layer list for classification results.

Reflects a ``core.ClassificationLayerModel``'s rows without owning any
classification state itself -- a Controller applies row signals back to the
Model, then re-renders the composited image.
"""

from __future__ import annotations

from typing import Optional

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal

from core import ClassificationLayer


class _LayerRowWidget(QtWidgets.QWidget):
    """One layer row: visibility toggle, name/pixel count, opacity slider."""

    visibilityChanged = pyqtSignal(int, bool)
    opacityChanged = pyqtSignal(int, float)

    def __init__(self, layer: ClassificationLayer, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("classificationLayerRow")
        self.class_id = layer.class_id

        self._toggle = QtWidgets.QCheckBox(self)
        self._toggle.setObjectName("layerVisibilityToggle")
        self._toggle.setChecked(layer.visible)
        self._toggle.setToolTip("Show or hide this class")
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)

        self._name_label = QtWidgets.QLabel(layer.name, self)
        self._name_label.setObjectName("layerNameLabel")
        self._count_label = QtWidgets.QLabel(f"{layer.pixel_count:,} px", self)
        self._count_label.setObjectName("layerPixelCountLabel")

        self._opacity_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal, self)
        self._opacity_slider.setObjectName("layerOpacitySlider")
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(round(layer.opacity * 100))
        self._opacity_slider.setToolTip("Layer opacity")

        self._opacity_value_label = QtWidgets.QLabel(f"{round(layer.opacity * 100)}%", self)
        self._opacity_value_label.setObjectName("layerOpacityValueLabel")
        self._opacity_value_label.setFixedWidth(34)
        self._opacity_value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        name_column = QtWidgets.QVBoxLayout()
        name_column.setSpacing(0)
        name_column.addWidget(self._name_label)
        name_column.addWidget(self._count_label)

        header_row = QtWidgets.QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)
        header_row.addWidget(self._toggle)
        header_row.addLayout(name_column, 1)

        opacity_row = QtWidgets.QHBoxLayout()
        opacity_row.setContentsMargins(0, 0, 0, 0)
        opacity_row.setSpacing(6)
        opacity_row.addWidget(self._opacity_slider, 1)
        opacity_row.addWidget(self._opacity_value_label)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        layout.addLayout(header_row)
        layout.addLayout(opacity_row)

        self._toggle.toggled.connect(self._on_toggled)
        self._opacity_slider.valueChanged.connect(self._on_slider_changed)

    def _on_toggled(self, checked: bool) -> None:
        self.visibilityChanged.emit(self.class_id, checked)

    def _on_slider_changed(self, value: int) -> None:
        self._opacity_value_label.setText(f"{value}%")
        self.opacityChanged.emit(self.class_id, value / 100.0)


class ClassificationLayerPanel(QtWidgets.QWidget):
    """Fixed-width side panel listing one row per classification layer."""

    visibilityChanged = pyqtSignal(int, bool)
    opacityChanged = pyqtSignal(int, float)
    setAllVisibleRequested = pyqtSignal(bool)
    globalOpacityChanged = pyqtSignal(float)
    outlineModeChanged = pyqtSignal(bool)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("classificationLayerPanel")

        title = QtWidgets.QLabel("Layers", self)
        title.setObjectName("classificationLayerPanelTitle")

        self._empty_label = QtWidgets.QLabel(
            "Run classification to see layers.", self
        )
        self._empty_label.setObjectName("classificationLayerPanelEmpty")
        self._empty_label.setWordWrap(True)

        self._toggle_all_button = QtWidgets.QPushButton("Disable All", self)
        self._toggle_all_button.setObjectName("layerToggleAllButton")
        self._toggle_all_button.setToolTip("Show or hide every class at once")
        self._toggle_all_button.clicked.connect(self._on_toggle_all_clicked)
        # A row's own toggle also flips the aggregate state this button shows.
        self.visibilityChanged.connect(lambda *_: self._update_toggle_all_button_text())

        self._outline_toggle_button = QtWidgets.QPushButton(
            "Borders Only", self
        )
        self._outline_toggle_button.setObjectName("layerOutlineToggleButton")
        self._outline_toggle_button.setCheckable(True)
        self._outline_toggle_button.setToolTip(
            "Show only each class's boundary instead of a solid fill"
        )
        self._outline_toggle_button.toggled.connect(self._on_outline_toggled)

        header_controls_row = QtWidgets.QHBoxLayout()
        header_controls_row.setContentsMargins(0, 0, 0, 0)
        header_controls_row.setSpacing(6)
        header_controls_row.addWidget(self._toggle_all_button, 1)
        header_controls_row.addWidget(self._outline_toggle_button, 1)

        global_opacity_label = QtWidgets.QLabel("Overall opacity", self)
        global_opacity_label.setObjectName("globalOpacityLabel")

        self._global_opacity_slider = QtWidgets.QSlider(
            Qt.Orientation.Horizontal, self
        )
        self._global_opacity_slider.setObjectName("globalOpacitySlider")
        self._global_opacity_slider.setRange(0, 100)
        self._global_opacity_slider.setValue(100)
        self._global_opacity_slider.setToolTip(
            "Scale every layer's opacity together"
        )
        self._global_opacity_slider.valueChanged.connect(
            self._on_global_opacity_slider_changed
        )

        self._global_opacity_value_label = QtWidgets.QLabel("100%", self)
        self._global_opacity_value_label.setObjectName(
            "globalOpacityValueLabel"
        )
        self._global_opacity_value_label.setFixedWidth(34)
        self._global_opacity_value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        global_opacity_row = QtWidgets.QHBoxLayout()
        global_opacity_row.setContentsMargins(0, 0, 0, 0)
        global_opacity_row.setSpacing(6)
        global_opacity_row.addWidget(self._global_opacity_slider, 1)
        global_opacity_row.addWidget(self._global_opacity_value_label)

        self._rows_container = QtWidgets.QWidget(self)
        self._rows_layout = QtWidgets.QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(6)
        self._rows_layout.addStretch(1)

        self._scroll_area = QtWidgets.QScrollArea(self)
        self._scroll_area.setObjectName("classificationLayerScrollArea")
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._scroll_area.setWidget(self._rows_container)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(self._empty_label)
        layout.addLayout(header_controls_row)
        layout.addWidget(global_opacity_label)
        layout.addLayout(global_opacity_row)
        layout.addWidget(self._scroll_area, 1)

        self._rows: dict[int, _LayerRowWidget] = {}
        self._update_empty_state()

    def set_layers(
        self,
        layers: tuple[ClassificationLayer, ...],
        *,
        global_opacity: float = 1.0,
        outline_mode: bool = False,
    ) -> None:
        """Rebuild every row from scratch to reflect ``layers``.

        Called both after a brand new classification result arrives and
        when switching back to a resolution level with an existing result
        (see ``ClassificationController.refresh_display``), so a full
        clear-and-rebuild (mirroring the block-signals-and-rebuild pattern
        in ``docs/classification_layer_api.md``) is simplest and avoids
        reordering/diffing logic this panel does not need. ``global_opacity``
        and ``outline_mode`` reflect that result's own
        ``ClassificationLayerModel`` state, so switching back to a result
        the user had already faded or outlined restores those header
        controls too, not just the per-row values.
        """

        self._clear_rows()
        for layer in layers:
            row = _LayerRowWidget(layer, self._rows_container)
            row.visibilityChanged.connect(self.visibilityChanged)
            row.opacityChanged.connect(self.opacityChanged)
            self._rows[layer.class_id] = row
            self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)
        self._apply_global_controls(global_opacity, outline_mode)
        self._update_empty_state()

    def clear(self) -> None:
        """Remove all rows, reset the global controls, and show empty state."""

        self._clear_rows()
        self._reset_global_controls()
        self._update_empty_state()

    def _clear_rows(self) -> None:
        for row in self._rows.values():
            self._rows_layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()

    def _on_toggle_all_clicked(self) -> None:
        any_visible = any(row._toggle.isChecked() for row in self._rows.values())
        self.setAllVisibleRequested.emit(not any_visible)

    def _on_outline_toggled(self, checked: bool) -> None:
        self._outline_toggle_button.setText(
            "Show Fill" if checked else "Borders Only"
        )
        self.outlineModeChanged.emit(checked)

    def _on_global_opacity_slider_changed(self, value: int) -> None:
        self._global_opacity_value_label.setText(f"{value}%")
        self.globalOpacityChanged.emit(value / 100.0)

    def _apply_global_controls(self, global_opacity: float, outline_mode: bool) -> None:
        value = round(global_opacity * 100)
        self._global_opacity_slider.blockSignals(True)
        self._global_opacity_slider.setValue(value)
        self._global_opacity_slider.blockSignals(False)
        self._global_opacity_value_label.setText(f"{value}%")

        self._outline_toggle_button.blockSignals(True)
        self._outline_toggle_button.setChecked(outline_mode)
        self._outline_toggle_button.setText("Show Fill" if outline_mode else "Borders Only")
        self._outline_toggle_button.blockSignals(False)

    def _reset_global_controls(self) -> None:
        self._apply_global_controls(1.0, False)

    def _update_empty_state(self) -> None:
        has_rows = bool(self._rows)
        self._empty_label.setVisible(not has_rows)
        self._scroll_area.setVisible(has_rows)
        self._toggle_all_button.setEnabled(has_rows)
        self._outline_toggle_button.setEnabled(has_rows)
        self._global_opacity_slider.setEnabled(has_rows)
        if has_rows:
            self._update_toggle_all_button_text()

    def _update_toggle_all_button_text(self) -> None:
        any_visible = any(row._toggle.isChecked() for row in self._rows.values())
        self._toggle_all_button.setText("Disable All" if any_visible else "Enable All")
