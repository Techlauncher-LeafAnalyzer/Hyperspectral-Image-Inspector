"""Annotation tools and layer stack shown in the classification side panel.

Reflects a ``core.AnnotationLayerModel`` without owning it: every edit is
emitted as a signal and the Controller applies it to the Model.
"""

from __future__ import annotations

from typing import Optional

from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal

from core import AnnotationKind, AnnotationLayer, AnnotationStyle
from core.annotation_model import RGB

_LAYER_ID_ROLE = Qt.ItemDataRole.UserRole
_TOOL_LABELS: tuple[tuple[AnnotationKind, str, str], ...] = (
    (AnnotationKind.RECTANGLE, "Rect", "Draw a rectangle (click again to turn off)"),
    (AnnotationKind.ELLIPSE, "Circle", "Draw a circle or ellipse (click again to turn off)"),
    (
        AnnotationKind.ARROW,
        "Arrow",
        "Drag to draw an arrow in the stroke colour (click again to turn off)",
    ),
    (AnnotationKind.TEXT, "Text", "Click the image to add text (click again to turn off)"),
)


class _ColorButton(QtWidgets.QPushButton):
    """Button that shows a colour and asks for a new one when clicked."""

    colorChanged = pyqtSignal(tuple)

    def __init__(self, color: RGB, tooltip: str, parent=None) -> None:
        super().__init__(parent)
        self._color = color
        self.setFixedSize(34, 22)
        self.setToolTip(tooltip)
        self.setAccessibleName(tooltip)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clicked.connect(self._choose)
        self._paint()

    @property
    def color(self) -> RGB:
        return self._color

    def set_color(self, color: RGB) -> None:
        self._color = color
        self._paint()

    def _paint(self) -> None:
        pixmap = QtGui.QPixmap(22, 12)
        pixmap.fill(QtGui.QColor(*self._color))
        self.setIcon(QtGui.QIcon(pixmap))
        self.setIconSize(pixmap.size())

    def _choose(self) -> None:
        chosen = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(*self._color), self, self.toolTip()
        )
        if chosen.isValid():
            color = (chosen.red(), chosen.green(), chosen.blue())
            self.request_color(color)

    def request_color(self, color: RGB) -> None:
        self.set_color(color)
        self.colorChanged.emit(color)


class AnnotationToolsPage(QtWidgets.QWidget):
    """Toolbar, style controls and the stacked layer list."""

    toolChanged = pyqtSignal(object)  # AnnotationKind or None
    styleChanged = pyqtSignal(object)  # AnnotationStyle
    layerSelected = pyqtSignal(int)
    visibilityChanged = pyqtSignal(int, bool)
    moveRequested = pyqtSignal(int, int)
    deleteRequested = pyqtSignal(int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("annotationToolsPage")
        defaults = AnnotationStyle()
        self._tool_buttons: dict[AnnotationKind, QtWidgets.QToolButton] = {}
        # Not exclusive: clicking the active tool turns it off, which returns to
        # the default pan/select behaviour. Exclusivity is kept by hand.
        self._tool_group = QtWidgets.QButtonGroup(self)
        self._tool_group.setExclusive(False)

        tool_row = QtWidgets.QHBoxLayout()
        tool_row.setContentsMargins(0, 0, 0, 0)
        tool_row.setSpacing(5)
        for kind, label, tip in _TOOL_LABELS:
            button = QtWidgets.QToolButton(self)
            button.setObjectName("annotationToolButton")
            button.setText(label)
            button.setToolTip(tip)
            button.setAccessibleName(tip)
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            self._tool_group.addButton(button)
            self._tool_buttons[kind] = button
            tool_row.addWidget(button)
            button.clicked.connect(
                lambda checked, k=kind: self._on_tool_clicked(k, checked)
            )
        tool_row.addStretch(1)

        self._fill_button = _ColorButton(defaults.fill, "Fill colour (also text colour)")
        self._fill_check = QtWidgets.QCheckBox("Fill", self)
        self._fill_check.setToolTip("Fill shapes with the fill colour")
        self._fill_check.setChecked(defaults.fill_enabled)
        self._stroke_button = _ColorButton(defaults.stroke, "Stroke colour")
        self._stroke_width = QtWidgets.QSpinBox(self)
        self._stroke_width.setRange(0, 50)
        self._stroke_width.setValue(int(defaults.stroke_width))
        self._stroke_width.setToolTip("Stroke width in image pixels (0 = no outline)")
        self._font_size = QtWidgets.QSpinBox(self)
        self._font_size.setRange(6, 400)
        self._font_size.setValue(defaults.font_size)
        self._font_size.setToolTip("Text size in image pixels")

        self._opacity = QtWidgets.QSlider(Qt.Orientation.Horizontal, self)
        self._opacity.setRange(0, 100)
        self._opacity.setValue(round(defaults.opacity * 100))
        self._opacity.setToolTip("Opacity of the selected annotation")
        self._opacity.setAccessibleName("Annotation opacity")
        self._opacity_value = QtWidgets.QLabel(f"{self._opacity.value()}%", self)
        self._opacity_value.setFixedWidth(34)
        self._opacity_value.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        style_grid = QtWidgets.QGridLayout()
        style_grid.setContentsMargins(0, 0, 0, 0)
        style_grid.setHorizontalSpacing(7)
        style_grid.setVerticalSpacing(6)
        style_grid.addWidget(QtWidgets.QLabel("Fill", self), 0, 0)
        style_grid.addWidget(self._fill_button, 0, 1)
        style_grid.addWidget(self._fill_check, 0, 2)
        style_grid.addWidget(QtWidgets.QLabel("Stroke", self), 1, 0)
        style_grid.addWidget(self._stroke_button, 1, 1)
        style_grid.addWidget(self._stroke_width, 1, 2)
        style_grid.addWidget(QtWidgets.QLabel("Text size", self), 2, 0)
        style_grid.addWidget(self._font_size, 2, 1, 1, 2)
        style_grid.addWidget(QtWidgets.QLabel("Opacity", self), 3, 0)
        style_grid.addWidget(self._opacity, 3, 1, 1, 2)
        style_grid.addWidget(self._opacity_value, 3, 3)
        style_grid.setColumnStretch(2, 1)

        controls = QtWidgets.QFrame(self)
        controls.setObjectName("classificationLayerControls")
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(9)
        controls_layout.addLayout(tool_row)
        controls_layout.addLayout(style_grid)

        self._list = QtWidgets.QListWidget(self)
        self._list.setObjectName("annotationLayerList")
        self._list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setToolTip("Top of the list is drawn on top")

        self._up_button = self._small_button("Up", "Move the layer above its neighbour")
        self._down_button = self._small_button("Down", "Move the layer below its neighbour")
        self._delete_button = self._small_button("Delete", "Delete the selected layer")
        order_row = QtWidgets.QHBoxLayout()
        order_row.setContentsMargins(0, 0, 0, 0)
        order_row.setSpacing(6)
        for button in (self._up_button, self._down_button, self._delete_button):
            order_row.addWidget(button)

        self._empty_label = QtWidgets.QLabel(
            "No annotations yet\nPick a tool and draw on the image.", self
        )
        self._empty_label.setObjectName("classificationLayerPanelEmpty")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setWordWrap(True)

        layers_label = QtWidgets.QLabel("Layers", self)
        layers_label.setObjectName("classificationLayerSectionLabel")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(controls)
        layout.addWidget(layers_label)
        layout.addWidget(self._empty_label, 1)
        layout.addWidget(self._list, 1)
        layout.addLayout(order_row)

        self._fill_button.colorChanged.connect(self._emit_style)
        self._stroke_button.colorChanged.connect(self._emit_style)
        self._fill_check.toggled.connect(self._emit_style)
        self._stroke_width.valueChanged.connect(self._emit_style)
        self._font_size.valueChanged.connect(self._emit_style)
        self._opacity.valueChanged.connect(
            lambda value: self._opacity_value.setText(f"{value}%")
        )
        self._opacity.valueChanged.connect(self._emit_style)
        self._list.currentItemChanged.connect(self._on_current_changed)
        self._list.itemChanged.connect(self._on_item_changed)
        self._up_button.clicked.connect(lambda: self._request_move(1))
        self._down_button.clicked.connect(lambda: self._request_move(-1))
        self._delete_button.clicked.connect(self._request_delete)
        self._refreshing = False
        self._update_buttons()

    # ------------------------------------------------------------------ #

    def current_style(self) -> AnnotationStyle:
        return AnnotationStyle(
            fill=self._fill_button.color,
            fill_enabled=self._fill_check.isChecked(),
            stroke=self._stroke_button.color,
            stroke_width=float(self._stroke_width.value()),
            font_size=self._font_size.value(),
            opacity=self._opacity.value() / 100.0,
        )

    def set_style(self, style: AnnotationStyle) -> None:
        """Show ``style`` in the controls without emitting ``styleChanged``."""

        widgets = (self._fill_check, self._stroke_width, self._font_size, self._opacity)
        for widget in widgets:
            widget.blockSignals(True)
        self._fill_button.set_color(style.fill)
        self._stroke_button.set_color(style.stroke)
        self._fill_check.setChecked(style.fill_enabled)
        self._stroke_width.setValue(round(style.stroke_width))
        self._font_size.setValue(style.font_size)
        self._opacity.setValue(round(style.opacity * 100))
        self._opacity_value.setText(f"{self._opacity.value()}%")
        for widget in widgets:
            widget.blockSignals(False)

    def set_tool(self, kind: Optional[AnnotationKind]) -> None:
        """Show ``kind`` as the active tool (``None`` for none) without emitting."""

        for tool, button in self._tool_buttons.items():
            button.setChecked(tool is kind)

    def _on_tool_clicked(self, kind: AnnotationKind, checked: bool) -> None:
        self.set_tool(kind if checked else None)
        self.toolChanged.emit(kind if checked else None)

    def selected_layer_id(self) -> Optional[int]:
        item = self._list.currentItem()
        return None if item is None else int(item.data(_LAYER_ID_ROLE))

    def set_layers(
        self,
        layers: tuple[AnnotationLayer, ...],
        selected_id: Optional[int] = None,
    ) -> None:
        """Rebuild the list top layer first, keeping ``selected_id`` selected."""

        self._refreshing = True
        self._list.clear()
        for layer in reversed(layers):
            item = QtWidgets.QListWidgetItem(layer.name)
            item.setData(_LAYER_ID_ROLE, layer.layer_id)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable
            )
            item.setCheckState(
                Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked
            )
            self._list.addItem(item)
            if layer.layer_id == selected_id:
                self._list.setCurrentItem(item)
        self._refreshing = False
        has_layers = bool(layers)
        self._list.setVisible(has_layers)
        self._empty_label.setVisible(not has_layers)
        self._update_buttons()

    # ------------------------------------------------------------------ #

    def _small_button(self, text: str, tip: str) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text, self)
        button.setObjectName("layerToggleAllButton")
        button.setToolTip(tip)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    def _emit_style(self, *_args: object) -> None:
        self.styleChanged.emit(self.current_style())

    def _on_current_changed(self, current, _previous) -> None:
        self._update_buttons()
        if not self._refreshing and current is not None:
            self.layerSelected.emit(int(current.data(_LAYER_ID_ROLE)))

    def _on_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        if self._refreshing:
            return
        self.visibilityChanged.emit(
            int(item.data(_LAYER_ID_ROLE)),
            item.checkState() == Qt.CheckState.Checked,
        )

    def _request_move(self, delta: int) -> None:
        layer_id = self.selected_layer_id()
        if layer_id is not None:
            self.moveRequested.emit(layer_id, delta)

    def _request_delete(self) -> None:
        layer_id = self.selected_layer_id()
        if layer_id is not None:
            self.deleteRequested.emit(layer_id)

    def _update_buttons(self) -> None:
        row = self._list.currentRow()
        count = self._list.count()
        selected = row >= 0
        self._up_button.setEnabled(selected and row > 0)
        self._down_button.setEnabled(selected and row < count - 1)
        self._delete_button.setEnabled(selected)
