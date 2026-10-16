"""Annotation tools and layer stack shown in the classification side panel.

Reflects a ``core.AnnotationLayerModel`` without owning it: every edit is
emitted as a signal and the Controller applies it to the Model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import QRectF, QSize, Qt, pyqtSignal

from core import AnnotationKind, AnnotationLayer, AnnotationStyle
from core.annotation_model import MEASURABLE_KINDS, RGB

_LAYER_ID_ROLE = Qt.ItemDataRole.UserRole
_TOOL_LABELS: tuple[tuple[AnnotationKind, str, str], ...] = (
    (AnnotationKind.RECTANGLE, "Rectangle", "Draw a rectangle (click again to turn off)"),
    (AnnotationKind.ELLIPSE, "Circle", "Draw a circle or ellipse (click again to turn off)"),
    (
        AnnotationKind.ARROW,
        "Arrow",
        "Drag to draw an arrow in the stroke colour (click again to turn off)",
    ),
    (AnnotationKind.TEXT, "Text", "Click the image to add text (click again to turn off)"),
)
_ASSETS = Path(__file__).parent / "assets"
_TOOL_ICONS = {
    AnnotationKind.RECTANGLE: "annotation_rectangle.svg",
    AnnotationKind.ELLIPSE: "annotation_ellipse.svg",
    AnnotationKind.ARROW: "annotation_arrow.svg",
    AnnotationKind.TEXT: "annotation_text.svg",
}


class _ColorButton(QtWidgets.QPushButton):
    """Button that shows a colour and asks for a new one when clicked."""

    colorChanged = pyqtSignal(tuple)

    def __init__(self, color: RGB, tooltip: str, parent=None) -> None:
        super().__init__(parent)
        self._color = color
        self.setObjectName("annotationColorButton")
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
        color = QtGui.QColor(*self._color)
        ratio = self.devicePixelRatioF()
        pixmap = QtGui.QPixmap(round(16 * ratio), round(16 * ratio))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(QtGui.QColor("#a9bcb6"), 1))
        painter.setBrush(color)
        painter.drawRoundedRect(QRectF(0.5, 0.5, 15, 15), 4, 4)
        painter.end()
        self.setIcon(QtGui.QIcon(pixmap))
        self.setIconSize(QSize(16, 16))
        self.setText(color.name().upper())
        self.setAccessibleDescription(f"Current colour: {color.name().upper()}")

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
    measureChanged = pyqtSignal(int, bool)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("annotationToolsPage")
        defaults = AnnotationStyle()
        self._tool_buttons: dict[AnnotationKind, QtWidgets.QToolButton] = {}
        # Not exclusive: clicking the active tool turns it off, which returns to
        # the default pan/select behaviour. Exclusivity is kept by hand.
        self._tool_group = QtWidgets.QButtonGroup(self)
        self._tool_group.setExclusive(False)

        tool_grid = QtWidgets.QGridLayout()
        tool_grid.setContentsMargins(0, 0, 0, 0)
        tool_grid.setSpacing(6)
        for index, (kind, label, tip) in enumerate(_TOOL_LABELS):
            button = QtWidgets.QToolButton(self)
            button.setObjectName("annotationToolButton")
            button.setText(label)
            button.setToolTip(tip)
            button.setAccessibleName(tip)
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setIcon(QtGui.QIcon(str(_ASSETS / _TOOL_ICONS[kind])))
            button.setIconSize(QSize(16, 16))
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            self._tool_group.addButton(button)
            self._tool_buttons[kind] = button
            tool_grid.addWidget(button, index // 2, index % 2)
            button.clicked.connect(
                lambda checked, k=kind: self._on_tool_clicked(k, checked)
            )
        tool_grid.setColumnStretch(0, 1)
        tool_grid.setColumnStretch(1, 1)

        tool_hint = QtWidgets.QLabel(
            "Click the active tool again to pan or select.", self
        )
        tool_hint.setObjectName("annotationHint")
        tool_hint.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Minimum
        )
        tool_hint.setWordWrap(True)

        self._fill_button = _ColorButton(defaults.fill, "Fill colour (also text colour)")
        self._fill_check = QtWidgets.QCheckBox("Fill shapes", self)
        self._fill_check.setObjectName("annotationFillCheck")
        self._fill_check.setToolTip("Fill shapes with the fill colour")
        self._fill_check.setChecked(defaults.fill_enabled)
        self._stroke_button = _ColorButton(defaults.stroke, "Stroke colour")
        self._stroke_width = QtWidgets.QSpinBox(self)
        self._stroke_width.setObjectName("annotationSizeInput")
        self._stroke_width.setRange(0, 50)
        self._stroke_width.setValue(int(defaults.stroke_width))
        self._stroke_width.setSuffix(" px")
        self._stroke_width.setAccessibleName("Stroke width in image pixels")
        self._stroke_width.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self._stroke_width.setToolTip("Stroke width in image pixels (0 = no outline)")
        self._font_size = QtWidgets.QSpinBox(self)
        self._font_size.setObjectName("annotationSizeInput")
        self._font_size.setRange(6, 400)
        self._font_size.setValue(defaults.font_size)
        self._font_size.setSuffix(" px")
        self._font_size.setAccessibleName("Text size in image pixels")
        self._font_size.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self._font_size.setToolTip("Text size in image pixels")

        self._opacity = QtWidgets.QSlider(Qt.Orientation.Horizontal, self)
        self._opacity.setObjectName("annotationOpacitySlider")
        self._opacity.setRange(0, 100)
        self._opacity.setValue(round(defaults.opacity * 100))
        self._opacity.setToolTip("Opacity of the selected annotation")
        self._opacity.setAccessibleName("Annotation opacity")
        self._opacity_value = QtWidgets.QLabel(f"{self._opacity.value()}%", self)
        self._opacity_value.setObjectName("annotationOpacityValue")
        self._opacity_value.setMinimumWidth(38)
        self._opacity_value.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        colors_grid = QtWidgets.QGridLayout()
        colors_grid.setContentsMargins(0, 0, 0, 0)
        colors_grid.setHorizontalSpacing(8)
        colors_grid.setVerticalSpacing(5)
        colors_grid.addWidget(self._field_label("Fill / text", self._fill_button), 0, 0)
        colors_grid.addWidget(self._field_label("Stroke", self._stroke_button), 0, 1)
        colors_grid.addWidget(self._fill_button, 1, 0)
        colors_grid.addWidget(self._stroke_button, 1, 1)
        colors_grid.addWidget(self._fill_check, 2, 0, 1, 2)
        colors_grid.setColumnStretch(0, 1)
        colors_grid.setColumnStretch(1, 1)

        sizes_grid = QtWidgets.QGridLayout()
        sizes_grid.setContentsMargins(0, 0, 0, 0)
        sizes_grid.setHorizontalSpacing(8)
        sizes_grid.setVerticalSpacing(5)
        sizes_grid.addWidget(self._field_label("Stroke width", self._stroke_width), 0, 0)
        sizes_grid.addWidget(self._field_label("Text size", self._font_size), 0, 1)
        sizes_grid.addWidget(self._stroke_width, 1, 0)
        sizes_grid.addWidget(self._font_size, 1, 1)
        sizes_grid.setColumnStretch(0, 1)
        sizes_grid.setColumnStretch(1, 1)

        opacity_header = QtWidgets.QHBoxLayout()
        opacity_header.addWidget(self._field_label("Opacity", self._opacity))
        opacity_header.addStretch(1)
        opacity_header.addWidget(self._opacity_value)
        opacity_layout = QtWidgets.QVBoxLayout()
        opacity_layout.setSpacing(2)
        opacity_layout.addLayout(opacity_header)
        opacity_layout.addWidget(self._opacity)

        controls = QtWidgets.QFrame(self)
        controls.setObjectName("annotationAppearance")
        controls_layout = QtWidgets.QVBoxLayout(controls)
        # Preserve every row's minimum height when the hint above wraps. The
        # scroll area can grow its content instead of overlapping controls.
        controls_layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(10)
        controls_layout.addWidget(self._section_label("Appearance"))
        controls_layout.addLayout(colors_grid)
        controls_layout.addLayout(sizes_grid)
        controls_layout.addLayout(opacity_layout)

        self._measure_check = QtWidgets.QCheckBox("Measure index means", self)
        self._measure_check.setObjectName("annotationFillCheck")
        self._measure_check.setToolTip(
            "Add the pixels inside the selected rectangle or circle to the Classes "
            "tab, with their index means kept up to date as it moves"
        )
        self._measure_check.setEnabled(False)

        self._list = QtWidgets.QListWidget(self)
        self._list.setObjectName("annotationLayerList")
        self._list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setMinimumHeight(110)
        self._list.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Ignored
        )
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._list.setSpacing(3)
        self._list.setIconSize(QSize(16, 16))
        self._list.setToolTip("Top of the list is drawn on top")
        self._list.setAccessibleName("Annotation layers, front to back")

        self._up_button = self._small_button("↑ Up", "Move the layer above its neighbour")
        self._down_button = self._small_button("↓ Down", "Move the layer below its neighbour")
        self._delete_button = self._small_button("Delete", "Delete the selected layer")
        self._delete_button.setObjectName("annotationDeleteButton")
        order_row = QtWidgets.QHBoxLayout()
        order_row.setContentsMargins(0, 0, 0, 0)
        order_row.setSpacing(6)
        for button in (self._up_button, self._down_button, self._delete_button):
            order_row.addWidget(button, 1)

        self._empty_label = QtWidgets.QLabel(
            "<b>No annotations yet</b><br><br>"
            "Choose a drawing tool above,<br>then draw on the image.", self
        )
        self._empty_label.setObjectName("annotationEmptyState")
        self._empty_label.setMinimumHeight(110)
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Ignored
        )

        self._layer_count = QtWidgets.QLabel("0", self)
        self._layer_count.setObjectName("annotationLayerCount")
        self._layer_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layers_header = QtWidgets.QHBoxLayout()
        layers_header.setSpacing(7)
        layers_header.addWidget(self._section_label("Layers"))
        layers_header.addWidget(self._layer_count)
        layers_header.addStretch(1)
        layers_hint = QtWidgets.QLabel("Top layer appears in front.", self)
        layers_hint.setObjectName("annotationHint")
        layers_hint.setWordWrap(True)
        layers_heading = QtWidgets.QVBoxLayout()
        layers_heading.setSpacing(3)
        layers_heading.addLayout(layers_header)
        layers_heading.addWidget(layers_hint)

        content = QtWidgets.QWidget(self)
        content.setObjectName("annotationContent")
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 2, 0)
        layout.setSpacing(8)
        layout.addWidget(self._section_label("Drawing tools"))
        layout.addLayout(tool_grid)
        layout.addWidget(tool_hint)
        layout.addWidget(controls)
        layout.addWidget(self._measure_check)
        layout.addLayout(layers_heading)
        layout.addWidget(self._empty_label, 1)
        layout.addWidget(self._list, 1)
        layout.addLayout(order_row)

        # Let short windows scroll the inspector without compressing controls
        # or forcing the image workspace beyond the available height.
        scroll = QtWidgets.QScrollArea(self)
        scroll.setObjectName("annotationScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        page_layout = QtWidgets.QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(scroll)

        self._fill_button.colorChanged.connect(self._emit_style)
        self._stroke_button.colorChanged.connect(self._emit_style)
        self._fill_check.toggled.connect(self._emit_style)
        self._stroke_width.valueChanged.connect(self._emit_style)
        self._font_size.valueChanged.connect(self._emit_style)
        self._opacity.valueChanged.connect(
            lambda value: self._opacity_value.setText(f"{value}%")
        )
        self._opacity.valueChanged.connect(self._emit_style)
        self._measure_check.toggled.connect(self._on_measure_toggled)
        self._list.currentItemChanged.connect(self._on_current_changed)
        self._list.itemChanged.connect(self._on_item_changed)
        self._up_button.clicked.connect(lambda: self._request_move(1))
        self._down_button.clicked.connect(lambda: self._request_move(-1))
        self._delete_button.clicked.connect(self._request_delete)
        self._refreshing = False
        self._layers_by_id: dict[int, AnnotationLayer] = {}
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
        self._layers_by_id = {layer.layer_id: layer for layer in layers}
        self._list.clear()
        for layer in reversed(layers):
            item = QtWidgets.QListWidgetItem(layer.name)
            item.setIcon(QtGui.QIcon(str(_ASSETS / _TOOL_ICONS[layer.kind])))
            item.setToolTip(f"{layer.name}\nUncheck to hide this annotation.")
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
        self._update_measure_check()
        has_layers = bool(layers)
        self._layer_count.setText(str(len(layers)))
        self._layer_count.setAccessibleName(f"{len(layers)} annotation layers")
        self._list.setVisible(has_layers)
        self._empty_label.setVisible(not has_layers)
        self._update_buttons()

    # ------------------------------------------------------------------ #

    def _section_label(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text, self)
        label.setObjectName("annotationSectionLabel")
        return label

    def _field_label(self, text: str, buddy: QtWidgets.QWidget) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text, self)
        label.setObjectName("annotationFieldLabel")
        label.setBuddy(buddy)
        return label

    def _small_button(self, text: str, tip: str) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text, self)
        button.setObjectName("annotationLayerAction")
        button.setToolTip(tip)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    def _emit_style(self, *_args: object) -> None:
        self.styleChanged.emit(self.current_style())

    def _update_measure_check(self) -> None:
        """Enable the measure box for a selected rectangle/circle, showing its state."""

        selected_id = self.selected_layer_id()
        layer = self._layers_by_id.get(selected_id) if selected_id is not None else None
        measurable = layer is not None and layer.kind in MEASURABLE_KINDS
        self._measure_check.blockSignals(True)
        self._measure_check.setEnabled(measurable)
        self._measure_check.setChecked(bool(measurable and layer.measure_index))
        self._measure_check.blockSignals(False)

    def _on_measure_toggled(self, checked: bool) -> None:
        layer_id = self.selected_layer_id()
        if layer_id is not None:
            self.measureChanged.emit(layer_id, checked)

    def _on_current_changed(self, current, _previous) -> None:
        self._update_buttons()
        if not self._refreshing:
            self._update_measure_check()
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
