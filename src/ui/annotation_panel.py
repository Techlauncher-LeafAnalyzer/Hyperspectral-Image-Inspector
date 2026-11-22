"""Drawing tools and their options, shown in the segmentation side panel.

The toolbox (:attr:`AnnotationToolsPage.tool_palette`) sits in the left stripe;
the page itself holds the options: appearance of the selected or next shape,
and the class vector masks the Pen edits. Annotation layers are listed with
every other layer in :class:`~ui.classification_layer_panel.ClassificationLayerPanel`.
Edits are emitted as signals; the Controller applies them to the Model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import QRectF, QSize, Qt, pyqtSignal

from core import AnchorMode, AnnotationKind, AnnotationLayer, AnnotationStyle
from core.annotation_model import MEASURABLE_KINDS, RGB

_LAYER_ID_ROLE = Qt.ItemDataRole.UserRole
# Vector-list entry that makes the pen draw a new class.
_NEW_CLASS_ID = -1
# Emitted by ``toolChanged`` for the vector-mask pen, which is not an
# annotation kind: it edits class outlines rather than adding a layer.
PEN_TOOL = "pen"
_TOOL_LABELS: tuple[tuple[AnnotationKind | str, str, str], ...] = (
    (AnnotationKind.RECTANGLE, "Rectangle", "Draw a rectangle (click again to turn off)"),
    (AnnotationKind.ELLIPSE, "Circle", "Draw a circle or ellipse (click again to turn off)"),
    (
        AnnotationKind.ARROW,
        "Arrow",
        "Drag to draw an arrow in the stroke colour (click again to turn off)",
    ),
    (AnnotationKind.TEXT, "Text", "Click the image to add text (click again to turn off)"),
    (
        PEN_TOOL,
        "Pen (vector mask)",
        "Draw a new class, or edit a class's vector mask outline "
        "(click again to turn off)",
    ),
)
_ASSETS = Path(__file__).parent / "assets"
_PALETTE_BUTTON_SIZE = 34
_TOOL_ICONS = {
    AnnotationKind.RECTANGLE: "annotation_rectangle.svg",
    AnnotationKind.ELLIPSE: "annotation_ellipse.svg",
    AnnotationKind.ARROW: "annotation_arrow.svg",
    AnnotationKind.TEXT: "annotation_text.svg",
    PEN_TOOL: "annotation_pen.svg",
}
_ANCHOR_MODE_TIPS = {
    AnchorMode.CORNER: "Corner: handles move independently, for a sharp turn",
    AnchorMode.SMOOTH: "Smooth: handles stay in line so the curve flows through",
    AnchorMode.SYMMETRIC: "Symmetric: smooth, with both handles the same length",
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
    """The toolbox and the options of the active tool or selected annotation."""

    toolChanged = pyqtSignal(object)  # AnnotationKind or None
    styleChanged = pyqtSignal(object)  # AnnotationStyle
    measureChanged = pyqtSignal(int, bool)
    # Vector masks: choose the mask the pen edits/draws into (class ID), or
    # -1 for "New class".
    vectorMaskSelected = pyqtSignal(int)
    # Convert the selected point to an AnchorMode.
    anchorModeRequested = pyqtSignal(object)
    deleteAnchorRequested = pyqtSignal()
    # Discard a class's vector mask and restore its segmented pixels.
    vectorRevertRequested = pyqtSignal(int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("annotationToolsPage")
        defaults = AnnotationStyle()
        self._tool_buttons: dict[AnnotationKind | str, QtWidgets.QToolButton] = {}
        # Not exclusive: clicking the active tool turns it off, which returns to
        # the default pan/select behaviour. Exclusivity is kept by hand.
        self._tool_group = QtWidgets.QButtonGroup(self)
        self._tool_group.setExclusive(False)

        # Photoshop-style toolbox: a column of square icon buttons. The page
        # owns it, but the window places it at the bottom of the left stripe.
        self.tool_palette = QtWidgets.QFrame(self)
        self.tool_palette.setObjectName("annotationToolPalette")
        self.tool_palette.hide()  # until the window places it in the stripe
        self.tool_palette.setAccessibleName("Drawing tools")
        self.tool_palette.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Maximum
        )
        palette_layout = QtWidgets.QVBoxLayout(self.tool_palette)
        palette_layout.setContentsMargins(0, 0, 0, 0)
        palette_layout.setSpacing(4)
        for kind, label, tip in _TOOL_LABELS:
            button = QtWidgets.QToolButton(self.tool_palette)
            button.setObjectName("annotationPaletteButton")
            button.setText(label)
            button.setToolTip(f"<b>{label}</b><br>{tip}")
            button.setAccessibleName(label)
            button.setAccessibleDescription(tip)
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setIcon(QtGui.QIcon(str(_ASSETS / _TOOL_ICONS[kind])))
            button.setIconSize(QSize(20, 20))
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            button.setFixedSize(_PALETTE_BUTTON_SIZE, _PALETTE_BUTTON_SIZE)
            self._tool_group.addButton(button)
            self._tool_buttons[kind] = button
            palette_layout.addWidget(button)
            button.clicked.connect(
                lambda checked, k=kind: self._on_tool_clicked(k, checked)
            )

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
        self._measure_check.setChecked(True)
        self._measure_check.setEnabled(False)

        vector_section = self._vector_section = self._build_vector_section()
        vector_section.hide()  # shown while the Pen is armed

        content = QtWidgets.QWidget(self)
        content.setObjectName("annotationContent")
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 2, 0)
        layout.setSpacing(8)
        layout.addWidget(vector_section)
        layout.addWidget(controls)
        layout.addWidget(self._measure_check)

        # No scroll area of its own: the class-layer inspector scrolls as one.
        page_layout = QtWidgets.QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(content)

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
        # The annotation selected in the layer list, whose options are shown.
        self._selected_layer: Optional[AnnotationLayer] = None
        self.set_vector_masks(())

    def _build_vector_section(self) -> QtWidgets.QFrame:
        """What the Pen edits (a class layer's outline) and its point controls."""

        section = QtWidgets.QFrame(self)
        section.setObjectName("annotationPenOptions")
        layout = QtWidgets.QVBoxLayout(section)
        layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(self._section_label("Pen"))

        # Which class layer the Pen edits; None draws a new class.
        self._vector_class: Optional[int] = None
        self._vector_target = QtWidgets.QLabel(self)
        self._vector_target.setObjectName("annotationHint")
        self._vector_target.setWordWrap(True)
        self._new_class_button = self._small_button(
            "New class", "Stop editing this outline; the next path starts a new class (Esc)"
        )
        self._new_class_button.clicked.connect(lambda: self.vectorMaskSelected.emit(_NEW_CLASS_ID))

        point_label = QtWidgets.QLabel("Selected point", self)
        point_label.setObjectName("annotationFieldLabel")
        self._anchor_mode_buttons: dict[AnchorMode, QtWidgets.QToolButton] = {}
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.setSpacing(4)
        mode_group = QtWidgets.QButtonGroup(self)
        mode_group.setExclusive(True)
        for mode in AnchorMode:
            button = QtWidgets.QToolButton(self)
            button.setObjectName("annotationToolButton")
            button.setText(mode.value)
            button.setToolTip(_ANCHOR_MODE_TIPS[mode])
            button.setAccessibleName(_ANCHOR_MODE_TIPS[mode])
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed
            )
            button.clicked.connect(lambda _checked, m=mode: self.anchorModeRequested.emit(m))
            mode_group.addButton(button)
            mode_row.addWidget(button, 1)
            self._anchor_mode_buttons[mode] = button

        self._delete_anchor_button = self._small_button(
            "Delete point", "Remove the selected point (Delete)"
        )
        self._delete_anchor_button.clicked.connect(self.deleteAnchorRequested)
        self._revert_vector_button = self._small_button(
            "Revert to pixels",
            "Discard this vector mask: a converted class gets its segmented "
            "pixels back, and a drawn class is removed",
        )
        self._revert_vector_button.setObjectName("annotationDeleteButton")
        self._revert_vector_button.clicked.connect(self._request_vector_revert)
        actions = QtWidgets.QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(6)
        actions.addWidget(self._delete_anchor_button, 1)
        actions.addWidget(self._revert_vector_button, 1)

        layout.addWidget(self._vector_target)
        layout.addWidget(self._new_class_button)
        layout.addWidget(point_label)
        layout.addLayout(mode_row)
        layout.addLayout(actions)
        self._vector_controls = (point_label, self._delete_anchor_button, self._revert_vector_button,
                                 self._new_class_button, *self._anchor_mode_buttons.values())
        return section

    def set_vector_masks(
        self,
        masks: tuple[tuple[int, str, RGB, int, int], ...],
        selected_class: Optional[int] = None,
    ) -> None:
        """Show which outline the Pen edits.

        ``masks`` lists outlined classes as ``(class_id, name, colour, paths,
        points)``; ``selected_class`` is the one being edited, or ``None``
        when the Pen draws a new class.
        """

        entries = {entry[0]: entry for entry in masks}
        entry = entries.get(selected_class)
        self._vector_class = None if entry is None else selected_class
        if entry is None:
            self._vector_target.setText(
                "Drawing a new class. Select a class layer to edit its outline."
            )
        else:
            _class_id, name, _color, path_count, anchor_count = entry
            self._vector_target.setText(
                f"Editing <b>{name}</b> · {path_count} path{'s' if path_count != 1 else ''}, "
                f"{anchor_count} points"
            )
        for widget in self._vector_controls:
            widget.setVisible(entry is not None)

    def set_anchor_state(self, mode: Optional[AnchorMode]) -> None:
        """Show the selected point's mode; ``None`` when no point is selected."""

        for anchor_mode, button in self._anchor_mode_buttons.items():
            button.setEnabled(mode is not None)
            button.setChecked(anchor_mode is mode)
        self._delete_anchor_button.setEnabled(mode is not None)

    def selected_vector_class(self) -> Optional[int]:
        """The class whose outline the Pen edits; ``None`` when it draws a new class."""

        return self._vector_class

    def _request_vector_revert(self) -> None:
        class_id = self.selected_vector_class()
        if class_id is not None:
            self.vectorRevertRequested.emit(class_id)

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

    def set_tool(self, kind: Optional[AnnotationKind | str]) -> None:
        """Show ``kind`` (or :data:`PEN_TOOL`) as the active tool without emitting."""

        for tool, button in self._tool_buttons.items():
            button.setChecked(tool == kind)
        self._vector_section.setVisible(kind == PEN_TOOL)

    def clear_tool(self) -> None:
        """Turn off the active tool, if any, as if its button were clicked."""

        if any(button.isChecked() for button in self._tool_buttons.values()):
            self.set_tool(None)
            self.toolChanged.emit(None)

    def _on_tool_clicked(self, kind: AnnotationKind | str, checked: bool) -> None:
        self.set_tool(kind if checked else None)
        self.toolChanged.emit(kind if checked else None)

    def selected_layer_id(self) -> Optional[int]:
        return None if self._selected_layer is None else self._selected_layer.layer_id

    def set_selected_layer(self, layer: Optional[AnnotationLayer]) -> None:
        """Show ``layer``'s options (its style and measuring); ``None`` clears it."""

        self._selected_layer = layer
        if layer is not None:
            self.set_style(layer.style)
        self._update_measure_check()

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

        layer = self._selected_layer
        measurable = layer is not None and layer.kind in MEASURABLE_KINDS
        self._measure_check.blockSignals(True)
        self._measure_check.setEnabled(measurable)
        self._measure_check.setChecked(layer is None or bool(measurable and layer.measure_index))
        self._measure_check.blockSignals(False)

    def _on_measure_toggled(self, checked: bool) -> None:
        layer_id = self.selected_layer_id()
        if layer_id is not None:
            self.measureChanged.emit(layer_id, checked)
