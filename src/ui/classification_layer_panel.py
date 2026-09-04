"""Layer inspector for classification results.

Reflects a ``core.ClassificationLayerModel`` without owning classification
state. A Controller applies emitted changes to the Model and re-renders the
composited image.
"""

from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal

from core import ClassificationLayer
from ui.classification_colors import RGBColor, classification_palette


_EXPANDED_WIDTH = 280
_EXPANDED_MINIMUM_WIDTH = 236
_COLLAPSED_WIDTH = 48
_COLLAPSE_DURATION_MS = 220


class _ColorSwatch(QtWidgets.QLabel):
    """Show a crisp class-colour marker without dynamic inline QSS."""

    def __init__(
        self,
        color: RGBColor,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._color = QtGui.QColor(*color)
        self.setObjectName("layerColorSwatch")
        self.setFixedSize(18, 12)
        self.setToolTip(f"Class color {self._color.name().upper()}")
        self.setAccessibleName(f"Class color {self._color.name().upper()}")
        pixmap = QtGui.QPixmap(18, 12)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(self._color.darker(118), 1))
        painter.setBrush(self._color)
        painter.drawRoundedRect(pixmap.rect().adjusted(1, 1, -1, -1), 4, 4)
        painter.end()
        self.setPixmap(pixmap)

    @property
    def color(self) -> RGBColor:
        return self._color.red(), self._color.green(), self._color.blue()


class _LayerRowWidget(QtWidgets.QWidget):
    """One class row with selection, colour, count, and opacity controls."""

    visibilityChanged = pyqtSignal(int, bool)
    opacityChanged = pyqtSignal(int, float)

    def __init__(
        self,
        layer: ClassificationLayer,
        color: RGBColor,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("classificationLayerRow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.class_id = layer.class_id

        self._toggle = QtWidgets.QCheckBox(self)
        self._toggle.setObjectName("layerVisibilityToggle")
        self._toggle.setChecked(layer.visible)
        self._toggle.setToolTip(f"Show or hide {layer.name}")
        self._toggle.setAccessibleName(f"Show {layer.name}")
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)

        self._name_label = QtWidgets.QLabel(layer.name, self)
        self._name_label.setObjectName("layerNameLabel")
        self._name_label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)

        self._color_swatch = _ColorSwatch(color, self)

        self._count_label = QtWidgets.QLabel(f"{layer.pixel_count:,} pixels", self)
        self._count_label.setObjectName("layerPixelCountLabel")

        self._opacity_caption = QtWidgets.QLabel("Opacity", self)
        self._opacity_caption.setObjectName("layerOpacityCaption")

        self._opacity_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal, self)
        self._opacity_slider.setObjectName("layerOpacitySlider")
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(round(layer.opacity * 100))
        self._opacity_slider.setToolTip(f"Adjust {layer.name} opacity")
        self._opacity_slider.setAccessibleName(f"{layer.name} opacity")

        self._opacity_value_label = QtWidgets.QLabel(
            f"{round(layer.opacity * 100)}%", self
        )
        self._opacity_value_label.setObjectName("layerOpacityValueLabel")
        self._opacity_value_label.setFixedWidth(34)
        self._opacity_value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        name_row = QtWidgets.QHBoxLayout()
        name_row.setContentsMargins(0, 0, 0, 0)
        name_row.setSpacing(8)
        name_row.addWidget(self._name_label, 1)
        name_row.addWidget(self._color_swatch, 0, Qt.AlignmentFlag.AlignVCenter)

        name_column = QtWidgets.QVBoxLayout()
        name_column.setContentsMargins(0, 0, 0, 0)
        name_column.setSpacing(1)
        name_column.addLayout(name_row)
        name_column.addWidget(self._count_label)

        header_row = QtWidgets.QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(10)
        header_row.addWidget(self._toggle, 0, Qt.AlignmentFlag.AlignTop)
        header_row.addLayout(name_column, 1)

        opacity_header = QtWidgets.QHBoxLayout()
        opacity_header.setContentsMargins(0, 0, 0, 0)
        opacity_header.setSpacing(6)
        opacity_header.addWidget(self._opacity_caption)
        opacity_header.addStretch(1)
        opacity_header.addWidget(self._opacity_value_label)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(11, 10, 11, 10)
        layout.setSpacing(7)
        layout.addLayout(header_row)
        layout.addLayout(opacity_header)
        layout.addWidget(self._opacity_slider)

        self._toggle.toggled.connect(self._on_toggled)
        self._opacity_slider.valueChanged.connect(self._on_slider_changed)
        self._set_visible_property(layer.visible)

    def _set_visible_property(self, visible: bool) -> None:
        self.setProperty("layerVisible", visible)
        self.style().unpolish(self)
        self.style().polish(self)

    def _on_toggled(self, checked: bool) -> None:
        self._set_visible_property(checked)
        self.visibilityChanged.emit(self.class_id, checked)

    def _on_slider_changed(self, value: int) -> None:
        self._opacity_value_label.setText(f"{value}%")
        self.opacityChanged.emit(self.class_id, value / 100.0)


class ClassificationLayerPanel(QtWidgets.QWidget):
    """Collapsible side inspector listing each classification layer."""

    visibilityChanged = pyqtSignal(int, bool)
    opacityChanged = pyqtSignal(int, float)
    setAllVisibleRequested = pyqtSignal(bool)
    globalOpacityChanged = pyqtSignal(float)
    outlineModeChanged = pyqtSignal(bool)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("classificationLayerPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.setMinimumWidth(_EXPANDED_MINIMUM_WIDTH)
        self.setMaximumWidth(_EXPANDED_WIDTH)
        self._collapsed = False

        self._title = QtWidgets.QLabel("Class layers", self)
        self._title.setObjectName("classificationLayerPanelTitle")

        self._layer_count_label = QtWidgets.QLabel("0", self)
        self._layer_count_label.setObjectName("classificationLayerCount")
        self._layer_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._header_details = QtWidgets.QWidget(self)
        self._header_details.setObjectName("classificationLayerHeaderDetails")
        header_details_layout = QtWidgets.QHBoxLayout(self._header_details)
        header_details_layout.setContentsMargins(0, 0, 0, 0)
        header_details_layout.setSpacing(7)
        header_details_layout.addWidget(self._title)
        header_details_layout.addWidget(self._layer_count_label)
        header_details_layout.addStretch(1)

        self._collapse_button = QtWidgets.QToolButton(self)
        self._collapse_button.setObjectName("classificationLayerCollapseButton")
        self._collapse_button.setText("‹")
        self._collapse_button.setToolTip("Collapse class layers")
        self._collapse_button.setAccessibleName("Collapse class layers")
        self._collapse_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._collapse_button.clicked.connect(self.toggle_collapsed)

        header = QtWidgets.QWidget(self)
        header.setObjectName("classificationLayerHeader")
        header.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        header.setFixedHeight(27)
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(5)
        header_layout.addWidget(self._header_details, 1)
        header_layout.addWidget(self._collapse_button)

        self._summary_label = QtWidgets.QLabel("No classes available", self)
        self._summary_label.setObjectName("classificationLayerSummary")

        self._toggle_all_button = QtWidgets.QPushButton("Select all", self)
        self._toggle_all_button.setObjectName("layerToggleAllButton")
        self._toggle_all_button.setToolTip("Select or deselect every class")
        self._toggle_all_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_all_button.clicked.connect(self._on_toggle_all_clicked)
        self.visibilityChanged.connect(lambda *_: self._update_layer_summary())

        self._outline_toggle_button = QtWidgets.QPushButton("Borders only", self)
        self._outline_toggle_button.setObjectName("layerOutlineToggleButton")
        self._outline_toggle_button.setCheckable(True)
        self._outline_toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._outline_toggle_button.setToolTip(
            "Show only each class boundary instead of a solid fill"
        )
        self._outline_toggle_button.toggled.connect(self._on_outline_toggled)

        header_controls_row = QtWidgets.QHBoxLayout()
        header_controls_row.setContentsMargins(0, 0, 0, 0)
        header_controls_row.setSpacing(7)
        header_controls_row.addWidget(self._toggle_all_button, 1)
        header_controls_row.addWidget(self._outline_toggle_button, 1)

        global_opacity_label = QtWidgets.QLabel("Overall opacity", self)
        global_opacity_label.setObjectName("globalOpacityLabel")

        self._global_opacity_value_label = QtWidgets.QLabel("100%", self)
        self._global_opacity_value_label.setObjectName("globalOpacityValueLabel")
        self._global_opacity_value_label.setFixedWidth(36)
        self._global_opacity_value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        global_opacity_header = QtWidgets.QHBoxLayout()
        global_opacity_header.setContentsMargins(0, 0, 0, 0)
        global_opacity_header.addWidget(global_opacity_label)
        global_opacity_header.addStretch(1)
        global_opacity_header.addWidget(self._global_opacity_value_label)

        self._global_opacity_slider = QtWidgets.QSlider(
            Qt.Orientation.Horizontal, self
        )
        self._global_opacity_slider.setObjectName("globalOpacitySlider")
        self._global_opacity_slider.setRange(0, 100)
        self._global_opacity_slider.setValue(100)
        self._global_opacity_slider.setToolTip("Scale every layer opacity together")
        self._global_opacity_slider.setAccessibleName("Overall layer opacity")
        self._global_opacity_slider.valueChanged.connect(
            self._on_global_opacity_slider_changed
        )

        controls = QtWidgets.QFrame(self)
        controls.setObjectName("classificationLayerControls")
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(9)
        controls_layout.addLayout(header_controls_row)
        controls_layout.addLayout(global_opacity_header)
        controls_layout.addWidget(self._global_opacity_slider)

        classes_label = QtWidgets.QLabel("Classes", self)
        classes_label.setObjectName("classificationLayerSectionLabel")

        self._empty_label = QtWidgets.QLabel(
            "No classification yet\nRun a classification to create class layers.", self
        )
        self._empty_label.setObjectName("classificationLayerPanelEmpty")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setWordWrap(True)

        self._rows_container = QtWidgets.QWidget(self)
        self._rows_container.setObjectName("classificationLayerRows")
        self._rows_container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._rows_layout = QtWidgets.QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 4, 0)
        self._rows_layout.setSpacing(7)
        self._rows_layout.addStretch(1)

        self._scroll_area = QtWidgets.QScrollArea(self)
        self._scroll_area.setObjectName("classificationLayerScrollArea")
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._scroll_area.setWidget(self._rows_container)

        self._body = QtWidgets.QWidget(self)
        self._body.setObjectName("classificationLayerBody")
        body_layout = QtWidgets.QVBoxLayout(self._body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(10)
        body_layout.addWidget(self._summary_label)
        body_layout.addWidget(controls)
        body_layout.addWidget(classes_label)
        body_layout.addWidget(self._empty_label, 1)
        body_layout.addWidget(self._scroll_area, 1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(11, 10, 11, 10)
        layout.setSpacing(10)
        layout.addWidget(header)
        layout.setAlignment(header, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self._body, 1)

        self._width_animation = QtCore.QPropertyAnimation(self, b"maximumWidth", self)
        self._width_animation.setDuration(_COLLAPSE_DURATION_MS)
        self._width_animation.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._width_animation.finished.connect(self._finish_width_animation)

        self._rows: dict[int, _LayerRowWidget] = {}
        self._update_empty_state()

    @property
    def is_collapsed(self) -> bool:
        return self._collapsed

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(
            _COLLAPSED_WIDTH if self._collapsed else _EXPANDED_WIDTH,
            super().sizeHint().height(),
        )

    def set_layers(
        self,
        layers: tuple[ClassificationLayer, ...],
        *,
        global_opacity: float = 1.0,
        outline_mode: bool = False,
    ) -> None:
        """Rebuild rows to reflect the Model while preserving global controls."""

        self._clear_rows()
        colors = classification_palette(layer.class_id for layer in layers)
        for layer in layers:
            row = _LayerRowWidget(layer, colors[layer.class_id], self._rows_container)
            row.visibilityChanged.connect(self.visibilityChanged)
            row.opacityChanged.connect(self.opacityChanged)
            self._rows[layer.class_id] = row
            self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)
        self._apply_global_controls(global_opacity, outline_mode)
        self._update_empty_state()

    def clear(self) -> None:
        """Remove all rows, reset global controls, and show the empty state."""

        self._clear_rows()
        self._reset_global_controls()
        self._update_empty_state()

    def toggle_collapsed(self) -> None:
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool, *, animate: bool = True) -> None:
        """Collapse to a compact rail or expand back to the full inspector."""

        if collapsed == self._collapsed and self._width_animation.state() != (
            QtCore.QAbstractAnimation.State.Running
        ):
            return

        self._width_animation.stop()
        self._collapsed = collapsed
        target_width = _COLLAPSED_WIDTH if collapsed else _EXPANDED_WIDTH
        self.setMinimumWidth(_COLLAPSED_WIDTH)

        if collapsed:
            self._body.hide()
            self._header_details.hide()
            self._collapse_button.setText("›")
            self._collapse_button.setToolTip("Expand class layers")
            self._collapse_button.setAccessibleName("Expand class layers")
        else:
            self._body.show()
            self._header_details.show()
            self._collapse_button.setText("‹")
            self._collapse_button.setToolTip("Collapse class layers")
            self._collapse_button.setAccessibleName("Collapse class layers")

        if not animate or not self.isVisible():
            self.setMaximumWidth(target_width)
            self._finish_width_animation()
            return

        self._width_animation.setStartValue(self.width())
        self._width_animation.setEndValue(target_width)
        self._width_animation.start()

    def _finish_width_animation(self) -> None:
        target_width = _COLLAPSED_WIDTH if self._collapsed else _EXPANDED_WIDTH
        self.setMaximumWidth(target_width)
        self.setMinimumWidth(
            _COLLAPSED_WIDTH if self._collapsed else _EXPANDED_MINIMUM_WIDTH
        )
        self.updateGeometry()

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
            "Show fill" if checked else "Borders only"
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
        self._outline_toggle_button.setText(
            "Show fill" if outline_mode else "Borders only"
        )
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
        self._layer_count_label.setText(str(len(self._rows)))
        self._update_layer_summary()

    def _update_layer_summary(self) -> None:
        visible_count = sum(row._toggle.isChecked() for row in self._rows.values())
        total_count = len(self._rows)
        if total_count:
            self._summary_label.setText(
                f"{visible_count} of {total_count} classes selected"
            )
            self._toggle_all_button.setText(
                "Deselect all" if visible_count else "Select all"
            )
        else:
            self._summary_label.setText("No classes available")
            self._toggle_all_button.setText("Select all")

    def _update_toggle_all_button_text(self) -> None:
        """Compatibility wrapper for callers that refresh aggregate text."""

        self._update_layer_summary()
