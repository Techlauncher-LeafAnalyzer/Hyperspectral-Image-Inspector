"""Segmentation inspector: the classifier, tool options and one layer list.

Every layer is listed together, like Photoshop's Layers panel: annotations
(front-most first), Ctrl+Click pins, then classification and drawn classes.
Each row shows or hides its layer and ticks whether its spectrum is viewed in
the Spectrum Inspector. The panel owns no state: Controllers apply emitted
changes to their Models and send the result back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal

from core import AnnotationKind, AnnotationLayer, ClassificationLayer
from core.annotation_model import MEASURABLE_KINDS
from ui.annotation_panel import AnnotationToolsPage
from ui.classification_colors import RGBColor, classification_palette
from ui.theme import INSPECTOR_PANEL_WIDTH
from ui.viewer import PixelValueEntry, format_pixel_values_html


_DEFAULT_WIDTH = INSPECTOR_PANEL_WIDTH
# Narrow enough to leave the image room; wide enough for the classifier form.
_MINIMUM_WIDTH = 270
_MAXIMUM_WIDTH = 520
_GRIP_WIDTH = 6


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


class _ResizeGrip(QtWidgets.QWidget):
    """Thin strip on the panel's left edge; dragging it asks for a new width."""

    widthRequested = pyqtSignal(int)

    def __init__(self, panel: QtWidgets.QWidget) -> None:
        super().__init__(panel)
        self.setObjectName("classificationLayerResizeGrip")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(Qt.CursorShape.SplitHCursor)
        self.setToolTip("Drag to resize the panel")
        self._drag_origin: Optional[tuple[float, int]] = None

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = (event.globalPosition().x(), self.parentWidget().width())
            event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag_origin is not None:
            start_x, start_width = self._drag_origin
            # The panel sits right of the image, so dragging left widens it.
            self.widthRequested.emit(round(start_width + start_x - event.globalPosition().x()))
            event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._drag_origin = None


class _LayerRow(QtWidgets.QWidget):
    """One row of the layer list: visibility, name, colour and Spectra.

    ``key`` names the layer (``"class:3"``, ``"annotation:7"``, ``"pin:0"``).
    Clicking the row selects the layer; double-clicking its name renames it.
    """

    clicked = pyqtSignal(str)
    renamed = pyqtSignal(str, str)
    shownChanged = pyqtSignal(str, bool)
    spectraToggled = pyqtSignal(str, bool)

    def __init__(
        self,
        key: str,
        name: str,
        color: RGBColor,
        details: str,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        visible: bool = True,
        spectra: Optional[bool] = False,
        renamable: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("classificationLayerRow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.key = key
        self._name = name

        self._toggle = QtWidgets.QCheckBox(self)
        self._toggle.setObjectName("layerVisibilityToggle")
        self._toggle.setChecked(visible)
        self._toggle.setToolTip(f"Show or hide {name}")
        self._toggle.setAccessibleName(f"Show {name}")
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)

        self._name_label = QtWidgets.QLabel(name, self)
        self._name_label.setObjectName("layerNameLabel")
        self._name_label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self._name_edit = QtWidgets.QLineEdit(self)
        self._name_edit.setObjectName("layerNameEdit")
        self._name_edit.setAccessibleName(f"Rename {name}")
        self._name_edit.hide()
        self._name_edit.editingFinished.connect(self._finish_rename)
        if renamable:
            self._name_label.setToolTip("Double-click to rename")
            self._name_label.installEventFilter(self)

        self._color_swatch = _ColorSwatch(color, self)
        self._count_label = QtWidgets.QLabel(details, self)
        self._count_label.setObjectName("layerPixelCountLabel")

        # None: this kind of layer has no spectrum (arrows, text).
        self._spectra_check = QtWidgets.QCheckBox("Spectra", self)
        self._spectra_check.setObjectName("layerSpectraCheck")
        self._spectra_check.setCursor(Qt.CursorShape.PointingHandCursor)
        self._spectra_check.setChecked(bool(spectra))
        self._spectra_check.setEnabled(spectra is not None)
        self._spectra_check.setToolTip(
            f"View {name} in the Spectrum Inspector" if spectra is not None
            else "Arrows and text enclose no pixels, so they have no spectrum"
        )
        self._spectra_check.setAccessibleName(f"View {name} in the Spectrum Inspector")
        self._spectra_check.toggled.connect(
            lambda checked: self.spectraToggled.emit(self.key, checked)
        )

        name_row = QtWidgets.QHBoxLayout()
        name_row.setContentsMargins(0, 0, 0, 0)
        name_row.setSpacing(8)
        name_row.addWidget(self._name_label, 1)
        name_row.addWidget(self._name_edit, 1)
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
        header_row.addWidget(self._spectra_check, 0, Qt.AlignmentFlag.AlignTop)

        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(11, 10, 11, 10)
        self._layout.setSpacing(7)
        self._layout.addLayout(header_row)
        self._means_label: Optional[QtWidgets.QLabel] = None

        self._toggle.toggled.connect(self._on_toggled)
        self._set_property("layerVisible", visible)
        self._set_property("layerSelected", False)

    @property
    def name(self) -> str:
        return self._name

    def has_spectrum(self) -> bool:
        return self._spectra_check.isEnabled()

    def set_spectra_shown(self, shown: bool) -> None:
        """Tick the Spectra box to match the inspector, without emitting."""

        self._spectra_check.blockSignals(True)
        self._spectra_check.setChecked(shown)
        self._spectra_check.blockSignals(False)

    def set_selected(self, selected: bool) -> None:
        self._set_property("layerSelected", selected)

    def is_selected(self) -> bool:
        return bool(self.property("layerSelected"))

    def _add_means(self, means: Optional[Mapping[str, PixelValueEntry]]) -> None:
        """Per-visualization means, drawn like the viewer's hover pixel values."""

        caption = QtWidgets.QLabel("Index means", self)
        caption.setObjectName("layerIndexMeansCaption")
        self._means_label = QtWidgets.QLabel(self)
        self._means_label.setObjectName("layerIndexMeansLabel")
        self._means_label.setTextFormat(Qt.TextFormat.RichText)
        self._means_label.setText(format_pixel_values_html(means))
        self._layout.addWidget(caption)
        self._layout.addWidget(self._means_label)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.key)
        super().mousePressEvent(event)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if (
            watched is self._name_label
            and event.type() == QtCore.QEvent.Type.MouseButtonDblClick
        ):
            self.begin_rename()
            return True
        return super().eventFilter(watched, event)

    def begin_rename(self) -> None:
        """Swap the name label for an inline editor."""

        self._name_edit.setText(self._name)
        self._name_label.hide()
        self._name_edit.show()
        self._name_edit.setFocus()
        self._name_edit.selectAll()

    def _finish_rename(self) -> None:
        # editingFinished fires on Enter and again on focus loss; only act
        # while the editor is still showing.
        if not self._name_edit.isVisible():
            return
        new_name = self._name_edit.text().strip()
        self._name_edit.hide()
        self._name_label.show()
        if new_name and new_name != self._name:
            self.renamed.emit(self.key, new_name)

    def _set_property(self, name: str, value: bool) -> None:
        self.setProperty(name, value)
        self.style().unpolish(self)
        self.style().polish(self)

    def _on_toggled(self, checked: bool) -> None:
        self._set_property("layerVisible", checked)
        self.shownChanged.emit(self.key, checked)


class _LayerRowWidget(_LayerRow):
    """A class layer: a classification class or a class drawn with the Pen."""

    visibilityChanged = pyqtSignal(int, bool)
    opacityChanged = pyqtSignal(int, float)
    renameRequested = pyqtSignal(int, str)

    def __init__(
        self,
        layer: ClassificationLayer,
        color: RGBColor,
        means: Optional[Mapping[str, PixelValueEntry]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        vectorized: bool = False,
        spectra: bool = False,
    ) -> None:
        details = f"{layer.pixel_count:,} pixels"
        if vectorized:
            details += " · vector mask"
        super().__init__(
            f"class:{layer.class_id}", layer.name, color, details, parent,
            visible=layer.visible, spectra=spectra if layer.pixel_count > 0 else False,
        )
        self.class_id = layer.class_id
        self._spectra_check.setEnabled(layer.pixel_count > 0)
        if vectorized:
            self._count_label.setToolTip(
                "This class's region is set by its vector mask; edit it with the Pen"
            )

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
        opacity_header = QtWidgets.QHBoxLayout()
        opacity_header.setContentsMargins(0, 0, 0, 0)
        opacity_header.setSpacing(6)
        opacity_header.addWidget(self._opacity_caption)
        opacity_header.addStretch(1)
        opacity_header.addWidget(self._opacity_value_label)
        self._layout.addLayout(opacity_header)
        self._layout.addWidget(self._opacity_slider)
        if means is not None:
            self._add_means(means)

        self.shownChanged.connect(lambda _key, shown: self.visibilityChanged.emit(self.class_id, shown))
        self.renamed.connect(lambda _key, name: self.renameRequested.emit(self.class_id, name))
        self._opacity_slider.valueChanged.connect(self._on_slider_changed)

    def _on_slider_changed(self, value: int) -> None:
        self._opacity_value_label.setText(f"{value}%")
        self.opacityChanged.emit(self.class_id, value / 100.0)


_ANNOTATION_KIND_NAMES = {
    AnnotationKind.RECTANGLE: "Rectangle",
    AnnotationKind.ELLIPSE: "Circle",
    AnnotationKind.ARROW: "Arrow",
    AnnotationKind.TEXT: "Text",
}


class _AnnotationRowWidget(_LayerRow):
    """An annotation layer. Rectangles and circles are regions with means and
    a spectrum; arrows and text are shown but have neither."""

    def __init__(
        self,
        layer: AnnotationLayer,
        measurement: Optional[tuple[int, Optional[Mapping[str, PixelValueEntry]]]],
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        spectra: bool = False,
    ) -> None:
        kind = _ANNOTATION_KIND_NAMES[layer.kind]
        region = layer.kind in MEASURABLE_KINDS
        details = kind
        if region and measurement is not None:
            pixel_count, _means = measurement
            details = (
                f"{kind} · {pixel_count:,} pixels" if pixel_count else f"{kind} · outside the image"
            )
        super().__init__(
            f"annotation:{layer.layer_id}", layer.name, tuple(layer.style.stroke[:3]),
            details, parent, visible=layer.visible, spectra=spectra if region else None,
        )
        self.annotation_id = layer.layer_id
        # Measured: a visible rectangle or circle whose pixels are summarised.
        self.measured = region and measurement is not None
        if region and measurement is not None and measurement[1]:
            self._add_means(measurement[1])


class _PinRowWidget(_LayerRow):
    """A Ctrl+Click pin: one pixel's spectrum, numbered like its image marker."""

    def __init__(
        self,
        key: str,
        number: int,
        row: int,
        column: int,
        color: RGBColor,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        visible: bool = True,
        spectra: bool = True,
    ) -> None:
        super().__init__(
            key, f"Pin #{number}", color, f"Pixel ({row}, {column})", parent,
            visible=visible, spectra=spectra, renamable=False,
        )
        self._toggle.setToolTip(f"Show or hide Pin #{number}'s marker on the image")


class ClassificationLayerPanel(QtWidgets.QWidget):
    """Side inspector for segmentation: classifier controls, the options of the
    drawing tool, and every layer. Resizable by its left-edge grip; shown and
    hidden as a whole by the Segment pill on the right stripe."""

    visibilityChanged = pyqtSignal(int, bool)
    opacityChanged = pyqtSignal(int, float)
    renameRequested = pyqtSignal(int, str)
    setAllVisibleRequested = pyqtSignal(bool)
    globalOpacityChanged = pyqtSignal(float)
    outlineModeChanged = pyqtSignal(bool)
    exportRequested = pyqtSignal()
    classSpectraRequested = pyqtSignal()
    # Convert class regions into editable vector masks.
    vectorizeRequested = pyqtSignal()
    regionRenameRequested = pyqtSignal(int, str)
    # A row was clicked: its layer key ("class:3", "annotation:7", "pin:0").
    layerSelected = pyqtSignal(str)
    # A row's Spectra box: view (or stop viewing) that layer's spectrum.
    spectraToggled = pyqtSignal(str, bool)
    annotationVisibilityChanged = pyqtSignal(int, bool)
    # Annotation ID, +1 to bring it forward or -1 to send it back.
    annotationMoveRequested = pyqtSignal(int, int)
    annotationDeleteRequested = pyqtSignal(int)
    pinVisibilityChanged = pyqtSignal(str, bool)
    pinDeleteRequested = pyqtSignal(str)
    # Class ID, +1 to move it up the layer stack or -1 down.
    classMoveRequested = pyqtSignal(int, int)
    classDeleteRequested = pyqtSignal(int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("classificationLayerPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.setMinimumWidth(_MINIMUM_WIDTH)
        self.setMaximumWidth(_DEFAULT_WIDTH)
        self._panel_width = _DEFAULT_WIDTH
        self._resize_grip = _ResizeGrip(self)
        self._resize_grip.widthRequested.connect(self.set_panel_width)

        self._title = QtWidgets.QLabel("Layers", self)
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

        self._summary_label = QtWidgets.QLabel("No layers yet", self)
        self._summary_label.setObjectName("classificationLayerSummary")

        self._toggle_all_button = QtWidgets.QPushButton("Show all", self)
        self._toggle_all_button.setObjectName("layerToggleAllButton")
        self._toggle_all_button.setToolTip("Show or hide every layer")
        self._toggle_all_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_all_button.clicked.connect(self._on_toggle_all_clicked)

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

        self._export_button = QtWidgets.QPushButton("Export CSV", self)
        self._export_button.setObjectName("layerExportButton")
        self._export_button.setToolTip(
            "Save class names, sizes, and index means to a CSV file for analysis"
        )
        self._export_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_button.clicked.connect(self.exportRequested)

        self._spectra_button = QtWidgets.QPushButton("All spectra", self)
        self._spectra_button.setObjectName("layerSpectraButton")
        self._spectra_button.setToolTip(
            "View the mean spectrum (±1σ) of every shown class and region in the "
            "Spectrum Inspector, and compare how separable they are"
        )
        self._spectra_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._spectra_button.clicked.connect(self.classSpectraRequested)

        # Splits the plant class into one class per leaf; the classification
        # controller enables it after a supervised run and handles clicks.
        self.separate_leaves_button = QtWidgets.QPushButton("Separate leaves", self)
        self.separate_leaves_button.setObjectName("separateLeavesButton")
        self.separate_leaves_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.separate_leaves_button.setEnabled(False)

        self._vectorize_button = QtWidgets.QPushButton("Outline classes…", self)
        self._vectorize_button.setObjectName("vectorizeButton")
        self._vectorize_button.setIcon(
            QtGui.QIcon(str(Path(__file__).parent / "assets" / "vector_mask.svg"))
        )
        self._vectorize_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._vectorize_button.setToolTip(
            "Convert class regions into vector masks with editable Bézier "
            "outlines, then reshape them with the Pen"
        )
        self._vectorize_button.setEnabled(False)
        self._vectorize_button.clicked.connect(self.vectorizeRequested)

        export_row = QtWidgets.QHBoxLayout()
        export_row.setContentsMargins(0, 0, 0, 0)
        export_row.setSpacing(7)
        export_row.addWidget(self._export_button, 1)
        export_row.addWidget(self._spectra_button, 1)

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
        controls_layout.addLayout(export_row)
        controls_layout.addWidget(self.separate_leaves_button)
        controls_layout.addWidget(self._vectorize_button)
        controls_layout.addLayout(global_opacity_header)
        controls_layout.addWidget(self._global_opacity_slider)

        layers_label = QtWidgets.QLabel("Layers", self)
        layers_label.setObjectName("classificationLayerSectionLabel")

        self._empty_label = QtWidgets.QLabel(
            "No layers yet\nClassify the image, draw with the toolbox, or "
            "Ctrl+Click a pixel to pin its spectrum.", self
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

        self._up_button = self._layer_action("↑ Up", "Move the selected layer up the stack")
        self._down_button = self._layer_action("↓ Down", "Move the selected layer down the stack")
        self._delete_button = self._layer_action("Delete", "Delete the selected layer")
        self._delete_button.setObjectName("annotationDeleteButton")
        self._up_button.clicked.connect(lambda: self._move_selected(1))
        self._down_button.clicked.connect(lambda: self._move_selected(-1))
        self._delete_button.clicked.connect(self._delete_selected)
        actions_row = QtWidgets.QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(6)
        for button in (self._up_button, self._down_button, self._delete_button):
            actions_row.addWidget(button, 1)

        # The options of the active drawing tool or selected annotation.
        self.annotation_page = AnnotationToolsPage(self)
        self.annotation_page.toolChanged.connect(lambda _kind: self.update_tool_options())

        self._body = QtWidgets.QWidget(self)
        self._body.setObjectName("classificationLayerBody")
        body_layout = QtWidgets.QVBoxLayout(self._body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(10)
        body_layout.addWidget(self.annotation_page)
        body_layout.addWidget(self._summary_label)
        body_layout.addWidget(controls)
        body_layout.addWidget(layers_label)
        body_layout.addWidget(self._empty_label)
        # Rows take their natural height; the whole inspector scrolls.
        body_layout.addWidget(self._rows_container)
        body_layout.addLayout(actions_row)
        body_layout.addStretch(1)

        # Everything scrolls together as one sidebar: no inner scroll areas.
        self._content = QtWidgets.QWidget(self)
        self._content.setObjectName("classificationLayerContent")
        layout = QtWidgets.QVBoxLayout(self._content)
        layout.setContentsMargins(11, 10, 11, 10)
        layout.setSpacing(10)
        layout.addWidget(header)
        layout.setAlignment(header, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self._body, 1)
        self._scroll_area = QtWidgets.QScrollArea(self)
        self._scroll_area.setObjectName("classificationLayerScrollArea")
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._scroll_area.setWidget(self._content)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._scroll_area)

        self._rows: dict[int, _LayerRowWidget] = {}
        self._annotation_rows: dict[int, _AnnotationRowWidget] = {}
        self._pin_rows: dict[str, _PinRowWidget] = {}
        # Keys of the layers whose spectrum the Spectrum Inspector shows.
        self._spectra_keys: frozenset[str] = frozenset()
        self._selected_key: Optional[str] = None
        self.update_tool_options()
        self._update_empty_state()

    def _layer_action(self, text: str, tip: str) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text, self)
        button.setObjectName("annotationLayerAction")
        button.setToolTip(tip)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    @property
    def panel_width(self) -> int:
        """Width the panel takes (user-adjustable with the grip)."""

        return self._panel_width

    def set_panel_width(self, width: int) -> None:
        """Set the panel width, clamped to the supported range."""

        self._panel_width = max(_MINIMUM_WIDTH, min(_MAXIMUM_WIDTH, width))
        self.setMaximumWidth(self._panel_width)
        self.updateGeometry()

    def set_classifier_controls(self, controls: QtWidgets.QWidget) -> None:
        """Show the classifier form at the top of the inspector."""

        controls.setParent(self._body)
        self._body.layout().insertWidget(0, controls)
        controls.show()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._resize_grip.setGeometry(0, 0, _GRIP_WIDTH, self.height())
        self._resize_grip.raise_()

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(self._panel_width, super().sizeHint().height())

    def set_layers(
        self,
        layers: tuple[ClassificationLayer, ...],
        *,
        global_opacity: float = 1.0,
        outline_mode: bool = False,
        means: Optional[Mapping[int, Mapping[str, PixelValueEntry]]] = None,
        fixed_colors: Optional[Mapping[int, RGBColor]] = None,
        vector_class_ids: frozenset[int] = frozenset(),
    ) -> None:
        """Rebuild rows to reflect the Model while preserving global controls.

        ``means`` maps a class ID to that class's per-visualization means,
        keyed by visualization name; classes without an entry show none.
        ``fixed_colors`` pins swatch colours, matching the viewer's class map.
        Classes in ``vector_class_ids`` are marked as set by a vector mask.
        """

        self._clear_rows()
        colors = classification_palette(
            (layer.class_id for layer in layers), fixed_colors
        )
        means = means or {}
        # Top of the stack first, as in Photoshop's Layers panel.
        for layer in reversed(layers):
            row = _LayerRowWidget(
                layer,
                colors[layer.class_id],
                means.get(layer.class_id),
                self._rows_container,
                vectorized=layer.class_id in vector_class_ids,
                spectra=f"class:{layer.class_id}" in self._spectra_keys,
            )
            row.visibilityChanged.connect(self.visibilityChanged)
            row.visibilityChanged.connect(lambda *_: self._update_layer_summary())
            row.opacityChanged.connect(self.opacityChanged)
            row.renameRequested.connect(self.renameRequested)
            self._adopt(row)
            self._rows[layer.class_id] = row
        self._apply_global_controls(global_opacity, outline_mode)
        self._order_rows()

    def set_annotations(
        self,
        layers: tuple[AnnotationLayer, ...],
        measurements: Optional[
            Mapping[int, tuple[int, Optional[Mapping[str, PixelValueEntry]]]]
        ] = None,
    ) -> None:
        """List every annotation, front-most first, above the pins and classes.

        ``measurements`` maps a measured rectangle or circle's ID to its
        ``(pixel_count, means)``. Class rows are left untouched, so this is
        cheap enough to call while a shape is being dragged.
        """

        self._clear_group(self._annotation_rows)
        measurements = measurements or {}
        for layer in reversed(layers):
            row = _AnnotationRowWidget(
                layer, measurements.get(layer.layer_id), self._rows_container,
                spectra=f"annotation:{layer.layer_id}" in self._spectra_keys,
            )
            row.shownChanged.connect(
                lambda _key, shown, layer_id=layer.layer_id:
                    self.annotationVisibilityChanged.emit(layer_id, shown)
            )
            row.renamed.connect(
                lambda _key, name, layer_id=layer.layer_id:
                    self.regionRenameRequested.emit(layer_id, name)
            )
            self._adopt(row)
            self._annotation_rows[layer.layer_id] = row
        self._order_rows()

    def set_pins(
        self, pins: tuple[tuple[str, int, int, int, RGBColor, bool], ...]
    ) -> None:
        """List pins as ``(key, number, row, column, colour, shown on the image)``."""

        self._clear_group(self._pin_rows)
        for key, number, row_index, column, color, marker_visible in pins:
            row = _PinRowWidget(
                key, number, row_index, column, color, self._rows_container,
                visible=marker_visible, spectra=key in self._spectra_keys,
            )
            row.shownChanged.connect(self.pinVisibilityChanged)
            self._adopt(row)
            self._pin_rows[key] = row
        self._order_rows()

    def set_spectra_keys(self, keys: frozenset[str]) -> None:
        """Tick the Spectra box of exactly the layers in ``keys``."""

        self._spectra_keys = frozenset(keys)
        for row in self._all_rows():
            row.set_spectra_shown(row.key in self._spectra_keys)

    def set_selected_layer(self, key: Optional[str]) -> None:
        """Highlight the layer ``key`` (``None`` clears), without emitting."""

        self._selected_key = key
        for row in self._all_rows():
            row.set_selected(row.key == key)
        self._update_layer_actions()
        self.update_tool_options()

    @property
    def selected_layer(self) -> Optional[str]:
        return self._selected_key

    def row_for(self, key: str) -> Optional[_LayerRow]:
        return next((row for row in self._all_rows() if row.key == key), None)

    def _all_rows(self) -> tuple[_LayerRow, ...]:
        return (*self._annotation_rows.values(), *self._pin_rows.values(), *self._rows.values())

    def _adopt(self, row: _LayerRow) -> None:
        row.clicked.connect(self._on_row_clicked)
        row.spectraToggled.connect(self.spectraToggled)
        row.set_selected(row.key == self._selected_key)

    def _order_rows(self) -> None:
        """Annotations (front first), then pins, then classes, then the stretch."""

        for index, row in enumerate(self._all_rows()):
            self._rows_layout.removeWidget(row)
            self._rows_layout.insertWidget(index, row)
        if self._selected_key is not None and self.row_for(self._selected_key) is None:
            self._selected_key = None
        self._update_layer_actions()
        self.update_tool_options()
        self._update_empty_state()

    def _clear_group(self, rows: dict) -> None:
        for row in rows.values():
            self._rows_layout.removeWidget(row)
            row.deleteLater()
        rows.clear()

    def _on_row_clicked(self, key: str) -> None:
        self.set_selected_layer(key)
        self.layerSelected.emit(key)

    def _selected_id(self, prefix: str) -> Optional[int]:
        key = self._selected_key or ""
        return int(key.split(":", 1)[1]) if key.startswith(prefix) else None

    def _selected_annotation(self) -> Optional[int]:
        return self._selected_id("annotation:")

    def _move_selected(self, delta: int) -> None:
        layer_id = self._selected_annotation()
        class_id = self._selected_id("class:")
        if layer_id is not None:
            self.annotationMoveRequested.emit(layer_id, delta)
        elif class_id is not None:
            self.classMoveRequested.emit(class_id, delta)

    def _delete_selected(self) -> None:
        layer_id = self._selected_annotation()
        class_id = self._selected_id("class:")
        if layer_id is not None:
            self.annotationDeleteRequested.emit(layer_id)
        elif class_id is not None:
            self.classDeleteRequested.emit(class_id)
        elif (self._selected_key or "").startswith("pin:"):
            self.pinDeleteRequested.emit(self._selected_key)

    def _update_layer_actions(self) -> None:
        # Annotations and classes each reorder within their own stack; rows
        # list the top of each stack first. Pins have no order.
        for group, selected in (
            (list(self._annotation_rows), self._selected_annotation()),
            (list(self._rows), self._selected_id("class:")),
        ):
            if selected in group:
                index = group.index(selected)
                self._up_button.setEnabled(index > 0)
                self._down_button.setEnabled(index < len(group) - 1)
                self._delete_button.setEnabled(True)
                return
        self._up_button.setEnabled(False)
        self._down_button.setEnabled(False)
        self._delete_button.setEnabled((self._selected_key or "").startswith("pin:"))

    def update_tool_options(self) -> None:
        """Show the tool options while drawing or with an annotation selected."""

        page = self.annotation_page
        drawing = any(button.isChecked() for button in page._tool_buttons.values())
        page.setVisible(drawing or self._selected_annotation() is not None)

    def clear(self) -> None:
        """Remove all rows, reset global controls, and show the empty state."""

        self._clear_rows()
        self._clear_group(self._annotation_rows)
        self._reset_global_controls()
        self._order_rows()

    def _clear_rows(self) -> None:
        for row in self._rows.values():
            self._rows_layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()

    def _on_toggle_all_clicked(self) -> None:
        # Every layer is shown or hidden together, as in one layer stack.
        show = not any(row._toggle.isChecked() for row in self._all_rows())
        for row in (*self._annotation_rows.values(), *self._pin_rows.values()):
            row._toggle.setChecked(show)
        if self._rows:
            self.setAllVisibleRequested.emit(show)
        self._update_layer_summary()

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
        has_content = bool(self._all_rows())
        self._empty_label.setVisible(not has_content)
        self._rows_container.setVisible(has_content)
        self._toggle_all_button.setEnabled(has_content)
        self._outline_toggle_button.setEnabled(has_rows)
        self._vectorize_button.setEnabled(has_rows)
        self._export_button.setEnabled(
            has_rows or any(row.measured for row in self._annotation_rows.values())
        )
        self._spectra_button.setEnabled(
            any(row.has_spectrum() for row in (*self._rows.values(), *self._annotation_rows.values()))
        )
        self._global_opacity_slider.setEnabled(has_rows)
        self._layer_count_label.setText(str(len(self._all_rows())))
        self._update_layer_summary()

    def _update_layer_summary(self) -> None:
        rows = self._all_rows()
        visible_count = sum(row._toggle.isChecked() for row in rows)
        total_count = len(rows)
        if total_count:
            self._summary_label.setText(f"{visible_count} of {total_count} layers shown")
            self._toggle_all_button.setText("Hide all" if visible_count else "Show all")
        else:
            self._summary_label.setText("No layers yet")
            self._toggle_all_button.setText("Show all")

    def _update_toggle_all_button_text(self) -> None:
        """Compatibility wrapper for callers that refresh aggregate text."""

        self._update_layer_summary()
