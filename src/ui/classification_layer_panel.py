"""Layer inspector for classification results.

Reflects a ``core.ClassificationLayerModel`` without owning classification
state. A Controller applies emitted changes to the Model and re-renders the
composited image.
"""

from __future__ import annotations

from typing import Mapping, Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal

from core import ClassificationLayer
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


def _make_inspect_button(name: str, parent: QtWidgets.QWidget) -> QtWidgets.QPushButton:
    """The per-row button that adds one class/region to the Spectrum Inspector."""

    button = QtWidgets.QPushButton("Inspect", parent)
    button.setObjectName("layerInspectButton")
    button.setToolTip(f"Add the mean spectrum of {name} to the Spectrum Inspector")
    button.setAccessibleName(f"Inspect {name} spectrum")
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    return button


class _LayerRowWidget(QtWidgets.QWidget):
    """One class row with selection, colour, count, and opacity controls."""

    visibilityChanged = pyqtSignal(int, bool)
    opacityChanged = pyqtSignal(int, float)
    renameRequested = pyqtSignal(int, str)
    inspectRequested = pyqtSignal(int)

    def __init__(
        self,
        layer: ClassificationLayer,
        color: RGBColor,
        means: Optional[Mapping[str, PixelValueEntry]] = None,
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

        self._name = layer.name
        self._name_label = QtWidgets.QLabel(layer.name, self)
        self._name_label.setObjectName("layerNameLabel")
        self._name_label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self._name_label.setToolTip("Double-click to rename")
        self._name_label.installEventFilter(self)

        self._name_edit = QtWidgets.QLineEdit(self)
        self._name_edit.setObjectName("layerNameEdit")
        self._name_edit.setAccessibleName(f"Rename {layer.name}")
        self._name_edit.hide()
        self._name_edit.editingFinished.connect(self._finish_rename)

        self._color_swatch = _ColorSwatch(color, self)
        self._inspect_button = _make_inspect_button(layer.name, self)
        self._inspect_button.setEnabled(layer.pixel_count > 0)
        self._inspect_button.clicked.connect(lambda: self.inspectRequested.emit(self.class_id))

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
        header_row.addWidget(self._inspect_button, 0, Qt.AlignmentFlag.AlignTop)

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

        # Per-class means of each cached visualization, drawn with the same
        # swatch-and-value markup as the viewer's hover pixel-value overlay.
        self._means_label: Optional[QtWidgets.QLabel] = None
        if means is not None:
            means_caption = QtWidgets.QLabel("Index means", self)
            means_caption.setObjectName("layerIndexMeansCaption")
            self._means_label = QtWidgets.QLabel(self)
            self._means_label.setObjectName("layerIndexMeansLabel")
            self._means_label.setTextFormat(Qt.TextFormat.RichText)
            self._means_label.setText(format_pixel_values_html(means))
            layout.addWidget(means_caption)
            layout.addWidget(self._means_label)

        self._toggle.toggled.connect(self._on_toggled)
        self._opacity_slider.valueChanged.connect(self._on_slider_changed)
        self._set_visible_property(layer.visible)

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
            self.renameRequested.emit(self.class_id, new_name)

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


class _RegionRowWidget(QtWidgets.QWidget):
    """Read-only row for an annotation region: name, pixel count, index means.

    Styled like a class row so the measured region reads as one more class.
    Double-click the name to rename the annotation.
    """

    renameRequested = pyqtSignal(int, str)
    # The row's checkbox: whether "Class spectra" includes this region. It
    # does not hide the annotation (that lives on the Annotations tab).
    selectionChanged = pyqtSignal(int, bool)
    inspectRequested = pyqtSignal(int)

    def __init__(
        self,
        annotation_id: int,
        name: str,
        color: RGBColor,
        pixel_count: int,
        means: Optional[Mapping[str, PixelValueEntry]],
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        selected: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("classificationLayerRow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setProperty("layerVisible", True)
        self.annotation_id = annotation_id
        self._name = name
        self.setToolTip(f"{name}: measured from the annotation drawn on the image")

        self._name_label = QtWidgets.QLabel(name, self)
        self._name_label.setObjectName("layerNameLabel")
        self._name_label.setToolTip("Double-click to rename")
        self._name_label.installEventFilter(self)
        self._name_edit = QtWidgets.QLineEdit(self)
        self._name_edit.setObjectName("layerNameEdit")
        self._name_edit.setAccessibleName(f"Rename {name}")
        self._name_edit.hide()
        self._name_edit.editingFinished.connect(self._finish_rename)
        swatch = _ColorSwatch(color, self)
        self._toggle = QtWidgets.QCheckBox(self)
        self._toggle.setObjectName("layerVisibilityToggle")
        self._toggle.setChecked(selected)
        self._toggle.setToolTip(f"Include {name} when plotting class spectra")
        self._toggle.setAccessibleName(f"Select {name}")
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle.toggled.connect(self._on_toggled)
        self._inspect_button = _make_inspect_button(name, self)
        self._inspect_button.setEnabled(pixel_count > 0)
        self._inspect_button.clicked.connect(
            lambda: self.inspectRequested.emit(self.annotation_id)
        )
        self._count_label = QtWidgets.QLabel(f"{pixel_count:,} pixels · annotation", self)
        self._count_label.setObjectName("layerPixelCountLabel")

        name_row = QtWidgets.QHBoxLayout()
        name_row.setContentsMargins(0, 0, 0, 0)
        name_row.setSpacing(8)
        name_row.addWidget(self._name_label, 1)
        name_row.addWidget(self._name_edit, 1)
        name_row.addWidget(swatch, 0, Qt.AlignmentFlag.AlignVCenter)

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
        header_row.addWidget(self._inspect_button, 0, Qt.AlignmentFlag.AlignTop)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(11, 10, 11, 10)
        layout.setSpacing(5)
        layout.addLayout(header_row)

        self._means_label: Optional[QtWidgets.QLabel] = None
        if means:
            caption = QtWidgets.QLabel("Index means", self)
            caption.setObjectName("layerIndexMeansCaption")
            self._means_label = QtWidgets.QLabel(self)
            self._means_label.setObjectName("layerIndexMeansLabel")
            self._means_label.setTextFormat(Qt.TextFormat.RichText)
            self._means_label.setText(format_pixel_values_html(means))
            layout.addWidget(caption)
            layout.addWidget(self._means_label)
        elif pixel_count == 0:
            self._count_label.setText("Outside the image")
        self._set_selected_property(selected)

    def is_selected(self) -> bool:
        return self._toggle.isChecked()

    def _set_selected_property(self, selected: bool) -> None:
        self.setProperty("layerVisible", selected)
        self.style().unpolish(self)
        self.style().polish(self)

    def _on_toggled(self, checked: bool) -> None:
        self._set_selected_property(checked)
        self.selectionChanged.emit(self.annotation_id, checked)

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
        # editingFinished fires on Enter and again on focus loss.
        if not self._name_edit.isVisible():
            return
        new_name = self._name_edit.text().strip()
        self._name_edit.hide()
        self._name_label.show()
        if new_name and new_name != self._name:
            self.renameRequested.emit(self.annotation_id, new_name)


class ClassificationLayerPanel(QtWidgets.QWidget):
    """Side inspector for segmentation: classifier controls, class layers and
    annotations. Resizable by its left-edge grip; shown and hidden as a whole
    by the Segment pill on the right stripe."""

    visibilityChanged = pyqtSignal(int, bool)
    opacityChanged = pyqtSignal(int, float)
    renameRequested = pyqtSignal(int, str)
    setAllVisibleRequested = pyqtSignal(bool)
    globalOpacityChanged = pyqtSignal(float)
    outlineModeChanged = pyqtSignal(bool)
    backgroundChanged = pyqtSignal(str)
    exportRequested = pyqtSignal()
    classSpectraRequested = pyqtSignal()
    regionRenameRequested = pyqtSignal(int, str)
    # Inspect one class or region: "class:<id>" or "annotation:<id>".
    inspectRequested = pyqtSignal(str)
    # "classes" or "annotations"
    modeChanged = pyqtSignal(str)

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

        self._export_button = QtWidgets.QPushButton("Export CSV", self)
        self._export_button.setObjectName("layerExportButton")
        self._export_button.setToolTip(
            "Save class names, sizes, and index means to a CSV file for analysis"
        )
        self._export_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_button.clicked.connect(self.exportRequested)

        self._spectra_button = QtWidgets.QPushButton("Class spectra", self)
        self._spectra_button.setObjectName("layerSpectraButton")
        self._spectra_button.setToolTip(
            "Plot the mean spectrum (±1σ) of every selected class and measured "
            "annotation region, and compare how separable they are"
        )
        self._spectra_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._spectra_button.clicked.connect(self.classSpectraRequested)

        # Splits the plant class into one class per leaf; the classification
        # controller enables it after a supervised run and handles clicks.
        self.separate_leaves_button = QtWidgets.QPushButton("Separate leaves", self)
        self.separate_leaves_button.setObjectName("separateLeavesButton")
        self.separate_leaves_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.separate_leaves_button.setEnabled(False)

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

        background_label = QtWidgets.QLabel("Visualization", self)
        background_label.setObjectName("layerBackgroundLabel")
        self._background_combo = QtWidgets.QComboBox(self)
        self._background_combo.setObjectName("layerBackgroundCombo")
        self._background_combo.setToolTip(
            "Choose the visualization shown beneath the class layers"
        )
        self._background_combo.setAccessibleName("Background visualization")
        self._background_combo.activated.connect(self._on_background_activated)

        background_row = QtWidgets.QHBoxLayout()
        background_row.setContentsMargins(0, 0, 0, 0)
        background_row.setSpacing(7)
        background_row.addWidget(background_label)
        background_row.addWidget(self._background_combo, 1)

        controls = QtWidgets.QFrame(self)
        controls.setObjectName("classificationLayerControls")
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(9)
        controls_layout.addLayout(background_row)
        controls_layout.addLayout(header_controls_row)
        controls_layout.addLayout(export_row)
        controls_layout.addWidget(self.separate_leaves_button)
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

        self._body = QtWidgets.QWidget(self)
        self._body.setObjectName("classificationLayerBody")
        body_layout = QtWidgets.QVBoxLayout(self._body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(10)
        body_layout.addWidget(self._summary_label)
        body_layout.addWidget(controls)
        body_layout.addWidget(classes_label)
        body_layout.addWidget(self._empty_label, 1)
        # Rows take their natural height; the whole inspector scrolls.
        body_layout.addWidget(self._rows_container, 1)

        self.annotation_page = AnnotationToolsPage(self)

        self._classes_mode_button = self._make_mode_button("Classes", "Show class layers")
        self._annotations_mode_button = self._make_mode_button(
            "Annotations", "Show annotation tools and layers"
        )
        self._classes_mode_button.setChecked(True)
        mode_group = QtWidgets.QButtonGroup(self)
        mode_group.setExclusive(True)
        mode_group.addButton(self._classes_mode_button)
        mode_group.addButton(self._annotations_mode_button)
        self._classes_mode_button.clicked.connect(lambda: self.set_mode("classes"))
        self._annotations_mode_button.clicked.connect(
            lambda: self.set_mode("annotations")
        )
        self._mode_row = QtWidgets.QWidget(self)
        self._mode_row.setObjectName("classificationLayerHeaderDetails")
        mode_layout = QtWidgets.QHBoxLayout(self._mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(0)
        mode_layout.addWidget(self._classes_mode_button)
        mode_layout.addWidget(self._annotations_mode_button)

        self._stack = QtWidgets.QStackedWidget(self)
        self._stack.setObjectName("classificationLayerBody")
        self._stack.addWidget(self._body)
        self._stack.addWidget(self.annotation_page)

        # Everything scrolls together as one sidebar: no inner scroll areas.
        self._content = QtWidgets.QWidget(self)
        self._content.setObjectName("classificationLayerContent")
        layout = QtWidgets.QVBoxLayout(self._content)
        layout.setContentsMargins(11, 10, 11, 10)
        layout.setSpacing(10)
        layout.addWidget(header)
        layout.setAlignment(header, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self._mode_row)
        layout.addWidget(self._stack, 1)
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
        self._fit_stack_to_page()

        self._rows: dict[int, _LayerRowWidget] = {}
        self._region_rows: dict[int, _RegionRowWidget] = {}
        # Region rows are rebuilt on every geometry change; keep their
        # selection by annotation ID so a drag doesn't reset it.
        self._region_selected: dict[int, bool] = {}
        self._update_empty_state()

    def _make_mode_button(self, text: str, tip: str) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text, self)
        button.setObjectName("layerModeButton")
        button.setCheckable(True)
        button.setToolTip(tip)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    @property
    def mode(self) -> str:
        return "annotations" if self._stack.currentWidget() is self.annotation_page else "classes"

    def set_mode(self, mode: str) -> None:
        """Switch between the class layers and the annotation tools/layers."""

        if mode not in ("classes", "annotations"):
            raise ValueError(f"Unknown layer panel mode: {mode!r}")
        annotations = mode == "annotations"
        self._classes_mode_button.setChecked(not annotations)
        self._annotations_mode_button.setChecked(annotations)
        self._title.setText("Annotations" if annotations else "Class layers")
        self._layer_count_label.setVisible(not annotations)
        self._stack.setCurrentWidget(self.annotation_page if annotations else self._body)
        self._fit_stack_to_page()
        if mode != getattr(self, "_announced_mode", "classes"):
            self._announced_mode = mode
            self.modeChanged.emit(mode)

    def _fit_stack_to_page(self) -> None:
        """Size the page stack to the shown page, so the scroll length is
        the current mode's content rather than the taller of the two."""

        for index in range(self._stack.count()):
            page = self._stack.widget(index)
            vertical = (
                QtWidgets.QSizePolicy.Policy.Preferred if page is self._stack.currentWidget()
                else QtWidgets.QSizePolicy.Policy.Ignored
            )
            page.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, vertical)
        self._stack.updateGeometry()

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
        """Show the classifier form at the top of the Classes page."""

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
    ) -> None:
        """Rebuild rows to reflect the Model while preserving global controls.

        ``means`` maps a class ID to that class's per-visualization means,
        keyed by visualization name; classes without an entry show none.
        ``fixed_colors`` pins swatch colours, matching the viewer's class map.
        """

        self._clear_rows()
        colors = classification_palette(
            (layer.class_id for layer in layers), fixed_colors
        )
        means = means or {}
        for layer in layers:
            row = _LayerRowWidget(
                layer,
                colors[layer.class_id],
                means.get(layer.class_id),
                self._rows_container,
            )
            row.visibilityChanged.connect(self.visibilityChanged)
            row.opacityChanged.connect(self.opacityChanged)
            row.renameRequested.connect(self.renameRequested)
            row.inspectRequested.connect(
                lambda class_id: self.inspectRequested.emit(f"class:{class_id}")
            )
            # Class rows stay above any measured-region rows.
            self._rows_layout.insertWidget(len(self._rows), row)
            self._rows[layer.class_id] = row
        self._apply_global_controls(global_opacity, outline_mode)
        self._update_empty_state()

    def set_regions(
        self,
        regions: tuple[tuple[int, str, RGBColor, int, Optional[Mapping[str, PixelValueEntry]]], ...],
    ) -> None:
        """Show measured annotation regions after the class rows.

        Each entry is ``(annotation_id, name, colour, pixel_count, means)``.
        Class rows are left untouched, so this is cheap to call while a shape
        is being dragged.
        """

        self._clear_region_rows()
        for annotation_id, name, color, pixel_count, means in regions:
            row = _RegionRowWidget(
                annotation_id, name, color, pixel_count, means, self._rows_container,
                selected=self._region_selected.get(annotation_id, True),
            )
            row.renameRequested.connect(self.regionRenameRequested)
            row.selectionChanged.connect(self._on_region_selection_changed)
            row.inspectRequested.connect(
                lambda region_id: self.inspectRequested.emit(f"annotation:{region_id}")
            )
            self._region_rows[annotation_id] = row
            self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)
        self._update_empty_state()

    def set_backgrounds(self, names: tuple[str, ...], selected: str) -> None:
        """List the visualizations selectable beneath the class layers."""

        self._background_combo.blockSignals(True)
        self._background_combo.clear()
        self._background_combo.addItems(names)
        index = self._background_combo.findText(selected)
        self._background_combo.setCurrentIndex(max(index, 0))
        self._background_combo.blockSignals(False)
        self._background_combo.setEnabled(len(names) > 1)

    def _on_background_activated(self, index: int) -> None:
        self.backgroundChanged.emit(self._background_combo.itemText(index))

    def clear(self) -> None:
        """Remove all rows, reset global controls, and show the empty state."""

        self._clear_rows()
        self._clear_region_rows()
        self.set_backgrounds((), "")
        self._reset_global_controls()
        self._update_empty_state()

    def _clear_rows(self) -> None:
        for row in self._rows.values():
            self._rows_layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()

    def _clear_region_rows(self) -> None:
        for row in self._region_rows.values():
            self._rows_layout.removeWidget(row)
            row.deleteLater()
        self._region_rows.clear()

    def is_region_selected(self, annotation_id: int) -> bool:
        """Whether the region's row is ticked (new regions start ticked)."""

        return self._region_selected.get(annotation_id, True)

    def _on_region_selection_changed(self, annotation_id: int, selected: bool) -> None:
        self._region_selected[annotation_id] = selected
        self._update_layer_summary()

    def _on_toggle_all_clicked(self) -> None:
        # Classes and regions are selected together, as one list of classes.
        rows = (*self._rows.values(), *self._region_rows.values())
        select = not any(row._toggle.isChecked() for row in rows)
        for row in self._region_rows.values():
            row._toggle.setChecked(select)
        if self._rows:
            self.setAllVisibleRequested.emit(select)

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
        has_content = has_rows or bool(self._region_rows)
        self._empty_label.setVisible(not has_content)
        self._rows_container.setVisible(has_content)
        self._toggle_all_button.setEnabled(has_content)
        self._outline_toggle_button.setEnabled(has_rows)
        self._export_button.setEnabled(has_content)
        self._spectra_button.setEnabled(has_content)
        self._global_opacity_slider.setEnabled(has_rows)
        self._layer_count_label.setText(str(len(self._rows)))
        self._update_layer_summary()

    def _update_layer_summary(self) -> None:
        rows = (*self._rows.values(), *self._region_rows.values())
        visible_count = sum(row._toggle.isChecked() for row in rows)
        total_count = len(rows)
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
