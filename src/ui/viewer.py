from __future__ import annotations

import math
from typing import Callable, Mapping, NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFontDatabase, QImage, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import QGraphicsRectItem, QGraphicsTextItem, QLabel, QMenu

from ui.annotation_render import make_annotation_item
from core.annotation_model import ROTATABLE_KINDS
from core import (
    OPTIONAL_VISUALIZATION_MODES,
    AnnotationKind,
    AnnotationLayer,
    AnnotationStyle,
)
from ui.theme import (
    CROP_OVERLAY_COLOR,
    CROP_SELECTION_COLOR,
    PROMPT_NEGATIVE_COLOR,
    PROMPT_POINT_RADIUS,
    PROMPT_POSITIVE_COLOR,
    VIEWER_SCENE_BACKGROUND,
    VIEWER_WATERMARK_COLOR,
    VIEWER_WATERMARK_POINT_SIZE,
)

# Annotation items sit above the photo (z=1) and mask (z=3) but below the crop
# overlays (z=10/11), in layer order. Layers are spaced finely so the whole
# stack stays below the overlays however many layers there are.
_ANNOTATION_BASE_Z = 4
_ANNOTATION_Z_STEP = 0.0005
# Edge length, in screen pixels, of a resize handle.
_HANDLE_PX = 8
_SELECTION_PADDING = 3
# Extra screen pixels, beyond the stroke, that still count as clicking an annotation.
_HIT_SLACK_PX = 4
# A press/release closer than this (screen px) is a click, not a drag.
_CLICK_SLOP_PX = 3
_HANDLE_EDGES = {
    "tl": ("l", "t"), "t": ("t",), "tr": ("r", "t"), "r": ("r",),
    "br": ("r", "b"), "b": ("b",), "bl": ("l", "b"), "l": ("l",),
}
# Direction each resize handle points, in degrees clockwise from +x, unrotated.
_HANDLE_BASE_ANGLE = {
    "r": 0, "br": 45, "b": 90, "bl": 135, "l": 180, "tl": 225, "t": 270, "tr": 315,
}
# Screen distance from a shape's top edge to its rotate handle, and handle radius.
_ROTATE_HANDLE_PX = 24
_ROTATE_KNOB_PX = 5
# Holding Shift while rotating snaps to this many degrees.
_ROTATE_SNAP_DEGREES = 15


def _rotate_point(point: QPointF, center: QPointF, degrees: float) -> QPointF:
    """Rotate ``point`` about ``center``; positive is clockwise on screen (y down)."""

    if not degrees:
        return QPointF(point)
    radians = math.radians(degrees)
    cos, sin = math.cos(radians), math.sin(radians)
    dx, dy = point.x() - center.x(), point.y() - center.y()
    return QPointF(center.x() + dx * cos - dy * sin, center.y() + dx * sin + dy * cos)


def _resize_cursor(handle: str, rotation: float) -> Qt.CursorShape:
    """Resize cursor matching the handle's on-screen direction once rotated."""

    angle = (_HANDLE_BASE_ANGLE[handle] + rotation) % 180
    return (
        Qt.CursorShape.SizeHorCursor,
        Qt.CursorShape.SizeFDiagCursor,
        Qt.CursorShape.SizeVerCursor,
        Qt.CursorShape.SizeBDiagCursor,
    )[int(round(angle / 45)) % 4]


def _in_ellipse(bounds: QtCore.QRectF, point: QPointF) -> bool:
    rx, ry = bounds.width() / 2, bounds.height() / 2
    if rx <= 0 or ry <= 0:
        return False
    c = bounds.center()
    return ((point.x() - c.x()) / rx) ** 2 + ((point.y() - c.y()) / ry) ** 2 <= 1


def _distance_to_segment(point: QPointF, line: QtCore.QLineF) -> float:
    dx, dy = line.dx(), line.dy()
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return QtCore.QLineF(point, line.p1()).length()
    t = ((point.x() - line.x1()) * dx + (point.y() - line.y1()) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return QtCore.QLineF(point, QPointF(line.x1() + t * dx, line.y1() + t * dy)).length()


class PixelValueEntry(NamedTuple):
    """A single visualization's reading at one pixel.

    ``color`` is the on-screen RGB colour for that visualization at that
    pixel (i.e. ``display_rgb[row, column]``), used to draw the swatch tile
    beside the numeric/tuple ``value`` in the overlay.
    """
    value: object
    color: tuple[int, int, int]


# Signature for the callback that supplies visualization values for a single hovered pixel.
PixelValueProvider = Callable[
    [int, int], Mapping[str, PixelValueEntry | int]
]


# All visualization modes except HyperCube, in display order.
VISUALIZATION_NAMES = (
    "RGB", "NDVI", "EVI", "MCARI", "MTVI", "OSAVI", "PRI", "NDWI", "NDMI"
)


# Modes that only make sense for some cameras; hidden unless the image has them.
OPTIONAL_VISUALIZATION_NAMES = frozenset(mode.value for mode in OPTIONAL_VISUALIZATION_MODES)


def format_pixel_values_html(
    values: Mapping[str, PixelValueEntry | int],
) -> str:
    """Render visualization readings as rich text: a colour swatch, then the value.

    Shared by the hover pixel-value overlay and the classification layer
    panel's per-class means, so both read identically. A missing entry shows
    an empty swatch and an em dash.
    """

    rows: list[str] = []
    class_id = values.get("Class")
    if isinstance(class_id, (int, np.integer)):
        rows.append(
            '<tr><td width="10" height="10"></td>'
            f'<td style="padding-left:6px;">Class: {int(class_id)}</td></tr>'
        )
    for name in VISUALIZATION_NAMES:
        if name in OPTIONAL_VISUALIZATION_NAMES and name not in values:
            continue
        entry = values.get(name)
        if not isinstance(entry, PixelValueEntry):
            swatch_color = None
            text = "—"
        else:
            swatch_color = entry.color
            if isinstance(entry.value, tuple):
                formatted = ", ".join(f"{component:.0f}" for component in entry.value)
                text = f"({formatted})"
            elif isinstance(entry.value, (int, float)):
                text = f"{entry.value:.3f}"
            else:
                text = "—"
        if swatch_color is not None:
            hex_color = "#{:02x}{:02x}{:02x}".format(*swatch_color)
            swatch = f'<td width="10" height="10" bgcolor="{hex_color}"></td>'
        else:
            swatch = '<td width="10" height="10"></td>'
        rows.append(
            f'<tr>{swatch}'
            f'<td style="padding-left:6px;">{name}: {text}</td></tr>'
        )
    return f'<table cellspacing="3" cellpadding="0">{"".join(rows)}</table>'


class HSIViewer(QtWidgets.QGraphicsView):
    """Interactive graphics view for hyperspectral image display and annotation.

    Emits signals instead of holding a back-reference to the main window,
    keeping the dependency graph acyclic.
    """

    spectrumPlotRequested = pyqtSignal(QPointF)
    meanIndexRequested    = pyqtSignal(str)
    cropRequested         = pyqtSignal(QtCore.QRectF)
    # Payload is a list of (x, y) image-pixel vertex tuples, already clamped
    # to the image. Kept separate from `cropRequested` so the rectangle path
    # and its regression tests stay untouched.
    polygonCropRequested  = pyqtSignal(list)
    # A finished drag with an annotation shape tool: (kind, image-pixel rect).
    annotationShapeDrawn  = pyqtSignal(object, QtCore.QRectF)
    # A click with the text tool, at an image-pixel position.
    annotationTextRequested = pyqtSignal(QPointF)
    # A finished drag with the arrow tool: (tail, head) in image pixels.
    annotationArrowDrawn = pyqtSignal(QPointF, QPointF)
    # The selected annotation was moved/resized: (layer_id, x, y, width, height),
    # emitted continuously during the drag and once more at its end.
    annotationGeometryChanged = pyqtSignal(int, float, float, float, float)
    # The selected annotation was rotated: (layer_id, degrees clockwise).
    annotationRotationChanged = pyqtSignal(int, float)
    annotationInteractionFinished = pyqtSignal()
    # A plain click (no drag, no tool) landed on an annotation.
    annotationPicked = pyqtSignal(int)
    # A plain click (no drag) landed on empty image area.
    annotationDeselected = pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        # --- annotation state ---
        self.input_points: NDArray[np.uint32] = np.empty((0, 2), dtype=np.uint32)
        self.input_labels: list[int] = []

        # --- crop mode state ---
        self._cropping: bool = False
        self._crop_start: Optional[QPointF] = None
        self._crop_rect_item: Optional[QGraphicsRectItem] = None
        self._crop_overlay_item: Optional[QtWidgets.QGraphicsPathItem] = None

        # --- polygon crop mode state ---
        self._polygon_cropping: bool = False
        self._polygon_points: list[QPointF] = []
        self._polygon_item: Optional[QtWidgets.QGraphicsPathItem] = None

        # --- annotation state ---
        self._annotation_tool: Optional[AnnotationKind] = None
        self._annotation_style_provider: Callable[[], AnnotationStyle] = AnnotationStyle
        self._annotation_start: Optional[QPointF] = None
        self._annotation_preview: Optional[QtWidgets.QGraphicsItem] = None
        self._annotation_items: list[QtWidgets.QGraphicsItem] = []
        self._annotation_items_by_id: dict[int, QtWidgets.QGraphicsItem] = {}
        self._annotation_layers: tuple[AnnotationLayer, ...] = ()
        self._selected_layer: Optional[AnnotationLayer] = None
        # (layer_id, "move" or handle name, press point, layer at press, bounds at press)
        self._annotation_interaction: Optional[
            tuple[int, str, QPointF, AnnotationLayer, QtCore.QRectF]
        ] = None
        self._pointer_view_pos: Optional[QtCore.QPoint] = None
        self._pointer_modifiers = Qt.KeyboardModifier.NoModifier
        self._press_view_pos: Optional[QtCore.QPoint] = None

        # --- image data ---
        self._zoom: int = 0
        self._empty: bool = True
        self._pending_view_state: Optional[tuple[float, QPointF]] = None
        self._pending_fit: bool = False
        self.rgb: Optional[NDArray[np.uint8]] = None
        self.mask_array: Optional[NDArray[np.uint8]] = None

        # --- pixel value overlay ---
        self.pixel_value_provider: PixelValueProvider = self._no_pixel_values
        # Optional index names the current image can compute (see above).
        self.available_optional_indices: frozenset[str] = frozenset()
        self._pixel_overlay_enabled: bool = False
        self._pixel_overlay = QLabel(self.viewport())
        self._pixel_overlay.setObjectName("pixelValueOverlay")
        self._pixel_overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._pixel_overlay.setTextFormat(Qt.TextFormat.RichText)
        self._pixel_overlay.hide()
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

        # --- scene items ---
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self.mask_pixmapitem = QtWidgets.QGraphicsPixmapItem()

        self._scene.addItem(self._photo)
        self._scene.addItem(self.mask_pixmapitem)
        self.setScene(self._scene)

        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(VIEWER_SCENE_BACKGROUND))
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.text_item = QGraphicsTextItem("APPN-Tech")
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
        font.setPointSize(VIEWER_WATERMARK_POINT_SIZE)
        font.setBold(True)
        font.setItalic(True)
        self.text_item.setFont(font)
        self.text_item.setDefaultTextColor(VIEWER_WATERMARK_COLOR)
        self.text_item.setTextWidth(-1)
        self.text_item.setPos(
            -self.text_item.boundingRect().center().x(),
            -self.text_item.boundingRect().center().y(),
        )
        self._scene.addItem(self.text_item)
        self.setSceneRect(self.text_item.sceneBoundingRect())

        self.text_item.setZValue(0)
        self._photo.setZValue(1)
        self.mask_pixmapitem.setZValue(3)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def has_photo(self) -> bool:
        return not self._empty

    def photo_size(self) -> Optional[QtCore.QSize]:
        """Pixel dimensions of the currently displayed photo, or None if empty."""
        return self._photo.pixmap().size() if self.has_photo() else None

    def fit_in_view(self) -> None:
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.has_photo():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(
                    viewrect.width() / scenerect.width(),
                    viewrect.height() / scenerect.height(),
                )
                self.scale(factor, factor)
            self._zoom = 0

    def set_photo(self, pixmap: Optional[QPixmap] = None) -> None:
        self._clear()
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self._photo.setPixmap(QtGui.QPixmap())
        self._apply_drag_mode()
        self._apply_idle_cursor()
        self.fit_in_view()
        if self.has_photo():
            # The viewport may not have its final size yet (e.g. right after
            # startup, or in a tab that has never been shown), so the fit
            # above can be computed against stale geometry. Keep reapplying
            # it on resize until the layout settles, mirroring queue_view_state.
            self._pending_fit = True
            QtCore.QTimer.singleShot(300, self._clear_pending_fit)

    def get_view_state(self) -> Optional[tuple[float, QPointF]]:
        """Return (scale_factor, scene_center) describing the current pan/zoom."""
        if not self.has_photo():
            return None
        return self.transform().m11(), self.mapToScene(self.viewport().rect()).boundingRect().center()

    def set_view_state(self, state: tuple[float, QPointF]) -> None:
        """Apply a (scale_factor, scene_center) pair captured via `get_view_state`."""
        if not self.has_photo():
            return
        scale_factor, center = state
        self.resetTransform()
        self.scale(scale_factor, scale_factor)
        self.centerOn(center)
        rect = self._photo.pixmap().rect()
        viewrect = self.viewport().rect()
        if rect.width() and rect.height() and viewrect.width() and viewrect.height():
            fit_scale = min(
                viewrect.width() / rect.width(),
                viewrect.height() / rect.height(),
            )
            step = 1.125  # keep consistent with wheelEvent()
            ratio = scale_factor / fit_scale if fit_scale else 1.0
            steps = float(np.log(ratio) / np.log(step)) if ratio > 0 else 0.0
            self._zoom = max(0, int(round(steps)))
        else:
            # Fall back to a non-zero zoom so wheel-down doesn't immediately snap to fit.
            self._zoom = 1

    def queue_view_state(self, state: tuple[float, QPointF]) -> None:
        """Apply `state` now, and again on every resize for a short window.

        A viewer whose tab has never been shown before doesn't have its
        final viewport size yet, so `centerOn` inside `set_view_state`
        can compute against stale/default geometry, and the surrounding
        layout may still be settling across several resize events. Keep
        reapplying until geometry stops changing, then stop.
        """
        self._pending_view_state = state
        self.set_view_state(state)
        QtCore.QTimer.singleShot(300, self._clear_pending_view_state)

    def _clear_pending_view_state(self) -> None:
        self._pending_view_state = None

    def _clear_pending_fit(self) -> None:
        self._pending_fit = False

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._pending_view_state is not None:
            # The scale was already applied when the state was queued; only
            # the center needs recomputing as the viewport settles.
            self.centerOn(self._pending_view_state[1])
        elif self._pending_fit:
            self.fit_in_view()

    def set_annotation_tool(
        self,
        kind: Optional[AnnotationKind],
        style_provider: Optional[Callable[[], AnnotationStyle]] = None,
    ) -> None:
        """Select the drawing tool; ``None`` restores normal pan/zoom."""

        self._cancel_annotation_drag()
        self._annotation_tool = kind
        if style_provider is not None:
            self._annotation_style_provider = style_provider
        self._apply_drag_mode()
        self._apply_idle_cursor()

    def set_annotations(
        self,
        layers: tuple[AnnotationLayer, ...],
        selected_id: Optional[int] = None,
    ) -> None:
        """Replace the drawn annotations with ``layers`` (bottom to top)."""

        for item in self._annotation_items:
            self._scene.removeItem(item)
        self._annotation_items = []
        self._annotation_items_by_id = {}
        self._annotation_layers = tuple(layer for layer in layers if layer.visible)
        self._selected_layer = None
        for order, layer in enumerate(layers):
            if not layer.visible:
                continue
            item = make_annotation_item(layer)
            item.setZValue(_ANNOTATION_BASE_Z + order * _ANNOTATION_Z_STEP)
            self._scene.addItem(item)
            self._annotation_items.append(item)
            self._annotation_items_by_id[layer.layer_id] = item
            if layer.layer_id == selected_id:
                self._selected_layer = layer
        selected = self._selected_layer
        if selected is not None:
            top_z = _ANNOTATION_BASE_Z + (len(layers) + 1) * _ANNOTATION_Z_STEP
            outline = self._scene.addPolygon(
                QtGui.QPolygonF(self._layer_corners(selected, grow=_SELECTION_PADDING)),
                QPen(QColor(0, 120, 215), 0, Qt.PenStyle.DashLine),
            )
            outline.setZValue(top_z)
            self._annotation_items.append(outline)
            for name, point in self._annotation_handles(selected).items():
                if name == "rotate":
                    continue
                handle = QGraphicsRectItem(
                    -_HANDLE_PX / 2, -_HANDLE_PX / 2, _HANDLE_PX, _HANDLE_PX
                )
                handle.setFlag(
                    QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
                )
                handle.setPen(QPen(QColor(0, 120, 215), 0))
                handle.setBrush(QBrush(QColor(255, 255, 255)))
                handle.setPos(point)
                handle.setZValue(top_z)
                self._scene.addItem(handle)
                self._annotation_items.append(handle)
            if selected.kind in ROTATABLE_KINDS:
                self._add_rotate_handle(selected, top_z)
        self._apply_idle_cursor()

    def _add_rotate_handle(self, layer: AnnotationLayer, z: float) -> None:
        """Draw a knob on a stalk above the shape's top edge.

        The stalk is a screen-sized child of an anchor on the top edge, so it
        keeps its length at any zoom; hit-testing derives the same position.
        """

        rect = self._selection_rect(layer)
        center = rect.center()
        anchor = _rotate_point(QPointF(center.x(), rect.top()), center, layer.rotation)
        stalk = _rotate_point(
            QPointF(0, -_ROTATE_HANDLE_PX), QPointF(0, 0), layer.rotation
        )
        root = QGraphicsRectItem(-0.5, -0.5, 1, 1)
        root.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        root.setPen(QPen(Qt.PenStyle.NoPen))
        root.setPos(anchor)
        root.setZValue(z)
        pen = QPen(QColor(0, 120, 215), 0)
        line = QtWidgets.QGraphicsLineItem(0, 0, stalk.x(), stalk.y(), root)
        line.setPen(pen)
        r = _ROTATE_KNOB_PX
        knob = QtWidgets.QGraphicsEllipseItem(stalk.x() - r, stalk.y() - r, 2 * r, 2 * r, root)
        knob.setPen(pen)
        knob.setBrush(QBrush(QColor(255, 255, 255)))
        self._scene.addItem(root)
        self._annotation_items.append(root)

    def draw_circle(self, point: NDArray[np.uint32], label: int) -> None:
        """Draw a foreground (green) or background (red) prompt circle."""
        color = PROMPT_POSITIVE_COLOR if label == 1 else PROMPT_NEGATIVE_COLOR
        radius = PROMPT_POINT_RADIUS
        x, y = float(point[0]), float(point[1])
        self._scene.addEllipse(
            x - radius, y - radius, radius * 2, radius * 2,
            QPen(color), QBrush(color),
        )

    # ------------------------------------------------------------------ #
    # Qt event overrides                                                   #
    # ------------------------------------------------------------------ #

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if self.has_photo():
            if event.angleDelta().y() > 0:
                factor = 1.125
                self._zoom += 1
            else:
                factor = 1 / 1.125
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fit_in_view()
            else:
                self._zoom = 0
            self._apply_idle_cursor()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self._remember_pointer(event)
        self._press_view_pos = event.position().toPoint()
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.has_photo()
            and not self._cropping
            and not self._polygon_cropping
            and not event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            # A selected annotation under the cursor wins over every tool, so
            # it can be moved/resized without switching tools first.
            point = self.mapToScene(event.position().toPoint())
            mode = self._selected_annotation_hit(point)
            if mode is not None and self._selected_layer is not None:
                layer = self._selected_layer
                self._annotation_interaction = (
                    layer.layer_id, mode, point, layer, self._annotation_bounds(layer)
                )
                self._apply_idle_cursor()
                return
        if self._annotation_tool is not None and self.has_photo():
            if event.button() == Qt.MouseButton.LeftButton:
                point = self._clamped_scene_point(event)
                if self._annotation_tool is AnnotationKind.TEXT:
                    self.annotationTextRequested.emit(point)
                else:
                    self._annotation_start = point
            return
        if self._polygon_cropping:
            if event.button() == Qt.MouseButton.LeftButton:
                self._add_polygon_vertex(
                    self.mapToScene(event.position().toPoint())
                )
            elif event.button() == Qt.MouseButton.RightButton:
                self._finish_polygon_crop()
            return
        if self._cropping:
            if event.button() == Qt.MouseButton.LeftButton:
                self._crop_start = self.mapToScene(event.position().toPoint())
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._polygon_cropping:
            self._finish_polygon_crop()
            return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        self._remember_pointer(event)
        if self._annotation_interaction is not None:
            self._drag_annotation(
                self._clamped_scene_point(event),
                self.mapToScene(event.position().toPoint()),
                bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier),
            )
            self._apply_idle_cursor()
            return
        if self._annotation_start is not None:
            self._update_annotation_preview(self._clamped_scene_point(event))
            return
        if self._polygon_cropping:
            if self._polygon_points:
                self._update_polygon_overlay(
                    self.mapToScene(event.position().toPoint())
                )
            return
        if self._cropping and self._crop_start is not None:
            current = self.mapToScene(event.position().toPoint())
            image_rect = QtCore.QRectF(self._photo.pixmap().rect())
            selection = (
                QtCore.QRectF(self._crop_start, current)
                .normalized()
                .intersected(image_rect)
            )

            if selection.isEmpty():
                if self._crop_rect_item is not None:
                    self._scene.removeItem(self._crop_rect_item)
                    self._crop_rect_item = None
                if self._crop_overlay_item is not None:
                    self._scene.removeItem(self._crop_overlay_item)
                    self._crop_overlay_item = None
                return

            overlay_path = QPainterPath()
            overlay_path.addRect(image_rect)
            overlay_path.addRect(selection)
            overlay_path.setFillRule(Qt.FillRule.OddEvenFill)

            if self._crop_overlay_item is None:
                self._crop_overlay_item = self._scene.addPath(
                    overlay_path,
                    QPen(Qt.PenStyle.NoPen),
                    QBrush(CROP_OVERLAY_COLOR),
                )
                self._crop_overlay_item.setZValue(10)
            else:
                self._crop_overlay_item.setPath(overlay_path)

            if self._crop_rect_item is None:
                self._crop_rect_item = self._scene.addRect(
                    selection,
                    QPen(CROP_SELECTION_COLOR, 0, Qt.PenStyle.DashLine),
                )
                self._crop_rect_item.setZValue(11)
            else:
                self._crop_rect_item.setRect(selection)
            return
        super().mouseMoveEvent(event)
        if event.buttons() == Qt.MouseButton.NoButton:
            self._apply_idle_cursor()
        self._update_pixel_overlay(event.position().toPoint())

    def enterEvent(self, event: QtGui.QEnterEvent) -> None:
        super().enterEvent(event)
        self._pointer_view_pos = self.viewport().mapFromGlobal(QtGui.QCursor.pos())
        self._pointer_modifiers = QtWidgets.QApplication.keyboardModifiers()
        self._apply_idle_cursor()

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        super().leaveEvent(event)
        self._pixel_overlay.hide()
        self._pointer_view_pos = None
        self._apply_idle_cursor()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._remember_pointer(event)
        if self._annotation_interaction is not None:
            if event.button() == Qt.MouseButton.LeftButton:
                self._annotation_interaction = None
                self.annotationInteractionFinished.emit()
                self._apply_idle_cursor()
            return
        if self._annotation_tool is not None:
            start, kind = self._annotation_start, self._annotation_tool
            if start is not None and event.button() == Qt.MouseButton.LeftButton:
                end = self._clamped_scene_point(event)
                self._cancel_annotation_drag()
                is_click = (
                    self._press_view_pos is not None
                    and (event.position().toPoint() - self._press_view_pos).manhattanLength()
                    <= _CLICK_SLOP_PX
                )
                if is_click:
                    picked = self._annotation_at(start)
                    if picked is None:
                        self.annotationDeselected.emit()
                    else:
                        self.annotationPicked.emit(picked)
                    self._apply_idle_cursor()
                    return
                if kind is AnnotationKind.ARROW:
                    if QtCore.QLineF(start, end).length() >= 1:
                        self.annotationArrowDrawn.emit(start, end)
                else:
                    rect = QtCore.QRectF(start, end).normalized()
                    if rect.width() >= 1 and rect.height() >= 1:
                        self.annotationShapeDrawn.emit(kind, rect)
                self._apply_idle_cursor()
            return
        if self._cropping:
            if event.button() != Qt.MouseButton.LeftButton:
                self._end_crop_mode()
                return

            end_pos = self.mapToScene(event.position().toPoint())
            crop_rect = None
            if self._crop_start is not None:
                image_rect = QtCore.QRectF(self._photo.pixmap().rect())
                crop_rect = (
                    QtCore.QRectF(self._crop_start, end_pos)
                    .normalized()
                    .intersected(image_rect)
                )

            self._end_crop_mode()
            if crop_rect is not None and crop_rect.width() >= 1 and crop_rect.height() >= 1:
                self.cropRequested.emit(crop_rect)
            return

        super().mouseReleaseEvent(event)

        if (
            event.button() == Qt.MouseButton.LeftButton
            and not event.modifiers() & Qt.KeyboardModifier.ControlModifier
            and self._press_view_pos is not None
            and (event.position().toPoint() - self._press_view_pos).manhattanLength() <= _CLICK_SLOP_PX
        ):
            picked = self._annotation_at(self.mapToScene(event.position().toPoint()))
            if picked is not None:
                self.annotationPicked.emit(picked)
            else:
                self.annotationDeselected.emit()

        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            # Qt's own ScrollHandDrag release handling just reset the cursor
            # to its default open-hand grab icon; reapply ours on top of it.
            self._apply_idle_cursor()
            return

        clicked_pos = self.mapToScene(event.position().toPoint())

        input_point = np.array(
            [clicked_pos.x(), clicked_pos.y()], dtype=np.uint32
        )
        self.input_points = np.vstack((self.input_points, input_point))

        if event.button() == Qt.MouseButton.LeftButton:
            self.input_labels.append(1)
            self.draw_circle(input_point, 1)
        elif event.button() == Qt.MouseButton.RightButton:
            self.input_labels.append(0)
            self.draw_circle(input_point, 0)
        else:
            return

        if not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self._render_mask()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if self._polygon_cropping:
            if event.key() == Qt.Key.Key_Escape:
                self._end_crop_mode()
                return
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._finish_polygon_crop()
                return
            if event.key() == Qt.Key.Key_Backspace and self._polygon_points:
                self._polygon_points.pop()
                self._update_polygon_overlay(None)
                return
        if event.key() == Qt.Key.Key_Escape and self._annotation_start is not None:
            self._cancel_annotation_drag()
            return
        if event.key() == Qt.Key.Key_Escape and self._cropping:
            self._end_crop_mode()
            return
        if event.key() == Qt.Key.Key_Control:
            self._pointer_modifiers |= Qt.KeyboardModifier.ControlModifier
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
            self._apply_idle_cursor()

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Control:
            self._pointer_modifiers &= ~Qt.KeyboardModifier.ControlModifier
            self._apply_drag_mode()
            self._apply_idle_cursor()
        if event.key() == Qt.Key.Key_Shift:
            self._render_mask()

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        scene_pos = self.mapToScene(event.pos())
        menu = self._build_context_menu(scene_pos)
        menu.exec(event.globalPos())

    def _build_context_menu(self, scene_pos: QPointF) -> QMenu:
        """Build the viewer menu separately so its state remains testable."""
        menu = QMenu(self)
        menu.setObjectName("viewerContextMenu")
        menu.setAccessibleName("Viewer actions")
        menu.setMinimumWidth(252)

        icon_dir = QtCore.QFileInfo(__file__).absoluteDir().filePath("assets")

        spectrum_action = menu.addAction(
            "Spectrum Plot", lambda: self.spectrumPlotRequested.emit(scene_pos)
        )

        index_menu = QMenu("Index Mean", menu)
        index_menu.setObjectName("viewerIndexMenu")
        index_menu.setAccessibleName("Vegetation index mean")
        index_menu.setMinimumWidth(190)
        for name in VISUALIZATION_NAMES:
            if name == "RGB" or (
                name in OPTIONAL_VISUALIZATION_NAMES
                and name not in self.available_optional_indices
            ):
                continue
            index_menu.addAction(
                name, lambda checked=False, name=name: self.meanIndexRequested.emit(name)
            )
        menu.addMenu(index_menu)

        pixel_values_action = menu.addAction("Show Pixel Values")
        pixel_values_action.setCheckable(True)
        pixel_values_action.setChecked(self._pixel_overlay_enabled)
        pixel_values_action.toggled.connect(self._set_pixel_overlay_enabled)

        crop_action = None
        polygon_action = None
        if self.has_photo():
            menu.addSeparator()
            crop_action = menu.addAction("Crop", self._begin_crop_mode)
            polygon_action = menu.addAction(
                "Crop to Polygon", self._begin_polygon_crop_mode
            )
            polygon_action.setToolTip(
                "Click each corner; double-click, right-click or Enter to "
                "apply, Backspace to undo a corner, Esc to cancel"
            )

        spectrum_action.setIcon(QtGui.QIcon(f"{icon_dir}/spectrum_plot.svg"))
        index_menu.setIcon(QtGui.QIcon(f"{icon_dir}/index_mean.svg"))
        pixel_values_action.setIcon(QtGui.QIcon(f"{icon_dir}/pixel_values.svg"))
        if crop_action is not None:
            crop_action.setIcon(QtGui.QIcon(f"{icon_dir}/crop.svg"))
        if polygon_action is not None:
            polygon_action.setIcon(QtGui.QIcon(f"{icon_dir}/crop.svg"))

        return menu

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _unrotated_rect(self, layer: AnnotationLayer) -> QtCore.QRectF:
        """Scene-space box of ``layer`` before rotation (arrows: their extent)."""

        if layer.kind is AnnotationKind.TEXT:
            item = self._annotation_items_by_id.get(layer.layer_id)
            if item is not None:
                return item.boundingRect().translated(item.pos())
            return QtCore.QRectF(layer.x, layer.y, 0, 0)
        return QtCore.QRectF(
            QPointF(layer.x, layer.y),
            QPointF(layer.x + layer.width, layer.y + layer.height),
        ).normalized()

    def _layer_corners(self, layer: AnnotationLayer, grow: float = 0.0) -> list[QPointF]:
        """Corners (tl, tr, br, bl) of the layer's box with rotation applied."""

        base = self._unrotated_rect(layer)
        box = base.adjusted(-grow, -grow, grow, grow)
        corners = [box.topLeft(), box.topRight(), box.bottomRight(), box.bottomLeft()]
        return [_rotate_point(c, base.center(), layer.rotation) for c in corners]

    def _annotation_bounds(self, layer: AnnotationLayer) -> QtCore.QRectF:
        """Axis-aligned scene bounds of ``layer``, including any rotation."""

        return QtGui.QPolygonF(self._layer_corners(layer)).boundingRect()

    def _selection_rect(self, layer: AnnotationLayer) -> QtCore.QRectF:
        """The same padded box used by the outline and its handles."""

        pad = _SELECTION_PADDING
        return self._unrotated_rect(layer).adjusted(-pad, -pad, pad, pad)

    def _annotation_handles(self, layer: AnnotationLayer) -> dict[str, QPointF]:
        if layer.kind is AnnotationKind.ARROW:
            return {
                "tail": QPointF(layer.x, layer.y),
                "head": QPointF(layer.x + layer.width, layer.y + layer.height),
            }
        rect = self._selection_rect(layer)
        center = rect.center()
        handles: dict[str, QPointF] = {}
        if layer.kind is not AnnotationKind.TEXT:
            c = center
            local = {
                "tl": rect.topLeft(), "t": QPointF(c.x(), rect.top()), "tr": rect.topRight(),
                "r": QPointF(rect.right(), c.y()), "br": rect.bottomRight(),
                "b": QPointF(c.x(), rect.bottom()), "bl": rect.bottomLeft(),
                "l": QPointF(rect.left(), c.y()),
            }
            handles = {n: _rotate_point(p, c, layer.rotation) for n, p in local.items()}
        if layer.kind in ROTATABLE_KINDS:
            reach = rect.top() - _ROTATE_HANDLE_PX / self._view_scale()
            handles["rotate"] = _rotate_point(
                QPointF(center.x(), reach), center, layer.rotation
            )
        return handles

    def _view_scale(self) -> float:
        return self.transform().m11() or 1.0

    def _layer_contains(self, layer: AnnotationLayer, point: QPointF) -> bool:
        """Whether ``point`` selects ``layer``.

        The target extends a few screen pixels past the stroke so thin lines are
        easy to grab. An unfilled rectangle or circle only responds near its
        outline, so its empty interior counts as empty image.
        """

        slack = _HIT_SLACK_PX / self._view_scale()
        tolerance = layer.style.stroke_width / 2 + slack
        if layer.kind is AnnotationKind.ARROW:
            line = QtCore.QLineF(
                QPointF(layer.x, layer.y),
                QPointF(layer.x + layer.width, layer.y + layer.height),
            )
            return _distance_to_segment(point, line) <= tolerance
        bounds = self._unrotated_rect(layer)
        point = _rotate_point(point, bounds.center(), -layer.rotation)
        if layer.kind is AnnotationKind.TEXT:
            return bounds.adjusted(-slack, -slack, slack, slack).contains(point)
        outer = bounds.adjusted(-tolerance, -tolerance, tolerance, tolerance)
        inner = bounds.adjusted(tolerance, tolerance, -tolerance, -tolerance)
        hollow = not layer.style.fill_enabled and inner.width() > 0 and inner.height() > 0
        if layer.kind is AnnotationKind.ELLIPSE:
            if not _in_ellipse(outer, point):
                return False
            return not (hollow and _in_ellipse(inner, point))
        if not outer.contains(point):
            return False
        return not (hollow and inner.contains(point))

    def _selected_annotation_hit(self, point: QPointF) -> Optional[str]:
        """Return ``"move"`` / a handle name if ``point`` is on the selected layer."""

        layer = self._selected_layer
        if layer is None:
            return None
        tolerance = (_HANDLE_PX / 2 + 2) / self._view_scale()
        for name, handle in self._annotation_handles(layer).items():
            if abs(point.x() - handle.x()) <= tolerance and abs(point.y() - handle.y()) <= tolerance:
                return name
        return "move" if self._layer_contains(layer, point) else None

    def _annotation_at(self, point: QPointF) -> Optional[int]:
        """Topmost visible annotation under ``point``, if any."""

        for layer in reversed(self._annotation_layers):
            if self._layer_contains(layer, point):
                return layer.layer_id
        return None

    def _drag_annotation(
        self, point: QPointF, raw_point: QPointF, snap: bool = False
    ) -> None:
        """Apply a drag; ``point`` is clamped to the image, ``raw_point`` is not."""

        if self._annotation_interaction is None:
            return
        layer_id, mode, press, layer, bounds = self._annotation_interaction
        image = QtCore.QRectF(self._photo.pixmap().rect())
        if mode == "move":
            dx, dy = point.x() - press.x(), point.y() - press.y()
            if bounds.width() <= image.width():
                dx = min(max(dx, image.left() - bounds.left()), image.right() - bounds.right())
            if bounds.height() <= image.height():
                dy = min(max(dy, image.top() - bounds.top()), image.bottom() - bounds.bottom())
            geometry = (layer.x + dx, layer.y + dy, layer.width, layer.height)
        elif mode == "rotate":
            center = self._unrotated_rect(layer).center()
            now = math.degrees(math.atan2(raw_point.y() - center.y(), raw_point.x() - center.x()))
            then = math.degrees(math.atan2(press.y() - center.y(), press.x() - center.x()))
            degrees = layer.rotation + now - then
            if snap:
                degrees = round(degrees / _ROTATE_SNAP_DEGREES) * _ROTATE_SNAP_DEGREES
            self.annotationRotationChanged.emit(layer_id, degrees)
            return
        elif layer.kind is AnnotationKind.ARROW:
            tail = QPointF(layer.x, layer.y)
            head = QPointF(layer.x + layer.width, layer.y + layer.height)
            if mode == "tail":
                tail = point
            else:
                head = point
            if QtCore.QLineF(tail, head).length() < 1:
                return
            geometry = (tail.x(), tail.y(), head.x() - tail.x(), head.y() - tail.y())
        else:
            # Resize in the shape's own frame so rotated edges follow the
            # cursor, then re-centre so the opposite edge stays put on screen.
            base = self._unrotated_rect(layer)
            center = base.center()
            # Handles sit on the padded selection border. Use displacement
            # from the actual press so grabbing one never jumps the shape.
            delta = (
                _rotate_point(raw_point, center, -layer.rotation)
                - _rotate_point(press, center, -layer.rotation)
            )
            left, top, right, bottom = base.left(), base.top(), base.right(), base.bottom()
            edges = _HANDLE_EDGES[mode]
            grip = QPointF(
                left if "l" in edges else right if "r" in edges else center.x(),
                top if "t" in edges else bottom if "b" in edges else center.y(),
            )
            target = _rotate_point(grip + delta, center, layer.rotation)
            target = QPointF(
                min(max(target.x(), image.left()), image.right()),
                min(max(target.y(), image.top()), image.bottom()),
            )
            delta = _rotate_point(target, center, -layer.rotation) - grip
            if "l" in edges:
                left += delta.x()
            if "r" in edges:
                right += delta.x()
            if "t" in edges:
                top += delta.y()
            if "b" in edges:
                bottom += delta.y()
            rect = QtCore.QRectF(QPointF(left, top), QPointF(right, bottom)).normalized()
            if rect.width() < 1 or rect.height() < 1:
                return
            new_center = _rotate_point(rect.center(), center, layer.rotation)
            geometry = (
                new_center.x() - rect.width() / 2,
                new_center.y() - rect.height() / 2,
                rect.width(),
                rect.height(),
            )
        self.annotationGeometryChanged.emit(layer_id, *geometry)

    def _cursor_for_hit(self, mode: str) -> Qt.CursorShape:
        if mode == "move":
            return Qt.CursorShape.SizeAllCursor
        if mode == "rotate":
            return Qt.CursorShape.PointingHandCursor
        if mode in ("tail", "head"):
            return Qt.CursorShape.CrossCursor
        rotation = self._selected_layer.rotation if self._selected_layer else 0.0
        return _resize_cursor(mode, rotation)

    def _remember_pointer(self, event: QtGui.QMouseEvent) -> None:
        self._pointer_view_pos = event.position().toPoint()
        self._pointer_modifiers = event.modifiers()

    def _clamped_scene_point(self, event: QtGui.QMouseEvent) -> QPointF:
        point = self.mapToScene(event.position().toPoint())
        bounds = QtCore.QRectF(self._photo.pixmap().rect())
        return QPointF(
            min(max(point.x(), bounds.left()), bounds.right()),
            min(max(point.y(), bounds.top()), bounds.bottom()),
        )

    def _update_annotation_preview(self, current: QPointF) -> None:
        if self._annotation_start is None or self._annotation_tool is None:
            return
        start = self._annotation_start
        if self._annotation_tool is AnnotationKind.ARROW:
            geometry = (start.x(), start.y(), current.x() - start.x(), current.y() - start.y())
        else:
            rect = QtCore.QRectF(start, current).normalized()
            geometry = (rect.x(), rect.y(), rect.width(), rect.height())
        if self._annotation_preview is not None:
            self._scene.removeItem(self._annotation_preview)
        preview = make_annotation_item(
            AnnotationLayer(
                -1,
                "",
                self._annotation_tool,
                *geometry,
                style=self._annotation_style_provider(),
            )
        )
        preview.setZValue(_ANNOTATION_BASE_Z + 5)
        self._scene.addItem(preview)
        self._annotation_preview = preview

    def _cancel_annotation_drag(self) -> None:
        self._annotation_start = None
        if self._annotation_preview is not None:
            self._scene.removeItem(self._annotation_preview)
            self._annotation_preview = None

    def _apply_drag_mode(self) -> None:
        if not self.has_photo() or self._annotation_tool is not None:
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        else:
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)

    def _clear(self) -> None:
        self.text_item.setVisible(False)
        self.mask_pixmapitem.setPixmap(QPixmap())
        self.input_points = np.empty((0, 2), dtype=np.uint32)
        self.input_labels = []
        self._pixel_overlay.hide()

    def _begin_crop_mode(self) -> None:
        self._cropping = True
        self._crop_start = None
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        self._apply_idle_cursor()

    def _begin_polygon_crop_mode(self) -> None:
        """Start collecting polygon vertices for an irregular crop."""
        self._polygon_cropping = True
        self._polygon_points = []
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        self._apply_idle_cursor()
        self.setFocus(Qt.FocusReason.OtherFocusReason)

    def _add_polygon_vertex(self, point: QPointF) -> None:
        image_rect = QtCore.QRectF(self._photo.pixmap().rect())
        clamped = QPointF(
            min(max(point.x(), image_rect.left()), image_rect.right()),
            min(max(point.y(), image_rect.top()), image_rect.bottom()),
        )
        self._polygon_points.append(clamped)
        self._update_polygon_overlay(None)

    def _update_polygon_overlay(self, cursor: Optional[QPointF]) -> None:
        """Redraw the in-progress polygon and its dimmed exterior.

        ``cursor`` previews the edge back to the pointer without committing a
        vertex. The exterior uses the same odd-even fill trick as the
        rectangle overlay, so a polygon needs no new drawing machinery.
        """
        if not self._polygon_points:
            return
        image_rect = QtCore.QRectF(self._photo.pixmap().rect())
        polygon = QtGui.QPolygonF(
            self._polygon_points + ([cursor] if cursor is not None else [])
        )

        overlay_path = QPainterPath()
        overlay_path.addRect(image_rect)
        overlay_path.addPolygon(polygon)
        overlay_path.setFillRule(Qt.FillRule.OddEvenFill)
        if self._crop_overlay_item is None:
            self._crop_overlay_item = self._scene.addPath(
                overlay_path, QPen(Qt.PenStyle.NoPen), QBrush(CROP_OVERLAY_COLOR)
            )
            self._crop_overlay_item.setZValue(10)
        else:
            self._crop_overlay_item.setPath(overlay_path)

        outline = QPainterPath()
        outline.addPolygon(polygon)
        outline.closeSubpath()
        if self._polygon_item is None:
            self._polygon_item = self._scene.addPath(
                outline, QPen(CROP_SELECTION_COLOR, 0, Qt.PenStyle.DashLine)
            )
            self._polygon_item.setZValue(11)
        else:
            self._polygon_item.setPath(outline)

    def _finish_polygon_crop(self) -> None:
        points = list(self._polygon_points)
        self._end_crop_mode()
        if len(points) >= 3:
            self.polygonCropRequested.emit(
                [(point.x(), point.y()) for point in points]
            )

    def _end_crop_mode(self) -> None:
        if self._crop_rect_item is not None:
            self._scene.removeItem(self._crop_rect_item)
            self._crop_rect_item = None
        if self._crop_overlay_item is not None:
            self._scene.removeItem(self._crop_overlay_item)
            self._crop_overlay_item = None
        if self._polygon_item is not None:
            self._scene.removeItem(self._polygon_item)
            self._polygon_item = None
        self._cropping = False
        self._crop_start = None
        self._polygon_cropping = False
        self._polygon_points = []
        if self.has_photo():
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self._apply_idle_cursor()

    def _render_mask(self) -> None:
        if self.mask_array is None:
            return
        rgba = np.zeros((*self.mask_array.shape, 4), dtype=np.uint8)
        rgba[:, :, 2] = self.mask_array * 255   # blue channel
        rgba[:, :, 3] = self.mask_array * 128   # 50 % alpha
        self.mask_pixmapitem.setPixmap(QPixmap.fromImage(QImage(
            rgba.data, rgba.shape[1], rgba.shape[0],
            QImage.Format.Format_RGBA8888,
        )))

    def _set_pixel_overlay_enabled(self, enabled: bool) -> None:
        self._pixel_overlay_enabled = enabled
        if not enabled:
            self._pixel_overlay.hide()
        self._apply_idle_cursor()

    def _apply_idle_cursor(self) -> None:
        """Refresh the viewport cursor after pointer, tool or selection changes.

        QGraphicsView's pan handling sets an explicit viewport cursor. Apply
        ours after that handling, and keep the active drag cursor even when
        the controller rebuilds the selected annotation during a drag.
        """
        if self._annotation_interaction is not None:
            mode = self._annotation_interaction[1]
            cursor = (
                Qt.CursorShape.ClosedHandCursor if mode == "move"
                else self._cursor_for_hit(mode)
            )
        elif (self._cropping or self._polygon_cropping
              or self._pointer_modifiers & Qt.KeyboardModifier.ControlModifier):
            cursor = Qt.CursorShape.CrossCursor
        else:
            point = None
            if (self.has_photo() and self._pointer_view_pos is not None
                    and self.viewport().rect().contains(self._pointer_view_pos)):
                point = self.mapToScene(self._pointer_view_pos)
            mode = self._selected_annotation_hit(point) if point is not None else None
            if mode is not None:
                cursor = self._cursor_for_hit(mode)
            elif self._annotation_tool is not None and self.has_photo():
                cursor = Qt.CursorShape.CrossCursor
            elif point is not None and self._annotation_at(point) is not None:
                cursor = Qt.CursorShape.PointingHandCursor
            elif self._pixel_overlay_enabled:
                cursor = Qt.CursorShape.PointingHandCursor
            elif self.has_photo():
                cursor = Qt.CursorShape.OpenHandCursor
            else:
                cursor = Qt.CursorShape.ArrowCursor
        self.viewport().setCursor(cursor)

    def _update_pixel_overlay(self, view_pos: QtCore.QPoint) -> None:
        if not self._pixel_overlay_enabled or not self.has_photo():
            self._pixel_overlay.hide()
            return

        scene_pos = self.mapToScene(view_pos)
        photo_rect = self._photo.pixmap().rect()
        pixel = scene_pos.toPoint()
        if not photo_rect.contains(pixel):
            self._pixel_overlay.hide()
            return

        values = self.pixel_value_provider(pixel.y(), pixel.x())
        self._pixel_overlay.setText(format_pixel_values_html(values))
        self._pixel_overlay.adjustSize()
        self._position_pixel_overlay(view_pos)
        self._pixel_overlay.show()
        self._pixel_overlay.raise_()

    def _position_pixel_overlay(self, view_pos: QtCore.QPoint) -> None:
        offset = QtCore.QPoint(16, 16)
        target = view_pos + offset
        bounds = self.viewport().rect()
        target.setX(min(target.x(), max(0, bounds.width() - self._pixel_overlay.width())))
        target.setY(min(target.y(), max(0, bounds.height() - self._pixel_overlay.height())))
        self._pixel_overlay.move(target)

    @staticmethod
    def _no_pixel_values(row: int, column: int) -> Mapping[str, PixelValueEntry]:
        """Default provider until the controller wires in a real one."""
        return {}
