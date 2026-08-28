from __future__ import annotations

from enum import Enum, auto
from typing import Callable, Mapping, NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFontDatabase, QImage, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import QGraphicsRectItem, QGraphicsTextItem, QLabel, QMenu


class PromptMode(Enum):
    """
    Whether the input is prompted via a grouping of points or a rectangular box
    """
    POINTS = auto()
    BOXES  = auto()


class PixelValueEntry(NamedTuple):
    """A single visualization's reading at one pixel.

    ``color`` is the on-screen RGB colour for that visualization at that
    pixel (i.e. ``display_rgb[row, column]``), used to draw the swatch tile
    beside the numeric/tuple ``value`` in the overlay.
    """
    value: object
    color: tuple[int, int, int]


# Signature for the callback that supplies visualization values for a single
# hovered pixel. Keyed by the visualization names in VISUALIZATION_NAMES.
# This is the seam a future visualization model plugs into: assign
# `viewer.pixel_value_provider = ...` the same way the controller already
# assigns `viewer.rgb`/`viewer.mask_array` after loading an image.
PixelValueProvider = Callable[[int, int], Mapping[str, PixelValueEntry]]


class HSIViewer(QtWidgets.QGraphicsView):
    """Interactive graphics view for hyperspectral image display and annotation.

    Emits signals instead of holding a back-reference to the main window,
    keeping the dependency graph acyclic.
    """

    photoClicked          = pyqtSignal(QPointF)
    historyChanged        = pyqtSignal(bool, bool)   # (can_undo, can_redo)
    spectrumPlotRequested = pyqtSignal(QPointF)
    meanIndexRequested    = pyqtSignal(str)
    cropRequested         = pyqtSignal(QtCore.QRectF)

    # All visualization modes except HyperCube, in display order.
    VISUALIZATION_NAMES = ("RGB", "NDVI", "EVI", "MCARI", "MTVI", "OSAVI", "PRI")

    # Placeholder values used until a real pixel_value_provider is wired in.
    _DUMMY_PIXEL_VALUES: Mapping[str, PixelValueEntry] = {
        "RGB": PixelValueEntry((128, 128, 128), (128, 128, 128)),
        "NDVI": PixelValueEntry(0.42, (145, 207, 96)),
        "EVI": PixelValueEntry(0.31, (166, 217, 106)),
        "MCARI": PixelValueEntry(0.55, (94, 201, 98)),
        "MTVI": PixelValueEntry(0.67, (33, 145, 140)),
        "OSAVI": PixelValueEntry(0.38, (215, 173, 96)),
        "PRI": PixelValueEntry(-0.05, (94, 79, 162)),
    }

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        # --- annotation state ---
        self.prompt_mode: PromptMode = PromptMode.POINTS
        self.is_split: bool = False
        self.input_points: NDArray[np.uint32] = np.empty((0, 2), dtype=np.uint32)
        self.input_labels: list[int] = []
        self.input_box: Optional[NDArray[np.float64]] = None
        self.start_point: Optional[QPointF] = None
        self.rect_item: Optional[QGraphicsRectItem] = None
        self.new_input_points: NDArray[np.uint32] = np.empty((0, 2), dtype=np.uint32)
        self.newInputBoxList: list[NDArray[np.float64]] = []
        self.allInputBoxList: list[NDArray[np.float64]] = []
        self.history: list[tuple[str, object]] = []
        self.redo_stack: list[tuple[str, object]] = []

        # --- crop mode state ---
        self._cropping: bool = False
        self._crop_start: Optional[QPointF] = None
        self._crop_rect_item: Optional[QGraphicsRectItem] = None
        self._crop_overlay_item: Optional[QtWidgets.QGraphicsPathItem] = None

        # --- image data ---
        self._zoom: int = 0
        self._empty: bool = True
        self._pending_view_state: Optional[tuple[float, QPointF]] = None
        self._pending_fit: bool = False
        self.rgb: Optional[NDArray[np.uint8]] = None
        self.mask_array: Optional[NDArray[np.uint8]] = None
        self.avatarArray: Optional[NDArray[np.uint8]] = None

        # --- pixel value overlay ---
        self.pixel_value_provider: PixelValueProvider = self._dummy_pixel_values
        self._pixel_overlay_enabled: bool = False
        self._pixel_overlay = QLabel(self.viewport())
        self._pixel_overlay.setObjectName("pixelValueOverlay")
        self._pixel_overlay.setStyleSheet(
            "QLabel#pixelValueOverlay {"
            "  background-color: rgba(20, 20, 20, 200);"
            "  color: #f5f5f5;"
            "  border: 1px solid #555555;"
            "  border-radius: 4px;"
            "  padding: 4px 6px;"
            "  font-size: 11px;"
            "}"
        )
        self._pixel_overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._pixel_overlay.setTextFormat(Qt.TextFormat.RichText)
        self._pixel_overlay.hide()
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

        # --- scene items ---
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self.avatar = QtWidgets.QGraphicsPixmapItem()
        self.mask_pixmapitem = QtWidgets.QGraphicsPixmapItem()

        self._scene.addItem(self._photo)
        self._scene.addItem(self.avatar)
        self._scene.addItem(self.mask_pixmapitem)
        self.setScene(self._scene)

        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.text_item = QGraphicsTextItem("APPN-Tech")
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
        font.setPointSize(45)
        font.setBold(True)
        font.setItalic(True)
        self.text_item.setFont(font)
        self.text_item.setDefaultTextColor(QColor("#eaecee"))
        self.text_item.setTextWidth(-1)
        self.text_item.setPos(
            -self.text_item.boundingRect().center().x(),
            -self.text_item.boundingRect().center().y(),
        )
        self._scene.addItem(self.text_item)
        self.setSceneRect(self.text_item.sceneBoundingRect())

        self.text_item.setZValue(0)
        self._photo.setZValue(1)
        self.avatar.setZValue(2)
        self.mask_pixmapitem.setZValue(3)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def has_photo(self) -> bool:
        return not self._empty

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
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
            self._photo.setPixmap(QtGui.QPixmap())
        self._apply_idle_cursor()
        self.fit_in_view()
        if self.has_photo():
            # The viewport may not have its final size yet (e.g. right after
            # startup, or in a tab that has never been shown), so the fit
            # above can be computed against stale geometry. Keep reapplying
            # it on resize until the layout settles, mirroring queue_view_state.
            self._pending_fit = True
            QtCore.QTimer.singleShot(300, self._clear_pending_fit)

    def set_avatar(self, pixmap: QPixmap) -> None:
        self.avatar.setPixmap(pixmap)

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

    def set_mask(self, mask: NDArray[np.uint8]) -> None:
        """Accept a new mask array and render it as a blue RGBA overlay."""
        self.mask_array = mask
        self._render_mask()

    def draw_circle(self, point: NDArray[np.uint32], label: int) -> None:
        """Draw a foreground (green) or background (red) prompt circle."""
        color = Qt.GlobalColor.green if label == 1 else Qt.GlobalColor.red
        radius = 5.0
        x, y = float(point[0]), float(point[1])
        self._scene.addEllipse(
            x - radius, y - radius, radius * 2, radius * 2,
            QPen(color), QBrush(color),
        )

    def draw_rectangle(self, start: QPointF, end: QPointF) -> None:
        """Draw a blue bounding-box prompt rectangle."""
        self.rect_item = self._scene.addRect(
            QtCore.QRectF(start, end), QPen(Qt.GlobalColor.blue)
        )

    def undo(self) -> None:
        if not self.history:
            return
        prompt_type, prompt_data = self.history.pop()
        self.redo_stack.append((prompt_type, prompt_data))

        if prompt_type == "point":
            self.input_points = self.input_points[:-1]
            if self.input_labels:
                self.input_labels.pop()
        elif prompt_type == "box":
            self.start_point = None
            self.rect_item = None
            self.input_box = None

        items = self._scene.items()
        if items:
            self._scene.removeItem(items[0])
        self._render_mask()
        self.historyChanged.emit(bool(self.history), bool(self.redo_stack))

    def redo(self) -> None:
        if not self.redo_stack:
            return
        prompt_type, prompt_data = self.redo_stack.pop()
        self.history.append((prompt_type, prompt_data))

        if prompt_type == "point":
            self.input_points = np.vstack((self.input_points, prompt_data[1]))
            self.input_labels.append(prompt_data[0])
            self.draw_circle(prompt_data[1], prompt_data[0])
        elif prompt_type == "rectangle":
            self.input_box = prompt_data
            self.draw_rectangle(
                QPointF(float(prompt_data[0]), float(prompt_data[1])),
                QPointF(float(prompt_data[2]), float(prompt_data[3])),
            )

        self._render_mask()
        self.historyChanged.emit(bool(self.history), bool(self.redo_stack))

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

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._cropping:
            if event.button() == Qt.MouseButton.LeftButton:
                self._crop_start = self.mapToScene(event.position().toPoint())
            return
        if self._photo.isUnderMouse():
            self.photoClicked.emit(self.mapToScene(event.position().toPoint()))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
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
                    QBrush(QColor(0, 0, 0, 140)),
                )
                self._crop_overlay_item.setZValue(10)
            else:
                self._crop_overlay_item.setPath(overlay_path)

            if self._crop_rect_item is None:
                self._crop_rect_item = self._scene.addRect(
                    selection,
                    QPen(Qt.GlobalColor.yellow, 0, Qt.PenStyle.DashLine),
                )
                self._crop_rect_item.setZValue(11)
            else:
                self._crop_rect_item.setRect(selection)
            return
        super().mouseMoveEvent(event)
        self._update_pixel_overlay(event.position().toPoint())

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        super().leaveEvent(event)
        self._pixel_overlay.hide()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
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

        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            # Qt's own ScrollHandDrag release handling just reset the cursor
            # to its default open-hand grab icon; reapply ours on top of it.
            self._apply_idle_cursor()
            return

        clicked_pos = self.mapToScene(event.position().toPoint())

        if self.prompt_mode == PromptMode.POINTS:
            input_point = np.array(
                [clicked_pos.x(), clicked_pos.y()], dtype=np.uint32
            )
            self.input_points = np.vstack((self.input_points, input_point))
            if self.is_split:
                self.new_input_points = np.vstack((self.new_input_points, input_point))

            if event.button() == Qt.MouseButton.LeftButton:
                self.input_labels.append(1)
                action: tuple[str, object] = ("point", (1, input_point))
                self.draw_circle(input_point, 1)
            elif event.button() == Qt.MouseButton.RightButton:
                self.input_labels.append(0)
                action = ("point", (0, input_point))
                self.draw_circle(input_point, 0)
            else:
                return
            self.history.append(action)

        elif self.prompt_mode == PromptMode.BOXES and self.start_point is not None:
            self.draw_rectangle(self.start_point, clicked_pos)
            rect = np.array([
                self.start_point.x(), self.start_point.y(),
                clicked_pos.x(), clicked_pos.y(),
            ])
            self.input_box = rect
            self.newInputBoxList.append(rect.copy())
            self.allInputBoxList.append(rect.copy())
            self.history.append(("box", rect))
            self.start_point = None

        self.historyChanged.emit(bool(self.history), bool(self.redo_stack))

        if not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self._render_mask()
            if self.is_split:
                self.new_input_points = np.empty((0, 2), dtype=np.uint32)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape and self._cropping:
            self._end_crop_mode()
            return
        if event.key() == Qt.Key.Key_Control:
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.CrossCursor)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Control:
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
            self._apply_idle_cursor()
        if event.key() == Qt.Key.Key_Shift:
            self.newInputBoxList = []
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
        clear_action = menu.addAction("Clear Selection", self._clear_selection)

        index_menu = QMenu("Index Mean", menu)
        index_menu.setObjectName("viewerIndexMenu")
        index_menu.setAccessibleName("Vegetation index mean")
        index_menu.setMinimumWidth(190)
        for name in self.VISUALIZATION_NAMES:
            if name == "RGB":
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
        if self.has_photo():
            menu.addSeparator()
            crop_action = menu.addAction("Crop", self._begin_crop_mode)

        spectrum_action.setIcon(QtGui.QIcon(f"{icon_dir}/spectrum_plot.svg"))
        clear_action.setIcon(QtGui.QIcon(f"{icon_dir}/clear_selection.svg"))
        index_menu.setIcon(QtGui.QIcon(f"{icon_dir}/index_mean.svg"))
        pixel_values_action.setIcon(QtGui.QIcon(f"{icon_dir}/pixel_values.svg"))
        if crop_action is not None:
            crop_action.setIcon(QtGui.QIcon(f"{icon_dir}/crop.svg"))

        return menu

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _clear(self) -> None:
        self.text_item.setVisible(False)
        self.mask_pixmapitem.setPixmap(QPixmap())
        self.input_points = np.empty((0, 2), dtype=np.uint32)
        self.input_labels = []
        self.history = []
        self.redo_stack = []
        self._pixel_overlay.hide()

    def _clear_selection(self) -> None:
        self._clear()
        self.historyChanged.emit(False, False)

    def _begin_crop_mode(self) -> None:
        self._cropping = True
        self._crop_start = None
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def _end_crop_mode(self) -> None:
        if self._crop_rect_item is not None:
            self._scene.removeItem(self._crop_rect_item)
            self._crop_rect_item = None
        if self._crop_overlay_item is not None:
            self._scene.removeItem(self._crop_overlay_item)
            self._crop_overlay_item = None
        self._cropping = False
        self._crop_start = None
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
        """Set the viewport cursor for the current (non-dragging) state.

        Left untouched while cropping or while Ctrl-crosshair mode is active,
        since those manage their own cursor. Otherwise shows a pointer while
        pixel-value inspection is on, or the usual pan/grab cursor.

        Targets the viewport specifically, not the view itself: ScrollHandDrag
        sets an explicit cursor on the viewport widget, which takes priority
        over whatever cursor the enclosing view has while the mouse is over
        it, so overriding self.setCursor() here would have no visible effect.
        """
        if self._cropping:
            return
        if self._pixel_overlay_enabled:
            self.viewport().setCursor(Qt.CursorShape.PointingHandCursor)
        elif self.has_photo():
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)

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
        self._pixel_overlay.setText(self._format_pixel_values(values))
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

    def _format_pixel_values(self, values: Mapping[str, PixelValueEntry]) -> str:
        rows: list[str] = []
        for name in self.VISUALIZATION_NAMES:
            entry = values.get(name)
            if entry is None:
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

    @classmethod
    def _dummy_pixel_values(cls, row: int, column: int) -> Mapping[str, PixelValueEntry]:
        """Placeholder provider; real per-pixel values arrive with the
        visualization model integration."""
        return cls._DUMMY_PIXEL_VALUES
