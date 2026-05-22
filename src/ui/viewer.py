from __future__ import annotations

from enum import Enum, auto
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QImage, QPen, QPixmap
from PyQt6.QtWidgets import QGraphicsRectItem, QGraphicsTextItem, QMenu


class PromptMode(Enum):
    POINTS = auto()
    BOXES  = auto()


class HSIViewer(QtWidgets.QGraphicsView):
    """Interactive graphics view for hyperspectral image display and annotation.

    Emits signals instead of holding a back-reference to the main window,
    keeping the dependency graph acyclic.
    """

    photoClicked          = pyqtSignal(QPointF)
    historyChanged        = pyqtSignal(bool, bool)   # (can_undo, can_redo)
    spectrumPlotRequested = pyqtSignal(QPointF)
    meanIndexRequested    = pyqtSignal(str)

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

        # --- image data ---
        self._zoom: int = 0
        self._empty: bool = True
        self.rgb: Optional[NDArray[np.uint8]] = None
        self.mask_array: Optional[NDArray[np.uint8]] = None
        self.avatarArray: Optional[NDArray[np.uint8]] = None

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

        self.text_item = QGraphicsTextItem("APPN-Tech")
        font = self.text_item.font()
        font.setPointSize(45)
        font.setBold(True)
        font.setItalic(True)
        self.text_item.setFont(font)
        self.text_item.setDefaultTextColor(QColor("#eaecee"))
        self.text_item.setTextWidth(400)
        self.text_item.setPos(
            -self.text_item.boundingRect().width() / 2,
            -self.text_item.boundingRect().height() / 2,
        )
        self._scene.addItem(self.text_item)

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
        self.fit_in_view()

    def set_avatar(self, pixmap: QPixmap) -> None:
        self.avatar.setPixmap(pixmap)

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
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fit_in_view()
            else:
                self._zoom = 0

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._photo.isUnderMouse():
            self.photoClicked.emit(self.mapToScene(event.position().toPoint()))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        super().mouseReleaseEvent(event)
        clicked_pos = self.mapToScene(event.position().toPoint())

        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            return

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
        if event.key() == Qt.Key.Key_Control:
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.CrossCursor)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Control:
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        if event.key() == Qt.Key.Key_Shift:
            self.newInputBoxList = []
            self._render_mask()

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        scene_pos = self.mapToScene(event.pos())
        menu = QMenu(self)
        menu.addAction("Spectrum Plot", lambda: self.spectrumPlotRequested.emit(scene_pos))
        menu.addAction("Clear Selection", self._clear_selection)
        index_menu = QMenu("Index Mean", self)
        index_menu.addAction("NDVI", lambda: self.meanIndexRequested.emit("NDVI"))
        index_menu.addAction("EVI", lambda: self.meanIndexRequested.emit("EVI"))
        menu.addMenu(index_menu)
        menu.exec(event.globalPos())

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

    def _clear_selection(self) -> None:
        self._clear()
        self.historyChanged.emit(False, False)

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
