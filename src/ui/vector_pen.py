"""Pen tool for editing class vector masks on an :class:`~ui.viewer.HSIViewer`.

The viewer forwards mouse and key events here while the pen is armed. The
tool draws outlines, anchors and handles as scene items and reports every
edit as a whole new :class:`core.VectorMask` through the viewer's
``vectorMaskEdited`` signal, or ``vectorClassDrawn`` for a path drawn with no
mask chosen, which starts a new class; the Controller owns the Model and
sends the masks back with :meth:`VectorPenTool.set_masks`.

Interaction follows Illustrator's pen:

* click empty image to start a path, adding corner points; drag while
  placing a point to pull out smooth handles; click the first point (or
  press Enter, double-click or right-click) to close the path;
* click an outline to select it; click the selected outline to add a point;
* drag a point or a handle to reshape; Alt-drag a handle to break a smooth
  point into a corner; Alt-drag a point to pull new handles out of it;
* Alt-click or double-click a point to toggle it between corner and smooth;
* Delete/Backspace removes the selected point, Esc cancels or deselects.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple, Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QBrush, QColor, QPainterPath, QPen

from core import Anchor, AnchorMode, VectorMask, VectorPath
from core.vector_mask import MIN_PATH_ANCHORS

if TYPE_CHECKING:
    from ui.viewer import HSIViewer

# Above annotations (z=4..) and below pin markers (z=9) and crop overlays.
_VECTOR_Z = 6
# Screen sizes, in pixels, of anchor markers and handle knobs.
_ANCHOR_PX = 7
_KNOB_PX = 3.5
# Screen pixels around a point/outline that still count as hitting it.
_HIT_PX = 6
# A drag shorter than this (screen px) while placing a point keeps it a corner.
_PULL_SLOP_PX = 3
_SELECTION_COLOR = QColor(0, 120, 215)

# (class_id, path_index, anchor_index); anchor_index is -1 for a whole path.
VectorSelection = tuple[int, int, int]


class VectorOutline(NamedTuple):
    """One class's vector mask as shown on the canvas."""

    class_id: int
    color: tuple[int, int, int]
    mask: VectorMask


def vector_painter_path(path: VectorPath) -> QPainterPath:
    """Return the closed Qt path of a :class:`VectorPath`."""

    first = path.anchors[0]
    painter_path = QPainterPath(QPointF(first.x, first.y))
    for index in range(len(path)):
        p0, c1, c2, p3 = path.segment(index)
        if c1 == p0 and c2 == p3:
            painter_path.lineTo(QPointF(*p3))
        else:
            painter_path.cubicTo(QPointF(*c1), QPointF(*c2), QPointF(*p3))
    painter_path.closeSubpath()
    return painter_path


def _cosmetic_pen(color: QColor, width: float, style=Qt.PenStyle.SolidLine) -> QPen:
    pen = QPen(color, width, style)
    pen.setCosmetic(True)
    return pen


class VectorPenTool:
    """State, drawing and hit-testing for the pen; owned by one viewer."""

    def __init__(self, viewer: "HSIViewer") -> None:
        self._viewer = viewer
        self.active = False
        self._outlines: tuple[VectorOutline, ...] = ()
        self._selection: Optional[VectorSelection] = None
        # New paths are added to this class's mask; None starts a new class.
        self._draw_target: Optional[int] = None
        self._items: list[QtWidgets.QGraphicsItem] = []
        # Bounding boxes of every outline, for cheap hit-test rejection.
        self._bounds: list[tuple[int, int, QtCore.QRectF]] = []
        self._draft: list[Anchor] = []
        self._draft_pulling = False
        self._draft_press: Optional[QPointF] = None
        self._cursor: Optional[QPointF] = None
        # (kind, class_id, path_index, anchor_index, side, path at press,
        #  press point, moved); kind is "anchor", "handle" or "pull".
        self._drag: Optional[list] = None

    # ------------------------------------------------------------------ #
    # State from the Controller                                           #
    # ------------------------------------------------------------------ #

    def set_active(self, active: bool) -> None:
        self.active = bool(active)
        self._draft = []
        self._drag = None
        self.redraw()

    def set_masks(
        self,
        outlines: tuple[VectorOutline, ...],
        selection: Optional[VectorSelection],
        draw_target: Optional[int],
    ) -> None:
        self._outlines = tuple(outlines)
        self._selection = selection if self._path_for(selection) is not None else None
        if selection is not None and self._selection is not None:
            path = self._path_for(selection)
            if selection[2] >= len(path):
                self._selection = (selection[0], selection[1], -1)
        self._draw_target = draw_target
        self.redraw()

    @property
    def selection(self) -> Optional[VectorSelection]:
        return self._selection

    @property
    def is_drafting(self) -> bool:
        return bool(self._draft)

    @property
    def is_dragging(self) -> bool:
        return self._drag is not None

    # ------------------------------------------------------------------ #
    # Events forwarded by the viewer                                      #
    # ------------------------------------------------------------------ #

    def press(self, point: QPointF, alt: bool) -> None:
        if self._draft:
            first = self._draft[0]
            if len(self._draft) >= MIN_PATH_ANCHORS and self._near(point, first.point):
                self.finish_draft()
                return
            self._add_draft_anchor(point)
            return
        selected = self._selected_path()
        if selected is not None:
            class_id, path_index, path = selected
            handle = self._handle_at(point)
            if handle is not None:
                self._drag = ["handle", class_id, path_index, handle[0], handle[1],
                              path, point, False]
                return
            anchor = self._anchor_at(point)
            if anchor is not None:
                self._select((class_id, path_index, anchor))
                self._drag = ["pull" if alt else "anchor", class_id, path_index, anchor,
                              None, path, point, False]
                return
        hit = self._outline_at(point)
        if hit is not None:
            class_id, path_index, segment, t = hit
            path = self._path_for((class_id, path_index, -1))
            if selected is not None and (class_id, path_index) == selected[:2]:
                path, anchor = path.insert_anchor(segment, t)
                self._emit(class_id, path_index, path, True)
                self._select((class_id, path_index, anchor))
                self._drag = ["anchor", class_id, path_index, anchor, None,
                              path, point, False]
            else:
                self._select((class_id, path_index, -1))
            return
        self._add_draft_anchor(point)

    def move(self, point: QPointF, alt: bool) -> None:
        self._cursor = point
        if self._drag is not None:
            self._drag_to(point, alt)
            return
        if self._draft:
            if self._draft_pulling and self._draft_press is not None and (
                self._screen_distance(point, self._draft_press) > _PULL_SLOP_PX
            ):
                last = self._draft[-1]
                dx, dy = point.x() - last.x, point.y() - last.y
                self._draft[-1] = Anchor(
                    last.x, last.y, -dx, -dy, dx, dy, AnchorMode.SYMMETRIC
                )
            self.redraw()

    def release(self, alt: bool) -> None:
        if self._drag is not None:
            kind, class_id, path_index, anchor, _side, path, _press, moved = self._drag
            self._drag = None
            if moved:
                current = self._path_for((class_id, path_index, -1))
                if current is not None:
                    self._emit(class_id, path_index, current, True)
            elif kind == "pull":
                # Alt-click without dragging: toggle corner/smooth.
                self._emit(class_id, path_index, path.toggle_mode(anchor), True)
            self._viewer.vectorEditFinished.emit()
            return
        self._draft_pulling = False
        self._draft_press = None

    def double_click(self, point: QPointF) -> None:
        if self._draft:
            # The double-click's first press already placed a point here.
            if len(self._draft) > 1 and self._near(
                QPointF(*self._draft[-1].point), self._draft[-2].point
            ):
                self._draft.pop()
            self.finish_draft()
            return
        selected = self._selected_path()
        if selected is None:
            return
        anchor = self._anchor_at(point)
        if anchor is not None:
            class_id, path_index, path = selected
            self._emit(class_id, path_index, path.toggle_mode(anchor), True)

    def key(self, key: int) -> bool:
        """Handle a key press; return whether the pen consumed it."""

        if self._draft:
            if key == Qt.Key.Key_Escape:
                self.cancel_draft()
            elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.finish_draft()
            elif key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
                self._draft.pop()
                self.redraw()
            else:
                return False
            return True
        if key == Qt.Key.Key_Escape and self._selection is not None:
            self._select(None)
            return True
        if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            return self.delete_selected_anchor()
        return False

    def delete_selected_anchor(self) -> bool:
        """Remove the selected point; the whole path once too few remain."""

        selected = self._selected_path()
        if selected is None or self._selection[2] < 0:
            return False
        class_id, path_index, path = selected
        remaining = path.remove_anchor(self._selection[2])
        self._emit(class_id, path_index, remaining, True)
        self._select(None if remaining is None else (class_id, path_index, -1))
        return True

    def finish_draft(self) -> None:
        draft, self._draft = self._draft, []
        self._draft_pulling = False
        target = self._draw_target
        if len(draft) < MIN_PATH_ANCHORS:
            self.redraw()
            return
        if target is None:
            # The Controller creates the class and sends it back selected.
            self.redraw()
            self._viewer.vectorClassDrawn.emit(VectorMask((VectorPath(tuple(draft)),)))
            return
        mask = self._mask_for(target) or VectorMask()
        updated = mask.added(VectorPath(tuple(draft)))
        self._viewer.vectorMaskEdited.emit(target, updated, True)
        self._select((target, len(updated.paths) - 1, -1))

    def cancel_draft(self) -> None:
        self._draft = []
        self._draft_pulling = False
        self.redraw()

    def cursor_shape(self, point: Optional[QPointF]) -> Qt.CursorShape:
        if self._drag is not None:
            return Qt.CursorShape.ClosedHandCursor
        if point is not None and not self._draft and self._selected_path() is not None:
            if self._handle_at(point) is not None or self._anchor_at(point) is not None:
                return Qt.CursorShape.PointingHandCursor
        return Qt.CursorShape.CrossCursor

    # ------------------------------------------------------------------ #
    # Drawing                                                             #
    # ------------------------------------------------------------------ #

    def redraw(self) -> None:
        scene = self._viewer.scene()
        for item in self._items:
            if item.scene() is scene:
                scene.removeItem(item)
        self._items = []
        self._bounds = []
        if not self.active or not self._viewer.has_photo():
            return
        selection = self._selection
        for outline in self._outlines:
            color = QColor(*outline.color)
            for path_index, path in enumerate(outline.mask.paths):
                painter_path = vector_painter_path(path)
                self._bounds.append(
                    (outline.class_id, path_index, painter_path.controlPointRect())
                )
                chosen = selection is not None and selection[:2] == (
                    outline.class_id, path_index
                )
                self._add_path(painter_path, QColor(0, 0, 0, 170), 3.5 if chosen else 2.5)
                self._add_path(painter_path, color, 2.0 if chosen else 1.2)
        selected = self._selected_path()
        if selected is not None:
            self._draw_handles_and_anchors(selected[2], selection[2])
        if self._draft:
            self._draw_draft()

    def _add_path(self, path: QPainterPath, color: QColor, width: float) -> None:
        item = self._viewer.scene().addPath(path, _cosmetic_pen(color, width))
        item.setZValue(_VECTOR_Z)
        self._items.append(item)

    def _draw_handles_and_anchors(self, path: VectorPath, selected_anchor: int) -> None:
        scene = self._viewer.scene()
        handle_pen = _cosmetic_pen(_SELECTION_COLOR, 1)
        for anchor_index, side, knob in self._visible_handles():
            anchor = path.anchors[anchor_index]
            line = scene.addLine(
                QtCore.QLineF(QPointF(anchor.x, anchor.y), knob), handle_pen
            )
            line.setZValue(_VECTOR_Z + 0.1)
            self._items.append(line)
            self._add_marker(knob, "knob", False)
        for index, anchor in enumerate(path.anchors):
            self._add_marker(
                QPointF(anchor.x, anchor.y), anchor.mode.value, index == selected_anchor
            )

    def _add_marker(self, point: QPointF, shape: str, filled: bool) -> None:
        """A screen-sized marker: square = corner, circle = smooth,
        diamond = symmetric, small circle = handle knob."""

        half = _ANCHOR_PX / 2
        if shape == "knob":
            item = QtWidgets.QGraphicsEllipseItem(-_KNOB_PX, -_KNOB_PX, 2 * _KNOB_PX, 2 * _KNOB_PX)
        elif shape == AnchorMode.SMOOTH.value:
            item = QtWidgets.QGraphicsEllipseItem(-half, -half, 2 * half, 2 * half)
        elif shape == AnchorMode.SYMMETRIC.value:
            reach = half * 1.3
            item = QtWidgets.QGraphicsPolygonItem(QtGui.QPolygonF([
                QPointF(0, -reach), QPointF(reach, 0), QPointF(0, reach), QPointF(-reach, 0)
            ]))
        else:
            item = QtWidgets.QGraphicsRectItem(-half, -half, 2 * half, 2 * half)
        item.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        item.setPen(_cosmetic_pen(_SELECTION_COLOR, 1.2))
        item.setBrush(QBrush(_SELECTION_COLOR if filled else QColor(255, 255, 255)))
        item.setPos(point)
        item.setZValue(_VECTOR_Z + 0.2)
        self._viewer.scene().addItem(item)
        self._items.append(item)

    def _draw_draft(self) -> None:
        anchors = list(self._draft)
        if self._cursor is not None and not self._draft_pulling:
            anchors.append(Anchor(self._cursor.x(), self._cursor.y()))
        painter_path = QPainterPath(QPointF(*anchors[0].point))
        for start, end in zip(anchors, anchors[1:]):
            painter_path.cubicTo(
                QPointF(*start.out_handle), QPointF(*end.in_handle), QPointF(*end.point)
            )
        self._add_path(painter_path, QColor(0, 0, 0, 170), 2.5)
        draft_pen = _cosmetic_pen(_SELECTION_COLOR, 1.5)
        item = self._viewer.scene().addPath(painter_path, draft_pen)
        item.setZValue(_VECTOR_Z + 0.05)
        self._items.append(item)
        last = self._draft[-1]
        if last.has_out_handle:
            for handle in (last.in_handle, last.out_handle):
                line = self._viewer.scene().addLine(
                    QtCore.QLineF(QPointF(*last.point), QPointF(*handle)),
                    _cosmetic_pen(_SELECTION_COLOR, 1),
                )
                line.setZValue(_VECTOR_Z + 0.1)
                self._items.append(line)
                self._add_marker(QPointF(*handle), "knob", False)
        for index, anchor in enumerate(self._draft):
            # The first point is filled to show where to click to close.
            self._add_marker(QPointF(*anchor.point), anchor.mode.value, index == 0)

    # ------------------------------------------------------------------ #
    # Hit testing and edits                                               #
    # ------------------------------------------------------------------ #

    def _tolerance(self) -> float:
        return _HIT_PX / (self._viewer.transform().m11() or 1.0)

    def _near(self, point: QPointF, target: tuple[float, float]) -> bool:
        return math.hypot(point.x() - target[0], point.y() - target[1]) <= self._tolerance()

    def _screen_distance(self, a: QPointF, b: QPointF) -> float:
        return math.hypot(a.x() - b.x(), a.y() - b.y()) * (self._viewer.transform().m11() or 1.0)

    def _mask_for(self, class_id: int) -> Optional[VectorMask]:
        for outline in self._outlines:
            if outline.class_id == class_id:
                return outline.mask
        return None

    def _path_for(self, selection: Optional[VectorSelection]) -> Optional[VectorPath]:
        if selection is None:
            return None
        mask = self._mask_for(selection[0])
        if mask is None or not 0 <= selection[1] < len(mask.paths):
            return None
        return mask.paths[selection[1]]

    def _selected_path(self) -> Optional[tuple[int, int, VectorPath]]:
        path = self._path_for(self._selection)
        if path is None:
            return None
        return self._selection[0], self._selection[1], path

    def _visible_handles(self) -> list[tuple[int, str, QPointF]]:
        """Handles of the selected point and the facing handles of its neighbours."""

        selected = self._selected_path()
        if selected is None or self._selection[2] < 0:
            return []
        path = selected[2]
        index = self._selection[2]
        count = len(path)
        handles = []
        for anchor_index, side in (
            (index, "in"), (index, "out"),
            ((index - 1) % count, "out"), ((index + 1) % count, "in"),
        ):
            anchor = path.anchors[anchor_index]
            present = anchor.has_in_handle if side == "in" else anchor.has_out_handle
            if present:
                point = anchor.in_handle if side == "in" else anchor.out_handle
                handles.append((anchor_index, side, QPointF(*point)))
        return handles

    def _handle_at(self, point: QPointF) -> Optional[tuple[int, str]]:
        for anchor_index, side, knob in self._visible_handles():
            if self._near(point, (knob.x(), knob.y())):
                return anchor_index, side
        return None

    def _anchor_at(self, point: QPointF) -> Optional[int]:
        selected = self._selected_path()
        if selected is None:
            return None
        best, best_distance = None, self._tolerance()
        for index, anchor in enumerate(selected[2].anchors):
            distance = math.hypot(point.x() - anchor.x, point.y() - anchor.y)
            if distance <= best_distance:
                best, best_distance = index, distance
        return best

    def _outline_at(self, point: QPointF) -> Optional[tuple[int, int, int, float]]:
        """Nearest outline within reach: ``(class_id, path, segment, t)``.

        The selected path is tested first so a click on it never selects a
        neighbour drawn on top.
        """

        tolerance = self._tolerance()
        candidates = sorted(
            self._bounds,
            key=lambda entry: self._selection is None or entry[:2] != self._selection[:2],
        )
        best, best_distance = None, tolerance
        for class_id, path_index, bounds in candidates:
            if not bounds.adjusted(-tolerance, -tolerance, tolerance, tolerance).contains(point):
                continue
            path = self._path_for((class_id, path_index, -1))
            segment, t, distance = path.closest_point(point.x(), point.y())
            if distance <= best_distance:
                best, best_distance = (class_id, path_index, segment, t), distance
                if self._selection is not None and (class_id, path_index) == self._selection[:2]:
                    break
        return best

    def _drag_to(self, point: QPointF, alt: bool) -> None:
        kind, class_id, path_index, anchor, side, path, press, _moved = self._drag
        if not self._drag[7] and self._screen_distance(point, press) <= _PULL_SLOP_PX:
            return
        self._drag[7] = True
        if kind == "handle":
            updated = path.move_handle(anchor, side, point.x(), point.y(), break_handles=alt)
        elif kind == "pull":
            updated = path.pull_handles(anchor, point.x(), point.y())
        else:
            start = path.anchors[anchor]
            updated = path.move_anchor(
                anchor,
                start.x + point.x() - press.x(),
                start.y + point.y() - press.y(),
            )
        self._emit(class_id, path_index, updated, False)

    def _add_draft_anchor(self, point: QPointF) -> None:
        self._draft.append(Anchor(point.x(), point.y()))
        self._draft_pulling = True
        self._draft_press = point
        self._cursor = point
        self.redraw()

    def _emit(
        self, class_id: int, path_index: int, path: Optional[VectorPath], final: bool
    ) -> None:
        """Report the edited mask, and show it straight away.

        The Controller echoes the mask back via :meth:`set_masks`; updating
        locally first keeps a drag responsive and its next step consistent.
        """

        mask = self._mask_for(class_id)
        if mask is None:
            return
        updated = mask.with_path(path_index, path)
        self._outlines = tuple(
            outline._replace(mask=updated) if outline.class_id == class_id else outline
            for outline in self._outlines
        )
        self._viewer.vectorMaskEdited.emit(class_id, updated, final)
        self.redraw()

    def _select(self, selection: Optional[VectorSelection]) -> None:
        self._selection = selection
        self.redraw()
        self._viewer.vectorSelectionChanged.emit(*(selection or (-1, -1, -1)))
