"""Stacked annotation layers drawn over an image.

Qt-free Model: Controllers translate these layers into scene items. Layers are
stored bottom-to-top, so a layer's index is its paint order (higher is drawn
above). Coordinates are image pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from enum import Enum
from typing import Optional

from .errors import AnnotationError

RGB = tuple[int, int, int]

_MAX_NAME_TEXT = 18


class AnnotationKind(Enum):
    RECTANGLE = "Rectangle"
    ELLIPSE = "Circle"
    ARROW = "Arrow"
    TEXT = "Text"


ROTATABLE_KINDS = frozenset({AnnotationKind.RECTANGLE, AnnotationKind.TEXT})
# Shapes that enclose an area of pixels, so can report index means inside it.
MEASURABLE_KINDS = frozenset({AnnotationKind.RECTANGLE, AnnotationKind.ELLIPSE})


@dataclass(frozen=True)
class AnnotationStyle:
    """Appearance shared by every kind; text uses ``fill`` as its colour."""

    fill: RGB = (255, 255, 255)
    fill_enabled: bool = False
    stroke: RGB = (0, 0, 0)
    stroke_width: float = 1.0
    font_size: int = 18
    opacity: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.opacity <= 1.0:
            raise AnnotationError("Opacity must be between 0 and 1.")
        if self.stroke_width < 0:
            raise AnnotationError("Stroke width cannot be negative.")
        if self.font_size < 1:
            raise AnnotationError("Text size must be at least 1.")


@dataclass(frozen=True)
class AnnotationLayer:
    """One drawn object.

    Shapes use ``x``/``y``/``width``/``height`` as their bounding box. Text uses
    ``x``/``y`` as its top-left anchor. An arrow runs from ``(x, y)`` to
    ``(x + width, y + height)``, so ``width``/``height`` may be negative.

    ``rotation`` is in degrees, clockwise, about the centre of the unrotated
    box. Only rectangles and text rotate; ``x``/``y``/``width``/``height`` always
    describe the unrotated box.

    ``measure_index`` asks for the mean of each visualization over the pixels
    the shape encloses; only rectangles and ellipses support it. The Model
    enables measurement by default when creating those shapes.
    """

    layer_id: int
    name: str
    kind: AnnotationKind
    x: float
    y: float
    width: float = 0.0
    height: float = 0.0
    text: str = ""
    style: AnnotationStyle = field(default_factory=AnnotationStyle)
    visible: bool = True
    rotation: float = 0.0
    measure_index: bool = False


class AnnotationLayerModel:
    """Ordered stack of :class:`AnnotationLayer` objects."""

    def __init__(self) -> None:
        self._layers: list[AnnotationLayer] = []
        self._next_id = 0
        self._kind_counts: dict[AnnotationKind, int] = {}
        self._undo: list[tuple[AnnotationLayer, ...]] = []
        self._redo: list[tuple[AnnotationLayer, ...]] = []
        self._coalesce_key: Optional[tuple[str, int]] = None

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> bool:
        """Restore the layer stack from before the last change."""

        if not self._undo:
            return False
        self._redo.append(tuple(self._layers))
        self._layers = list(self._undo.pop())
        self._coalesce_key = None
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(tuple(self._layers))
        self._layers = list(self._redo.pop())
        self._coalesce_key = None
        return True

    def break_coalescing(self) -> None:
        """Make the next style edit a new undo step (e.g. after reselecting)."""

        self._coalesce_key = None

    def _record(self, coalesce_key: Optional[tuple[str, int]] = None) -> None:
        """Snapshot the stack before a change.

        Consecutive edits sharing ``coalesce_key`` (a slider drag, a spinbox
        held down) collapse into a single undo step.
        """

        self._redo.clear()
        if coalesce_key is not None and coalesce_key == self._coalesce_key:
            return
        self._coalesce_key = coalesce_key
        self._undo.append(tuple(self._layers))

    @property
    def layers(self) -> tuple[AnnotationLayer, ...]:
        """Layers ordered bottom to top."""

        return tuple(self._layers)

    def __len__(self) -> int:
        return len(self._layers)

    def get(self, layer_id: int) -> AnnotationLayer:
        return self._layers[self._index_of(layer_id)]

    def add_shape(
        self,
        kind: AnnotationKind,
        x: float,
        y: float,
        width: float,
        height: float,
        style: AnnotationStyle,
    ) -> AnnotationLayer:
        if kind is AnnotationKind.TEXT:
            raise AnnotationError("Use add_text() to create text annotations.")
        if kind is AnnotationKind.ARROW:
            raise AnnotationError("Use add_arrow() to create arrows.")
        if width <= 0 or height <= 0:
            raise AnnotationError("Shapes need a positive width and height.")
        return self._append(kind, self._auto_name(kind), x, y, width, height, "", style)

    def add_arrow(
        self, x1: float, y1: float, x2: float, y2: float, style: AnnotationStyle
    ) -> AnnotationLayer:
        if x1 == x2 and y1 == y2:
            raise AnnotationError("Arrows need a non-zero length.")
        return self._append(
            AnnotationKind.ARROW,
            self._auto_name(AnnotationKind.ARROW),
            x1,
            y1,
            x2 - x1,
            y2 - y1,
            "",
            style,
        )

    def add_text(
        self, x: float, y: float, text: str, style: AnnotationStyle
    ) -> AnnotationLayer:
        if not text.strip():
            raise AnnotationError("Text annotations cannot be empty.")
        label = text.strip().splitlines()[0]
        if len(label) > _MAX_NAME_TEXT:
            label = label[: _MAX_NAME_TEXT - 1] + "…"
        self._kind_counts[AnnotationKind.TEXT] = (
            self._kind_counts.get(AnnotationKind.TEXT, 0) + 1
        )
        return self._append(
            AnnotationKind.TEXT, f"Text: {label}", x, y, 0.0, 0.0, text, style
        )

    def remove(self, layer_id: int) -> None:
        index = self._index_of(layer_id)
        self._record()
        del self._layers[index]

    def move(self, layer_id: int, delta: int) -> bool:
        """Shift a layer ``delta`` places up (positive) or down; clamp at the ends.

        Returns whether the order actually changed.
        """

        index = self._index_of(layer_id)
        target = max(0, min(len(self._layers) - 1, index + delta))
        if target == index:
            return False
        self._record()
        self._layers.insert(target, self._layers.pop(index))
        return True

    def set_visible(self, layer_id: int, visible: bool) -> None:
        self._update(layer_id, visible=visible)

    def set_style(self, layer_id: int, style: AnnotationStyle) -> None:
        current = self.get(layer_id).style
        changed = tuple(
            f.name for f in fields(style) if getattr(style, f.name) != getattr(current, f.name)
        )
        if not changed:
            return
        # A drag on one control emits many values for the same field; those
        # collapse into one undo step, while editing a different field starts
        # a new one.
        self._update(layer_id, coalesce_key=("style:" + ",".join(changed), layer_id), style=style)

    def set_geometry(
        self, layer_id: int, x: float, y: float, width: float, height: float
    ) -> None:
        """Move or resize a layer; see :class:`AnnotationLayer` for the meaning.

        Text only has a position, so its extent is ignored. Calls for the same
        layer collapse into one undo step until :meth:`break_coalescing`, so a
        mouse drag is a single step however many moves it reports.
        """

        layer = self.get(layer_id)
        if layer.kind is AnnotationKind.TEXT:
            width = height = 0.0
        elif layer.kind is AnnotationKind.ARROW:
            if width == 0 and height == 0:
                raise AnnotationError("Arrows need a non-zero length.")
        elif width <= 0 or height <= 0:
            raise AnnotationError("Shapes need a positive width and height.")
        if (x, y, width, height) == (layer.x, layer.y, layer.width, layer.height):
            return
        self._update(
            layer_id,
            coalesce_key=("geometry", layer_id),
            x=x,
            y=y,
            width=width,
            height=height,
        )

    def set_rotation(self, layer_id: int, degrees: float) -> None:
        """Rotate a rectangle or text layer; the angle is wrapped to (-180, 180]."""

        layer = self.get(layer_id)
        if layer.kind not in ROTATABLE_KINDS:
            raise AnnotationError(f"{layer.kind.value} annotations cannot be rotated.")
        wrapped = (degrees + 180.0) % 360.0 - 180.0
        if wrapped == -180.0:
            wrapped = 180.0
        if wrapped == layer.rotation:
            return
        self._update(layer_id, coalesce_key=("rotation", layer_id), rotation=wrapped)

    def rename(self, layer_id: int, name: str) -> None:
        """Give a layer a user-facing name; the layer id stays its identity."""

        cleaned = str(name).strip()
        if not cleaned:
            raise AnnotationError("An annotation name cannot be empty.")
        if cleaned == self.get(layer_id).name:
            return
        self._update(layer_id, name=cleaned)

    def set_measure_index(self, layer_id: int, enabled: bool) -> None:
        """Turn index-mean measurement of the pixels inside a shape on or off."""

        layer = self.get(layer_id)
        if enabled and layer.kind not in MEASURABLE_KINDS:
            raise AnnotationError(
                f"{layer.kind.value} annotations do not enclose an area to measure."
            )
        if bool(enabled) == layer.measure_index:
            return
        self._update(layer_id, measure_index=bool(enabled))

    def set_text(self, layer_id: int, text: str) -> None:
        if self.get(layer_id).kind is not AnnotationKind.TEXT:
            raise AnnotationError("Only text annotations have editable text.")
        if not text.strip():
            raise AnnotationError("Text annotations cannot be empty.")
        self._update(layer_id, text=text)

    def clear(self) -> None:
        if self._layers:
            self._record()
        self._layers.clear()

    # ------------------------------------------------------------------ #

    def _append(
        self,
        kind: AnnotationKind,
        name: str,
        x: float,
        y: float,
        width: float,
        height: float,
        text: str,
        style: AnnotationStyle,
    ) -> AnnotationLayer:
        layer = AnnotationLayer(
            self._next_id, name, kind, x, y, width, height, text, style,
            measure_index=kind in MEASURABLE_KINDS,
        )
        self._record()
        self._next_id += 1
        self._layers.append(layer)
        return layer

    def _auto_name(self, kind: AnnotationKind) -> str:
        count = self._kind_counts.get(kind, 0) + 1
        self._kind_counts[kind] = count
        return f"{kind.value} {count}"

    def _index_of(self, layer_id: int) -> int:
        for index, layer in enumerate(self._layers):
            if layer.layer_id == layer_id:
                return index
        raise AnnotationError(f"Annotation layer {layer_id} does not exist.")

    def _update(
        self,
        layer_id: int,
        *,
        coalesce_key: Optional[tuple[str, int]] = None,
        **changes: object,
    ) -> None:
        index = self._index_of(layer_id)
        self._record(coalesce_key)
        self._layers[index] = replace(self._layers[index], **changes)

    def selected_or_none(self, layer_id: Optional[int]) -> Optional[AnnotationLayer]:
        """Return the layer for ``layer_id``, or ``None`` if unset or removed."""

        if layer_id is None:
            return None
        try:
            return self.get(layer_id)
        except AnnotationError:
            return None
