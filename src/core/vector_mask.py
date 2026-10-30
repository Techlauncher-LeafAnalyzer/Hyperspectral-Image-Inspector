"""Editable Bézier outlines ("vector masks") for classification classes.

Qt-free Model. A class's pixel mask can be traced into closed cubic Bézier
paths, edited anchor by anchor like an Illustrator/Photoshop vector mask, and
rasterized back into pixels. The rasterized mask then *is* the class region:
:func:`apply_vector_masks` rewrites the classification result, so spectra,
index means and CSV export all follow the edited outline.

Coordinates are image pixels with ``(0, 0)`` at the top-left corner of the
top-left pixel. A pixel is inside a mask when its centre is inside, matching
:func:`core.annotation_region.region_mask` and :func:`core.roi.polygon_mask`.
Paths of one mask combine with the even-odd rule, as in a compound path: a
path drawn inside another cuts a hole.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Mapping, Optional

import numpy as np

from .classification_model import (
    SupervisedClassificationResult,
    UnsupervisedClassificationResult,
)
from .errors import ClassificationError

ClassificationResult = (
    UnsupervisedClassificationResult | SupervisedClassificationResult
)
Point = tuple[float, float]

# Fewest anchors that still enclose an area.
MIN_PATH_ANCHORS = 3
# Bézier handle length as a fraction of the neighbouring side, for handles
# generated automatically (tracing, or converting a corner to smooth).
_AUTO_HANDLE_FRACTION = 1.0 / 3.0
# Flattened curve segments are at most about this long, in pixels.
_FLATTEN_STEP_PX = 1.0
_MAX_FLATTEN_STEPS = 256


class AnchorMode(Enum):
    """How an anchor's two direction handles relate, as in Illustrator.

    ``CORNER`` handles move independently (and are usually retracted), giving
    a sharp turn. ``SMOOTH`` keeps the handles on one line so the curve flows
    through the anchor, but each may have its own length. ``SYMMETRIC`` also
    keeps both handles the same length.
    """

    CORNER = "Corner"
    SMOOTH = "Smooth"
    SYMMETRIC = "Symmetric"


@dataclass(frozen=True, slots=True)
class Anchor:
    """One anchor point and its handles, stored relative to the point.

    ``in_handle`` shapes the segment arriving at the anchor and
    ``out_handle`` the segment leaving it. A zero offset is a retracted
    handle, which makes that side of the anchor straight.
    """

    x: float
    y: float
    in_dx: float = 0.0
    in_dy: float = 0.0
    out_dx: float = 0.0
    out_dy: float = 0.0
    mode: AnchorMode = AnchorMode.CORNER

    @property
    def point(self) -> Point:
        return (self.x, self.y)

    @property
    def in_handle(self) -> Point:
        return (self.x + self.in_dx, self.y + self.in_dy)

    @property
    def out_handle(self) -> Point:
        return (self.x + self.out_dx, self.y + self.out_dy)

    @property
    def has_in_handle(self) -> bool:
        return bool(self.in_dx or self.in_dy)

    @property
    def has_out_handle(self) -> bool:
        return bool(self.out_dx or self.out_dy)


@dataclass(frozen=True, slots=True)
class VectorPath:
    """A closed path of cubic segments; segment ``i`` runs anchor ``i`` → ``i+1``.

    Editing methods return a new path and leave this one unchanged.
    """

    anchors: tuple[Anchor, ...]

    def __post_init__(self) -> None:
        if len(self.anchors) < MIN_PATH_ANCHORS:
            raise ClassificationError(
                f"A vector path needs at least {MIN_PATH_ANCHORS} anchor points."
            )

    def __len__(self) -> int:
        return len(self.anchors)

    def segment(self, index: int) -> tuple[Point, Point, Point, Point]:
        """Return the Bézier control points ``(p0, c1, c2, p3)`` of a segment."""

        start = self.anchors[index % len(self.anchors)]
        end = self.anchors[(index + 1) % len(self.anchors)]
        return start.point, start.out_handle, end.in_handle, end.point

    def flatten(self) -> np.ndarray:
        """Return the outline as an ``(N, 2)`` polyline of ``(x, y)`` points.

        The polyline is implicitly closed (the first point is not repeated).
        Curved segments are split into roughly pixel-long pieces.
        """

        pieces = [
            _flatten_segment(*self.segment(index))[:-1]
            for index in range(len(self.anchors))
        ]
        return np.concatenate(pieces, axis=0)

    def closest_point(self, x: float, y: float) -> tuple[int, float, float]:
        """Return ``(segment_index, t, distance)`` of the outline point nearest ``(x, y)``."""

        best = (0, 0.0, math.inf)
        target = np.array([x, y], dtype=np.float64)
        for index in range(len(self.anchors)):
            control = self.segment(index)
            steps = _segment_steps(control)
            t = np.linspace(0.0, 1.0, steps * 4 + 1)
            points = _bezier_points(control, t)
            distances = np.hypot(*(points - target).T)
            nearest = int(np.argmin(distances))
            if distances[nearest] < best[2]:
                best = (index, float(t[nearest]), float(distances[nearest]))
        return best

    def area(self) -> float:
        """Unsigned area enclosed by the flattened outline."""

        return abs(_signed_area(self.flatten()))

    # -- editing -------------------------------------------------------- #

    def move_anchor(self, index: int, x: float, y: float) -> "VectorPath":
        """Move an anchor; its handles travel with it."""

        anchor = self.anchors[index]
        return self._with_anchor(index, replace(anchor, x=float(x), y=float(y)))

    def move_handle(
        self,
        index: int,
        side: str,
        x: float,
        y: float,
        *,
        break_handles: bool = False,
    ) -> "VectorPath":
        """Drag one handle (``side`` is ``"in"`` or ``"out"``) to ``(x, y)``.

        A smooth anchor turns its opposite handle to stay in line, keeping
        that handle's length; a symmetric anchor mirrors it exactly. With
        ``break_handles`` (Alt-drag in Illustrator) the anchor becomes a
        corner, so the opposite handle stays where it is.
        """

        if side not in ("in", "out"):
            raise ClassificationError(f"Unknown handle side: {side!r}")
        anchor = self.anchors[index]
        dx, dy = float(x) - anchor.x, float(y) - anchor.y
        mode = AnchorMode.CORNER if break_handles else anchor.mode
        if side == "in":
            other = (anchor.out_dx, anchor.out_dy)
        else:
            other = (anchor.in_dx, anchor.in_dy)
        if mode is AnchorMode.SYMMETRIC:
            other = (-dx, -dy)
        elif mode is AnchorMode.SMOOTH:
            length = math.hypot(dx, dy)
            other_length = math.hypot(*other)
            if length > 0:
                other = (-dx / length * other_length, -dy / length * other_length)
        if side == "in":
            updated = replace(
                anchor, in_dx=dx, in_dy=dy, out_dx=other[0], out_dy=other[1], mode=mode
            )
        else:
            updated = replace(
                anchor, out_dx=dx, out_dy=dy, in_dx=other[0], in_dy=other[1], mode=mode
            )
        return self._with_anchor(index, updated)

    def pull_handles(self, index: int, x: float, y: float) -> "VectorPath":
        """Pull symmetric handles out of an anchor towards ``(x, y)``.

        Matches dragging from an anchor with the pen in Illustrator: the out
        handle follows the cursor and the in handle mirrors it.
        """

        anchor = self.anchors[index]
        dx, dy = float(x) - anchor.x, float(y) - anchor.y
        updated = replace(
            anchor, out_dx=dx, out_dy=dy, in_dx=-dx, in_dy=-dy, mode=AnchorMode.SYMMETRIC
        )
        return self._with_anchor(index, updated)

    def set_mode(self, index: int, mode: AnchorMode) -> "VectorPath":
        """Convert an anchor, as Illustrator's *Convert anchor point* does.

        To corner: both handles retract, so the anchor is a sharp point.
        To smooth or symmetric: an anchor without handles gets handles
        tangent to its neighbours; existing handles are lined up along their
        average direction (and, for symmetric, given their average length).
        """

        anchor = self.anchors[index]
        if mode is AnchorMode.CORNER:
            updated = Anchor(anchor.x, anchor.y, mode=AnchorMode.CORNER)
            return self._with_anchor(index, updated)
        if not (anchor.has_in_handle or anchor.has_out_handle):
            updated = _auto_smooth_anchor(
                self.anchors[index - 1], anchor, self.anchors[(index + 1) % len(self)], mode
            )
            return self._with_anchor(index, updated)
        in_length = math.hypot(anchor.in_dx, anchor.in_dy)
        out_length = math.hypot(anchor.out_dx, anchor.out_dy)
        # Direction of travel through the anchor, from both handles.
        dir_x = anchor.out_dx - anchor.in_dx
        dir_y = anchor.out_dy - anchor.in_dy
        norm = math.hypot(dir_x, dir_y)
        if norm == 0:
            return self._with_anchor(index, replace(anchor, mode=mode))
        dir_x, dir_y = dir_x / norm, dir_y / norm
        if mode is AnchorMode.SYMMETRIC:
            in_length = out_length = (in_length + out_length) / 2.0
        updated = replace(
            anchor,
            in_dx=-dir_x * in_length,
            in_dy=-dir_y * in_length,
            out_dx=dir_x * out_length,
            out_dy=dir_y * out_length,
            mode=mode,
        )
        return self._with_anchor(index, updated)

    def toggle_mode(self, index: int) -> "VectorPath":
        """Corner ↔ smooth, like Alt-clicking an anchor with the pen."""

        corner = self.anchors[index].mode is AnchorMode.CORNER
        return self.set_mode(index, AnchorMode.SMOOTH if corner else AnchorMode.CORNER)

    def insert_anchor(self, segment_index: int, t: float) -> tuple["VectorPath", int]:
        """Split a segment at ``t`` without changing the outline's shape.

        Returns the new path and the index of the inserted anchor. The split
        uses de Casteljau's construction, so the neighbours' handles shorten
        to keep the same curve.
        """

        count = len(self.anchors)
        segment_index %= count
        t = min(max(float(t), 0.0), 1.0)
        start = self.anchors[segment_index]
        end_index = (segment_index + 1) % count
        end = self.anchors[end_index]
        anchors = list(self.anchors)
        if not (start.has_out_handle or end.has_in_handle):
            # A straight segment gains a plain corner point, as in Illustrator.
            anchors.insert(segment_index + 1, Anchor(
                start.x + (end.x - start.x) * t, start.y + (end.y - start.y) * t
            ))
            return VectorPath(tuple(anchors)), segment_index + 1
        p0, c1, c2, p3 = (np.asarray(p, dtype=np.float64) for p in self.segment(segment_index))
        a = p0 + (c1 - p0) * t
        b = c1 + (c2 - c1) * t
        c = c2 + (p3 - c2) * t
        d = a + (b - a) * t
        e = b + (c - b) * t
        split = d + (e - d) * t
        start = replace(start, out_dx=a[0] - start.x, out_dy=a[1] - start.y)
        end = replace(end, in_dx=c[0] - end.x, in_dy=c[1] - end.y)
        middle = Anchor(
            float(split[0]),
            float(split[1]),
            float(d[0] - split[0]),
            float(d[1] - split[1]),
            float(e[0] - split[0]),
            float(e[1] - split[1]),
            AnchorMode.SMOOTH,
        )
        anchors[segment_index] = start
        anchors[end_index] = end
        anchors.insert(segment_index + 1, middle)
        return VectorPath(tuple(anchors)), segment_index + 1

    def remove_anchor(self, index: int) -> Optional["VectorPath"]:
        """Delete an anchor; ``None`` when too few would remain to enclose an area."""

        if len(self.anchors) <= MIN_PATH_ANCHORS:
            return None
        anchors = list(self.anchors)
        del anchors[index]
        return VectorPath(tuple(anchors))

    def _with_anchor(self, index: int, anchor: Anchor) -> "VectorPath":
        anchors = list(self.anchors)
        anchors[index] = anchor
        return VectorPath(tuple(anchors))


@dataclass(frozen=True, slots=True)
class VectorMask:
    """Every outline of one class, combined with the even-odd rule."""

    paths: tuple[VectorPath, ...] = ()

    @property
    def anchor_count(self) -> int:
        return sum(len(path) for path in self.paths)

    def with_path(self, index: int, path: Optional[VectorPath]) -> "VectorMask":
        """Replace path ``index`` (``None`` removes it)."""

        paths = list(self.paths)
        if path is None:
            del paths[index]
        else:
            paths[index] = path
        return VectorMask(tuple(paths))

    def added(self, path: VectorPath) -> "VectorMask":
        return VectorMask((*self.paths, path))

    def rasterize(self, shape: tuple[int, int]) -> np.ndarray:
        """Return an HxW boolean mask of the pixels whose centres are inside."""

        return rasterize_polygons((path.flatten() for path in self.paths), shape)


@dataclass(frozen=True, slots=True)
class VectorTraceOptions:
    """How closely :func:`trace_mask` follows the pixels, like Image Trace.

    ``tolerance`` is how far, in pixels, the simplified outline may stray
    from the pixel boundary (higher gives fewer anchors and smoother paths).
    Turns sharper than ``corner_angle`` degrees become corner anchors; the
    rest become smooth. Regions and holes smaller than ``min_area`` pixels are
    dropped, which removes speckle from noisy classifications.
    """

    tolerance: float = 1.0
    corner_angle: float = 70.0
    min_area: float = 25.0

    def __post_init__(self) -> None:
        if not self.tolerance > 0:
            raise ClassificationError("Trace tolerance must be greater than zero.")
        if not 0.0 <= self.corner_angle <= 180.0:
            raise ClassificationError("Corner angle must be between 0 and 180 degrees.")
        if self.min_area < 0:
            raise ClassificationError("Minimum area cannot be negative.")


def trace_mask(
    mask: np.ndarray, options: VectorTraceOptions = VectorTraceOptions()
) -> VectorMask:
    """Trace a boolean HxW mask into a :class:`VectorMask`.

    Boundaries are found with marching squares between pixel centres, so the
    unsimplified outline rasterizes back to exactly ``mask``. Each contour is
    then simplified (Ramer–Douglas–Peucker) and fitted with smooth or corner
    anchors.
    """

    import contourpy  # Installed with matplotlib; imported lazily as it is heavy.

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ClassificationError("A vector mask can only be traced from a 2-D mask.")
    if not values.any():
        return VectorMask()
    rows, columns = values.shape
    # A frame of background closes every contour, even along the image edge.
    padded = np.pad(values, 1).astype(np.float64)
    generator = contourpy.contour_generator(
        x=np.arange(-1, columns + 1) + 0.5,
        y=np.arange(-1, rows + 1) + 0.5,
        z=padded,
        line_type="Separate",
    )
    paths: list[VectorPath] = []
    for line in generator.lines(0.5):
        points = np.asarray(line, dtype=np.float64)
        if len(points) > 1 and np.array_equal(points[0], points[-1]):
            points = points[:-1]
        if len(points) < MIN_PATH_ANCHORS or abs(_signed_area(points)) < options.min_area:
            continue
        simplified = _simplify_closed(points, options.tolerance)
        if len(simplified) < MIN_PATH_ANCHORS:
            continue
        paths.append(_fit_anchors(simplified, options.corner_angle))
    return VectorMask(tuple(paths))


def rasterize_polygons(
    polygons: Iterable[np.ndarray], shape: tuple[int, int]
) -> np.ndarray:
    """Even-odd fill of closed ``(N, 2)`` ``(x, y)`` polylines at pixel centres.

    Each edge toggles every pixel to the right of where it crosses a row of
    pixel centres; a running parity along each row then marks the inside.
    The work is proportional to the edges and rows they cross, not the image.
    """

    rows, columns = int(shape[0]), int(shape[1])
    toggles = np.zeros((rows, columns + 1), dtype=np.uint8)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float64)
        if len(points) < 3:
            continue
        x0, y0 = points[:, 0], points[:, 1]
        x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
        low = np.minimum(y0, y1)
        high = np.maximum(y0, y1)
        # Rows whose centre (row + 0.5) lies in [low, high).
        first = np.clip(np.ceil(low - 0.5), 0, rows).astype(np.int64)
        last = np.clip(np.ceil(high - 0.5), 0, rows).astype(np.int64)
        counts = last - first
        keep = counts > 0
        if not keep.any():
            continue
        counts = counts[keep]
        edge = np.repeat(np.flatnonzero(keep), counts)
        offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        row = np.repeat(first[keep], counts) + offsets
        centre = row + 0.5
        ex0, ey0, ex1, ey1 = x0[edge], y0[edge], x1[edge], y1[edge]
        x = ex0 + (centre - ey0) * (ex1 - ex0) / (ey1 - ey0)
        # First pixel whose centre lies to the right of the crossing.
        column = np.clip(np.floor(x - 0.5).astype(np.int64) + 1, 0, columns)
        np.add.at(toggles, (row, column), 1)
    inside = np.cumsum(toggles[:, :columns], axis=1, dtype=np.int64) % 2
    return inside.astype(bool)


def apply_vector_masks(
    result: ClassificationResult, masks: Mapping[int, VectorMask]
) -> ClassificationResult:
    """Return ``result`` with each vectorized class's pixels set by its outline.

    A vectorized class gives up its original pixels and takes exactly the
    pixels its outline covers, claiming them from any other class. Pixels a
    vectorized class releases go back to whichever un-vectorized class held
    them originally, or become unclassified (``-1``) when the class itself
    held them. Where outlines overlap, the later entry in ``masks`` wins.

    A mask whose class ID ``result`` lacks adds that class, so a class can be
    drawn from scratch. A K-means result's IDs must stay ``0..K-1``, so any
    IDs skipped on the way to a new one are added as empty classes.
    """

    if not masks:
        return result
    if any(int(class_id) < 0 for class_id in masks):
        raise ClassificationError("Class IDs for vector masks cannot be negative.")
    added = sorted(set(masks).difference(_result_class_ids(result)))
    if added:
        result = _with_empty_classes(result, added)
    class_ids = _result_class_ids(result)
    base = np.asarray(result.class_map)
    class_map = np.array(base, dtype=np.int32, copy=True)
    for class_id in masks:
        class_map[base == class_id] = -1
    for class_id, mask in masks.items():
        class_map[mask.rasterize(base.shape)] = class_id
    id_array = np.asarray(class_ids, dtype=np.int32)[:, None, None]
    one_hot_masks = (class_map[None, :, :] == id_array).astype(np.uint8)
    counts = one_hot_masks.reshape(len(class_ids), -1).sum(axis=1, dtype=np.int64)
    for array in (class_map, one_hot_masks, counts):
        array.setflags(write=False)
    return replace(
        result,
        class_map=class_map,
        one_hot_masks=one_hot_masks,
        class_pixel_counts=counts,
    )


class VectorMaskModel:
    """The vector masks of one classification result, with undo history.

    Keyed by class ID in the order classes were vectorized, which is also the
    order :func:`apply_vector_masks` paints them in.
    """

    def __init__(self) -> None:
        self._masks: dict[int, VectorMask] = {}
        self._undo: list[dict[int, VectorMask]] = []
        self._redo: list[dict[int, VectorMask]] = []
        self._coalesce_key: Optional[object] = None

    @property
    def masks(self) -> Mapping[int, VectorMask]:
        return dict(self._masks)

    def __contains__(self, class_id: object) -> bool:
        return class_id in self._masks

    def __bool__(self) -> bool:
        return bool(self._masks)

    def get(self, class_id: int) -> Optional[VectorMask]:
        return self._masks.get(class_id)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def set_masks(self, masks: Mapping[int, VectorMask]) -> None:
        """Add or replace several classes' masks as one undo step."""

        self._record()
        for class_id, mask in masks.items():
            self._masks[int(class_id)] = mask

    def set_mask(
        self, class_id: int, mask: VectorMask, *, coalesce_key: Optional[object] = None
    ) -> None:
        """Replace one class's mask.

        Consecutive calls sharing ``coalesce_key`` (the moves of one drag)
        collapse into a single undo step until :meth:`break_coalescing`.
        """

        if self._masks.get(class_id) == mask:
            return
        self._record(coalesce_key)
        self._masks[int(class_id)] = mask

    def remove(self, class_id: int) -> None:
        """Discard a class's vector mask, restoring its original pixels."""

        if class_id not in self._masks:
            return
        self._record()
        del self._masks[class_id]

    def break_coalescing(self) -> None:
        self._coalesce_key = None

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(dict(self._masks))
        self._masks = self._undo.pop()
        self._coalesce_key = None
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(dict(self._masks))
        self._masks = self._redo.pop()
        self._coalesce_key = None
        return True

    def _record(self, coalesce_key: Optional[object] = None) -> None:
        self._redo.clear()
        if coalesce_key is not None and coalesce_key == self._coalesce_key:
            return
        self._coalesce_key = coalesce_key
        self._undo.append(dict(self._masks))


# ---------------------------------------------------------------------- #


def _result_class_ids(result: ClassificationResult) -> tuple[int, ...]:
    if isinstance(result, SupervisedClassificationResult):
        return tuple(int(value) for value in result.class_ids)
    return tuple(range(result.n_classes))


def _with_empty_classes(
    result: ClassificationResult, class_ids: list[int]
) -> ClassificationResult:
    """Return ``result`` with an empty class for each of ``class_ids`` added."""

    if isinstance(result, SupervisedClassificationResult):
        new_ids = tuple(class_ids)
    else:
        new_ids = tuple(range(result.n_classes, max(class_ids) + 1))
    count = len(new_ids)
    masks = np.asarray(result.one_hot_masks)
    one_hot_masks = np.concatenate(
        [masks, np.zeros((count, *masks.shape[1:]), dtype=masks.dtype)]
    )
    if isinstance(result, SupervisedClassificationResult):
        samples = np.asarray(result.training_sample_counts)
        return replace(
            result,
            class_ids=(*result.class_ids, *new_ids),
            one_hot_masks=one_hot_masks,
            training_sample_counts=np.concatenate(
                [samples, np.zeros(count, dtype=samples.dtype)]
            ),
        )
    centers = np.asarray(result.cluster_centers)
    # A drawn class has no spectral centre.
    blank = np.full(
        (count, *centers.shape[1:]), np.nan, dtype=np.result_type(centers.dtype, np.float32)
    )
    return replace(
        result,
        one_hot_masks=one_hot_masks,
        cluster_centers=np.concatenate([centers, blank]),
    )


def _bezier_points(
    control: tuple[Point, Point, Point, Point], t: np.ndarray
) -> np.ndarray:
    p0, c1, c2, p3 = (np.asarray(p, dtype=np.float64) for p in control)
    t = t[:, None]
    u = 1.0 - t
    return u**3 * p0 + 3 * u**2 * t * c1 + 3 * u * t**2 * c2 + t**3 * p3


def _segment_steps(control: tuple[Point, Point, Point, Point]) -> int:
    p0, c1, c2, p3 = control
    if c1 == p0 and c2 == p3:
        return 1
    length = math.dist(p0, c1) + math.dist(c1, c2) + math.dist(c2, p3)
    return max(2, min(_MAX_FLATTEN_STEPS, math.ceil(length / _FLATTEN_STEP_PX)))


def _flatten_segment(*control: Point) -> np.ndarray:
    steps = _segment_steps(control)
    return _bezier_points(control, np.linspace(0.0, 1.0, steps + 1))


def _signed_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _simplify_closed(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Ramer–Douglas–Peucker for a closed polyline.

    The ring is cut at its first point and the point farthest from it, and
    each half simplified as an open polyline.
    """

    far = int(np.argmax(np.hypot(*(points - points[0]).T)))
    if far == 0:
        return points[:1]
    ring = np.concatenate([points, points[:1]], axis=0)
    first = _simplify_open(ring[: far + 1], tolerance)
    second = _simplify_open(ring[far:], tolerance)
    return np.concatenate([first[:-1], second[:-1]], axis=0)


def _simplify_open(points: np.ndarray, tolerance: float) -> np.ndarray:
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        if end - start < 2:
            continue
        a, b = points[start], points[end]
        inner = points[start + 1 : end]
        chord = b - a
        length = math.hypot(*chord)
        if length == 0:
            distances = np.hypot(*(inner - a).T)
        else:
            offset = inner - a
            distances = np.abs(chord[0] * offset[:, 1] - chord[1] * offset[:, 0]) / length
        index = int(np.argmax(distances))
        if distances[index] > tolerance:
            split = start + 1 + index
            keep[split] = True
            stack.append((start, split))
            stack.append((split, end))
    return points[keep]


def _turn_angle(previous: Point, point: Point, following: Point) -> float:
    """Degrees the outline turns at ``point`` (0 = straight on)."""

    ax, ay = point[0] - previous[0], point[1] - previous[1]
    bx, by = following[0] - point[0], following[1] - point[1]
    norm = math.hypot(ax, ay) * math.hypot(bx, by)
    if norm == 0:
        return 0.0
    cosine = max(-1.0, min(1.0, (ax * bx + ay * by) / norm))
    return math.degrees(math.acos(cosine))


def _auto_smooth_anchor(
    previous: Anchor, anchor: Anchor, following: Anchor, mode: AnchorMode
) -> Anchor:
    """Handles tangent to the line through the neighbours (Catmull–Rom style)."""

    dir_x = following.x - previous.x
    dir_y = following.y - previous.y
    norm = math.hypot(dir_x, dir_y)
    if norm == 0:
        return replace(anchor, mode=mode)
    dir_x, dir_y = dir_x / norm, dir_y / norm
    in_length = math.dist(previous.point, anchor.point) * _AUTO_HANDLE_FRACTION
    out_length = math.dist(anchor.point, following.point) * _AUTO_HANDLE_FRACTION
    if mode is AnchorMode.SYMMETRIC:
        in_length = out_length = (in_length + out_length) / 2.0
    return Anchor(
        anchor.x,
        anchor.y,
        -dir_x * in_length,
        -dir_y * in_length,
        dir_x * out_length,
        dir_y * out_length,
        mode,
    )


def _fit_anchors(points: np.ndarray, corner_angle: float) -> VectorPath:
    corners = [Anchor(float(x), float(y)) for x, y in points]
    count = len(corners)
    anchors = []
    for index, anchor in enumerate(corners):
        previous = corners[index - 1]
        following = corners[(index + 1) % count]
        if _turn_angle(previous.point, anchor.point, following.point) > corner_angle:
            anchors.append(anchor)
        else:
            anchors.append(
                _auto_smooth_anchor(previous, anchor, following, AnchorMode.SMOOTH)
            )
    return VectorPath(tuple(anchors))
