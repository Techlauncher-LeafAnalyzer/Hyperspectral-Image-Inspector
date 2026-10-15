"""Detect the four corner markers of a Leaf2QR pattern and derive a crop.

The pattern is four square fiducial (QR-like) tags placed at the corners of a
rectangle around the plant. The useful region is the "fat plus" left over when
the four tags are cut out of the rectangle that bounds them, so the polygon's
notches sit exactly on the tags' inner borders.

Detection is intentionally dependency-light (numpy + scipy): the tags are the
dark, roughly square blobs of matching size that form a rectangle.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import ndimage as ndi

# A pixel can belong to a tag when it is darker than this fraction of the local
# background. Relative, because uncalibrated cube renders have a tinted, uneven
# background (~150-200) while white-balanced previews are ~255. The tags' outer
# border fades to ~0.7 of the background on the lit side.
_CORE_RATIO = 0.6  # tag bodies; strict so grey margins beside the paper don't merge
_EDGE_RATIO = 0.8  # faded tag borders, only accepted right next to a body
_EDGE_PAD_PX = 8
_BACKGROUND_WINDOW_FRACTION = 0.12  # of the image's shorter side; > tag size
_MERGE_ITERATIONS = 6
_MIN_SIDE_FRACTION = 0.04  # tag side relative to the image's shorter side
_MAX_SIDE_FRACTION = 0.30
_MAX_ASPECT = 1.35
_MAX_SIZE_RATIO = 1.35
_MAX_ALIGN_FRACTION = 0.25  # row/column misalignment relative to tag size
_MIN_FILL = 0.35


@dataclass(frozen=True)
class MarkerBox:
    """Axis-aligned tag bounds; ``right``/``bottom`` are exclusive."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def center(self) -> tuple[float, float]:
        return (self.left + self.right) / 2, (self.top + self.bottom) / 2


@dataclass(frozen=True)
class MarkerLayout:
    top_left: MarkerBox
    top_right: MarkerBox
    bottom_left: MarkerBox
    bottom_right: MarkerBox


def _candidates(rgb: np.ndarray) -> list[MarkerBox]:
    grey = np.asarray(rgb, dtype=np.float32).mean(axis=2)
    height, width = grey.shape
    short = min(height, width)
    window = max(3, int(_BACKGROUND_WINDOW_FRACTION * short) | 1)
    background = ndi.maximum_filter(grey, size=window)
    background = ndi.uniform_filter(background, size=window)
    scaled = np.maximum(background, 1.0)
    core = grey < _CORE_RATIO * scaled
    edge = grey < _EDGE_RATIO * scaled
    labels, _ = ndi.label(ndi.binary_dilation(core, iterations=_MERGE_ITERATIONS))
    boxes: list[MarkerBox] = []
    for index, region in enumerate(ndi.find_objects(labels), start=1):
        if region is None:
            continue
        body = (labels == index) & core
        by, bx = np.nonzero(body[region])
        if len(by) == 0:
            continue
        # Grow into the faded border, but only a few pixels beyond the body.
        near = ndi.binary_dilation(body[region], iterations=_EDGE_PAD_PX)
        ys, xs = np.nonzero(near & edge[region])
        if len(ys) == 0:
            continue
        left = int(xs.min()) + region[1].start
        top = int(ys.min()) + region[0].start
        box = MarkerBox(
            left, top, int(xs.max()) + region[1].start + 1,
            int(ys.max()) + region[0].start + 1,
        )
        if box.left <= 0 or box.top <= 0 or box.right >= width or box.bottom >= height:
            continue  # scene edges / frame, not a tag
        if not _MIN_SIDE_FRACTION * short <= min(box.width, box.height):
            continue
        if max(box.width, box.height) > _MAX_SIDE_FRACTION * short:
            continue
        if max(box.width, box.height) / min(box.width, box.height) > _MAX_ASPECT:
            continue
        if len(ys) / (box.width * box.height) < _MIN_FILL:
            continue
        boxes.append(box)
    return boxes


def _as_layout(group: tuple[MarkerBox, ...]) -> MarkerLayout | None:
    by_y = sorted(group, key=lambda b: b.center[1])
    top = sorted(by_y[:2], key=lambda b: b.center[0])
    bottom = sorted(by_y[2:], key=lambda b: b.center[0])
    layout = MarkerLayout(top[0], top[1], bottom[0], bottom[1])

    sides = [max(b.width, b.height) for b in group]
    if max(sides) / min(sides) > _MAX_SIZE_RATIO:
        return None
    tol = _MAX_ALIGN_FRACTION * float(np.mean(sides))
    tl, tr, bl, br = (b.center for b in (top[0], top[1], bottom[0], bottom[1]))
    if abs(tl[1] - tr[1]) > tol or abs(bl[1] - br[1]) > tol:
        return None
    if abs(tl[0] - bl[0]) > tol or abs(tr[0] - br[0]) > tol:
        return None
    # The tags must be separated by more than their own size.
    if tr[0] - tl[0] < 1.5 * min(sides) or bl[1] - tl[1] < 1.5 * min(sides):
        return None
    return layout


def detect_markers(rgb: np.ndarray) -> MarkerLayout | None:
    """Find the four corner tags in an ``(H, W, 3)`` image, or ``None``."""

    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return None
    boxes = _candidates(rgb)
    best: tuple[float, MarkerLayout] | None = None
    for group in combinations(boxes, 4):
        layout = _as_layout(group)
        if layout is None:
            continue
        # Prefer the largest rectangle if several groups qualify.
        span = (layout.top_right.center[0] - layout.top_left.center[0]) * (
            layout.bottom_left.center[1] - layout.top_left.center[1]
        )
        if best is None or span > best[0]:
            best = (span, layout)
    return None if best is None else best[1]


def plus_polygon(layout: MarkerLayout) -> list[tuple[float, float]]:
    """Fat-plus polygon: the markers' bounding box minus the four tags.

    Vertices are ``(x, y)`` pixel-edge coordinates, clockwise from the top-left
    notch. Each notch corner lies on a tag's inner border.
    """

    tl, tr, bl, br = (
        layout.top_left, layout.top_right, layout.bottom_left, layout.bottom_right
    )
    x0 = min(tl.left, bl.left)
    x1 = max(tr.right, br.right)
    y0 = min(tl.top, tr.top)
    y1 = max(bl.bottom, br.bottom)
    return [
        (tl.right, y0), (tr.left, y0),
        (tr.left, tr.bottom), (x1, tr.bottom),
        (x1, br.top), (br.left, br.top),
        (br.left, y1), (bl.right, y1),
        (bl.right, bl.top), (x0, bl.top),
        (x0, tl.bottom), (tl.right, tl.bottom),
    ]


def find_plus_crop(rgb: np.ndarray) -> list[tuple[float, float]] | None:
    """Polygon for the automatic Leaf2QR crop, or ``None`` if no pattern."""

    layout = detect_markers(rgb)
    return None if layout is None else plus_polygon(layout)
