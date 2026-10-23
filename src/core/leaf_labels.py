"""Burn each separated leaf's number into a rendered class map.

After :func:`core.separate_leaves`, leaves are named "Leaf 1", "Leaf 2", ...
This module places that number just beneath each leaf so a saved image (or a
batch preview) can be matched to the layer panel and CSV rows without the
colour legend. Rendering uses Pillow only, so the Classification tab and the
batch pipeline draw identical labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_TEXT_FILL = (255, 255, 255)
_TEXT_STROKE = (0, 0, 0)
# Gap, in pixels, between a leaf's lowest pixel and the top of its label.
_LABEL_GAP = 2


@dataclass(frozen=True, slots=True)
class LeafLabel:
    """Text and its anchor: ``(x, y)`` is the top-centre of the label."""

    class_id: int
    text: str
    x: int
    y: int


def leaf_label_size(image_shape: tuple[int, int]) -> int:
    """Font pixel size scaled to the image: legible but small."""

    return max(10, min(int(min(image_shape) / 18), 48))


def place_leaf_labels(
    class_map: np.ndarray,
    numbers: Mapping[int, int],
) -> tuple[LeafLabel, ...]:
    """Anchor each leaf's number just below the leaf, centred on its columns.

    A leaf touching the bottom edge has no room underneath, so its label sits
    inside the leaf instead, at its lowest row that still fits the text.
    """

    class_map = np.asarray(class_map)
    rows = class_map.shape[0]
    height = leaf_label_size(class_map.shape) + 2
    labels: list[LeafLabel] = []
    for class_id, number in numbers.items():
        ys, xs = np.nonzero(class_map == class_id)
        if ys.size == 0:
            continue
        x = int(round(float(xs.mean())))
        y = int(ys.max()) + _LABEL_GAP
        if y + height > rows:
            y = max(0, rows - height)
        labels.append(LeafLabel(int(class_id), str(number), x, y))
    return tuple(labels)


def draw_leaf_labels(
    rgb: np.ndarray,
    labels: Iterable[LeafLabel],
    visible_ids: Iterable[int] | None = None,
) -> np.ndarray:
    """Return a copy of ``rgb`` with each label drawn as outlined white text.

    ``visible_ids`` limits drawing to those classes (hidden layers lose their
    number too); ``None`` draws every label.
    """

    visible = None if visible_ids is None else {int(i) for i in visible_ids}
    labels = [label for label in labels if visible is None or label.class_id in visible]
    if not labels:
        return rgb
    image = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    size = leaf_label_size(rgb.shape[:2])
    font = ImageFont.load_default(size=size)
    stroke = max(1, size // 8)
    for label in labels:
        draw.text(
            (label.x, label.y), label.text, fill=_TEXT_FILL, font=font, anchor="mt",
            stroke_width=stroke, stroke_fill=_TEXT_STROKE,
        )
    return np.asarray(image, dtype=np.uint8)
