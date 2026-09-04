"""Shared display colours for classification results and their layer controls."""

from __future__ import annotations

from collections.abc import Iterable

from PyQt6.QtGui import QColor


RGBColor = tuple[int, int, int]


def classification_palette(class_ids: Iterable[int]) -> dict[int, RGBColor]:
    """Return the stable colour assigned to each class in display order."""

    ordered_ids = tuple(int(class_id) for class_id in class_ids)
    class_count = max(len(ordered_ids), 1)
    palette: dict[int, RGBColor] = {}
    for color_index, class_id in enumerate(ordered_ids):
        color = QColor.fromHsvF(color_index / class_count, 0.72, 0.92)
        palette[class_id] = (color.red(), color.green(), color.blue())
    return palette
