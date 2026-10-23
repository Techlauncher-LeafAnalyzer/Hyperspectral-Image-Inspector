"""Shared display colours for classification results and their layer controls."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from PyQt6.QtGui import QColor


RGBColor = tuple[int, int, int]


def classification_palette(
    class_ids: Iterable[int],
    fixed_colors: Mapping[int, RGBColor] | None = None,
) -> dict[int, RGBColor]:
    """Return the stable colour assigned to each class in display order.

    Classes in ``fixed_colors`` keep that colour; the hues are spread evenly
    over the remaining classes only.
    """

    fixed_colors = fixed_colors or {}
    ordered_ids = tuple(int(class_id) for class_id in class_ids)
    hued_ids = [class_id for class_id in ordered_ids if class_id not in fixed_colors]
    class_count = max(len(hued_ids), 1)
    hues: dict[int, RGBColor] = {}
    for color_index, class_id in enumerate(hued_ids):
        color = QColor.fromHsvF(color_index / class_count, 0.72, 0.92)
        hues[class_id] = (color.red(), color.green(), color.blue())
    return {
        class_id: tuple(fixed_colors[class_id]) if class_id in fixed_colors else hues[class_id]
        for class_id in ordered_ids
    }
