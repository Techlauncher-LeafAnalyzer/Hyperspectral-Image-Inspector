"""Index means over the pixels enclosed by a rectangle, circle or polygon.

Qt-free Model. An annotation is drawn in image-pixel coordinates, so the pixels
it encloses can be found without the View: a pixel belongs to the region when
its centre lies inside the shape. Rectangles may be rotated about their centre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .annotation_model import MEASURABLE_KINDS, AnnotationKind, AnnotationLayer
from .classification_layer_model import ClassVisualizationMean
from .errors import AnnotationError
from .roi import polygon_mask
from .visualization_model import VisualizationMode, VisualizationResult


@dataclass(frozen=True, slots=True)
class RegionMeasurement:
    """Pixels enclosed by one annotation and the visualization means over them.

    ``means`` is keyed by visualization name (``VisualizationMode.value``).
    A visualization with no finite value inside the region is omitted.
    """

    pixel_count: int
    means: Mapping[str, ClassVisualizationMean]


def _region_window(
    layer: AnnotationLayer, shape: tuple[int, int]
) -> tuple[tuple[slice, slice], np.ndarray] | None:
    """Return the shape's bounding-box slices and the mask inside them.

    ``None`` when the shape lies wholly outside the image. Only pixels near the
    shape are tested, so a small shape on a large image is cheap to measure.
    """

    if layer.kind not in MEASURABLE_KINDS:
        raise AnnotationError(
            f"{layer.kind.value} annotations do not enclose an area to measure."
        )
    rows, columns = shape
    if layer.kind is AnnotationKind.POLYGON:
        return _polygon_window(layer, shape)
    half_width = layer.width / 2.0
    half_height = layer.height / 2.0
    centre_x = layer.x + half_width
    centre_y = layer.y + half_height
    if layer.rotation:
        reach_x = reach_y = math.hypot(half_width, half_height)
    else:
        reach_x, reach_y = half_width, half_height
    first_column = max(0, math.floor(centre_x - reach_x) - 1)
    last_column = min(columns, math.ceil(centre_x + reach_x) + 1)
    first_row = max(0, math.floor(centre_y - reach_y) - 1)
    last_row = min(rows, math.ceil(centre_y + reach_y) + 1)
    if first_column >= last_column or first_row >= last_row:
        return None

    dx = np.arange(first_column, last_column) + 0.5 - centre_x
    dy = np.arange(first_row, last_row) + 0.5 - centre_y
    angle = math.radians(layer.rotation)
    cos, sin = math.cos(angle), math.sin(angle)
    # Undo the clockwise (y-down) rotation to get shape-local coordinates.
    u = dx[None, :] * cos + dy[:, None] * sin
    v = -dx[None, :] * sin + dy[:, None] * cos
    if layer.kind is AnnotationKind.ELLIPSE:
        inside = (u / half_width) ** 2 + (v / half_height) ** 2 <= 1.0
    else:
        inside = (np.abs(u) <= half_width) & (np.abs(v) <= half_height)
    return (slice(first_row, last_row), slice(first_column, last_column)), inside


def _polygon_window(
    layer: AnnotationLayer, shape: tuple[int, int]
) -> tuple[tuple[slice, slice], np.ndarray] | None:
    """Rasterize a polygon inside its bounding box, as a crop's ROI is."""

    rows, columns = shape
    first_column = max(0, math.floor(layer.x))
    last_column = min(columns, math.ceil(layer.x + layer.width))
    first_row = max(0, math.floor(layer.y))
    last_row = min(rows, math.ceil(layer.y + layer.height))
    if first_column >= last_column or first_row >= last_row:
        return None
    local = [(x - first_column, y - first_row) for x, y in layer.points]
    inside = polygon_mask(local, (last_row - first_row, last_column - first_column))
    return (slice(first_row, last_row), slice(first_column, last_column)), inside


def region_mask(layer: AnnotationLayer, shape: tuple[int, int]) -> np.ndarray:
    """Return an HxW boolean mask of the pixels enclosed by ``layer``."""

    mask = np.zeros(shape, dtype=bool)
    window = _region_window(layer, shape)
    if window is not None:
        slices, inside = window
        mask[slices] = inside
    return mask


def measure_region(
    layer: AnnotationLayer,
    visualizations: Mapping[VisualizationMode, VisualizationResult],
) -> RegionMeasurement:
    """Mean of every visualization over the pixels ``layer`` encloses.

    Matches :meth:`ClassificationLayerModel.visualization_means`: the mean RGB
    triple for RGB, the finite-value mean for an index, each with the colour
    that represents it. Visualizations whose raster does not match the first
    one's shape are skipped.
    """

    shape = next(
        (tuple(result.display_rgb.shape[:2]) for result in visualizations.values()),
        None,
    )
    if shape is None:
        return RegionMeasurement(0, {})
    window = _region_window(layer, shape)
    pixel_count = 0 if window is None else int(np.count_nonzero(window[1]))
    if window is None or pixel_count == 0:
        return RegionMeasurement(0, {})
    slices, inside = window
    means: dict[str, ClassVisualizationMean] = {}
    for mode, visualization in visualizations.items():
        if tuple(visualization.display_rgb.shape[:2]) != shape:
            continue
        if visualization.values is None:
            mean_rgb = visualization.display_rgb[slices][inside].mean(
                axis=0, dtype=np.float64
            )
            value = tuple(float(channel) for channel in mean_rgb)
            color = tuple(int(round(channel)) for channel in mean_rgb)
        else:
            if visualization.values.shape != shape:
                continue
            selected = visualization.values[slices][inside]
            finite = selected[np.isfinite(selected)]
            if finite.size == 0:
                continue
            value = float(np.mean(finite, dtype=np.float64))
            color = visualization.color_for(value)
            if color is None:
                continue
        means[mode.value] = ClassVisualizationMean(value=value, color=color)
    return RegionMeasurement(pixel_count, means)
