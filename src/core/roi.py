"""Region-of-interest masking for non-rectangular crops.

A polygon crop cannot be expressed as a SPy sub-image: ``SpyFile`` advertises
a rectangular ``(rows, columns, bands)`` shape and reads rectangular regions.
An irregular selection is therefore stored as a bounding-box crop (handled by
:meth:`core.HSIData.crop`) plus a boolean region-of-interest mask that marks
which pixels inside that box the user actually selected.

:class:`Masked` is the carrier that threads the mask through Model code. It is
a small monadic container: :meth:`Masked.unit` lifts a plain array into it,
:meth:`Masked.map` applies a shape-preserving function without disturbing the
mask, and :meth:`Masked.bind` chains a function that returns another
``Masked``. Aggregations that must *honour* the mask use the eliminators
:meth:`Masked.select`, :meth:`Masked.fill`, and :meth:`Masked.scatter`.

The container deliberately never injects ``NaN`` into values handed back to
SPy or torch. ``SuperResolutionService`` and the supervised classifiers reject
non-finite cubes outright, so masking is applied at each aggregation point
rather than inside the lazy cube reads.

This module is UI-independent; polygons arrive as plain vertex arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Optional, Sequence, TypeVar

import numpy as np
import numpy.typing as npt
from matplotlib.path import Path as MplPath

T = TypeVar("T")
U = TypeVar("U")

BoolMask = npt.NDArray[np.bool_]


def polygon_mask(
    vertices: Sequence[tuple[float, float]],
    shape: tuple[int, int],
) -> BoolMask:
    """Rasterize a polygon into an ``(rows, columns)`` boolean mask.

    ``vertices`` are ``(x, y)`` pairs in pixel coordinates of the same frame
    as ``shape``; the polygon is implicitly closed. A pixel is inside when its
    centre is inside, which matches how the selection was drawn on screen.

    Raises:
        ValueError: Fewer than three vertices, or a non-positive shape.
    """

    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("A polygon region needs at least three (x, y) vertices.")
    rows, columns = int(shape[0]), int(shape[1])
    if rows < 1 or columns < 1:
        raise ValueError(f"Cannot rasterize a polygon into a {rows}x{columns} frame.")

    grid_y, grid_x = np.mgrid[0:rows, 0:columns]
    centres = np.column_stack(
        (grid_x.ravel() + 0.5, grid_y.ravel() + 0.5)
    ).astype(np.float64)
    inside = MplPath(points, closed=False).contains_points(centres)
    return inside.reshape(rows, columns)


def combine(existing: Optional[BoolMask], addition: Optional[BoolMask]) -> Optional[BoolMask]:
    """Intersect two optional masks defined over the same frame.

    ``None`` means "every pixel selected", so it is the identity here. Nested
    polygon crops compose by intersection once both are expressed in the
    current bounding box's coordinates.
    """

    if existing is None:
        return addition
    if addition is None:
        return existing
    if existing.shape != addition.shape:
        raise ValueError(
            f"Cannot combine masks of shape {existing.shape} and {addition.shape}."
        )
    return np.logical_and(existing, addition)


@dataclass(frozen=True)
class Masked(Generic[T]):
    """An array paired with the region of interest that selects its pixels.

    ``mask`` is ``None`` when every pixel counts, which keeps the uncropped
    and rectangle-cropped paths allocation-free. ``mask`` otherwise has the
    array's leading ``(rows, columns)`` shape.
    """

    value: T
    mask: Optional[BoolMask] = None

    # -- monadic interface ------------------------------------------------ #

    @classmethod
    def unit(cls, value: T) -> "Masked[T]":
        """Lift an unmasked value into the container."""

        return cls(value, None)

    def map(self, function: Callable[[T], U]) -> "Masked[U]":
        """Apply a shape-preserving function, carrying the mask through."""

        return Masked(function(self.value), self.mask)

    def bind(self, function: Callable[[T], "Masked[U]"]) -> "Masked[U]":
        """Chain a function that introduces a mask of its own."""

        result = function(self.value)
        return Masked(result.value, combine(self.mask, result.mask))

    # -- eliminators ------------------------------------------------------ #

    @property
    def is_masked(self) -> bool:
        return self.mask is not None

    @property
    def count(self) -> int:
        """Number of selected pixels."""

        array = np.asarray(self.value)
        if self.mask is None:
            return int(array.shape[0] * array.shape[1]) if array.ndim >= 2 else array.size
        return int(np.count_nonzero(self.mask))

    def select(self) -> np.ndarray:
        """Return only the selected pixels.

        An ``(H, W)`` array yields shape ``(N,)``; an ``(H, W, B)`` cube
        yields ``(N, B)``. This is what aggregations such as percentile
        stretching and clustering should consume.
        """

        array = np.asarray(self.value)
        if self.mask is None:
            return array.reshape(-1) if array.ndim <= 2 else array.reshape(-1, array.shape[-1])
        return array[self.mask]

    def fill(self, fill_value: float = np.nan) -> np.ndarray:
        """Return the array with unselected pixels replaced by ``fill_value``.

        Used for display and statistics arrays only. ``NaN`` is the useful
        default: ``VisualizationService._percentile_stretch`` already drops
        non-finite values, and Index Mean already uses ``np.nanmean``, so
        filling makes both mask-correct without further change.
        """

        array = np.asarray(self.value)
        if self.mask is None:
            return array
        output = array.astype(np.result_type(array.dtype, np.float32), copy=True)
        output[~self.mask] = fill_value
        return output

    def scatter(self, values: np.ndarray, fill_value: float | int) -> np.ndarray:
        """Place a per-selected-pixel result back into the full frame.

        ``values`` must have one entry per selected pixel, in the order
        :meth:`select` produced them. Unselected pixels receive ``fill_value``.
        """

        array = np.asarray(self.value)
        frame = array.shape[:2]
        flat = np.asarray(values)
        if self.mask is None:
            return flat.reshape(frame + flat.shape[1:])
        output = np.full(frame + flat.shape[1:], fill_value, dtype=flat.dtype)
        output[self.mask] = flat
        return output
