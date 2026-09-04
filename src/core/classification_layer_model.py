"""View-neutral layer management for classification results.

The Controller creates one :class:`ClassificationLayerModel` after either
classification workflow completes.  The object owns only lightweight layer
state (names and visibility); masks remain the immutable one-hot arrays from
the classification result.  No Qt types are used here.

Visibility operations do not read the hyperspectral cube.  Vegetation-index
analysis delegates the scientific calculation to :class:`VisualizationService`
once, then reuses the resulting raster for every class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Callable, Iterable

import numpy as np

from .classification_model import (
    SupervisedClassificationResult,
    UnsupervisedClassificationResult,
)
from .errors import CancelledError, ClassificationError
from .hsi_data import HSIData
from .visualization_model import (
    VisualizationMode,
    VisualizationRequest,
    VisualizationResult,
    VisualizationService,
)


ClassificationResult = (
    UnsupervisedClassificationResult | SupervisedClassificationResult
)
ProgressCallback = Callable[[int, str], None]
CancellationCheck = Callable[[], bool]

INDEX_MODES = frozenset(
    {
        VisualizationMode.NDVI,
        VisualizationMode.EVI,
        VisualizationMode.MCARI,
        VisualizationMode.MTVI,
        VisualizationMode.OSAVI,
        VisualizationMode.PRI,
    }
)


@dataclass(frozen=True, slots=True)
class ClassificationLayer:
    """Immutable layer-row data for a View.

    A Controller can rebuild its checkbox/list rows from ``model.layers``
    after a visibility or naming change. ``class_id`` is the stable identity;
    list position must not be used as an ID for supervised results.
    """

    class_id: int
    name: str
    pixel_count: int
    visible: bool
    opacity: float


@dataclass(frozen=True, slots=True)
class ClassificationLayerComposite:
    """RGB and RGBA representations of the currently visible layers.

    ``display_rgb`` has hidden/faded pixels blended toward the requested
    background colour or base image. ``display_rgba`` retains original RGB
    everywhere and uses each pixel's class opacity as its alpha, making it
    suitable for a transparent overlay.
    """

    display_rgb: np.ndarray
    display_rgba: np.ndarray
    visible_mask: np.ndarray
    visible_class_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ClassIndexStatistics:
    """Summary of one analytical vegetation-index raster inside one class."""

    class_id: int
    class_name: str
    pixel_count: int
    finite_pixel_count: int
    image_fraction: float
    mean: float | None
    median: float | None
    standard_deviation: float | None
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True, slots=True)
class ClassificationIndexAnalysis:
    """One index calculation partitioned by classification masks.

    The full :class:`VisualizationResult` is retained so a Controller can use
    the same false-colour image, raw values, band mapping, and title already
    understood by the visualization UI. Per-class masked rasters are generated
    only on request to avoid storing ``classes x rows x columns`` float arrays.
    """

    visualization: VisualizationResult
    statistics: tuple[ClassIndexStatistics, ...]
    class_ids: tuple[int, ...]
    _one_hot_masks: np.ndarray = field(repr=False)

    def statistics_for_class(self, class_id: int) -> ClassIndexStatistics:
        """Return the precomputed statistics for ``class_id``."""

        normalized = _normalize_class_id(class_id)
        for item in self.statistics:
            if item.class_id == normalized:
                return item
        raise ClassificationError(f"Unknown classification class ID: {class_id}")

    def mask_for_class(self, class_id: int) -> np.ndarray:
        """Return the original read-only HxW binary mask for ``class_id``."""

        index = self._class_index(class_id)
        return self._one_hot_masks[index]

    def masked_values(
        self,
        class_id: int,
        *,
        fill_value: float = np.nan,
    ) -> np.ndarray:
        """Return an HxW index raster with pixels outside the class filled.

        Use this when an index View needs spatial detail for one class. For a
        chart or table, use :meth:`statistics_for_class` to avoid allocation.
        """

        values = self.visualization.values
        if values is None:  # Defensive: index analysis never accepts RGB.
            raise ClassificationError("Index analysis has no analytical values.")
        mask = self.mask_for_class(class_id).astype(bool, copy=False)
        output = np.full(values.shape, fill_value, dtype=np.float32)
        output[mask] = values[mask]
        output.setflags(write=False)
        return output

    def display_rgba_for_class(self, class_id: int) -> np.ndarray:
        """Return the index false-colour rendering as one transparent layer."""

        mask = self.mask_for_class(class_id).astype(bool, copy=False)
        rgb = self.visualization.display_rgb
        rgba = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
        rgba[mask, :3] = rgb[mask]
        rgba[mask, 3] = 255
        rgba.setflags(write=False)
        return rgba

    def _class_index(self, class_id: int) -> int:
        normalized = _normalize_class_id(class_id)
        try:
            return self.class_ids.index(normalized)
        except ValueError as exc:
            raise ClassificationError(
                f"Unknown classification class ID: {class_id}"
            ) from exc


class ClassificationLayerModel:
    """Mutable visibility/name state over an immutable classification result.

    Keep this object in Controller state for as long as the classification
    result is current. Create a new instance after reclassification, cropping,
    or loading another cube because those actions invalidate the masks.
    """

    def __init__(self, result: ClassificationResult) -> None:
        self._result = result
        masks = np.asarray(result.one_hot_masks)
        if masks.ndim != 3:
            raise ClassificationError(
                "Classification one-hot masks must have shape (classes, rows, columns)."
            )
        if not (
            np.issubdtype(masks.dtype, np.bool_)
            or np.issubdtype(masks.dtype, np.integer)
        ):
            raise ClassificationError("Classification masks must be binary arrays.")
        if np.any((masks != 0) & (masks != 1)):
            raise ClassificationError("Classification masks must contain only 0 and 1.")
        if np.any(masks.sum(axis=0, dtype=np.int64) > 1):
            raise ClassificationError("Classification mask layers cannot overlap.")

        if isinstance(result, SupervisedClassificationResult):
            class_ids = tuple(int(value) for value in result.class_ids)
        elif isinstance(result, UnsupervisedClassificationResult):
            class_ids = tuple(range(result.n_classes))
        else:
            raise ClassificationError("Unsupported classification result type.")
        if len(class_ids) != masks.shape[0] or len(set(class_ids)) != len(class_ids):
            raise ClassificationError(
                "Classification class IDs do not match the one-hot mask layers."
            )

        self._one_hot_masks = masks
        self._class_ids = class_ids
        self._pixel_counts = tuple(
            int(np.count_nonzero(mask)) for mask in self._one_hot_masks
        )
        self._names = {class_id: f"Class {class_id}" for class_id in class_ids}
        self._visibility = {class_id: True for class_id in class_ids}
        self._opacity = {class_id: 1.0 for class_id in class_ids}
        self._global_opacity = 1.0
        self._outline_mode = False

    @property
    def result(self) -> ClassificationResult:
        """Return the immutable source classification result."""

        return self._result

    @property
    def image_shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self._one_hot_masks.shape[1:])

    @property
    def class_ids(self) -> tuple[int, ...]:
        return self._class_ids

    @property
    def visible_class_ids(self) -> tuple[int, ...]:
        return tuple(
            class_id for class_id in self._class_ids if self._visibility[class_id]
        )

    @property
    def global_opacity(self) -> float:
        """Return the master opacity scale applied on top of every layer."""

        return self._global_opacity

    @property
    def outline_mode(self) -> bool:
        """Return whether layers currently render as borders only, not fills."""

        return self._outline_mode

    @property
    def layers(self) -> tuple[ClassificationLayer, ...]:
        """Return current layer rows in stable classification order."""

        return tuple(
            ClassificationLayer(
                class_id=class_id,
                name=self._names[class_id],
                pixel_count=self._pixel_counts[index],
                visible=self._visibility[class_id],
                opacity=self._opacity[class_id],
            )
            for index, class_id in enumerate(self._class_ids)
        )

    def mask_for_class(self, class_id: int) -> np.ndarray:
        """Return the source result's read-only HxW mask for one layer."""

        return self._one_hot_masks[self._class_index(class_id)]

    def rename_class(self, class_id: int, name: str) -> None:
        """Set a user-facing layer name while preserving its stable class ID."""

        normalized = self._require_class(class_id)
        cleaned = str(name).strip()
        if not cleaned:
            raise ClassificationError("A classification layer name cannot be empty.")
        self._names[normalized] = cleaned

    def set_class_visible(self, class_id: int, visible: bool) -> None:
        """Update one checkbox/eye state."""

        normalized = self._require_class(class_id)
        if not isinstance(visible, (bool, np.bool_)):
            raise ClassificationError("Layer visibility must be true or false.")
        self._visibility[normalized] = bool(visible)

    def set_class_opacity(self, class_id: int, opacity: float) -> None:
        """Update one layer's opacity, independent of its visibility state."""

        normalized = self._require_class(class_id)
        self._opacity[normalized] = _normalize_opacity(opacity)

    def set_visible_classes(self, class_ids: Iterable[int]) -> None:
        """Atomically replace the visible set, useful for restoring UI state."""

        normalized = tuple(_normalize_class_id(value) for value in class_ids)
        if len(set(normalized)) != len(normalized):
            raise ClassificationError("Visible class IDs must be unique.")
        unknown = set(normalized).difference(self._class_ids)
        if unknown:
            raise ClassificationError(
                f"Unknown classification class IDs: {sorted(unknown)}"
            )
        selected = set(normalized)
        for class_id in self._class_ids:
            self._visibility[class_id] = class_id in selected

    def show_only(self, class_id: int) -> None:
        """Hide all layers except one, matching a layer-panel solo action."""

        normalized = self._require_class(class_id)
        self.set_visible_classes((normalized,))

    def set_all_visible(self, visible: bool = True) -> None:
        """Show or hide every classification layer."""

        if not isinstance(visible, (bool, np.bool_)):
            raise ClassificationError("Layer visibility must be true or false.")
        for class_id in self._class_ids:
            self._visibility[class_id] = bool(visible)

    def set_global_opacity(self, opacity: float) -> None:
        """Scale every layer's effective opacity together, like a group opacity.

        Independent of each layer's own ``opacity``: the stored per-layer
        values are preserved, so restoring the global scale to ``1.0``
        reveals the same relative fades the user had set per class.
        """

        self._global_opacity = _normalize_opacity(opacity)

    def set_outline_mode(self, enabled: bool) -> None:
        """Switch every visible layer between a solid fill and border-only.

        Border-only mode paints just each class region's boundary pixels at
        its opacity, leaving the interior fully transparent so the base
        image or background shows through -- useful for inspecting class
        boundaries without obscuring the underlying picture.
        """

        if not isinstance(enabled, (bool, np.bool_)):
            raise ClassificationError("Outline mode must be true or false.")
        self._outline_mode = bool(enabled)

    def visible_mask(self) -> np.ndarray:
        """Return a read-only HxW boolean union of all visible classes."""

        indices = [
            index
            for index, class_id in enumerate(self._class_ids)
            if self._visibility[class_id]
        ]
        if indices:
            mask = np.any(self._one_hot_masks[indices].astype(bool), axis=0)
        else:
            mask = np.zeros(self.image_shape, dtype=bool)
        mask.setflags(write=False)
        return mask

    def _effective_alpha(self) -> np.ndarray:
        """Return an HxW blend factor: per-class opacity where visible, else 0.

        Class masks are disjoint (enforced in ``__init__``), so classes can be
        painted independently without any overlap/accumulation to resolve.
        The result is additionally scaled by ``global_opacity`` and, in
        ``outline_mode``, restricted to each class region's boundary pixels.
        """

        alpha = np.zeros(self.image_shape, dtype=np.float32)
        for index, class_id in enumerate(self._class_ids):
            if not self._visibility[class_id]:
                continue
            mask = self._one_hot_masks[index].astype(bool, copy=False)
            if self._outline_mode:
                mask = _boundary_mask(mask)
            alpha[mask] = self._opacity[class_id] * self._global_opacity
        alpha.setflags(write=False)
        return alpha

    def compose_rgb(
        self,
        rgb: np.ndarray,
        *,
        background_color: tuple[int, int, int] = (0, 0, 0),
    ) -> ClassificationLayerComposite:
        """Apply current visibility to true-colour RGB data.

        This is the main API for the requested human-eye inspection workflow.
        It never false-colours the source pixels.
        """

        return self.compose_display(rgb, background_color=background_color)

    def compose_display(
        self,
        display_rgb: np.ndarray,
        *,
        background_color: tuple[int, int, int] = (0, 0, 0),
        base_rgb: np.ndarray | None = None,
    ) -> ClassificationLayerComposite:
        """Apply visibility and opacity to any RGB rendering, including an
        index colormap.

        Hidden pixels and faded (partially opaque) pixels are blended toward
        ``base_rgb`` when given -- typically the true-colour image beneath a
        false-coloured classification overlay -- or toward the flat
        ``background_color`` otherwise. A layer at the default opacity of
        ``1.0`` behaves exactly like a hard visible/hidden switch.
        """

        rgb = self._validated_rgb(display_rgb)
        visible_mask = self.visible_mask()
        alpha = self._effective_alpha()
        alpha_channel = alpha[..., np.newaxis]

        if base_rgb is not None:
            background_arr = self._validated_rgb(base_rgb).astype(np.float32)
        else:
            background_arr = np.asarray(
                _normalize_rgb_color(background_color), dtype=np.float32
            )
        blended = rgb.astype(np.float32) * alpha_channel + background_arr * (
            1.0 - alpha_channel
        )
        composited = np.clip(blended, 0, 255).astype(np.uint8)

        rgba = np.empty((*self.image_shape, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        for array in (composited, rgba):
            array.setflags(write=False)
        return ClassificationLayerComposite(
            display_rgb=composited,
            display_rgba=rgba,
            visible_mask=visible_mask,
            visible_class_ids=self.visible_class_ids,
        )

    def rgb_layer(self, rgb: np.ndarray, class_id: int) -> np.ndarray:
        """Return one class as an independent transparent true-colour layer."""

        source = self._validated_rgb(rgb)
        mask = self.mask_for_class(class_id).astype(bool, copy=False)
        rgba = np.zeros((*self.image_shape, 4), dtype=np.uint8)
        rgba[mask, :3] = source[mask]
        rgba[mask, 3] = 255
        rgba.setflags(write=False)
        return rgba

    def analyze_index(
        self,
        data: HSIData,
        request: VisualizationRequest,
        *,
        visualization_service: VisualizationService | None = None,
        progress: ProgressCallback | None = None,
        is_cancelled: CancellationCheck | None = None,
    ) -> ClassificationIndexAnalysis:
        """Calculate one vegetation index and summarize it for every class.

        Run this method in a Controller worker because rendering reads bands
        from the hyperspectral cube. Use :meth:`analyze_visualization` instead
        when the Controller already has a cached ``VisualizationResult``.
        """

        self._validate_data_shape(data)
        service = visualization_service or VisualizationService()

        def render_progress(value: int, message: str) -> None:
            self._emit(progress, round(value * 0.7), message)

        visualization = service.render(
            data,
            request,
            progress=render_progress,
            is_cancelled=is_cancelled,
        )
        return self.analyze_visualization(
            visualization,
            progress=progress,
            is_cancelled=is_cancelled,
            progress_start=70,
        )

    def analyze_visualization(
        self,
        visualization: VisualizationResult,
        *,
        progress: ProgressCallback | None = None,
        is_cancelled: CancellationCheck | None = None,
        progress_start: int = 0,
    ) -> ClassificationIndexAnalysis:
        """Partition an already-rendered vegetation index by class masks."""

        if visualization.mode not in INDEX_MODES or visualization.values is None:
            raise ClassificationError(
                "Per-class analysis requires an NDVI, EVI, MCARI, MTVI, OSAVI, "
                "or PRI VisualizationResult."
            )
        if visualization.values.shape != self.image_shape:
            raise ClassificationError(
                f"Index raster shape {visualization.values.shape} does not match "
                f"classification shape {self.image_shape}."
            )
        expected_display_shape = (*self.image_shape, 3)
        if visualization.display_rgb.shape != expected_display_shape:
            raise ClassificationError(
                f"Index display shape {visualization.display_rgb.shape} does not "
                f"match expected {expected_display_shape}."
            )
        if not 0 <= progress_start <= 100:
            raise ClassificationError("Progress start must be between 0 and 100.")

        self._check_cancelled(is_cancelled)
        statistics: list[ClassIndexStatistics] = []
        total_pixels = int(np.prod(self.image_shape))
        remaining = 100 - progress_start
        for position, (class_id, mask) in enumerate(
            zip(self._class_ids, self._one_hot_masks, strict=True), start=1
        ):
            self._check_cancelled(is_cancelled)
            selected = visualization.values[mask.astype(bool, copy=False)]
            finite = selected[np.isfinite(selected)]
            pixel_count = int(selected.size)
            finite_count = int(finite.size)
            if finite_count:
                mean = float(np.mean(finite, dtype=np.float64))
                median = float(np.median(finite))
                standard_deviation = float(np.std(finite, dtype=np.float64))
                minimum = float(np.min(finite))
                maximum = float(np.max(finite))
            else:
                mean = median = standard_deviation = minimum = maximum = None
            statistics.append(
                ClassIndexStatistics(
                    class_id=class_id,
                    class_name=self._names[class_id],
                    pixel_count=pixel_count,
                    finite_pixel_count=finite_count,
                    image_fraction=(pixel_count / total_pixels if total_pixels else 0.0),
                    mean=mean,
                    median=median,
                    standard_deviation=standard_deviation,
                    minimum=minimum,
                    maximum=maximum,
                )
            )
            completed = progress_start + round(
                remaining * position / max(1, len(self._class_ids))
            )
            self._emit(
                progress,
                completed,
                f"Calculated {visualization.mode.value} for class {class_id}",
            )

        return ClassificationIndexAnalysis(
            visualization=visualization,
            statistics=tuple(statistics),
            class_ids=self._class_ids,
            _one_hot_masks=self._one_hot_masks,
        )

    def compose_index(
        self,
        analysis: ClassificationIndexAnalysis,
        *,
        background_color: tuple[int, int, int] = (0, 0, 0),
    ) -> ClassificationLayerComposite:
        """Apply current visibility to an index analysis's false-colour image."""

        if (
            analysis.class_ids != self._class_ids
            or analysis._one_hot_masks is not self._one_hot_masks
        ):
            raise ClassificationError(
                "Index analysis belongs to a different classification result."
            )
        return self.compose_display(
            analysis.visualization.display_rgb,
            background_color=background_color,
        )

    def _validated_rgb(self, rgb: np.ndarray) -> np.ndarray:
        values = np.asarray(rgb)
        expected = (*self.image_shape, 3)
        if values.shape != expected:
            raise ClassificationError(
                f"RGB display shape {values.shape} does not match expected {expected}."
            )
        if not np.issubdtype(values.dtype, np.number):
            raise ClassificationError("RGB display must contain numeric values.")
        if values.dtype == np.uint8:
            return values
        return np.clip(values, 0, 255).astype(np.uint8)

    def _validate_data_shape(self, data: HSIData) -> None:
        if (data.rows, data.columns) != self.image_shape:
            raise ClassificationError(
                f"Cube shape {(data.rows, data.columns)} does not match "
                f"classification shape {self.image_shape}."
            )

    def _class_index(self, class_id: int) -> int:
        normalized = self._require_class(class_id)
        return self._class_ids.index(normalized)

    def _require_class(self, class_id: int) -> int:
        normalized = _normalize_class_id(class_id)
        if normalized not in self._class_ids:
            raise ClassificationError(
                f"Unknown classification class ID: {class_id}"
            )
        return normalized

    @staticmethod
    def _emit(callback: ProgressCallback | None, value: int, message: str) -> None:
        if callback is not None:
            callback(value, message)

    @staticmethod
    def _check_cancelled(callback: CancellationCheck | None) -> None:
        if callback is not None and callback():
            raise CancelledError("Classification index analysis was cancelled.")


def _normalize_class_id(class_id: int) -> int:
    if isinstance(class_id, bool) or not isinstance(class_id, Integral):
        raise ClassificationError("Classification class ID must be an integer.")
    return int(class_id)


def _normalize_opacity(opacity: float) -> float:
    if isinstance(opacity, bool) or not isinstance(opacity, Real):
        raise ClassificationError("Layer opacity must be a number.")
    value = float(opacity)
    if not 0.0 <= value <= 1.0:
        raise ClassificationError("Layer opacity must be between 0.0 and 1.0.")
    return value


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    """Return ``mask`` pixels adjacent (4-connected) to a non-mask pixel.

    An image-edge pixel counts as adjacent to the outside, so a class region
    touching the frame is still fully outlined. Pure NumPy (no SciPy) keeps
    this dependency-free for the base install.
    """

    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return mask & ~interior


def _normalize_rgb_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    try:
        values = tuple(color)
    except TypeError as exc:
        raise ClassificationError("Background colour must be an RGB triplet.") from exc
    if len(values) != 3 or any(
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not 0 <= int(value) <= 255
        for value in values
    ):
        raise ClassificationError(
            "Background colour must contain three integers from 0 to 255."
        )
    return tuple(int(value) for value in values)
