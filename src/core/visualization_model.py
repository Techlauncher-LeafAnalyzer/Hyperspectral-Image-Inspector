"""SPy-backed visualization Model and its Controller-facing data contracts.

This module intentionally imports no Qt classes. Public operations are
synchronous and return plain dataclasses/NumPy arrays, allowing a Controller
to choose its own worker, signals, cancellation, and View implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Callable, Mapping

from matplotlib import colormaps
import numpy as np
from spectral import get_rgb

from .errors import CancelledError, VisualizationError
from .hsi_data import HSIData


# These callbacks execute on the same thread that invokes the Model method.
# A Qt Controller must bridge progress to the GUI thread using queued signals.
ProgressCallback = Callable[[int, str], None]
CancellationCheck = Callable[[], bool]


class VisualizationMode(StrEnum):
    """Stable mode identifiers for Controller actions and project JSON."""
    RGB = "RGB"
    BAND = "BAND"
    NDVI = "NDVI"
    EVI = "EVI"
    MCARI = "MCARI"
    MTVI = "MTVI"
    OSAVI = "OSAVI"
    PRI = "PRI"


@dataclass(frozen=True, slots=True)
class DisplayStretch:
    """Percentile limits used only to map raw values into display colors.

    Stretching never modifies analytical values returned in a result.
    """
    lower_percentile: float = 2.0
    upper_percentile: float = 98.0

    def __post_init__(self) -> None:
        if not 0 <= self.lower_percentile < self.upper_percentile <= 100:
            raise VisualizationError(
                "Display percentiles must satisfy 0 <= lower < upper <= 100."
            )


@dataclass(frozen=True, slots=True)
class VisualizationRequest:
    """Immutable visualization command constructed by a Controller.

    ``band_index`` is required only for ``BAND``. ``colormap`` applies to
    vegetation indices and must name a Matplotlib colormap; ``None`` selects
    the Model default.
    """
    mode: VisualizationMode | str
    band_index: int | None = None
    stretch: DisplayStretch = DisplayStretch()
    colormap: str | None = None


@dataclass(frozen=True, slots=True)
class VisualizationResult:
    """View-neutral output from :meth:`VisualizationService.render`.

    ``display_rgb`` is always ``uint8`` with shape ``(rows, columns, 3)``.
    ``values`` is the unstretched ``float32`` analytical raster for BAND/index
    modes and ``None`` for RGB. ``value_range`` describes the finite raw-data
    minimum/maximum, not the percentile display limits.

    Band mappings record the actual source bands used for reproducibility.
    The service does not cache this result; the Controller owns its arrays.
    """
    mode: VisualizationMode
    display_rgb: np.ndarray
    values: np.ndarray | None
    title: str
    value_range: tuple[float, float]
    band_indices: Mapping[str, int]
    band_wavelengths_nm: Mapping[str, float]
    colormap: str | None


@dataclass(frozen=True, slots=True)
class SpectrumResult:
    """Read-only wavelength and value arrays for one zero-based pixel."""
    row: int
    column: int
    wavelengths_nm: np.ndarray
    values: np.ndarray


@dataclass(frozen=True, slots=True)
class HypercubeData:
    """RGB plus orthogonal slices for a custom future cube renderer.

    The current SPy OpenGL View should use :class:`HypercubeViewData` instead.
    """
    top_rgb: np.ndarray
    row_side_values: np.ndarray
    column_side_values: np.ndarray
    wavelengths_nm: np.ndarray
    row_index: int
    column_index: int


@dataclass(frozen=True, slots=True)
class HypercubeViewData:
    """Downsampled surface payload for the interactive SPy cube View.

    ``surface_cube`` contains real values on four boundaries; its unused
    interior is zero and must not be used for scientific analysis. Index arrays
    map every sample back to the original :class:`HSIData`. Construct OpenGL/Qt
    widgets from this result only after returning to the GUI thread.
    """

    top_rgb: np.ndarray
    surface_cube: np.ndarray
    wavelengths_nm: np.ndarray
    row_indices: np.ndarray
    column_indices: np.ndarray
    band_indices: np.ndarray


class VisualizationService:
    """View-neutral visualization model backed by Spectral Python.

    Methods are synchronous and stateless. A Controller should run disk-reading
    operations in a worker, forward progress through thread-safe signals, and
    deliver the result to the GUI thread. Reuse is safe when reads on the same
    :class:`HSIData` are serialized.
    """

    RGB_TARGETS = MappingProxyType({"red": 660.0, "green": 550.0, "blue": 470.0})
    DEFAULT_COLORMAPS = MappingProxyType(
        {
            VisualizationMode.NDVI: "RdYlGn",
            VisualizationMode.EVI: "RdYlGn",
            VisualizationMode.MCARI: "viridis",
            VisualizationMode.MTVI: "viridis",
            VisualizationMode.OSAVI: "RdYlGn",
            VisualizationMode.PRI: "Spectral",
        }
    )

    @property
    def supported_modes(self) -> tuple[VisualizationMode, ...]:
        """Return supported 2D modes for populating Controller actions."""
        return tuple(VisualizationMode)

    def render(
        self,
        data: HSIData,
        request: VisualizationRequest,
        *,
        progress: ProgressCallback | None = None,
        is_cancelled: CancellationCheck | None = None,
    ) -> VisualizationResult:
        """Render RGB, one band, or a vegetation index.

        ``progress`` receives ``(integer_percent, message)`` on the calling
        thread. ``is_cancelled`` should be an inexpensive zero-argument check.
        Keep the previous image visible until this succeeds, then replace the
        View atomically with ``result.display_rgb``.

        Raises:
            VisualizationError: The request, band, stretch, or colormap is invalid.
            WavelengthError: Required wavelength coverage is unavailable.
            CancelledError: The Controller requested cancellation.
        """
        mode = self._coerce_mode(request.mode)
        self._check_cancelled(is_cancelled)
        self._emit(progress, 0, f"Preparing {mode.value}")
        if mode is VisualizationMode.RGB:
            result = self._render_rgb(data, request.stretch)
        elif mode is VisualizationMode.BAND:
            if request.band_index is None:
                raise VisualizationError("BAND mode requires band_index.")
            result = self._render_band(data, request.band_index, request.stretch)
        else:
            result = self._render_index(data, mode, request)
        self._check_cancelled(is_cancelled)
        self._emit(progress, 100, f"{mode.value} complete")
        return result

    def spectrum(self, data: HSIData, row: int, column: int) -> SpectrumResult:
        """Read the spectrum at a zero-based source-image coordinate.

        Controllers must undo View scaling, scroll offsets, and letterboxing
        before passing clicked widget coordinates to this method.
        """
        values = data.read_pixel(row, column)
        wavelengths = data.wavelengths_nm.copy()
        wavelengths.setflags(write=False)
        values.setflags(write=False)
        return SpectrumResult(row, column, wavelengths, values)

    def hypercube_data(
        self,
        data: HSIData,
        *,
        row_index: int | None = None,
        column_index: int | None = None,
        stretch: DisplayStretch = DisplayStretch(),
        max_side_points: int = 1024,
    ) -> HypercubeData:
        """Return RGB plus two sampled orthogonal spectral slices.

        This lower-level payload targets custom renderers. Sampling bounds
        memory while retaining all spectral bands; the call performs disk I/O.
        """
        if max_side_points < 2:
            raise VisualizationError("max_side_points must be at least 2.")
        row = data.rows // 2 if row_index is None else int(row_index)
        column = data.columns // 2 if column_index is None else int(column_index)
        if not 0 <= row < data.rows or not 0 <= column < data.columns:
            raise VisualizationError("Hypercube slice row or column is outside the cube.")
        top = self._render_rgb(data, stretch).display_rgb
        column_samples = np.linspace(
            0, data.columns - 1, min(data.columns, max_side_points), dtype=int
        )
        row_samples = np.linspace(
            0, data.rows - 1, min(data.rows, max_side_points), dtype=int
        )
        bands = list(range(data.bands))
        row_side = np.asarray(
            data.image.read_subimage([row], column_samples.tolist(), bands), dtype=np.float32
        )[0]
        column_side = np.asarray(
            data.image.read_subimage(row_samples.tolist(), [column], bands), dtype=np.float32
        )[:, 0, :]
        return HypercubeData(
            top_rgb=top,
            row_side_values=row_side,
            column_side_values=column_side,
            wavelengths_nm=data.wavelengths_nm.copy(),
            row_index=row,
            column_index=column,
        )

    def prepare_hypercube_view(
        self,
        data: HSIData,
        *,
        max_spatial_side: int = 256,
        max_spectral_bands: int = 192,
        stretch: DisplayStretch = DisplayStretch(),
        progress: ProgressCallback | None = None,
        is_cancelled: CancellationCheck | None = None,
    ) -> HypercubeViewData:
        """Prepare a compact cube whose surfaces retain real HSI values.

        SPy's OpenGL cube renders only the top texture and four boundary
        surfaces. Reading those surfaces directly avoids loading the complete
        hyperspectral volume into memory. Spatial sampling preserves the
        source aspect ratio and spectral sampling covers the wavelength range.

        Run this method in a Controller worker. Construct SPy's OpenGL widget
        on the GUI thread after receiving the result. Cancellation is checked
        between surface reads; progress is emitted at each completed stage.

        Raises:
            VisualizationError: A sampling limit is smaller than two.
            WavelengthError: RGB wavelengths are unavailable.
            CancelledError: Cancellation was observed between reads.
        """
        if max_spatial_side < 2 or max_spectral_bands < 2:
            raise VisualizationError("Hypercube sampling limits must be at least 2.")

        self._check_cancelled(is_cancelled)
        self._emit(progress, 0, "Preparing hypercube RGB surface")
        rgb = self._render_rgb(data, stretch).display_rgb

        spatial_scale = min(1.0, max_spatial_side / max(data.rows, data.columns))
        row_count = min(data.rows, max(2, round(data.rows * spatial_scale)))
        column_count = min(data.columns, max(2, round(data.columns * spatial_scale)))
        band_count = min(data.bands, max_spectral_bands)
        rows = np.unique(np.linspace(0, data.rows - 1, row_count, dtype=int))
        columns = np.unique(np.linspace(0, data.columns - 1, column_count, dtype=int))
        bands = np.unique(np.linspace(0, data.bands - 1, band_count, dtype=int))
        row_list = rows.tolist()
        column_list = columns.tolist()
        band_list = bands.tolist()

        top = np.ascontiguousarray(rgb[np.ix_(rows, columns)])
        surface = np.zeros((len(rows), len(columns), len(bands)), dtype=np.float32)

        self._check_cancelled(is_cancelled)
        self._emit(progress, 25, "Reading front cube surface")
        surface[-1, :, :] = np.asarray(
            data.image.read_subimage([data.rows - 1], column_list, band_list),
            dtype=np.float32,
        )[0]

        self._check_cancelled(is_cancelled)
        self._emit(progress, 45, "Reading right cube surface")
        surface[:, -1, :] = np.asarray(
            data.image.read_subimage(row_list, [data.columns - 1], band_list),
            dtype=np.float32,
        )[:, 0, :]

        self._check_cancelled(is_cancelled)
        self._emit(progress, 65, "Reading back cube surface")
        surface[0, :, :] = np.asarray(
            data.image.read_subimage([0], column_list, band_list), dtype=np.float32
        )[0]

        self._check_cancelled(is_cancelled)
        self._emit(progress, 85, "Reading left cube surface")
        surface[:, 0, :] = np.asarray(
            data.image.read_subimage(row_list, [0], band_list), dtype=np.float32
        )[:, 0, :]

        wavelengths = np.ascontiguousarray(data.wavelengths_nm[bands])
        self._check_cancelled(is_cancelled)
        self._emit(progress, 100, "Hypercube ready")
        return HypercubeViewData(
            top_rgb=top,
            surface_cube=surface,
            wavelengths_nm=wavelengths,
            row_indices=rows,
            column_indices=columns,
            band_indices=bands,
        )

    def _render_rgb(self, data: HSIData, stretch: DisplayStretch) -> VisualizationResult:
        indices = {
            name: data.nearest_band(target, tolerance_nm=20)
            for name, target in self.RGB_TARGETS.items()
        }
        rgb = get_rgb(
            data.image,
            (indices["red"], indices["green"], indices["blue"]),
            stretch=(stretch.lower_percentile / 100, stretch.upper_percentile / 100),
            stretch_all=True,
        )
        display = np.round(
            np.clip(np.nan_to_num(np.asarray(rgb, dtype=np.float32)), 0, 1) * 255
        ).astype(np.uint8)
        wavelengths = self._wavelength_map(data, indices)
        return VisualizationResult(
            mode=VisualizationMode.RGB,
            display_rgb=display,
            values=None,
            title=(
                f"RGB ({wavelengths['red']:g}, {wavelengths['green']:g}, "
                f"{wavelengths['blue']:g} nm)"
            ),
            value_range=(0.0, 1.0),
            band_indices=MappingProxyType(indices),
            band_wavelengths_nm=MappingProxyType(wavelengths),
            colormap=None,
        )

    def _render_band(
        self, data: HSIData, band_index: int, stretch: DisplayStretch
    ) -> VisualizationResult:
        values = data.read_band(band_index)
        normalized, value_range = self._percentile_stretch(values, stretch)
        display = np.round(np.repeat(normalized[:, :, None], 3, axis=2) * 255).astype(
            np.uint8
        )
        wavelength = float(data.wavelengths_nm[band_index])
        return VisualizationResult(
            mode=VisualizationMode.BAND,
            display_rgb=display,
            values=values,
            title=f"Band {band_index} ({wavelength:g} nm)",
            value_range=value_range,
            band_indices=MappingProxyType({"band": band_index}),
            band_wavelengths_nm=MappingProxyType({"band": wavelength}),
            colormap="gray",
        )

    def _render_index(
        self,
        data: HSIData,
        mode: VisualizationMode,
        request: VisualizationRequest,
    ) -> VisualizationResult:
        targets = self._index_targets(mode)
        indices = {
            name: data.nearest_band(wavelength, tolerance_nm=15)
            for name, wavelength in targets.items()
        }
        cube = data.read_bands(list(indices.values()))
        bands = {name: cube[:, :, position] for position, name in enumerate(indices)}
        values = self._calculate_index(mode, bands)
        values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        normalized, value_range = self._percentile_stretch(values, request.stretch)
        colormap = request.colormap or self.DEFAULT_COLORMAPS[mode]
        try:
            display = np.round(colormaps[colormap](normalized)[..., :3] * 255).astype(np.uint8)
        except KeyError as exc:
            raise VisualizationError(f"Unknown Matplotlib colormap: {colormap}") from exc
        return VisualizationResult(
            mode=mode,
            display_rgb=display,
            values=values,
            title=mode.value,
            value_range=value_range,
            band_indices=MappingProxyType(indices),
            band_wavelengths_nm=MappingProxyType(self._wavelength_map(data, indices)),
            colormap=colormap,
        )

    @staticmethod
    def _index_targets(mode: VisualizationMode) -> Mapping[str, float]:
        targets: dict[VisualizationMode, Mapping[str, float]] = {
            VisualizationMode.NDVI: {"nir": 800, "red": 670},
            VisualizationMode.EVI: {"nir": 800, "red": 670, "blue": 470},
            VisualizationMode.MCARI: {"red_edge": 700, "red": 670, "green": 550},
            VisualizationMode.MTVI: {"nir": 800, "red": 670, "green": 550},
            VisualizationMode.OSAVI: {"nir": 800, "red": 670},
            VisualizationMode.PRI: {"r531": 531, "r570": 570},
        }
        return targets[mode]

    def _calculate_index(
        self, mode: VisualizationMode, bands: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        if mode is VisualizationMode.NDVI:
            return self._safe_divide(bands["nir"] - bands["red"], bands["nir"] + bands["red"])
        if mode is VisualizationMode.EVI:
            return 2.5 * self._safe_divide(
                bands["nir"] - bands["red"],
                bands["nir"] + 6 * bands["red"] - 7.5 * bands["blue"] + 1,
            )
        if mode is VisualizationMode.MCARI:
            return (
                bands["red_edge"]
                - bands["red"]
                - 0.2 * (bands["red_edge"] - bands["green"])
            ) * self._safe_divide(bands["red_edge"], bands["red"])
        if mode is VisualizationMode.MTVI:
            return 1.2 * (
                1.2 * (bands["nir"] - bands["green"])
                - 2.5 * (bands["red"] - bands["green"])
            )
        if mode is VisualizationMode.OSAVI:
            return 1.16 * self._safe_divide(
                bands["nir"] - bands["red"], bands["nir"] + bands["red"] + 0.16
            )
        if mode is VisualizationMode.PRI:
            return self._safe_divide(
                bands["r531"] - bands["r570"], bands["r531"] + bands["r570"]
            )
        raise VisualizationError(f"No index implementation for {mode.value}.")

    @staticmethod
    def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float32),
            where=np.abs(denominator) > np.finfo(np.float32).eps,
        )

    @staticmethod
    def _percentile_stretch(
        values: np.ndarray, stretch: DisplayStretch
    ) -> tuple[np.ndarray, tuple[float, float]]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.zeros(values.shape, dtype=np.float32), (0.0, 0.0)
        low, high = np.percentile(
            finite, (stretch.lower_percentile, stretch.upper_percentile)
        )
        if high <= low:
            high = low + np.finfo(np.float32).eps
        normalized = np.clip((values - low) / (high - low), 0, 1).astype(np.float32)
        return normalized, (float(finite.min()), float(finite.max()))

    @staticmethod
    def _wavelength_map(data: HSIData, indices: Mapping[str, int]) -> dict[str, float]:
        return {name: float(data.wavelengths_nm[index]) for name, index in indices.items()}

    @staticmethod
    def _coerce_mode(mode: VisualizationMode | str) -> VisualizationMode:
        try:
            return mode if isinstance(mode, VisualizationMode) else VisualizationMode(str(mode).upper())
        except ValueError as exc:
            choices = ", ".join(item.value for item in VisualizationMode)
            raise VisualizationError(f"Unknown mode {mode!r}; choose one of {choices}.") from exc

    @staticmethod
    def _emit(progress: ProgressCallback | None, value: int, message: str) -> None:
        if progress is not None:
            progress(value, message)

    @staticmethod
    def _check_cancelled(is_cancelled: CancellationCheck | None) -> None:
        if is_cancelled is not None and is_cancelled():
            raise CancelledError("Visualization was cancelled.")
