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

from .custom_index import parse_custom_index
from .errors import CancelledError, VisualizationError, WavelengthError
from .hsi_data import HSIData
from .roi import roi_boundary


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
    NDWI = "NDWI"
    NDMI = "NDMI"
    CUSTOM = "CUSTOM"
    BRIGHTNESS = "BRIGHTNESS"


# Modes that only make sense for some cameras (e.g. Specim FX17's SWIR-only
# range); UI code hides their controls instead of merely disabling them.
OPTIONAL_VISUALIZATION_MODES = frozenset({VisualizationMode.NDWI, VisualizationMode.NDMI})

# Modes the user configures or that read the whole cube; Controllers render
# them on demand instead of with the eagerly cached vegetation indices.
ON_DEMAND_VISUALIZATION_MODES = frozenset(
    {VisualizationMode.CUSTOM, VisualizationMode.BRIGHTNESS}
)


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

    ``band_index`` is required only for ``BAND`` and ``formula`` only for
    ``CUSTOM`` (see :mod:`core.custom_index` for its syntax). ``colormap``
    applies to indices and must name a Matplotlib colormap; ``None`` selects
    the Model default.

    ``BRIGHTNESS`` normalizes the summed spectrum over the whole image unless
    ``segments`` is given: an integer label raster matching the image, whose
    every label ``>= 0`` is normalized on its own and whose negative labels
    are left as no-data. ``band_sum`` may carry a raster previously returned
    by :meth:`VisualizationService.band_sum` to skip re-reading the cube.
    """
    mode: VisualizationMode | str
    band_index: int | None = None
    stretch: DisplayStretch = DisplayStretch()
    colormap: str | None = None
    formula: str | None = None
    segments: np.ndarray | None = None
    band_sum: np.ndarray | None = None


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
    # Percentile limits mapped to the colormap's ends; ``None`` for RGB.
    display_limits: tuple[float, float] | None = None

    def color_for(self, value: float) -> tuple[int, int, int] | None:
        """Return the display colour this result's mapping gives ``value``.

        Matches the per-pixel rendering in ``display_rgb``, so a derived
        statistic (e.g. a class mean) can be shown in the same colour scale.
        """

        if self.display_limits is None or self.colormap is None:
            return None
        low, high = self.display_limits
        normalized = np.clip(np.float32((value - low) / (high - low)), 0, 1)
        rgba = colormaps[self.colormap](normalized)
        return tuple(int(channel) for channel in np.round(np.asarray(rgba[:3]) * 255))


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

    After a polygon crop, ``roi_mask`` is the region of interest sampled on the
    same ``(row_indices, column_indices)`` grid. The cube is then a prism
    extruded from that mask: ``surface_cube`` holds real spectra at every
    selected sample with an unselected (or out-of-frame) 4-neighbour, which is
    exactly where the prism's side walls are.
    """

    top_rgb: np.ndarray
    surface_cube: np.ndarray
    wavelengths_nm: np.ndarray
    row_indices: np.ndarray
    column_indices: np.ndarray
    band_indices: np.ndarray
    roi_mask: np.ndarray | None = None


class VisualizationService:
    """View-neutral visualization model backed by Spectral Python.

    Methods are synchronous and stateless. A Controller should run disk-reading
    operations in a worker, forward progress through thread-safe signals, and
    deliver the result to the GUI thread. Reuse is safe when reads on the same
    :class:`HSIData` are serialized.
    """

    RGB_TARGETS = MappingProxyType({"red": 660.0, "green": 550.0, "blue": 470.0})
    RGB_TOLERANCE_NM = 20.0
    # Display channels for cameras without visible coverage (e.g. Specim FX17,
    # 935-1720 nm), chosen to match the vendor preview. Cubes that lack any of
    # these wavelengths fall back to first/middle/last.
    SWIR_FALSE_COLOUR_TARGETS = MappingProxyType(
        {"red": 1340.0, "green": 1040.0, "blue": 1720.0}
    )
    INDEX_TOLERANCE_NM = 15.0
    DEFAULT_COLORMAPS = MappingProxyType(
        {
            VisualizationMode.NDVI: "RdYlGn",
            VisualizationMode.EVI: "RdYlGn",
            VisualizationMode.MCARI: "viridis",
            VisualizationMode.MTVI: "viridis",
            VisualizationMode.OSAVI: "RdYlGn",
            VisualizationMode.PRI: "Spectral",
            VisualizationMode.NDWI: "BrBG",
            VisualizationMode.NDMI: "BrBG",
            VisualizationMode.CUSTOM: "viridis",
            VisualizationMode.BRIGHTNESS: "inferno",
        }
    )
    # Bands summed per read when computing BRIGHTNESS, bounding peak memory
    # to this many full-frame float32 planes.
    BAND_SUM_CHUNK = 16

    @property
    def supported_modes(self) -> tuple[VisualizationMode, ...]:
        """Return supported 2D modes for populating Controller actions."""
        return tuple(VisualizationMode)

    def unavailable_reason(self, data: HSIData, mode: VisualizationMode | str) -> str | None:
        """Check wavelength capabilities without reading any image pixels."""
        if not data.is_loaded():
            return "Load an image first."
        mode = self._coerce_mode(mode)
        if mode in ON_DEMAND_VISUALIZATION_MODES:
            # CUSTOM bands are checked per formula when it is rendered.
            return None
        if mode not in (VisualizationMode.RGB, VisualizationMode.BAND):
            try:
                self._index_band_indices(data, mode)
            except WavelengthError as exc:
                return f"{mode.value} unavailable: {exc}"
        return None

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
        elif mode is VisualizationMode.CUSTOM:
            result = self._render_custom(data, request)
        elif mode is VisualizationMode.BRIGHTNESS:
            result = self._render_brightness(data, request, progress, is_cancelled)
        else:
            result = self._render_index(data, mode, request)
        self._check_cancelled(is_cancelled)
        self._emit(progress, 100, f"{mode.value} complete")
        return result

    def band_sum(
        self,
        data: HSIData,
        *,
        progress: ProgressCallback | None = None,
        is_cancelled: CancellationCheck | None = None,
    ) -> np.ndarray:
        """Return every pixel's spectrum summed over all bands, as ``float64``.

        Bands are read in chunks of :attr:`BAND_SUM_CHUNK`, so memory stays
        bounded however many bands the cube has. Pixels outside a polygon
        crop are NaN. The result can be passed back as
        ``VisualizationRequest.band_sum`` to re-normalize without new reads.
        """

        total = np.zeros((data.rows, data.columns), dtype=np.float64)
        for start in range(0, data.bands, self.BAND_SUM_CHUNK):
            stop = min(start + self.BAND_SUM_CHUNK, data.bands)
            self._check_cancelled(is_cancelled)
            self._emit(
                progress,
                round(90 * start / data.bands),
                f"Summing bands {start + 1}-{stop} of {data.bands}",
            )
            chunk = data.read_bands(range(start, stop))
            total += np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0).sum(
                axis=2, dtype=np.float64
            )
        return data.masked(total).fill()

    @staticmethod
    def normalize_by_segment(
        values: np.ndarray, segments: np.ndarray | None = None
    ) -> np.ndarray:
        """Min-max scale ``values`` to 0..1, per segment label when given.

        Each label ``>= 0`` in ``segments`` is scaled using only its own
        finite pixels; negative labels and non-finite inputs become NaN. A
        region of constant value maps to 0.
        """

        values = np.asarray(values, dtype=np.float64)
        normalized = np.full(values.shape, np.nan, dtype=np.float32)
        if segments is None:
            labels = np.zeros(values.shape, dtype=np.int64)
        else:
            labels = np.asarray(segments)
            if labels.shape != values.shape:
                raise VisualizationError(
                    f"Segment map shape {labels.shape} does not match the "
                    f"{values.shape} image."
                )
        valid = np.isfinite(values) & (labels >= 0)
        for label in np.unique(labels[valid]):
            region = valid & (labels == label)
            low = values[region].min()
            span = values[region].max() - low
            normalized[region] = (values[region] - low) / span if span > 0 else 0.0
        return normalized

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

        if data.roi_mask is not None:
            roi = np.ascontiguousarray(data.roi_mask[np.ix_(rows, columns)])
            if not roi.any():
                raise VisualizationError(
                    "The polygon crop is too small to display as a hypercube."
                )
            self._read_roi_walls(
                data, roi, rows, column_list, band_list, surface,
                progress, is_cancelled,
            )
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
                roi_mask=roi,
            )

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

    def _read_roi_walls(
        self,
        data: HSIData,
        roi: np.ndarray,
        rows: np.ndarray,
        column_list: list[int],
        band_list: list[int],
        surface: np.ndarray,
        progress: ProgressCallback | None,
        is_cancelled: CancellationCheck | None,
    ) -> None:
        """Fill ``surface`` with spectra at the sampled ROI's boundary cells.

        A polygon's walls follow its outline rather than the bounding box, so
        the four rectangular face reads are replaced by one read per sampled
        row covering only that row's boundary samples.
        """
        boundary = roi_boundary(roi)
        boundary_rows = np.flatnonzero(boundary.any(axis=1))
        for done, row in enumerate(boundary_rows):
            self._check_cancelled(is_cancelled)
            self._emit(
                progress,
                25 + round(70 * done / len(boundary_rows)),
                "Reading polygon cube surface",
            )
            cells = np.flatnonzero(boundary[row])
            values = np.asarray(
                data.image.read_subimage(
                    [int(rows[row])], [column_list[cell] for cell in cells], band_list
                ),
                dtype=np.float32,
            )[0]
            surface[row, cells, :] = values

    def _render_rgb(self, data: HSIData, stretch: DisplayStretch) -> VisualizationResult:
        false_colour = False
        try:
            indices = {
                name: data.nearest_band(target, tolerance_nm=self.RGB_TOLERANCE_NM)
                for name, target in self.RGB_TARGETS.items()
            }
        except WavelengthError:
            # Missing visible coverage is not a load failure. Retain all original
            # spectral data and use a fixed false-colour mapping for the display.
            if data.bands < 1 or data.wavelengths_nm.size != data.bands:
                raise
            try:
                indices = {
                    name: data.nearest_band(target, tolerance_nm=self.RGB_TOLERANCE_NM)
                    for name, target in self.SWIR_FALSE_COLOUR_TARGETS.items()
                }
            except WavelengthError:
                indices = {"red": 0, "green": data.bands // 2, "blue": data.bands - 1}
            false_colour = True
        if data.roi_mask is None:
            rgb = get_rgb(
                data.image,
                (indices["red"], indices["green"], indices["blue"]),
                stretch=(stretch.lower_percentile / 100, stretch.upper_percentile / 100),
                stretch_all=True,
            )
            rgb = np.asarray(rgb, dtype=np.float32)
        else:
            # SPy's get_rgb percentiles over the whole rectangle, which a
            # polygon crop's excluded corners would skew. Stretch each
            # channel over the region of interest only; `stretch_all=True`
            # is per-channel, so this matches its semantics.
            cube = data.read_bands(
                [indices["red"], indices["green"], indices["blue"]]
            )
            rgb = np.stack(
                [
                    self._percentile_stretch(
                        data.masked(cube[:, :, channel]).fill(), stretch
                    )[0]
                    for channel in range(3)
                ],
                axis=2,
            )
        display = np.round(
            np.clip(np.nan_to_num(rgb), 0, 1) * 255
        ).astype(np.uint8)
        wavelengths = self._wavelength_map(data, indices)
        return VisualizationResult(
            mode=VisualizationMode.RGB,
            display_rgb=display,
            values=None,
            title=(
                f"{'False-colour RGB' if false_colour else 'RGB'} "
                f"({wavelengths['red']:g}, {wavelengths['green']:g}, "
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
        values = data.masked(data.read_band(band_index)).fill()
        normalized, value_range, limits = self._percentile_stretch(values, stretch)
        # Excluded pixels stay NaN in `values` (so hover and statistics report
        # "no data") but must be finite before the uint8 display cast.
        display = np.round(
            np.repeat(np.nan_to_num(normalized)[:, :, None], 3, axis=2) * 255
        ).astype(np.uint8)
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
            display_limits=limits,
        )

    def _render_index(
        self,
        data: HSIData,
        mode: VisualizationMode,
        request: VisualizationRequest,
    ) -> VisualizationResult:
        indices = self._index_band_indices(data, mode)
        cube = data.read_bands(list(indices.values()))
        bands = {name: cube[:, :, position] for position, name in enumerate(indices)}
        values = self._calculate_index(mode, bands)
        values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        # Re-introduce NaN *after* neutralising genuine division artefacts, so
        # only ROI-excluded pixels stay non-finite. `_percentile_stretch` and
        # the Index Mean's `np.nanmean` then both skip them; Matplotlib
        # colormaps render them as the colormap's "bad" colour.
        values = data.masked(values).fill()
        normalized, value_range, limits = self._percentile_stretch(
            values, request.stretch
        )
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
            display_limits=limits,
        )

    def _render_custom(
        self, data: HSIData, request: VisualizationRequest
    ) -> VisualizationResult:
        if not request.formula:
            raise VisualizationError("CUSTOM mode requires a formula.")
        formula = parse_custom_index(request.formula)
        resolved = formula.resolve(data.wavelengths_nm)
        cube = data.read_bands([band.band_index for band in resolved])
        values = formula.evaluate(
            {band.requested_nm: cube[:, :, position] for position, band in enumerate(resolved)}
        )
        # Same treatment as the built-in indices: arithmetic artefacts become
        # 0 and only ROI-excluded pixels remain NaN.
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = data.masked(values).fill()
        normalized, value_range, limits = self._percentile_stretch(values, request.stretch)
        colormap = request.colormap or self.DEFAULT_COLORMAPS[VisualizationMode.CUSTOM]
        return VisualizationResult(
            mode=VisualizationMode.CUSTOM,
            display_rgb=self._apply_colormap(normalized, colormap),
            values=values,
            title=f"Custom index: {formula.text}",
            value_range=value_range,
            band_indices=MappingProxyType({band.token: band.band_index for band in resolved}),
            band_wavelengths_nm=MappingProxyType(
                {band.token: band.wavelength_nm for band in resolved}
            ),
            colormap=colormap,
            display_limits=limits,
        )

    def _render_brightness(
        self,
        data: HSIData,
        request: VisualizationRequest,
        progress: ProgressCallback | None,
        is_cancelled: CancellationCheck | None,
    ) -> VisualizationResult:
        total = request.band_sum
        if total is None:
            total = self.band_sum(data, progress=progress, is_cancelled=is_cancelled)
        elif total.shape != (data.rows, data.columns):
            raise VisualizationError("The supplied band sum does not match the image.")
        segments = request.segments
        if segments is not None and data.roi_mask is not None:
            segments = np.where(data.roi_mask, segments, -1)
        values = self.normalize_by_segment(data.masked(total).fill(), segments)
        colormap = request.colormap or self.DEFAULT_COLORMAPS[VisualizationMode.BRIGHTNESS]
        finite = values[np.isfinite(values)]
        if segments is None:
            scope = "whole image"
        else:
            count = np.unique(segments[segments >= 0]).size
            scope = f"per cluster, {count} cluster{'s' if count != 1 else ''}"
        ends = {"first": 0, "last": data.bands - 1}
        return VisualizationResult(
            mode=VisualizationMode.BRIGHTNESS,
            display_rgb=self._apply_colormap(values, colormap),
            values=values,
            title=f"Brightness (sum of {data.bands} bands, {scope})",
            value_range=(
                (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)
            ),
            band_indices=MappingProxyType(ends),
            band_wavelengths_nm=MappingProxyType(self._wavelength_map(data, ends)),
            colormap=colormap,
            display_limits=(0.0, 1.0),
        )

    @staticmethod
    def _apply_colormap(normalized: np.ndarray, colormap: str) -> np.ndarray:
        try:
            cmap = colormaps[colormap]
        except KeyError as exc:
            raise VisualizationError(f"Unknown Matplotlib colormap: {colormap}") from exc
        # NaN (no-data) pixels must be finite before the uint8 cast.
        return np.round(
            cmap(np.nan_to_num(normalized, nan=0.0))[..., :3] * 255
        ).astype(np.uint8)

    def _index_band_indices(self, data: HSIData, mode: VisualizationMode) -> dict[str, int]:
        """Share exact target/tolerance rules between rendering and UI gating."""
        return {
            name: data.nearest_band(wavelength, tolerance_nm=self.INDEX_TOLERANCE_NM)
            for name, wavelength in self._index_targets(mode).items()
        }

    @staticmethod
    def _index_targets(mode: VisualizationMode) -> Mapping[str, float]:
        targets: dict[VisualizationMode, Mapping[str, float]] = {
            VisualizationMode.NDVI: {"nir": 800, "red": 670},
            VisualizationMode.EVI: {"nir": 800, "red": 670, "blue": 470},
            VisualizationMode.MCARI: {"red_edge": 700, "red": 670, "green": 550},
            VisualizationMode.MTVI: {"nir": 800, "red": 670, "green": 550},
            VisualizationMode.OSAVI: {"nir": 800, "red": 670},
            VisualizationMode.PRI: {"r531": 531, "r570": 570},
            # Water indices adapted to start at the SWIR camera's 935 nm limit:
            # the NIR reference sits on the ~1070 nm plateau instead of the
            # usual 860/820 nm.
            VisualizationMode.NDWI: {"nir": 1070, "swir": 1240},
            VisualizationMode.NDMI: {"nir": 1070, "swir": 1650},
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
        if mode in (VisualizationMode.NDWI, VisualizationMode.NDMI):
            return self._safe_divide(
                bands["nir"] - bands["swir"], bands["nir"] + bands["swir"]
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
    ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float] | None]:
        """Return normalized values, the raw finite range, and the display limits."""
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.zeros(values.shape, dtype=np.float32), (0.0, 0.0), None
        low, high = np.percentile(
            finite, (stretch.lower_percentile, stretch.upper_percentile)
        )
        if high <= low:
            high = low + np.finfo(np.float32).eps
        normalized = np.clip((values - low) / (high - low), 0, 1).astype(np.float32)
        return (
            normalized,
            (float(finite.min()), float(finite.max())),
            (float(low), float(high)),
        )

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
