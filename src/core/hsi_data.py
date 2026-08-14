"""Shared hyperspectral dataset state for the application's Model layer.

The class retains the fields already consumed by the team Controller while
also exposing the lazy, validated read interface used by feature Models. This
keeps Model integration backward compatible with the current UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import numpy.typing as npt
from spectral.io.spyfile import SubImage

from .errors import VisualizationError, WavelengthError
from .hsi_utils import nearest_band_index


class _CroppedSpyFile(SubImage):
    """SPy sub-image with exact exclusive-end bounds for band reads.

    SPy 0.25's ``SubImage.read_band`` and ``read_bands`` subtract one from
    their exclusive end bounds, producing an image one row and column too
    small. These overrides retain lazy parent-file reads while keeping the
    advertised crop shape consistent with every Model operation.
    """

    def read_band(self, band: int) -> np.ndarray:
        values = self.parent.read_subregion(
            (self.row_offset, self.row_offset + self.nrows),
            (self.col_offset, self.col_offset + self.ncols),
            [band],
        )
        return np.asarray(values)[:, :, 0]

    def read_bands(self, bands: Sequence[int]) -> np.ndarray:
        return np.asarray(
            self.parent.read_subregion(
                (self.row_offset, self.row_offset + self.nrows),
                (self.col_offset, self.col_offset + self.ncols),
                list(bands),
            )
        )


class ImageFormat(Enum):
    """Header format used to open the current dataset."""

    ENVI = auto()
    PSI = auto()


class Functionality(Enum):
    """Legacy UI feature identifiers retained for Controller compatibility."""

    VISUALIZATION = auto()
    SUPER_RESOLUTION = auto()
    CALIBRATION = auto()
    CLASSIFICATION = auto()


@dataclass
class HSIData:
    """Single shared state object for a loaded hyperspectral capture.

    Controller compatibility fields such as ``spectral_obj``, ``wavelengths``,
    and ``rgb_array`` remain mutable. Feature Models consume the read-only-style
    aliases ``image`` and ``wavelengths_nm`` plus targeted read methods.

    Pixel data stay on disk inside SPy's lazy ``SpyFile`` until a read method
    is called. The Controller creates this object once and updates it in place,
    so panels can safely retain the shared reference. Controllers should
    serialize concurrent reads on one instance.
    """

    image_path: Optional[Path] = None
    header_path: Optional[Path] = None
    image_format: Optional[ImageFormat] = None
    spectral_obj: Optional[object] = None
    wavelengths: list[float] = field(default_factory=list)
    rgb_array: Optional[npt.NDArray[np.uint8]] = None
    mask_array: Optional[npt.NDArray[np.uint8]] = None
    selected_path: Optional[Path] = None
    metadata_map: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def create(
        cls,
        *,
        source_path: Path,
        header_path: Path,
        data_path: Path,
        image: Any,
        wavelengths_nm: np.ndarray,
        metadata: Mapping[str, Any],
        header_format: str,
    ) -> "HSIData":
        """Create fully populated state from :class:`core.HSIReader`.

        Controllers normally call ``HSIReader.open`` rather than this factory.
        """

        try:
            image_format = ImageFormat[header_format.upper()]
        except KeyError as exc:
            raise ValueError(f"Unknown header format: {header_format}") from exc
        return cls(
            image_path=Path(data_path),
            header_path=Path(header_path),
            image_format=image_format,
            spectral_obj=image,
            wavelengths=[float(value) for value in wavelengths_nm],
            selected_path=Path(source_path),
            metadata_map=MappingProxyType(dict(metadata)),
        )

    def is_loaded(self) -> bool:
        """Return whether a SPy image is available for Model operations."""

        return self.spectral_obj is not None

    def clear(self) -> None:
        """Reset all dataset and derived View state."""

        self.image_path = self.header_path = self.image_format = None
        self.spectral_obj = self.rgb_array = self.mask_array = None
        self.selected_path = None
        self.wavelengths = []
        self.metadata_map = {}

    def update_from(self, other: "HSIData") -> None:
        """Replace dataset contents without replacing this state object.

        Existing feature panels receive one ``HSIData`` instance during UI
        construction. Controllers should open into a temporary instance and
        call this method only after loading and initial rendering succeed.
        That preserves both panel references and the previously loaded cube
        when an import fails.
        """

        self.image_path = other.image_path
        self.header_path = other.header_path
        self.image_format = other.image_format
        self.spectral_obj = other.spectral_obj
        self.wavelengths = list(other.wavelengths)
        self.rgb_array = other.rgb_array
        self.mask_array = other.mask_array
        self.selected_path = other.selected_path
        self.metadata_map = MappingProxyType(dict(other.metadata_map))

    @property
    def source_path(self) -> Path:
        """Return the original user-selected path."""

        path = self.selected_path or self.image_path
        if path is None:
            raise VisualizationError("No hyperspectral dataset is loaded.")
        return path

    @property
    def data_path(self) -> Path:
        """Return the binary hyperspectral data path."""

        if self.image_path is None:
            raise VisualizationError("No hyperspectral data file is loaded.")
        return self.image_path

    @property
    def image(self) -> Any:
        """Return the lazy SPy ``SpyFile`` used by feature Models."""

        if self.spectral_obj is None:
            raise VisualizationError("No hyperspectral dataset is loaded.")
        return self.spectral_obj

    @property
    def wavelengths_nm(self) -> np.ndarray:
        """Return an immutable wavelength array in nanometres."""

        values = np.asarray(self.wavelengths, dtype=np.float64).copy()
        values.setflags(write=False)
        return values

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return a read-only metadata snapshot."""

        if self.metadata_map:
            return MappingProxyType(dict(self.metadata_map))
        raw = getattr(self.image, "metadata", {})
        return MappingProxyType(dict(raw))

    @property
    def header_format(self) -> str:
        """Return ``ENVI`` or ``PSI`` for serialization/status UI."""

        return self.image_format.name if self.image_format is not None else ""

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return ``(rows, columns, bands)`` without loading pixels."""

        return tuple(int(value) for value in self.image.shape)

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def columns(self) -> int:
        return self.shape[1]

    @property
    def bands(self) -> int:
        return self.shape[2]

    @property
    def storage_bytes(self) -> int:
        """Return the on-disk binary payload size."""

        return self.data_path.stat().st_size

    @property
    def estimated_float_bytes(self) -> int:
        """Estimate RAM needed to materialize the full cube as ``float32``."""

        return self.rows * self.columns * self.bands * np.dtype(np.float32).itemsize

    def nearest_band(
        self, wavelength_nm: float, *, tolerance_nm: float | None = None
    ) -> int:
        """Return the nearest zero-based band index for a wavelength."""

        wavelengths = self.wavelengths_nm
        if wavelengths.size != self.bands:
            raise WavelengthError(
                f"Cube has {self.bands} bands but {wavelengths.size} wavelengths."
            )
        index = nearest_band_index(
            wavelengths,
            wavelength_nm,
            tolerance_nm=tolerance_nm,
        )
        if index is None:
            nearest = nearest_band_index(wavelengths, wavelength_nm)
            assert nearest is not None
            raise WavelengthError(
                f"No band lies within {tolerance_nm:g} nm of {wavelength_nm:g} nm; "
                f"nearest is {wavelengths[nearest]:g} nm."
            )
        return index

    def read_band(self, band_index: int) -> np.ndarray:
        """Read one band as ``float32`` shaped ``(rows, columns)``."""

        if not 0 <= band_index < self.bands:
            raise VisualizationError(
                f"Band {band_index} is outside the valid range 0..{self.bands - 1}."
            )
        return np.asarray(self.image.read_band(band_index), dtype=np.float32)

    def read_bands(self, band_indices: Sequence[int]) -> np.ndarray:
        """Read bands as ``float32`` shaped ``(rows, columns, band_count)``."""

        indices = [int(index) for index in band_indices]
        if not indices:
            raise VisualizationError("At least one band must be requested.")
        if min(indices) < 0 or max(indices) >= self.bands:
            raise VisualizationError("One or more requested band indices are invalid.")
        return np.asarray(self.image.read_bands(indices), dtype=np.float32)

    def read_pixel(self, row: int, column: int) -> np.ndarray:
        """Read one spectrum as ``float32`` shaped ``(bands,)``."""

        if not 0 <= row < self.rows or not 0 <= column < self.columns:
            raise VisualizationError(
                f"Pixel ({row}, {column}) is outside a "
                f"{self.rows}x{self.columns} image."
            )
        return np.asarray(self.image.read_pixel(row, column), dtype=np.float32)

    def crop(
        self, left: float, top: float, right: float, bottom: float
    ) -> tuple[int, int] | None:
        """Crop display arrays and retain a lazy SPy-compatible cube view."""

        if self.rgb_array is None or self.mask_array is None:
            return None

        height, width = self.rgb_array.shape[:2]
        x1 = max(0, int(np.floor(min(left, right))))
        y1 = max(0, int(np.floor(min(top, bottom))))
        x2 = min(width, int(np.ceil(max(left, right))))
        y2 = min(height, int(np.ceil(max(top, bottom))))
        if x2 - x1 < 1 or y2 - y1 < 1:
            return None

        self.rgb_array = self.rgb_array[y1:y2, x1:x2, :].copy()
        self.mask_array = self.mask_array[y1:y2, x1:x2].copy()
        if self.spectral_obj is not None:
            self.spectral_obj = _CroppedSpyFile(
                self.spectral_obj,
                (y1, y2),
                (x1, x2),
            )
        return (x2 - x1, y2 - y1)
