"""UI-independent dark/bright radiometric calibration.

The service retains the sensor-column response of pushbroom calibration
captures.  It averages the central 21 lines of each frame, logically expands
the resulting detector row to the source's pre-crop line count, and applies the
source's current crop before calibration.  Output is streamed to a temporary
ENVI cube so production captures do not need to fit in memory and source files
are never modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from numbers import Integral, Real
from pathlib import Path
import re
import shutil
import tempfile
from typing import Callable

import numpy as np
from spectral.io import envi

from .errors import CalibrationError, CancelledError
from .hsi_data import HSIData
from .hsi_reader import DATA_EXTENSIONS, HSIReader


ProgressCallback = Callable[[int, str], None]
CancellationCheck = Callable[[], bool]
REFERENCE_ROW_COUNT = 21
MAX_CALIBRATION_PAIR_INTERVAL = timedelta(minutes=1)
_CAPTURE_TIMESTAMP = re.compile(
    r"^(?P<captured>\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})",
    re.IGNORECASE,
)
_CALIBRATION_FRAME_NAME = re.compile(
    r"^(?P<captured>\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})"
    r".*_calibFrame$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CalibrationFramePair:
    """Automatically resolved earlier-dark/later-bright frame pair."""

    dark_path: Path
    bright_path: Path
    dark_captured_at: datetime
    bright_captured_at: datetime
    source_captured_at: datetime

    @property
    def frame_interval_seconds(self) -> float:
        return (self.bright_captured_at - self.dark_captured_at).total_seconds()

    @property
    def seconds_before_source(self) -> float:
        return (self.source_captured_at - self.bright_captured_at).total_seconds()


class CalibrationFrameResolver:
    """Find the closest valid calibration pair preceding a source capture.

    Only same-folder ``.hdr`` files ending in ``_calibFrame`` are considered.
    A candidate must have a supported same-stem data file, precede the source,
    and form a pair captured no more than one minute apart. Of all valid pairs,
    the one with the latest bright-frame time is selected.
    """

    def resolve(self, source_path: str | Path) -> CalibrationFramePair | None:
        source = Path(source_path).expanduser().resolve()
        source_time = self._capture_time(source, calibration_frame=False)
        if source_time is None or not source.parent.is_dir():
            return None

        directory_entries = [
            path for path in source.parent.iterdir() if path.is_file()
        ]
        available_names = {path.name.casefold() for path in directory_entries}
        candidates: list[tuple[datetime, Path]] = []
        for path in directory_entries:
            if path.suffix.casefold() != ".hdr":
                continue
            captured_at = self._capture_time(path, calibration_frame=True)
            if captured_at is None or captured_at >= source_time:
                continue
            if not any(
                f"{path.stem}{extension}".casefold() in available_names
                for extension in DATA_EXTENSIONS
            ):
                continue
            candidates.append((captured_at, path.resolve()))

        candidates.sort(key=lambda item: item[0])
        valid_pairs: list[
            tuple[tuple[datetime, Path], tuple[datetime, Path]]
        ] = []
        for dark_index, dark in enumerate(candidates):
            for bright in candidates[dark_index + 1 :]:
                if bright[0] - dark[0] > MAX_CALIBRATION_PAIR_INTERVAL:
                    break
                valid_pairs.append((dark, bright))
        if not valid_pairs:
            return None

        dark, bright = max(
            valid_pairs,
            key=lambda pair: (pair[1][0], pair[0][0]),
        )
        return CalibrationFramePair(
            dark_path=dark[1],
            bright_path=bright[1],
            dark_captured_at=dark[0],
            bright_captured_at=bright[0],
            source_captured_at=source_time,
        )

    @staticmethod
    def _capture_time(
        path: Path, *, calibration_frame: bool
    ) -> datetime | None:
        pattern = _CALIBRATION_FRAME_NAME if calibration_frame else _CAPTURE_TIMESTAMP
        match = pattern.match(path.stem)
        if match is None:
            return None
        try:
            return datetime.strptime(
                match.group("captured"), "%Y-%m-%d--%H-%M-%S"
            )
        except ValueError:
            return None


def order_calibration_frame_paths(
    first_path: Path, second_path: Path
) -> tuple[Path, Path]:
    """Return ``(dark, bright)`` from timestamped calibration-frame names.

    Facility captures start with ``YYYY-MM-DD--HH-MM-SS`` and end with
    ``_calibFrame``. When both names follow that convention, the earlier frame
    is dark and the later frame is bright. Nonconforming names retain their
    caller-supplied order so hand-authored and legacy datasets remain usable.
    """

    paths = (Path(first_path), Path(second_path))
    matches = tuple(_CALIBRATION_FRAME_NAME.match(path.stem) for path in paths)
    if not all(matches):
        return paths
    captured = tuple(
        datetime.strptime(match.group("captured"), "%Y-%m-%d--%H-%M-%S")
        for match in matches
        if match is not None
    )
    if captured[0] == captured[1]:
        raise CalibrationError(
            "The selected calibration frames have the same capture time. Select "
            "the two distinct _calibFrame captures; the earlier must be dark and "
            "the later bright."
        )
    if captured[0] < captured[1]:
        return paths
    return paths[1], paths[0]


@dataclass(frozen=True, slots=True)
class CalibrationRequest:
    """Settings for dark/bright reference correction.

    ``line_chunk_size`` bounds source reads. Wavelengths are
    compared band-for-band because calibration must not silently resample
    reference spectra. ``minimum_response`` rejects detector positions where
    the averaged bright reference is not meaningfully brighter than dark.
    """

    line_chunk_size: int = 64
    wavelength_tolerance_nm: float = 0.5
    minimum_response: float = 1e-6

    def __post_init__(self) -> None:
        if isinstance(self.line_chunk_size, bool) or not isinstance(
            self.line_chunk_size, Integral
        ):
            raise CalibrationError("Calibration line chunk size must be an integer.")
        if self.line_chunk_size < 1:
            raise CalibrationError("Calibration line chunk size must be positive.")
        object.__setattr__(self, "line_chunk_size", int(self.line_chunk_size))
        if not isinstance(self.wavelength_tolerance_nm, Real) or not np.isfinite(
            self.wavelength_tolerance_nm
        ):
            raise CalibrationError("Wavelength tolerance must be finite.")
        if self.wavelength_tolerance_nm < 0:
            raise CalibrationError("Wavelength tolerance cannot be negative.")
        if not isinstance(self.minimum_response, Real) or not np.isfinite(
            self.minimum_response
        ):
            raise CalibrationError("Minimum bright/dark response must be finite.")
        if self.minimum_response <= 0:
            raise CalibrationError("Minimum bright/dark response must be positive.")


@dataclass
class CalibrationResult:
    """Calibrated reflectance cube and provenance for its reference captures.

    Keep this result alive while using ``data`` because it owns the temporary
    ENVI files backing the lazily readable calibrated cube.
    """

    data: HSIData
    input_shape: tuple[int, int, int]
    dark_reference_path: Path
    white_reference_path: Path
    reference_line_counts: tuple[int, int]
    reference_row_ranges: tuple[tuple[int, int], tuple[int, int]]
    original_input_shape: tuple[int, int, int]
    source_spatial_bounds: tuple[tuple[int, int], tuple[int, int]]
    reflectance_range: tuple[float, float]
    _storage: tempfile.TemporaryDirectory = field(repr=False, default=None)

    def cleanup(self) -> None:
        """Close the calibrated cube and remove its temporary ENVI files."""

        self.data.close()
        storage = self._storage
        self._storage = None
        if storage is not None:
            storage.cleanup()

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            # Interpreter shutdown and third-party mappings can make best-effort
            # cleanup fail; explicit Controller cleanup remains the primary path.
            pass


class CalibrationService:
    """Apply dark-current and bright-reference correction to an HSI cube."""

    def validate(
        self,
        source: HSIData,
        dark: HSIData,
        white: HSIData,
        request: CalibrationRequest = CalibrationRequest(),
        *,
        reference_source: HSIData | None = None,
    ) -> None:
        for label, data in (
            ("source", source),
            ("dark reference", dark),
            ("bright reference", white),
        ):
            if not data.is_loaded():
                raise CalibrationError(f"The {label} cube is not loaded.")
            if len(data.shape) != 3 or min(data.shape) < 1:
                raise CalibrationError(
                    f"The {label} must be a nonempty (lines, columns, bands) cube."
                )

        if reference_source is not None:
            if not reference_source.is_loaded() or source.shape != (
                reference_source.rows * 2,
                reference_source.columns * 2,
                reference_source.bands,
            ):
                raise CalibrationError(
                    "The high-resolution source must be 2× the current original image."
                )
        reference_geometry = reference_source or source
        original_rows, original_columns, original_bands = reference_geometry.original_shape
        if (
            original_rows < reference_geometry.rows
            or original_bands != source.bands
            or source.original_shape[0] < source.rows
        ):
            raise CalibrationError(
                "The source crop provenance is inconsistent with its current shape."
            )
        expected = (original_columns, source.bands)
        for label, data in (("dark", dark), ("bright", white)):
            if data.rows < REFERENCE_ROW_COUNT:
                raise CalibrationError(
                    f"The {label} reference has {data.rows} rows; at least "
                    f"{REFERENCE_ROW_COUNT} rows are required to average the middle "
                    f"{REFERENCE_ROW_COUNT}."
                )
            actual = (data.columns, data.bands)
            if actual != expected:
                raise CalibrationError(
                    f"The {label} reference has {actual[0]} columns and {actual[1]} "
                    f"bands; the original pre-crop source requires {expected[0]} "
                    f"columns and {expected[1]} bands. Reference row counts may "
                    "differ, but full-width columns and bands must match."
                )

        source_wavelengths = source.wavelengths_nm
        for label, data in (("dark", dark), ("bright", white)):
            wavelengths = data.wavelengths_nm
            if wavelengths.size != source_wavelengths.size:
                raise CalibrationError(
                    f"The {label} reference wavelength count does not match the source."
                )
            differences = np.abs(wavelengths - source_wavelengths)
            if not np.isfinite(differences).all():
                raise CalibrationError(
                    f"The {label} reference contains invalid wavelength metadata."
                )
            maximum = float(np.max(differences, initial=0.0))
            if maximum > request.wavelength_tolerance_nm:
                index = int(np.argmax(differences))
                raise CalibrationError(
                    f"The {label} reference wavelength at band {index} differs from "
                    f"the source by {maximum:.3g} nm; allowed tolerance is "
                    f"{request.wavelength_tolerance_nm:g} nm."
                )

    def calibrate(
        self,
        source: HSIData,
        dark: HSIData,
        white: HSIData,
        request: CalibrationRequest = CalibrationRequest(),
        *,
        progress: ProgressCallback | None = None,
        is_cancelled: CancellationCheck | None = None,
        reference_source: HSIData | None = None,
    ) -> CalibrationResult:
        """Return a lazy float32 reflectance cube computed without overwrites.

        Dark and bright cubes may contain different numbers of lines from the
        source and from each other, but each needs at least 21. They must
        describe the pre-crop detector columns and bands. The central 21 rows
        are averaged, expanded as a zero-copy view to the source's original
        row count, and then sliced by the source's current crop. For a 2× SR
        source, the cropped original detector profiles are interpolated onto
        high-resolution pixel centres before broadcasting. The standard
        correction ``(source - dark) / (bright - dark)`` is then applied.
        Values are intentionally not clipped to ``[0, 1]`` so analytical
        overshoot remains visible to callers.
        """

        self.validate(source, dark, white, request, reference_source=reference_source)
        self._check_cancelled(is_cancelled)
        self._emit(progress, 0, "Averaging middle 21 dark-reference rows")
        dark_mean, dark_row_range = self._mean_reference(
            dark,
            request.line_chunk_size,
            is_cancelled=is_cancelled,
        )
        self._emit(progress, 15, "Averaging middle 21 bright-reference rows")
        white_mean, white_row_range = self._mean_reference(
            white,
            request.line_chunk_size,
            is_cancelled=is_cancelled,
        )

        original_shape = source.original_shape
        row_bounds, column_bounds = source.spatial_bounds
        if reference_source is None:
            dark_aligned = self._expand_and_crop_reference(
                dark_mean, original_shape, row_bounds, column_bounds
            )
            white_aligned = self._expand_and_crop_reference(
                white_mean, original_shape, row_bounds, column_bounds
            )
            response = white_aligned[0] - dark_aligned[0]
            response_column_offset = column_bounds[0]
        else:
            low_left, low_right = reference_source.spatial_bounds[1]
            dark_row = self._interpolate_reference(
                dark_mean[low_left:low_right], source.columns
            )
            white_row = self._interpolate_reference(
                white_mean[low_left:low_right], source.columns
            )
            dark_aligned = np.broadcast_to(dark_row[None], source.shape)
            white_aligned = np.broadcast_to(white_row[None], source.shape)
            response = white_row - dark_row
            response_column_offset = 0
        invalid_response = (~np.isfinite(response)) | (
            response <= request.minimum_response
        )
        if np.any(invalid_response):
            count = int(np.count_nonzero(invalid_response))
            cropped_column, band = (
                int(value) for value in np.argwhere(invalid_response)[0]
            )
            column = response_column_offset + cropped_column
            raise CalibrationError(
                f"Bright reference is not brighter than dark at {count} "
                f"column/band position(s) (first: column {column}, band {band}). "
                "Check the selected frames and exposure.\n"
                "Try cropping before calibration."
            )

        required = source.estimated_float_bytes
        if shutil.disk_usage(tempfile.gettempdir()).free < required + 1024 * 1024:
            raise CalibrationError(
                "Insufficient temporary disk space: calibrated output needs "
                f"{required / 2**30:.2f} GiB."
            )

        storage = tempfile.TemporaryDirectory(
            prefix="hsi-calibration-", ignore_cleanup_errors=True
        )
        output = None
        try:
            metadata = {
                "lines": source.rows,
                "samples": source.columns,
                "bands": source.bands,
                "data type": 4,
                "interleave": "bip",
                "wavelength units": "nm",
                "wavelength": source.wavelengths,
                "description": (
                    f"Dark/bright calibrated reflectance from {source.source_path.name}"
                ),
            }
            for name in ("fwhm", "bbl", "band names"):
                if name in source.metadata:
                    metadata[name] = source.metadata[name]

            header = Path(storage.name) / "calibrated.hdr"
            image = envi.create_image(str(header), metadata=metadata, ext=".bip")
            output = image.open_memmap(writable=True, interleave="bip")
            minimum = np.inf
            maximum = -np.inf
            bands = list(range(source.bands))
            chunk_size = int(request.line_chunk_size)
            chunk_count = (source.rows + chunk_size - 1) // chunk_size

            for chunk_index, top in enumerate(range(0, source.rows, chunk_size), 1):
                self._check_cancelled(is_cancelled)
                bottom = min(source.rows, top + chunk_size)
                raw = np.asarray(
                    source.image.read_subregion(
                        (top, bottom), (0, source.columns), bands
                    ),
                    dtype=np.float32,
                )
                expected_shape = (bottom - top, source.columns, source.bands)
                if raw.shape != expected_shape:
                    raise CalibrationError(
                        f"Source read returned shape {raw.shape}; expected "
                        f"{expected_shape} for lines {top}:{bottom}."
                    )
                if not np.isfinite(raw).all():
                    raise CalibrationError(
                        f"Source contains NaN or infinite values in lines {top}:{bottom}."
                    )
                # The aligned references are broadcast views: this preserves the
                # specified expand-then-crop operation without allocating two
                # capture-sized calibration cubes (about 1.8 GiB each for the
                # facility's 1971 x 500 x 480 sample).
                dark_chunk = dark_aligned[top:bottom, :, :]
                response_chunk = (
                    white_aligned[top:bottom, :, :] - dark_chunk
                )
                calibrated = (raw - dark_chunk) / response_chunk
                if not np.isfinite(calibrated).all():
                    raise CalibrationError(
                        f"Calibration produced invalid values in lines {top}:{bottom}."
                    )
                output[top:bottom, :, :] = calibrated.astype(np.float32, copy=False)
                minimum = min(minimum, float(np.min(calibrated)))
                maximum = max(maximum, float(np.max(calibrated)))
                percent = 30 + round(65 * chunk_index / chunk_count)
                self._emit(
                    progress,
                    percent,
                    f"Calibrating source lines {top + 1}-{bottom} of {source.rows}",
                )

            output.flush()
            output = None
            self._check_cancelled(is_cancelled)
            calibrated_data = HSIReader().open(header)
            if source.roi_mask is not None:
                calibrated_data.roi_mask = source.roi_mask.copy()
            self._emit(progress, 100, "Radiometric calibration complete")
            return CalibrationResult(
                data=calibrated_data,
                input_shape=source.shape,
                dark_reference_path=dark.source_path,
                white_reference_path=white.source_path,
                reference_line_counts=(dark.rows, white.rows),
                reference_row_ranges=(dark_row_range, white_row_range),
                original_input_shape=original_shape,
                source_spatial_bounds=(row_bounds, column_bounds),
                reflectance_range=(minimum, maximum),
                _storage=storage,
            )
        except Exception as exc:
            output = None
            storage.cleanup()
            if isinstance(exc, (CalibrationError, CancelledError)):
                raise
            raise CalibrationError(f"Calibration failed: {exc}") from exc

    def _mean_reference(
        self,
        data: HSIData,
        chunk_size: int,
        *,
        is_cancelled: CancellationCheck | None,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        start, end = self._middle_row_bounds(data.rows)
        total = np.zeros((data.columns, data.bands), dtype=np.float64)
        bands = list(range(data.bands))
        for top in range(start, end, int(chunk_size)):
            self._check_cancelled(is_cancelled)
            bottom = min(end, top + int(chunk_size))
            values = np.asarray(
                data.image.read_subregion(
                    (top, bottom), (0, data.columns), bands
                ),
                dtype=np.float64,
            )
            expected_shape = (bottom - top, data.columns, data.bands)
            if values.shape != expected_shape:
                raise CalibrationError(
                    f"Reference read returned shape {values.shape}; expected "
                    f"{expected_shape}."
                )
            if not np.isfinite(values).all():
                raise CalibrationError(
                    f"Reference cube {data.source_path.name} contains NaN or "
                    f"infinite values in lines {top}:{bottom}."
                )
            total += values.sum(axis=0, dtype=np.float64)
        return total / REFERENCE_ROW_COUNT, (start, end)

    @staticmethod
    def _middle_row_bounds(row_count: int) -> tuple[int, int]:
        """Return the central 21-row half-open range.

        For an even number of rows the sensor midpoint is the first row of the
        lower half. The 21-row window is centred on that row; a 50-row frame
        therefore uses rows 15 through 35 (``15:36`` in Python notation).
        """

        middle = int(row_count) // 2
        start = middle - REFERENCE_ROW_COUNT // 2
        return start, start + REFERENCE_ROW_COUNT

    @staticmethod
    def _expand_and_crop_reference(
        detector_row: np.ndarray,
        original_shape: tuple[int, int, int],
        row_bounds: tuple[int, int],
        column_bounds: tuple[int, int],
    ) -> np.ndarray:
        """Logically span one detector row, then apply the current crop."""

        expanded = np.broadcast_to(detector_row[None, :, :], original_shape)
        aligned = expanded[
            row_bounds[0] : row_bounds[1],
            column_bounds[0] : column_bounds[1],
            :,
        ]
        expected = (
            row_bounds[1] - row_bounds[0],
            column_bounds[1] - column_bounds[0],
            original_shape[2],
        )
        if aligned.shape != expected:
            raise CalibrationError(
                f"Expanded calibration reference produced shape {aligned.shape}; "
                f"expected the cropped source shape {expected}."
            )
        return aligned

    @staticmethod
    def _interpolate_reference(detector_row: np.ndarray, columns: int) -> np.ndarray:
        """Interpolate a cropped detector profile onto 2× pixel centres."""

        low_columns = detector_row.shape[0]
        positions = (np.arange(columns, dtype=np.float64) + 0.5) / 2 - 0.5
        left = np.floor(positions).astype(np.intp)
        right = np.clip(left + 1, 0, low_columns - 1)
        weight = np.clip(positions - left, 0, 1)[:, None]
        left = np.clip(left, 0, low_columns - 1)
        return detector_row[left] * (1 - weight) + detector_row[right] * weight

    @staticmethod
    def _check_cancelled(check: CancellationCheck | None) -> None:
        if check is not None and check():
            raise CancelledError("Calibration cancelled.")

    @staticmethod
    def _emit(progress: ProgressCallback | None, value: int, message: str) -> None:
        if progress is not None:
            progress(value, message)
