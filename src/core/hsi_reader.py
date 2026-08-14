"""Controller-facing import service for ENVI and PSI hyperspectral pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from spectral.io import envi

from .errors import HSIFileError, HSIHeaderError
from .hsi_data import HSIData
from .hsi_utils import adapt_psi_header


DATA_EXTENSIONS = (".bil", ".bip", ".bsq", ".dat", ".img", ".raw")


class HSIReader:
    """Open native ENVI or PSI ``.hdr``/data pairs through Spectral Python.

    The service has no UI state. A Controller may reuse one instance, but it
    should retain only the returned :class:`HSIData` as application state.
    """

    def open(self, path: str | Path) -> HSIData:
        """Validate a selected header or data file and return lazy cube state.

        ``path`` may identify either side of a supported pair. PSI headers are
        adapted to deterministic temporary ENVI headers; source files are never
        modified. This call is synchronous, so use a worker for slow storage.

        A Controller should replace its current dataset only after this method
        succeeds, ensuring that failed imports do not discard a working cube.

        Raises:
            HSIFileError: A selected/paired file is missing or truncated.
            HSIHeaderError: Metadata is malformed, incomplete, or unsupported.
        """
        source_path = Path(path).expanduser().resolve()
        if not source_path.is_file():
            raise HSIFileError(f"Selected file does not exist: {source_path}")
        header_path, data_path = self._resolve_pair(source_path)
        header_format = self._detect_header_format(header_path)
        working_header = (
            header_path if header_format == "ENVI" else adapt_psi_header(header_path)
        )
        try:
            image = envi.open(str(working_header), str(data_path))
        except Exception as exc:
            raise HSIHeaderError(f"SPy could not open {data_path.name}: {exc}") from exc
        self._validate_data_size(image, data_path)
        wavelengths = self._read_wavelengths(image.metadata, int(image.nbands))
        return HSIData.create(
            source_path=source_path,
            header_path=header_path,
            data_path=data_path,
            image=image,
            wavelengths_nm=wavelengths,
            metadata=image.metadata,
            header_format=header_format,
        )

    def _resolve_pair(self, source_path: Path) -> tuple[Path, Path]:
        if source_path.suffix.lower() == ".hdr":
            for extension in DATA_EXTENSIONS:
                candidate = source_path.with_suffix(extension)
                if candidate.is_file():
                    return source_path, candidate
            raise HSIFileError(
                f"No data file was found beside {source_path.name}; expected one of "
                f"{', '.join(DATA_EXTENSIONS)}."
            )
        header_path = source_path.with_suffix(".hdr")
        if not header_path.is_file():
            raise HSIFileError(f"Paired header file is missing: {header_path}")
        return header_path, source_path

    @staticmethod
    def _detect_header_format(header_path: Path) -> str:
        try:
            first_line = header_path.read_text(
                encoding="utf-8", errors="strict"
            ).splitlines()[0].strip()
        except (OSError, UnicodeError, IndexError) as exc:
            raise HSIHeaderError(f"Header is empty or unreadable: {header_path}") from exc
        return "ENVI" if first_line.upper() == "ENVI" else "PSI"

    @staticmethod
    def _read_wavelengths(metadata: Mapping[str, Any], bands: int) -> np.ndarray:
        raw = metadata.get("wavelength")
        if raw is None:
            raise HSIHeaderError("Wavelength metadata is required for visualization models.")
        if isinstance(raw, str):
            raw = raw.strip("{}").split(",")
        try:
            wavelengths = np.asarray(
                [float(str(value).strip()) for value in raw], dtype=np.float64
            )
        except (TypeError, ValueError) as exc:
            raise HSIHeaderError("Wavelength metadata could not be parsed.") from exc
        if wavelengths.size != bands:
            raise HSIHeaderError(
                f"Header contains {wavelengths.size} wavelengths for {bands} bands."
            )
        if np.any(np.diff(wavelengths) <= 0):
            raise HSIHeaderError("Wavelengths must be strictly increasing.")
        return wavelengths

    @staticmethod
    def _validate_data_size(image: Any, data_path: Path) -> None:
        expected = int(np.prod(image.shape, dtype=np.int64)) * np.dtype(image.dtype).itemsize
        actual = data_path.stat().st_size
        if actual < expected:
            raise HSIFileError(
                f"Data file is truncated: expected at least {expected} bytes, found {actual}."
            )
