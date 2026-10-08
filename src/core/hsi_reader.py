"""Controller-facing import service for ENVI, PSI, and JSON cube metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from spectral.io import envi

from .errors import HSIFileError, HSIHeaderError
from .hsi_data import HSIData
from .hsi_utils import adapt_json_header, adapt_psi_header


DATA_EXTENSIONS = (".bil", ".bip", ".bsq", ".dat", ".img", ".raw")
METADATA_EXTENSIONS = (".hdr", ".json")
HSI_FILE_FILTER = "Hyperspectral Images (" + " ".join(
    f"*{extension}" for extension in (*METADATA_EXTENSIONS, *DATA_EXTENSIONS)
) + ");;All Files (*)"


class HSIReader:
    """Open ENVI/PSI headers or JSON metadata paired with spectral data.

    The service has no UI state. A Controller may reuse one instance, but it
    should retain only the returned :class:`HSIData` as application state.
    """

    def open(self, path: str | Path) -> HSIData:
        """Validate a selected header or data file and return lazy cube state.

        ``path`` may identify either side of a supported pair. PSI headers and
        JSON metadata are adapted to temporary ENVI headers; source files are
        never modified. JSON requires a separate binary cube. This call is
        synchronous, so use a worker for slow storage.

        A Controller should replace its current dataset only after this method
        succeeds, ensuring that failed imports do not discard a working cube.

        Raises:
            HSIFileError: A selected/paired file is missing or truncated.
            HSIHeaderError: Metadata is malformed, incomplete, or unsupported.
        """
        source_path = Path(path).expanduser().resolve()
        if not source_path.is_file():
            raise HSIFileError(f"Selected file does not exist: {source_path}")
        header_path, data_path = self.resolve_pair(source_path)
        header_format = self._detect_header_format(header_path)
        if header_format == "JSON":
            working_header = adapt_json_header(header_path)
        elif header_format == "PSI":
            working_header = adapt_psi_header(header_path)
        else:
            working_header = header_path
        try:
            image = envi.open(str(working_header), str(data_path))
        except Exception as exc:
            raise HSIHeaderError(f"SPy could not open {data_path.name}: {exc}") from exc
        try:
            wavelengths = self._read_wavelengths(image.metadata, int(image.nbands))
            if header_format == "JSON":
                self._validate_json_companion(image, data_path, wavelengths)
            self._validate_data_size(image, data_path, exact=header_format == "JSON")
        except (HSIFileError, HSIHeaderError):
            HSIData(spectral_obj=image).close()
            raise
        return HSIData.create(
            source_path=source_path,
            header_path=header_path,
            data_path=data_path,
            image=image,
            wavelengths_nm=wavelengths,
            metadata=image.metadata,
            header_format=header_format,
        )

    def _validate_json_companion(
        self, image: Any, data_path: Path, wavelengths: np.ndarray
    ) -> None:
        """Do not trust a same-stem JSON sidecar that contradicts the cube header.

        Specim JSON can describe the original acquisition, while the paired
        binary has subsequently been converted (e.g. uint16 to float32).
        Filename matching alone cannot establish metadata compatibility.
        """
        header = next((path for path in data_path.parent.iterdir()
                       if path.is_file() and path.name.casefold()
                       == f"{data_path.stem}.hdr".casefold()), None)
        if header is None:
            return
        try:
            reference_metadata = envi.read_envi_header(str(header))
        except Exception as exc:
            raise HSIHeaderError(f"Could not read companion header {header.name}: {exc}") from exc
        reference_wavelengths = self._read_wavelengths(reference_metadata, int(image.nbands))
        mismatches = []
        reference_shape = tuple(
            int(reference_metadata.get(field, -1)) for field in ("lines", "samples", "bands")
        )
        if tuple(image.shape) != reference_shape:
            mismatches.append("dimensions")
        if int(image.metadata.get("data type", -1)) != int(reference_metadata.get("data type", -2)):
            mismatches.append("data type")
        if int(image.offset) != int(reference_metadata.get("header offset", 0)):
            mismatches.append("header offset")
        for field in ("interleave", "byte order"):
            if str(image.metadata.get(field, "0")).strip().casefold() != str(
                reference_metadata.get(field, "0")
            ).strip().casefold():
                mismatches.append(field)
        if (wavelengths.shape != reference_wavelengths.shape or not np.allclose(
            wavelengths, reference_wavelengths, rtol=0, atol=1e-6
        )):
            mismatches.append("wavelengths")
        if mismatches:
            raise HSIHeaderError(
                f"JSON metadata conflicts with {header.name}: {', '.join(mismatches)}. "
                f"Select {header.name} to load this cube, or provide verified "
                "metadata for the selected data file."
            )

    @staticmethod
    def resolve_pair(path: str | Path) -> tuple[Path, Path]:
        """Find same-stem metadata/data without reading pixels, case-insensitively.

        An explicitly selected metadata file is retained. When selecting the
        binary cube, prefer .hdr over .json if both exist: camera JSON sidecars
        can contain acquisition metadata that differs from the final header.
        """
        source_path = Path(path).expanduser().resolve()
        if not source_path.is_file():
            raise HSIFileError(f"Selected file does not exist: {source_path}")
        suffix = source_path.suffix.casefold()
        if suffix not in (*METADATA_EXTENSIONS, *DATA_EXTENSIONS):
            raise HSIFileError(
                f"Unsupported hyperspectral file extension: {suffix!r}; "
                f"select metadata ({', '.join(METADATA_EXTENSIONS)}) or a supported cube."
            )
        try:
            files = {entry.name.casefold(): entry for entry in source_path.parent.iterdir()
                     if entry.is_file()}
        except OSError as exc:
            raise HSIFileError(f"Cannot scan the capture folder: {source_path.parent}") from exc
        if suffix in METADATA_EXTENSIONS:
            for extension in DATA_EXTENSIONS:
                candidate = files.get(f"{source_path.stem}{extension}".casefold())
                if candidate is not None:
                    return source_path, candidate
            raise HSIFileError(
                f"No data file was found beside {source_path.name}; expected one of "
                f"{', '.join(DATA_EXTENSIONS)} with the same filename stem. "
                "Metadata does not contain spectral pixels; copy the corresponding "
                "binary cube into this folder."
            )
        for extension in METADATA_EXTENSIONS:
            header_path = files.get(f"{source_path.stem}{extension}".casefold())
            if header_path is not None:
                return header_path, source_path
        raise HSIFileError(
            f"Paired metadata file is missing for {source_path.name}; expected "
            f"{', '.join(METADATA_EXTENSIONS)} with the same filename stem."
        )

    @staticmethod
    def _detect_header_format(header_path: Path) -> str:
        if header_path.suffix.casefold() == ".json":
            return "JSON"
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
        units = str(metadata.get("wavelength units", "nm")).strip().casefold()
        if units in {"um", "µm", "μm", "micron", "microns", "micrometer",
                     "micrometers", "micrometre", "micrometres"}:
            wavelengths *= 1000
        elif units not in {"nm", "nanometer", "nanometers", "nanometre", "nanometres"}:
            raise HSIHeaderError(f"Unsupported wavelength units: {units!r}; use nm or micrometers.")
        if not np.isfinite(wavelengths).all() or np.any(wavelengths <= 0):
            raise HSIHeaderError("Wavelengths must be finite and positive.")
        if np.any(np.diff(wavelengths) <= 0):
            raise HSIHeaderError("Wavelengths must be strictly increasing.")
        return wavelengths

    @staticmethod
    def _validate_data_size(image: Any, data_path: Path, *, exact: bool = False) -> None:
        expected = (int(image.offset) + int(np.prod(image.shape, dtype=np.int64))
                    * np.dtype(image.dtype).itemsize)
        actual = data_path.stat().st_size
        if actual < expected:
            raise HSIFileError(
                f"Data file is truncated: expected at least {expected} bytes, found {actual}."
            )
        if exact and actual != expected:
            raise HSIFileError(
                f"JSON metadata describes {expected} bytes, but {data_path.name} contains "
                f"{actual}. This JSON may describe a different or unconverted capture. "
                "Select the matching .hdr or provide verified JSON metadata."
            )
