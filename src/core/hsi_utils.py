"""Shared compatibility utilities for hyperspectral Model and Controller code.

New feature Models use :class:`core.HSIReader` and :class:`core.HSIData`.
Functions retained here support the existing team Controller and centralize
format/wavelength behavior so the two code paths cannot drift apart.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import numpy.typing as npt

from .errors import HSIHeaderError


def nearest_band_index(
    wavelengths: Sequence[float] | npt.NDArray[np.floating],
    target_nm: float,
    *,
    tolerance_nm: float | None = None,
) -> Optional[int]:
    """Return the nearest band index, or ``None`` when unavailable.

    This is the common wavelength lookup primitive. ``HSIData.nearest_band``
    wraps it with domain exceptions, while legacy helpers retain their
    optional-return API.
    """

    values = np.asarray(wavelengths, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        return None
    index = int(np.abs(values - float(target_nm)).argmin())
    if tolerance_nm is not None:
        difference = abs(float(values[index]) - float(target_nm))
        if difference > tolerance_nm:
            return None
    return index


def find_rgb_bands(wavelengths: list[float]) -> Optional[tuple[int, int, int]]:
    """Return legacy ``(red, green, blue)`` indices.

    The historical Controller targets 682.5/532.5/472.5 nm. When the capture
    lacks full visible coverage, evenly spaced bands are returned to preserve
    its prior fallback behavior. Feature Models use their explicit targets and
    tolerances through ``HSIData.nearest_band``.
    """

    if len(wavelengths) < 3:
        return None
    blue, red = 472.5, 682.5
    if wavelengths[0] > blue or wavelengths[-1] < red:
        last = len(wavelengths) - 1
        return (round(5 * last / 6), round(last / 2), round(last / 6))
    red_index = nearest_band_index(wavelengths, 682.5)
    green_index = nearest_band_index(wavelengths, 532.5)
    blue_index = nearest_band_index(wavelengths, 472.5)
    if red_index is None or green_index is None or blue_index is None:
        return None
    return red_index, green_index, blue_index


def find_red_nir_bands(wavelengths: list[float]) -> Optional[tuple[int, int]]:
    """Return legacy ``(red, NIR)`` indices, or ``None`` without coverage."""

    if len(wavelengths) < 2 or wavelengths[0] > 682.5 or wavelengths[-1] < 850.0:
        return None
    red = nearest_band_index(wavelengths, 682.5)
    nir = nearest_band_index(wavelengths, 850.0)
    return (red, nir) if red is not None and nir is not None else None


def read_psi_header(file_path: Path) -> dict[str, Any]:
    """Parse a PSI header using whitespace-tolerant, validated syntax.

    Keys are normalized to uppercase and ``WAVELENGTHS`` maps to
    ``list[float]``. :class:`HSIHeaderError` is raised for unreadable or invalid
    wavelength content instead of leaking low-level parsing exceptions.
    """

    values: dict[str, Any] = {}
    wavelengths: list[float] = []
    reading_wavelengths = False
    try:
        lines = Path(file_path).read_text(encoding="utf-8", errors="strict").splitlines()
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            marker = line.upper()
            if marker == "WAVELENGTHS":
                reading_wavelengths = True
                continue
            if marker == "WAVELENGTHS_END":
                reading_wavelengths = False
                continue
            if reading_wavelengths:
                wavelengths.append(float(line.split()[0]))
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                values[parts[0].upper()] = parts[1].strip()
    except (OSError, UnicodeError, ValueError) as exc:
        raise HSIHeaderError(f"Malformed PSI header {Path(file_path).name}: {exc}") from exc
    values["WAVELENGTHS"] = wavelengths
    return values


def _envi_data_type(bits: int, signed_value: Any = None) -> int:
    signed = str(signed_value or "0").lower() in {"1", "true", "yes", "signed"}
    if bits <= 8:
        return 1
    if bits <= 16:
        return 2 if signed else 12
    if bits <= 32:
        return 3 if signed else 13
    raise HSIHeaderError(f"Unsupported PSI bit depth: {bits}")


def _envi_byte_order(value: Any) -> int:
    key = str(value).strip().lower()
    if key in {"0", "i", "intel", "little", "littleendian", "little_endian"}:
        return 0
    if key in {"1", "m", "motorola", "big", "bigendian", "big_endian"}:
        return 1
    raise HSIHeaderError(f"Unsupported PSI byte order: {value}")


def create_envi_header(file_path: Path, meta: Mapping[str, Any]) -> None:
    """Write a correct ENVI adapter header from parsed PSI metadata.

    The source PSI bit depth is mapped to ENVI's data-type codes (for example,
    unsigned 12-bit camera data is stored as ENVI uint16/type 12).
    """

    required = ("NBANDS", "NROWS", "NCOLS", "NBITS", "LAYOUT")
    missing = [key for key in required if key not in meta]
    if missing:
        raise HSIHeaderError(f"PSI header is missing: {', '.join(missing)}.")
    try:
        bands = int(meta["NBANDS"])
        rows = int(meta["NROWS"])
        columns = int(meta["NCOLS"])
        bits = int(meta["NBITS"])
        wavelengths = [float(value) for value in meta.get("WAVELENGTHS", [])]
    except (TypeError, ValueError) as exc:
        raise HSIHeaderError("PSI dimensions, bit depth, and wavelengths are invalid.") from exc
    if len(wavelengths) != bands:
        raise HSIHeaderError(
            f"PSI header declares {bands} bands but contains "
            f"{len(wavelengths)} wavelengths."
        )
    interleave = str(meta["LAYOUT"]).lower()
    if interleave not in {"bil", "bip", "bsq"}:
        raise HSIHeaderError(f"Unsupported PSI layout: {interleave}")

    lines = [
        "ENVI",
        "description = {Hyperspectral Image Inspector PSI adapter}",
        f"samples = {columns}",
        f"lines = {rows}",
        f"bands = {bands}",
        "header offset = 0",
        "file type = ENVI Standard",
        f"data type = {_envi_data_type(bits, meta.get('SIGNED'))}",
        f"interleave = {interleave}",
        f"byte order = {_envi_byte_order(meta.get('BYTEORDER', 'I'))}",
        "wavelength units = Nanometers",
        "wavelength = {" + ", ".join(f"{value:g}" for value in wavelengths) + "}",
        f"sensor bit depth = {bits}",
    ]
    passthrough = {
        "integration time": meta.get("INTEGRATIONTIME"),
        "gain": meta.get("GAIN"),
        "chromatic correction": meta.get("CHROMATICCORRECTION"),
    }
    lines.extend(f"{key} = {value}" for key, value in passthrough.items() if value)
    Path(file_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def adapt_psi_header(header_path: Path) -> Path:
    """Return a cached ENVI adapter path without modifying the PSI source."""

    source = Path(header_path)
    try:
        digest = sha256(source.read_bytes()).hexdigest()[:16]
        cache_dir = (
            Path(tempfile.gettempdir())
            / "hyperspectral_image_inspector"
            / "envi_headers"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        output = cache_dir / f"{source.stem}-{digest}.hdr"
        if not output.exists():
            create_envi_header(output, read_psi_header(source))
        return output
    except HSIHeaderError:
        raise
    except OSError as exc:
        raise HSIHeaderError(
            f"Could not prepare a temporary ENVI adapter for {source.name}: {exc}"
        ) from exc


def numpy_to_qpixmap(
    image: npt.NDArray[np.uint8],
    alpha_mask: Optional[npt.NDArray[np.bool_]] = None,
) -> "QPixmap":
    """Convert ``uint8`` RGB data to QPixmap for the legacy Controller.

    ``alpha_mask`` marks the pixels to keep. Supply it for a non-rectangular
    (polygon-cropped) image: excluded pixels become fully transparent so the
    viewer's own background shows through, instead of rendering as black —
    black is a legitimate reflectance reading and must not be ambiguous with
    "outside the region". Without a mask the opaque RGB path is unchanged.

    This compatibility adapter imports Qt lazily. New Model code must not call
    it; conversion belongs at the Controller/View boundary.
    """

    from PyQt6.QtGui import QImage, QPixmap

    contiguous = np.ascontiguousarray(image, dtype=np.uint8)
    height, width, channels = contiguous.shape

    if alpha_mask is not None:
        if alpha_mask.shape != (height, width):
            raise ValueError(
                f"Alpha mask {alpha_mask.shape} does not match the "
                f"{(height, width)} display array."
            )
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[:, :, :3] = contiguous
        rgba[:, :, 3] = np.where(alpha_mask, 255, 0).astype(np.uint8)
        contiguous = np.ascontiguousarray(rgba)
        image_format = QImage.Format.Format_RGBA8888
    else:
        image_format = QImage.Format.Format_RGB888

    qimage = QImage(
        contiguous.data,
        width,
        height,
        int(contiguous.strides[0]),
        image_format,
    ).copy()
    return QPixmap.fromImage(qimage)
