"""View-neutral export Model for an already-rendered visualization.

The Controller decides which pixels are currently displayed and passes that
``uint8`` RGB array here. This module deliberately knows nothing about Qt,
widgets, zoom state, or application state; it only validates and persists the
display payload. Consequently the same API works for standard 2D results and
for a View-captured frame—including overlays, viewport zoom, or a hypercube
camera—after the View/Controller converts that frame to an RGB NumPy array.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Mapping

import numpy as np
import numpy.typing as npt
from PIL import Image

from .errors import VisualizationExportError


class ImageExportFormat(StrEnum):
    """Stable format identifiers suitable for Controller save filters."""

    PNG = "PNG"
    JPEG = "JPEG"
    TIFF = "TIFF"
    BMP = "BMP"


@dataclass(frozen=True, slots=True)
class VisualizationExportRequest:
    """Controller command describing where and how to save display pixels.

    A missing extension selects lossless PNG and appends ``.png``. Existing
    files are never replaced unless the Controller has already obtained user
    confirmation and sets ``overwrite=True``. ``jpeg_quality`` applies only
    to JPEG output.
    """

    output_path: str | Path
    overwrite: bool = False
    jpeg_quality: int = 95

    def __post_init__(self) -> None:
        if not 1 <= self.jpeg_quality <= 100:
            raise VisualizationExportError("JPEG quality must be between 1 and 100.")


@dataclass(frozen=True, slots=True)
class VisualizationExportResult:
    """Details a Controller can show after a successful export."""

    output_path: Path
    image_format: ImageExportFormat
    width: int
    height: int
    file_size_bytes: int
    sha256: str
    lossless: bool


class VisualizationExportService:
    """Atomically save the exact RGB display array selected by a Controller.

    The service is stateless and synchronous. A Controller should run it in a
    worker for large images, catch ``VisualizationExportError``, and update Qt
    only after the call returns. It must pass the array backing the current
    View—not raw HSI bands—so cropping, index coloring, and display stretching
    are preserved exactly for lossless formats. When presentation-only state
    such as overlays or zoom must be included, the View captures those pixels
    and the Controller passes the captured RGB array through the same method.
    """

    _SUFFIX_FORMATS: Mapping[str, ImageExportFormat] = MappingProxyType(
        {
            ".png": ImageExportFormat.PNG,
            ".jpg": ImageExportFormat.JPEG,
            ".jpeg": ImageExportFormat.JPEG,
            ".tif": ImageExportFormat.TIFF,
            ".tiff": ImageExportFormat.TIFF,
            ".bmp": ImageExportFormat.BMP,
        }
    )
    _LOSSLESS_FORMATS = frozenset(
        {ImageExportFormat.PNG, ImageExportFormat.TIFF, ImageExportFormat.BMP}
    )

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """Return extensions a Controller can use to build its save dialog."""

        return tuple(self._SUFFIX_FORMATS)

    def save_display(
        self,
        display_rgb: npt.NDArray[np.uint8],
        request: VisualizationExportRequest,
    ) -> VisualizationExportResult:
        """Validate and atomically save a displayed ``(H, W, 3)`` RGB array.

        PNG is recommended when pixels must round-trip exactly. JPEG stores the
        same visual composition but is inherently lossy. The input array is
        never modified and a failed encode leaves no partial destination file.

        Raises:
            VisualizationExportError: Pixels, destination, overwrite policy,
                extension, or image encoding is invalid.
        """

        pixels = self._validated_pixels(display_rgb)
        target, image_format = self._resolve_target(request.output_path)
        self._validate_destination(target, overwrite=request.overwrite)

        temporary_path: Path | None = None
        file_size_bytes = 0
        file_sha256 = ""
        try:
            # Encode beside the destination so os.replace is an atomic rename
            # on the same filesystem. The temporary file is always cleaned up.
            with tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=f".{target.stem}-",
                suffix=target.suffix,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)

            image = Image.fromarray(pixels)
            image.save(
                temporary_path,
                format=image_format.value,
                **self._save_options(image_format, request),
            )
            file_size_bytes = temporary_path.stat().st_size
            file_sha256 = self._file_sha256(temporary_path)

            # Recheck after encoding so a file created while the worker was
            # busy is not casually overwritten without Controller approval.
            if target.exists() and not request.overwrite:
                raise VisualizationExportError(
                    f"Export destination already exists: {target}"
                )
            os.replace(temporary_path, target)
            temporary_path = None
        except VisualizationExportError:
            raise
        except (OSError, ValueError) as exc:
            raise VisualizationExportError(
                f"Could not save visualization to {target}: {exc}"
            ) from exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    # The primary export failure remains the actionable error.
                    pass

        height, width, _ = pixels.shape
        return VisualizationExportResult(
            output_path=target,
            image_format=image_format,
            width=width,
            height=height,
            file_size_bytes=file_size_bytes,
            sha256=file_sha256,
            lossless=image_format in self._LOSSLESS_FORMATS,
        )

    @staticmethod
    def _validated_pixels(
        display_rgb: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.uint8]:
        if not isinstance(display_rgb, np.ndarray):
            raise VisualizationExportError("Display pixels must be a NumPy array.")
        if display_rgb.dtype != np.uint8:
            raise VisualizationExportError(
                "Display pixels must use uint8 RGB values in the range 0..255."
            )
        if display_rgb.ndim != 3 or display_rgb.shape[2] != 3:
            raise VisualizationExportError(
                "Display pixels must have shape (height, width, 3)."
            )
        if display_rgb.shape[0] < 1 or display_rgb.shape[1] < 1:
            raise VisualizationExportError("Display pixels cannot be empty.")
        return np.ascontiguousarray(display_rgb)

    def _resolve_target(
        self, output_path: str | Path
    ) -> tuple[Path, ImageExportFormat]:
        try:
            target = Path(output_path).expanduser()
        except TypeError as exc:
            raise VisualizationExportError("Export path must be text or a Path.") from exc
        if not target.name:
            raise VisualizationExportError("Export path must include a file name.")
        if not target.suffix:
            target = target.with_suffix(".png")
        suffix = target.suffix.lower()
        try:
            image_format = self._SUFFIX_FORMATS[suffix]
        except KeyError as exc:
            supported = ", ".join(self.supported_extensions)
            raise VisualizationExportError(
                f"Unsupported export extension {target.suffix!r}; choose {supported}."
            ) from exc
        return target.resolve(strict=False), image_format

    @staticmethod
    def _validate_destination(target: Path, *, overwrite: bool) -> None:
        if not target.parent.exists():
            raise VisualizationExportError(
                f"Export directory does not exist: {target.parent}"
            )
        if not target.parent.is_dir():
            raise VisualizationExportError(
                f"Export parent is not a directory: {target.parent}"
            )
        if target.exists():
            if target.is_dir():
                raise VisualizationExportError(
                    f"Export destination is a directory: {target}"
                )
            if not overwrite:
                raise VisualizationExportError(
                    f"Export destination already exists: {target}"
                )

    @staticmethod
    def _save_options(
        image_format: ImageExportFormat,
        request: VisualizationExportRequest,
    ) -> dict[str, object]:
        if image_format is ImageExportFormat.JPEG:
            return {"quality": request.jpeg_quality, "subsampling": 0}
        if image_format is ImageExportFormat.PNG:
            return {"compress_level": 6}
        if image_format is ImageExportFormat.TIFF:
            return {"compression": "tiff_deflate"}
        return {}

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
