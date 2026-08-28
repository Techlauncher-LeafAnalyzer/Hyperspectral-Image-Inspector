"""UI-independent, bounded-memory inference for the supplied MSDformer.

Like VisualizationService, calls are synchronous: Controllers must use a worker
and serialize access to the source HSIData. PyTorch and SciPy are lazy imports so
the rest of the inspector remains usable without the optional SR dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import tempfile
from typing import Callable

import numpy as np
from spectral.io import envi

from .errors import CancelledError, SuperResolutionError
from .hsi_data import HSIData
from .hsi_reader import HSIReader


DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "model" / "fin_msdformer.pth"
ProgressCallback = Callable[[int, str], None]
CancellationCheck = Callable[[], bool]


@dataclass(frozen=True)
class SuperResolutionRequest:
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    tile_size: int = 64
    context: int = 8
    device: str = "auto"


@dataclass
class SuperResolutionResult:
    """Full float32 cube, lazily readable through the existing HSIData API.

    Keep this result alive while using ``data``: it owns the temporary ENVI
    files. The source capture is never overwritten. Large-image tiling changes
    global attention context and is an approximation to whole-image inference.
    """

    data: HSIData
    input_shape: tuple[int, int, int]
    model_path: Path
    device: str
    tiled: bool
    scale: int = 2
    _storage: tempfile.TemporaryDirectory = field(repr=False, default=None)


class SuperResolutionService:
    """Run the fixed 480-band, 2× checkpoint without resampling spectral bands."""

    BANDS = 480
    SCALE = 2

    def __init__(self) -> None:
        self._model = None
        self._model_key = None

    def validate(self, data: HSIData, request: SuperResolutionRequest) -> None:
        if not data.is_loaded():
            raise SuperResolutionError("Load a hyperspectral image before running SR.")
        if len(data.shape) != 3 or min(data.shape) < 1:
            raise SuperResolutionError("SR requires a nonempty (rows, columns, bands) cube.")
        if data.bands != self.BANDS:
            raise SuperResolutionError(
                f"This MSDformer checkpoint requires exactly {self.BANDS} spectral bands; "
                f"the loaded image has {data.bands}. Use a compatible capture/model. "
                "Bands are not padded, dropped, or interpolated."
            )
        wavelengths = data.wavelengths_nm
        if (wavelengths.size != self.BANDS or not np.isfinite(wavelengths).all()
                or np.any(np.diff(wavelengths) <= 0)):
            raise SuperResolutionError("SR requires 480 finite, increasing wavelengths.")
        if (not isinstance(request.tile_size, int) or request.tile_size < 1
                or not isinstance(request.context, int) or request.context < 0):
            raise SuperResolutionError("Tile size must be positive and context nonnegative.")
        if request.device not in {"auto", "cpu", "cuda"}:
            raise SuperResolutionError("SR device must be auto, cpu, or cuda.")

    def _load_model(self, request: SuperResolutionRequest):
        path = Path(request.checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise SuperResolutionError(
                f"SR model not found: {path}. Place fin_msdformer.pth in the model directory."
            )
        try:
            import torch
            from scipy.ndimage import zoom
            from .sr.msdformer import MSDformer
        except ImportError as exc:
            raise SuperResolutionError(
                "SR requires PyTorch and SciPy. In the environment used to launch the app, "
                "run: python -m pip install -r requirements-sr.txt"
            ) from exc

        device = request.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise SuperResolutionError("CUDA is unavailable. Use auto or cpu.")
        key = (path, path.stat().st_mtime_ns, path.stat().st_size, device)
        if self._model_key != key:
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(checkpoint, dict):
                    raise ValueError("Expected a state_dict or a checkpoint containing 'model'.")
                state = checkpoint.get("model", checkpoint)
                model = MSDformer(8, 2, self.BANDS, self.SCALE, 240, 4)
                model.load_state_dict(state, strict=True)
                if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
                    raise ValueError("Checkpoint contains NaN or infinite weights.")
                self._model = model.to(device).eval()
                self._model_key = key
            except Exception as exc:
                raise SuperResolutionError(
                    f"Cannot load {path.name} as the 480-band, 2× MSDformer: {exc}"
                ) from exc
        return torch, zoom, self._model, device, path

    def run(
        self,
        data: HSIData,
        request: SuperResolutionRequest = SuperResolutionRequest(),
        *,
        progress: ProgressCallback | None = None,
        is_cancelled: CancellationCheck | None = None,
    ) -> SuperResolutionResult:
        """Infer all bands in source units; never apply RGB/display normalization.

        Each input tile is HWC float32. The network receives NCHW low-resolution
        pixels and per-band SciPy cubic interpolation at 2× size. Overlapping
        context is discarded when assembling tile interiors into an ENVI memmap.
        No spatial divisibility constraint or spectral resampling is imposed.
        """
        self.validate(data, request)
        self._check_cancelled(is_cancelled)
        self._emit(progress, 0, "Loading 480-band MSDformer (2×)")
        torch, zoom, model, device, path = self._load_model(request)
        self._check_cancelled(is_cancelled)
        rows, columns, bands = data.shape
        shape = (rows * self.SCALE, columns * self.SCALE, bands)
        required = int(np.prod(shape)) * np.dtype(np.float32).itemsize
        if shutil.disk_usage(tempfile.gettempdir()).free < required + 1024 * 1024:
            raise SuperResolutionError(
                f"Insufficient temporary disk space: SR output needs {required / 2**30:.2f} GiB."
            )
        storage = tempfile.TemporaryDirectory(prefix="hsi-sr-")
        output = None
        try:
            metadata = {
                "lines": shape[0], "samples": shape[1], "bands": bands,
                "data type": 4, "interleave": "bip", "wavelength units": "nm",
                "wavelength": data.wavelengths,
                "description": f"MSDformer 2x prediction from {data.source_path.name}",
            }
            # Spectral descriptors survive; spatial/georeferencing and storage
            # fields must not be copied unchanged onto a resized capture.
            for name in ("fwhm", "bbl", "band names"):
                if name in data.metadata:
                    metadata[name] = data.metadata[name]
            header = Path(storage.name) / "super_resolution.hdr"
            image = envi.create_image(str(header), metadata=metadata, ext=".bip")
            output = image.open_memmap(writable=True, interleave="bip")
            size, context = request.tile_size, request.context
            count = ((rows + size - 1) // size) * ((columns + size - 1) // size)
            completed = 0
            self._emit(progress, 5, f"Running on {device}: {count} tile(s)")
            with torch.inference_mode():
                for top in range(0, rows, size):
                    for left in range(0, columns, size):
                        self._check_cancelled(is_cancelled)
                        bottom, right = min(rows, top + size), min(columns, left + size)
                        y0, x0 = max(0, top - context), max(0, left - context)
                        y1, x1 = min(rows, bottom + context), min(columns, right + context)
                        cube = np.asarray(
                            data.image.read_subregion((y0, y1), (x0, x1)), dtype=np.float32
                        )
                        if cube.shape != (y1 - y0, x1 - x0, bands) or not np.isfinite(cube).all():
                            raise SuperResolutionError(
                                f"Invalid input tile at row {top}, column {left}: "
                                "unexpected shape or NaN/infinite pixels."
                            )
                        h, w, _ = cube.shape
                        cubic = np.empty((bands, 2 * h, 2 * w), dtype=np.float32)
                        for band in range(bands):
                            self._check_cancelled(is_cancelled)
                            cubic[band] = zoom(cube[:, :, band], self.SCALE, order=3)
                        low = torch.from_numpy(np.ascontiguousarray(cube.transpose(2, 0, 1)))[None].to(device)
                        baseline = torch.from_numpy(cubic)[None].to(device)
                        prediction = model(low, baseline)
                        expected = (1, bands, 2 * h, 2 * w)
                        if tuple(prediction.shape) != expected or not torch.isfinite(prediction).all():
                            raise SuperResolutionError(
                                f"Invalid model output: expected finite tensor {expected}, "
                                f"received {tuple(prediction.shape)}."
                            )
                        self._check_cancelled(is_cancelled)
                        values = prediction[0].cpu().numpy().transpose(1, 2, 0)
                        dy, dx = 2 * (top - y0), 2 * (left - x0)
                        output[2 * top:2 * bottom, 2 * left:2 * right] = values[
                            dy:dy + 2 * (bottom - top), dx:dx + 2 * (right - left)
                        ]
                        del low, baseline, prediction, values, cubic, cube
                        completed += 1
                        self._emit(progress, 5 + int(90 * completed / count),
                                   f"{device}: tile {completed}/{count}")
            output.flush()
            output = None
            result_data = HSIReader().open(header)
            self._check_cancelled(is_cancelled)
            self._emit(progress, 100, f"SR ready: {shape[1]} × {shape[0]}, {bands} bands")
            return SuperResolutionResult(result_data, data.shape, path, device,
                                         count > 1, _storage=storage)
        except Exception as exc:
            output = None
            storage.cleanup()
            if isinstance(exc, (CancelledError, SuperResolutionError)):
                raise
            raise SuperResolutionError(f"Super-Resolution failed: {exc}") from exc

    @staticmethod
    def _check_cancelled(check: CancellationCheck | None) -> None:
        if check is not None and check():
            raise CancelledError("Super-Resolution cancelled.")

    @staticmethod
    def _emit(progress: ProgressCallback | None, value: int, message: str) -> None:
        if progress is not None:
            progress(value, message)
