from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt


class ImageFormat(Enum):
    ENVI = auto()
    PSI  = auto()


class Functionality(Enum):
    VISUALIZATION    = auto()
    SUPER_RESOLUTION = auto()
    CALIBRATION      = auto()
    CLASSIFICATION   = auto()


@dataclass
class HSIData:
    """Single source of truth for all loaded hyperspectral image state.

    Created once in MainWindowController and passed by reference to each
    FeaturePanel on construction. MainWindowController loads this object and
    applies updates through HSIData methods; panels read from it.
    """

    image_path:   Optional[Path]                  = None
    header_path:  Optional[Path]                  = None
    image_format: Optional[ImageFormat]           = None
    spectral_obj: Optional[object]                = None
    wavelengths:  list[float]                     = field(default_factory=list)
    rgb_array:    Optional[npt.NDArray[np.uint8]] = None   # shape (H, W, 3)
    mask_array:   Optional[npt.NDArray[np.uint8]] = None   # shape (H, W)

    def is_loaded(self) -> bool:
        return self.spectral_obj is not None

    def clear(self) -> None:
        self.image_path = self.header_path = self.image_format = None
        self.spectral_obj = self.rgb_array = self.mask_array = None
        self.wavelengths = []

    def crop(self, left: float, top: float, right: float, bottom: float) -> tuple[int, int] | None:
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
            self.spectral_obj = self.spectral_obj[y1:y2, x1:x2, :]
        return (x2 - x1, y2 - y1)
