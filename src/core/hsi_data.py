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
    FeaturePanel on construction. Only MainWindowController._load_image writes
    to this object; panels read from it.
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
