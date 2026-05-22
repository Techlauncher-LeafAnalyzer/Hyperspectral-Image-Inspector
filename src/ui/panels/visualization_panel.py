from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PyQt6 import uic
from PyQt6.QtWidgets import QRadioButton, QWidget

from ui.panels.base_panel import FeaturePanel

if TYPE_CHECKING:
    from core.hsi_data import HSIData

_UI_PATH = Path(__file__).parents[2] / "qt" / "Visualization.ui"


class VisualizationPanel(FeaturePanel):
    """Panel for spectral-index visualization (RGB, NDVI, EVI, etc.)."""

    # Attributes injected by uic.loadUi at runtime:
    radioButton:   QRadioButton   # RGB
    radioButton_2: QRadioButton   # NDVI
    radioButton_3: QRadioButton   # EVI
    radioButton_4: QRadioButton   # MCARI
    radioButton_5: QRadioButton   # MTVI
    radioButton_6: QRadioButton   # OSAVI
    radioButton_7: QRadioButton   # PRI
    radioButton_8: QRadioButton   # Hypercube

    def __init__(self, hsi_data: "HSIData", parent: Optional[QWidget] = None) -> None:
        super().__init__(hsi_data, parent)
        uic.loadUi(_UI_PATH, self)
        self.setEnabled(False)

    def on_image_loaded(self) -> None:
        self.setEnabled(True)

    def reset(self) -> None:
        self.radioButton.setChecked(True)
        self.setEnabled(False)
