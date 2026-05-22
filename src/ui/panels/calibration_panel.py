from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PyQt6 import uic
from PyQt6.QtWidgets import QLineEdit, QPushButton, QWidget

from ui.panels.base_panel import FeaturePanel

if TYPE_CHECKING:
    from core.hsi_data import HSIData

_UI_PATH = Path(__file__).parents[2] / "qt" / "Calibration.ui"


class CalibrationPanel(FeaturePanel):
    """Panel for dark/reference calibration of raw hyperspectral data."""

    # Attributes injected by uic.loadUi at runtime:
    darkFileButton:      QPushButton
    darkFileEdit:        QLineEdit
    referenceFileButton: QPushButton
    referenceFileEdit:   QLineEdit
    calibrateButton:     QPushButton

    def __init__(self, hsi_data: "HSIData", parent: Optional[QWidget] = None) -> None:
        super().__init__(hsi_data, parent)
        uic.loadUi(_UI_PATH, self)
        self.setEnabled(False)

    def on_image_loaded(self) -> None:
        self.setEnabled(True)

    def reset(self) -> None:
        self.darkFileEdit.clear()
        self.referenceFileEdit.clear()
        self.setEnabled(False)
