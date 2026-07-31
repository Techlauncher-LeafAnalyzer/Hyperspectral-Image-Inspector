from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PyQt6 import uic
from PyQt6.QtWidgets import QProgressBar, QPushButton, QRadioButton, QWidget

from ui.panels.base_panel import FeaturePanel

if TYPE_CHECKING:
    from core.hsi_data import HSIData

_UI_PATH = Path(__file__).parents[2] / "qt" / "Super-resolution.ui"


class SuperResolutionPanel(FeaturePanel):
    """Panel for HSI super-resolution enhancement."""

    # Attributes injected by uic.loadUi at runtime:
    superResolutionButton: QPushButton
    lowResRadioButton:     QRadioButton
    highResRadioButton:    QRadioButton
    progressBar:           QProgressBar

    def __init__(self, hsi_data: "HSIData", parent: Optional[QWidget] = None) -> None:
        super().__init__(hsi_data, parent)
        uic.loadUi(_UI_PATH, self)
        self.progressBar.setValue(0)
        self.polish_controls(primary_buttons={"superResolutionButton"})
        self.setEnabled(False)

    def on_image_loaded(self) -> None:
        self.setEnabled(True)

    def reset(self) -> None:
        self.progressBar.setValue(0)
        self.setEnabled(False)
