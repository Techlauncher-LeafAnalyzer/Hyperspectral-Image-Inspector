from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PyQt6 import uic
from PyQt6.QtWidgets import QComboBox, QLineEdit, QPushButton, QWidget

from ui.panels.base_panel import FeaturePanel

if TYPE_CHECKING:
    from core.hsi_data import HSIData

_UI_PATH = Path(__file__).parents[2] / "qt" / "Classification.ui"


class ClassificationPanel(FeaturePanel):
    """Panel for unsupervised and supervised HSI classification."""

    # Unsupervised tab — attributes injected by uic.loadUi:
    numOfClassesEdit:        QLineEdit
    maxIterationsEdit:       QLineEdit
    unsupervisedClassifyButton: QPushButton

    # Supervised tab:
    lineEdit:    QLineEdit
    comboBox:    QComboBox
    pushButton:  QPushButton
    pushButton_2: QPushButton

    def __init__(self, hsi_data: "HSIData", parent: Optional[QWidget] = None) -> None:
        super().__init__(hsi_data, parent)
        uic.loadUi(_UI_PATH, self)
        self.numOfClassesEdit.setPlaceholderText("e.g. 5")
        self.maxIterationsEdit.setPlaceholderText("e.g. 20")
        self.lineEdit.setPlaceholderText("Select groundtruth file")
        self.polish_controls(
            primary_buttons={"unsupervisedClassifyButton", "pushButton_2"}
        )
        self.setEnabled(False)

    def on_image_loaded(self) -> None:
        self.setEnabled(True)

    def reset(self) -> None:
        self.numOfClassesEdit.clear()
        self.maxIterationsEdit.clear()
        self.lineEdit.clear()
        self.setEnabled(False)
