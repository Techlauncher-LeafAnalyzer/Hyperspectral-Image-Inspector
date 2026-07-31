from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import spectral.io.envi as envi
from spectral import get_rgb
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QFileDialog, QMessageBox

import core.hsi_utils as hsi_utils
from core.hsi_data import Functionality, HSIData
from ui.generated.MainWindow import Ui_MainWindow
from ui.panels.base_panel import FeaturePanel
from ui.panels.calibration_panel import CalibrationPanel
from ui.panels.classification_panel import ClassificationPanel
from ui.panels.super_resolution_panel import SuperResolutionPanel
from ui.panels.visualization_panel import VisualizationPanel


class MainWindowController(QtWidgets.QMainWindow, Ui_MainWindow):
    """Application controller for the Hyperspectral Image Inspector.

    Inherits the widget layout from ``Ui_MainWindow`` (auto-generated from
    ``qt/MainWindow.ui``) and wires all application logic on top of it.
    State is encapsulated in a single ``HSIData`` instance that is injected
    into each feature panel on construction.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self._hsi_data = HSIData()
        self._connect_signals()

    # ------------------------------------------------------------------ #
    # Private: signal wiring                                               #
    # ------------------------------------------------------------------ #

        self.actionLoadImage.triggered.connect(self._load_image)
        self.actionSaveImage.triggered.connect(self._save_image)
        self.darkFileButton.clicked.connect(self._select_dark_file)
        self.referenceFileButton.clicked.connect(self._select_reference_file)
        self.calibrateButton.setEnabled(False)
        self.calibrateButton.setToolTip("Calibration is not implemented yet")
    # ------------------------------------------------------------------ #
    # Private: image I/O                                                   #
    # ------------------------------------------------------------------ #

    def _select_dark_file(self) -> None:
        self._select_calibration_file(
            self.darkFileEdit,
            "Open Dark File",
        )

    def _select_reference_file(self) -> None:
        self._select_calibration_file(
            self.referenceFileEdit,
            "Open Reference File",
        )

    def _select_calibration_file(
        self,
        target_edit: QtWidgets.QLineEdit,
        dialog_title: str,
    ) -> None:
        file_path_str, _ = QFileDialog.getOpenFileName(
            self,
            dialog_title,
            "",
            (
                "Supported Images (*.bil *.bip *.bsq *.png *.jpg *.jpeg "
                "*.tif *.tiff);;All Files (*)"
            ),
        )
        if not file_path_str:
            return

        file_path = Path(file_path_str)
        target_edit.setText(str(file_path))
        target_edit.setToolTip(str(file_path))
        self.statusbar.showMessage(f"Selected {file_path.name}")

    def _load_image(self) -> None:
        image_path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open Hyperspectral Image",
            "",
            "Hyperspectral Images (*.bil *.bip *.bsq)",
        )
        if not image_path_str:
            return

        image_path = Path(image_path_str)
        header_path = image_path.with_suffix(".hdr")

        if not header_path.exists():
            QMessageBox.critical(self, "Error", "Header file not found!")
            return

        with header_path.open() as f:
            first_line = f.readline().strip()

        if first_line.startswith("BYTEORDER"):
            meta = hsi_utils.read_psi_header(header_path)
            header_path = image_path.with_name(image_path.stem + "_envi.hdr")
            hsi_utils.create_envi_header(header_path, meta)

        spectral_obj = envi.open(str(header_path), str(image_path))
        wavelengths = [float(w) for w in spectral_obj.metadata["wavelength"]]
        rgb_bands = hsi_utils.find_rgb_bands(wavelengths)
        rgb_array = (get_rgb(spectral_obj, rgb_bands) * 255).astype(np.uint8).copy()

        self._hsi_data.image_path   = image_path
        self._hsi_data.header_path  = header_path
        self._hsi_data.spectral_obj = spectral_obj
        self._hsi_data.wavelengths  = wavelengths
        self._hsi_data.rgb_array    = rgb_array
        self._hsi_data.mask_array   = np.zeros(rgb_array.shape[:2], dtype=np.uint8)

        self.imageFilePath.setText(str(image_path))
        self.statusbar.showMessage(f"Loaded {image_path.name}")
        pixmap = hsi_utils.numpy_to_qpixmap(rgb_array)

        self.viewer.rgb        = rgb_array
        self.viewer.mask_array = self._hsi_data.mask_array
        self.viewer.set_photo(pixmap)

        self.calibrationViewer.rgb        = rgb_array
        self.calibrationViewer.mask_array = self._hsi_data.mask_array
        self.calibrationViewer.set_photo(pixmap)

    def _save_image(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Private: viewer signal handlers                                      #
    # ------------------------------------------------------------------ #

    def _on_spectrum_plot(self, pos: QPointF) -> None:
        pass

    def _on_mean_index(self, index_name: str) -> None:
        pass
