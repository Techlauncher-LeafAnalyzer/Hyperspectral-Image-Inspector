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
        self._current_panel: Optional[FeaturePanel] = None
        self._panels: dict[Functionality, type[FeaturePanel]] = {
            Functionality.VISUALIZATION:    VisualizationPanel,
            Functionality.SUPER_RESOLUTION: SuperResolutionPanel,
            Functionality.CALIBRATION:      CalibrationPanel,
            Functionality.CLASSIFICATION:   ClassificationPanel,
        }

        self._configure_window()
        self._connect_signals()
        self._select_functionality(Functionality.VISUALIZATION)

    # ------------------------------------------------------------------ #
    # Private: visual setup                                                #
    # ------------------------------------------------------------------ #

    def _configure_window(self) -> None:
        self.setWindowTitle("Hyperspectral Image Inspector")
        self.resize(1180, 760)
        self.setMinimumSize(960, 620)

        self.centralwidget.layout().setContentsMargins(14, 14, 14, 12)
        self.centralwidget.layout().setSpacing(12)

        self.widget.setObjectName("navigationPanel")
        self.widget.setMinimumWidth(184)
        self.widget.setMaximumWidth(220)
        self._refresh_widget_style(self.widget)
        self.verticalLayout.setContentsMargins(12, 16, 12, 16)
        self.verticalLayout.setSpacing(8)

        self._nav_buttons = {
            Functionality.VISUALIZATION: self.visualizationButton,
            Functionality.SUPER_RESOLUTION: self.superResolutionButton,
            Functionality.CALIBRATION: self.calibrationButton,
            Functionality.CLASSIFICATION: self.classificationButton,
        }

        nav_labels = {
            Functionality.VISUALIZATION: "Visualization",
            Functionality.SUPER_RESOLUTION: "Super Resolution",
            Functionality.CALIBRATION: "Calibration",
            Functionality.CLASSIFICATION: "Classification",
        }

        for func, button in self._nav_buttons.items():
            button.setText(nav_labels[func])
            button.setCheckable(True)
            button.setProperty("navButton", True)
            button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            button.setMinimumHeight(42)
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            self._refresh_widget_style(button)
        self.verticalLayout.addStretch(1)

        self.frame.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.verticalLayout_4.setContentsMargins(16, 14, 16, 16)
        self.verticalLayout_4.setSpacing(10)
        self.verticalLayoutBottomRight.setSpacing(10)
        self.panelContainerLayout.setSpacing(10)

        self.label.setText("Image")
        self.label_2.setText("No image loaded")
        self.label_2.setWordWrap(False)
        self.label_2.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.line.setFixedHeight(1)

        self.splitter_3.setSizes([200, 980])
        self.splitter_2.setSizes([520, 220])
        self.splitter.setSizes([220])

        self.statusbar.showMessage("Ready. Load a hyperspectral image to begin.")

    def _refresh_widget_style(self, widget: QtWidgets.QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    # ------------------------------------------------------------------ #
    # Private: signal wiring                                               #
    # ------------------------------------------------------------------ #

    def _connect_signals(self) -> None:
        self.visualizationButton.clicked.connect(
            lambda: self._select_functionality(Functionality.VISUALIZATION)
        )
        self.superResolutionButton.clicked.connect(
            lambda: self._select_functionality(Functionality.SUPER_RESOLUTION)
        )
        self.calibrationButton.clicked.connect(
            lambda: self._select_functionality(Functionality.CALIBRATION)
        )
        self.classificationButton.clicked.connect(
            lambda: self._select_functionality(Functionality.CLASSIFICATION)
        )

        self.actionLoadImage.triggered.connect(self._load_image)
        self.actionSaveImage.triggered.connect(self._save_image)

        # Viewer outbound signals — eliminates the viewer.mainui back-reference.
        self.viewer.historyChanged.connect(self._on_history_changed)
        self.viewer.spectrumPlotRequested.connect(self._on_spectrum_plot)
        self.viewer.meanIndexRequested.connect(self._on_mean_index)

    # ------------------------------------------------------------------ #
    # Private: panel management                                            #
    # ------------------------------------------------------------------ #

    def _select_functionality(self, func: Functionality) -> None:
        """Swap the bottom-right feature panel using the named panelContainer."""
        if self._current_panel is not None:
            self.panelContainer.layout().removeWidget(self._current_panel)
            self._current_panel.deleteLater()

        panel = self._panels[func](self._hsi_data, self.panelContainer)
        self.panelContainer.layout().addWidget(panel)
        self._current_panel = panel

        if self._hsi_data.is_loaded():
            self._current_panel.on_image_loaded()

        for button_func, button in self._nav_buttons.items():
            button.setChecked(button_func == func)

    # ------------------------------------------------------------------ #
    # Private: image I/O                                                   #
    # ------------------------------------------------------------------ #

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

        self.label_2.setText(str(image_path))
        self.statusbar.showMessage(f"Loaded {image_path.name}")
        self.viewer.rgb        = rgb_array
        self.viewer.mask_array = self._hsi_data.mask_array
        self.viewer.set_photo(hsi_utils.numpy_to_qpixmap(rgb_array))

        if self._current_panel is not None:
            self._current_panel.on_image_loaded()

    def _save_image(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Private: viewer signal handlers                                      #
    # ------------------------------------------------------------------ #

    def _on_history_changed(self, can_undo: bool, can_redo: bool) -> None:
        """Respond to viewer annotation history changes."""
        pass  # actionUndo/actionRedo/actionClear to be added to MainWindow.ui

    def _on_spectrum_plot(self, pos: QPointF) -> None:
        pass

    def _on_mean_index(self, index_name: str) -> None:
        pass
