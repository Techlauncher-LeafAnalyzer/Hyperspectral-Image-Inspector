from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
from numpy.typing import NDArray
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QFileDialog, QMessageBox

import core.hsi_utils as hsi_utils
from core import (
    HSIData,
    HSIError,
    HSIReader,
    VisualizationError,
    VisualizationExportError,
    VisualizationExportRequest,
    VisualizationExportService,
    VisualizationMode,
    VisualizationRequest,
    VisualizationResult,
    VisualizationService,
    WavelengthError,
)
from ui.generated.MainWindow import Ui_MainWindow
from ui.spectrum_dialog import SpectrumDialog
from ui.tab_transition.handler import TabTransitionHandler
from ui.viewer import HSIViewer, PixelValueEntry


# Modes rendered eagerly after every image change so hover tooltips, mode
# switching, and index-mean lookups can read cached results instead of
# recomputing on demand. BAND is excluded (no band-index selector exists in
# the UI) and HyperCube is excluded (no hypercube view is wired up yet).
_CACHED_VISUALIZATION_MODES = (
    VisualizationMode.RGB,
    VisualizationMode.NDVI,
    VisualizationMode.EVI,
    VisualizationMode.MCARI,
    VisualizationMode.MTVI,
    VisualizationMode.OSAVI,
    VisualizationMode.PRI,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class _CropSnapshot:
    """A prior image state, kept around so a crop can be undone."""

    rgb_array:    NDArray[np.uint8]
    mask_array:   NDArray[np.uint8]
    spectral_obj: Optional[object]


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
        self._hsi_reader = HSIReader()
        self._visualization_service = VisualizationService()
        self._visualization_export_service = VisualizationExportService()
        self._active_visualization_mode: VisualizationMode = VisualizationMode.RGB
        self._visualization_results: dict[VisualizationMode, VisualizationResult] = {}
        self._crop_undo_stack: list[_CropSnapshot] = []
        self._crop_redo_stack: list[_CropSnapshot] = []
        self._configure_tabs()
        self._configure_file_menu()
        self._connect_signals()
        self._active_viewer = self._viewer_for_tab(self.tabWidget.currentIndex())

    # ------------------------------------------------------------------ #
    # Private: signal wiring                                               #
    # ------------------------------------------------------------------ #

    def _configure_tabs(self) -> None:
        tab_settings = (
            (self.tabWidget, "mainTabBar", "Application sections"),
            (
                self.classificationModeTabs,
                "classificationTabBar",
                "Classification mode",
            ),
        )
        self._tab_transitions: list[TabTransitionHandler] = []

        for tab_widget, object_name, accessible_name in tab_settings:
            tab_bar = tab_widget.tabBar()
            tab_bar.setObjectName(object_name)
            tab_bar.setAccessibleName(accessible_name)
            tab_bar.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            tab_bar.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            tab_bar.setExpanding(False)
            tab_bar.setElideMode(QtCore.Qt.TextElideMode.ElideRight)
            self._tab_transitions.append(TabTransitionHandler(tab_widget))

    def _configure_file_menu(self) -> None:
        """Fold the File menu into the main tab row as a ribbon-style dropdown."""
        assets_dir = Path(__file__).parent / "assets"
        self.actionLoadImage.setIcon(
            QtGui.QIcon(str(assets_dir / "folder_open.svg"))
        )
        self.actionLoadImage.setShortcut(
            QtGui.QKeySequence.StandardKey.Open
        )
        self.actionLoadImage.setStatusTip("Open a hyperspectral image")
        self.actionSaveImage.setIcon(
            QtGui.QIcon(str(assets_dir / "save_image.svg"))
        )
        self.actionSaveImage.setShortcut(
            QtGui.QKeySequence.StandardKey.Save
        )
        self.actionSaveImage.setStatusTip("Save the current image")

        self._file_menu = QtWidgets.QMenu(self)
        self._file_menu.setObjectName("fileMenu")
        self._file_menu.setAccessibleName("File actions")
        self._file_menu.setToolTipsVisible(True)
        self._file_menu.setMinimumWidth(220)
        self._file_menu.addAction(self.actionLoadImage)
        self._file_menu.addAction(self.actionSaveImage)

        self._file_menu_button = QtWidgets.QToolButton(self.tabWidget)
        self._file_menu_button.setObjectName("fileMenuButton")
        self._file_menu_button.setText("File")
        self._file_menu_button.setAccessibleName("File menu")
        self._file_menu_button.setAccessibleDescription(
            "Open the menu for loading and saving images"
        )
        self._file_menu_button.setToolTip("File actions")
        self._file_menu_button.setMenu(self._file_menu)
        self._file_menu_button.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup
        )
        self._file_menu_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly
        )
        self._file_menu_button.setCursor(
            QtCore.Qt.CursorShape.PointingHandCursor
        )
        self._file_menu_button.setFocusPolicy(
            QtCore.Qt.FocusPolicy.StrongFocus
        )
        self._file_menu.aboutToShow.connect(
            lambda: self._set_file_menu_open(True)
        )
        self._file_menu.aboutToHide.connect(
            lambda: self._set_file_menu_open(False)
        )
        self.tabWidget.setCornerWidget(
            self._file_menu_button, QtCore.Qt.Corner.TopLeftCorner
        )

    def _set_file_menu_open(self, is_open: bool) -> None:
        """Keep the File trigger visually active while its menu is open."""
        self._file_menu_button.setProperty("menuOpen", is_open)
        style = self._file_menu_button.style()
        style.unpolish(self._file_menu_button)
        style.polish(self._file_menu_button)
        self._file_menu_button.update()

    def _connect_signals(self) -> None:
        self.actionLoadImage.triggered.connect(self._load_image)
        self.actionSaveImage.triggered.connect(self._save_image)
        self.darkFileButton.clicked.connect(self._select_dark_file)
        self.referenceFileButton.clicked.connect(self._select_reference_file)
        self.pushButton.clicked.connect(self._select_groundtruth_file)
        self.highResButton.toggled.connect(
            self._update_super_resolution_view_state
        )
        self.lowResButton.setToolTip(
            "View the original file before Super-Resolution processing"
        )
        self.highResButton.setToolTip(
            "View the processed result after Super-Resolution"
        )
        self.calibrateButton.setEnabled(False)
        self.calibrateButton.setToolTip("Calibration is not implemented yet")
        self.runSuperResButton.setEnabled(False)
        self.runSuperResButton.setToolTip(
            "Load an image to test the Super-Resolution workflow"
        )
        self.runSuperResButton.clicked.connect(
            self._run_super_resolution_simulation
        )
        self._super_res_progress_animation = QtCore.QPropertyAnimation(
            self.superResProgressBar,
            b"value",
            self,
        )
        self._super_res_progress_animation.setDuration(2800)
        self._super_res_progress_animation.setStartValue(0)
        self._super_res_progress_animation.setEndValue(100)
        self._super_res_progress_animation.setEasingCurve(
            QtCore.QEasingCurve.Type.Linear
        )
        self._super_res_progress_animation.finished.connect(
            self._finish_super_resolution_simulation
        )
        self._update_super_resolution_view_state(
            self.highResButton.isChecked()
        )

        for viewer in self._all_viewers():
            viewer.cropRequested.connect(self._on_crop_requested)
            viewer.spectrumPlotRequested.connect(self._on_spectrum_plot)
            viewer.meanIndexRequested.connect(self._on_mean_index)
            viewer.pixel_value_provider = self._pixel_values_at

        mode_buttons = (
            (self.modeRGB, VisualizationMode.RGB),
            (self.modeNDVI, VisualizationMode.NDVI),
            (self.modeEVI, VisualizationMode.EVI),
            (self.modeMCARI, VisualizationMode.MCARI),
            (self.modeMTVI, VisualizationMode.MTVI),
            (self.modeOSAVI, VisualizationMode.OSAVI),
            (self.modePRI, VisualizationMode.PRI),
        )
        for button, mode in mode_buttons:
            button.toggled.connect(
                lambda checked, mode=mode: self._on_visualization_mode_toggled(mode, checked)
            )
        self.modeRGB.setChecked(True)
        self.modeHyperCube.setEnabled(False)
        self.modeHyperCube.setToolTip("Hypercube view is not implemented yet")

        QtGui.QShortcut(
            QtGui.QKeySequence.StandardKey.Undo,
            self,
            self._undo_crop,
        )
        QtGui.QShortcut(
            QtGui.QKeySequence.StandardKey.Redo,
            self,
            self._redo_crop,
        )

    def _update_super_resolution_view_state(
        self,
        show_processed: bool,
    ) -> None:
        """Describe the selected before/after state when processing is idle."""
        self.superResStatusStack.setCurrentWidget(self.superResIdlePage)

        if not self._hsi_data.is_loaded():
            status = "Load an image to compare the original and processed result"
        elif show_processed:
            status = "Processed result not generated — run Super-Resolution"
        else:
            status = "Showing the original file before processing"

        self.superResStatusText.setText(status)

    def _run_super_resolution_simulation(self) -> None:
        """Temporarily exercise the processing UI without running inference."""
        if (
            not self._hsi_data.is_loaded()
            or self._super_res_progress_animation.state()
            == QtCore.QAbstractAnimation.State.Running
        ):
            return

        self.highResButton.setChecked(True)
        self.lowResButton.setEnabled(False)
        self.highResButton.setEnabled(False)
        self.lowResButton.setToolTip(
            "View selection is locked while the simulation runs"
        )
        self.highResButton.setToolTip(
            "View selection is locked while the simulation runs"
        )
        self.runSuperResButton.setEnabled(False)
        self.runSuperResButton.setText("Processing…")
        self.runSuperResButton.setToolTip(
            "Super-Resolution progress simulation is running"
        )
        self.superResProgressBar.setValue(0)
        self.superResStatusStack.setCurrentWidget(self.superResProgressPage)
        self.statusbar.showMessage("Simulating Super-Resolution processing…")
        self._super_res_progress_animation.start()

    def _finish_super_resolution_simulation(self) -> None:
        """Restore controls while leaving the completed progress visible."""
        self._set_super_resolution_simulation_ready()
        self.statusbar.showMessage(
            "Super-Resolution progress simulation complete",
            5000,
        )

    def _set_super_resolution_simulation_ready(self) -> None:
        """Enable the temporary workflow and restore its idle labels."""
        self.lowResButton.setEnabled(True)
        self.highResButton.setEnabled(True)
        self.lowResButton.setToolTip(
            "View the original file before Super-Resolution processing"
        )
        self.highResButton.setToolTip(
            "View the processed result after Super-Resolution"
        )
        self.runSuperResButton.setEnabled(True)
        self.runSuperResButton.setText("Run Super-Resolution")
        self.runSuperResButton.setToolTip(
            "Temporarily simulate Super-Resolution processing progress"
        )

    def _reset_super_resolution_simulation(self) -> None:
        """Prepare the temporary workflow after an image is loaded."""
        self._super_res_progress_animation.stop()
        self.superResProgressBar.setValue(0)
        self._set_super_resolution_simulation_ready()
        self._update_super_resolution_view_state(
            self.highResButton.isChecked()
        )

        self.tabWidget.currentChanged.connect(self._on_tab_changed)

    # ------------------------------------------------------------------ #
    # Private: image I/O                                                   #
    # ------------------------------------------------------------------ #

    def _select_dark_file(self) -> None:
        self._select_supporting_file(
            self.darkFileEdit,
            "Open Dark File",
        )

    def _select_reference_file(self) -> None:
        self._select_supporting_file(
            self.referenceFileEdit,
            "Open Reference File",
        )

    def _select_groundtruth_file(self) -> None:
        self._select_supporting_file(
            self.lineEdit,
            "Open Groundtruth File",
        )

    def _select_supporting_file(
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
            "Hyperspectral Images (*.hdr *.bil *.bip *.bsq *.dat *.img *.raw)",
        )
        if not image_path_str:
            return

        self.load_image_from_path(Path(image_path_str))

    def load_image_from_path(self, image_path: Path) -> None:
        try:
            candidate = self._hsi_reader.open(image_path)
            result = self._visualization_service.render(
                candidate,
                VisualizationRequest(mode=VisualizationMode.RGB),
            )
        except HSIError as exc:
            QMessageBox.critical(self, "Unable to load image", str(exc))
            self.statusbar.showMessage("Image load failed")
            return
        except Exception as exc:  # Keep Qt's event loop alive for unexpected failures.
            LOGGER.exception("Unexpected hyperspectral import failure")
            QMessageBox.critical(
                self,
                "Unable to load image",
                f"An unexpected error occurred: {exc}",
            )
            self.statusbar.showMessage("Image load failed")
            return

        rgb_array = result.display_rgb
        candidate.rgb_array = rgb_array
        candidate.mask_array = np.zeros(rgb_array.shape[:2], dtype=np.uint8)
        self._hsi_data.update_from(candidate)
        loaded_path = self._hsi_data.data_path

        loaded_file_text = f"File Loaded: {loaded_path}"
        self.imageFilePath.setText(loaded_file_text)
        self.imageFilePath.setToolTip(str(loaded_path))
        self.superResFilePath.setText(loaded_file_text)
        self.superResFilePath.setToolTip(str(loaded_path))
        self.classificationFilePath.setText(loaded_file_text)
        self.classificationFilePath.setToolTip(str(loaded_path))
        self.unsupervisedClassifyButton.setEnabled(True)
        self.pushButton_2.setEnabled(True)
        self.statusbar.showMessage(f"Loaded {loaded_path.name}")
        self._crop_undo_stack.clear()
        self._crop_redo_stack.clear()
        self._active_visualization_mode = VisualizationMode.RGB
        self.modeRGB.setChecked(True)
        self._push_image_to_viewers()
        self._reset_super_resolution_simulation()

    def _save_image(self) -> None:
        if not self._hsi_data.is_loaded():
            QMessageBox.information(self, "Nothing to save", "Load an image first.")
            return

        result = self._visualization_results.get(self._active_visualization_mode)
        display_rgb = result.display_rgb if result is not None else self._hsi_data.rgb_array

        extensions = " ".join(
            f"*{ext}" for ext in self._visualization_export_service.supported_extensions
        )
        file_path_str, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", f"Images ({extensions})"
        )
        if not file_path_str:
            return

        output_path = Path(file_path_str)
        try:
            saved = self._visualization_export_service.save_display(
                display_rgb,
                VisualizationExportRequest(
                    output_path, overwrite=output_path.exists()
                ),
            )
        except VisualizationExportError as exc:
            QMessageBox.critical(self, "Unable to save image", str(exc))
            return

        self.statusbar.showMessage(f"Saved {saved.output_path.name}")

    # ------------------------------------------------------------------ #
    # Private: visualization mode / pixel values                          #
    # ------------------------------------------------------------------ #

    def _on_visualization_mode_toggled(self, mode: VisualizationMode, checked: bool) -> None:
        if not checked:
            return
        self._active_visualization_mode = mode
        if self._hsi_data.is_loaded():
            self._refresh_viewers_display()

    def _recompute_visualizations(self) -> None:
        self._visualization_results = {}
        for mode in _CACHED_VISUALIZATION_MODES:
            try:
                self._visualization_results[mode] = self._visualization_service.render(
                    self._hsi_data, VisualizationRequest(mode=mode)
                )
            except (VisualizationError, WavelengthError) as exc:
                LOGGER.info("Skipping %s visualization: %s", mode.value, exc)

    def _refresh_viewers_display(self) -> None:
        result = self._visualization_results.get(self._active_visualization_mode)
        display_rgb = result.display_rgb if result is not None else self._hsi_data.rgb_array
        pixmap = hsi_utils.numpy_to_qpixmap(display_rgb)
        for viewer in self._all_viewers():
            state = viewer.get_view_state()
            viewer.rgb        = self._hsi_data.rgb_array
            viewer.mask_array = self._hsi_data.mask_array
            viewer.set_photo(pixmap)
            if state is not None:
                viewer.queue_view_state(state)

    def _pixel_values_at(self, row: int, column: int) -> Mapping[str, PixelValueEntry]:
        values: dict[str, PixelValueEntry] = {}
        for mode, result in self._visualization_results.items():
            height, width = result.display_rgb.shape[:2]
            if not (0 <= row < height and 0 <= column < width):
                continue
            color = tuple(int(channel) for channel in result.display_rgb[row, column])
            if mode is VisualizationMode.RGB:
                values[mode.value] = PixelValueEntry(value=color, color=color)
            elif result.values is not None:
                values[mode.value] = PixelValueEntry(
                    value=float(result.values[row, column]), color=color
                )
        return values

    # ------------------------------------------------------------------ #
    # Private: viewer signal handlers                                      #
    # ------------------------------------------------------------------ #

    def _on_spectrum_plot(self, pos: QPointF) -> None:
        if not self._hsi_data.is_loaded():
            return

        row, column = int(pos.y()), int(pos.x())
        if not (0 <= row < self._hsi_data.rows and 0 <= column < self._hsi_data.columns):
            return

        try:
            result = self._visualization_service.spectrum(self._hsi_data, row, column)
        except HSIError as exc:
            QMessageBox.critical(self, "Unable to plot spectrum", str(exc))
            return

        dialog = SpectrumDialog(result, parent=self)
        dialog.exec()

    def _on_mean_index(self, index_name: str) -> None:
        if not self._hsi_data.is_loaded():
            return

        try:
            mode = VisualizationMode(index_name)
        except ValueError:
            return

        result = self._visualization_results.get(mode)
        if result is None or result.values is None:
            QMessageBox.information(
                self,
                "Index unavailable",
                f"{index_name} could not be computed for this image.",
            )
            return

        mean_value = float(np.nanmean(result.values))
        self.statusbar.showMessage(f"{index_name} mean: {mean_value:.4f}", 8000)

    def _on_crop_requested(self, rect: QtCore.QRectF) -> None:
        if not self._hsi_data.is_loaded():
            return

        self._crop_undo_stack.append(self._snapshot_current_state())
        self._crop_redo_stack.clear()

        cropped_size = self._hsi_data.crop(
            rect.left(),
            rect.top(),
            rect.right(),
            rect.bottom(),
        )
        if cropped_size is None:
            self._crop_undo_stack.pop()
            return

        self._push_image_to_viewers()
        self.statusbar.showMessage(
            f"Cropped to {cropped_size[0]}x{cropped_size[1]}"
        )

    # ------------------------------------------------------------------ #
    # Private: crop undo/redo                                             #
    # ------------------------------------------------------------------ #

    def _all_viewers(self) -> tuple:
        return (
            self.viewer,
            self.calibrationViewer,
            self.superResViewer,
            self.classificationViewer,
        )

    def _viewer_for_tab(self, index: int) -> Optional[HSIViewer]:
        page = self.tabWidget.widget(index)
        return page.findChild(HSIViewer) if page is not None else None

    def _on_tab_changed(self, index: int) -> None:
        new_viewer = self._viewer_for_tab(index)
        if new_viewer is None:
            return

        if self._active_viewer is not None and self._active_viewer is not new_viewer:
            state = self._active_viewer.get_view_state()
            if state is not None:
                new_viewer.queue_view_state(state)

        self._active_viewer = new_viewer

    def _snapshot_current_state(self) -> _CropSnapshot:
        return _CropSnapshot(
            rgb_array=self._hsi_data.rgb_array,
            mask_array=self._hsi_data.mask_array,
            spectral_obj=self._hsi_data.spectral_obj,
        )

    def _restore_snapshot(self, snapshot: _CropSnapshot) -> None:
        self._hsi_data.rgb_array    = snapshot.rgb_array
        self._hsi_data.mask_array   = snapshot.mask_array
        self._hsi_data.spectral_obj = snapshot.spectral_obj
        self._push_image_to_viewers()

    def _undo_crop(self) -> None:
        if not self._crop_undo_stack:
            return
        self._crop_redo_stack.append(self._snapshot_current_state())
        self._restore_snapshot(self._crop_undo_stack.pop())
        self.statusbar.showMessage("Crop undone")

    def _redo_crop(self) -> None:
        if not self._crop_redo_stack:
            return
        self._crop_undo_stack.append(self._snapshot_current_state())
        self._restore_snapshot(self._crop_redo_stack.pop())
        self.statusbar.showMessage("Crop redone")

    def _push_image_to_viewers(self) -> None:
        self._recompute_visualizations()
        self._refresh_viewers_display()
