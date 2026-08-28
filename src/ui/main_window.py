from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Mapping, Optional

import numpy as np
from numpy.typing import NDArray
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QFileDialog, QMessageBox

import core.hsi_utils as hsi_utils
from core import (
    CancelledError,
    ClassificationError,
    ClassificationService,
    HSIData,
    HSIError,
    HSIReader,
    SuperResolutionRequest,
    SuperResolutionResult,
    SuperResolutionService,
    SupervisedClassificationRequest,
    SupervisedClassificationResult,
    SupervisedClassifierType,
    TrainingFilePair,
    TrainingPairResolver,
    UnsupervisedClassificationRequest,
    UnsupervisedClassificationResult,
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
from ui.index_mean_dialog import IndexMeanDialog
from ui.hypercube_controller import HypercubeController
from ui.spectrum_dialog import SpectrumDialog
from ui.super_resolution_worker import SuperResolutionWorker
from ui.tab_transition.handler import TabTransitionHandler
from ui.viewer import HSIViewer, PixelValueEntry


# Modes rendered eagerly after every image change so hover tooltips, mode
# switching, and index-mean lookups can read cached results instead of
# recomputing on demand. BAND is excluded (no band-index selector exists in
# the UI) and HyperCube is excluded (it has its own dedicated background
# worker, driven by HypercubeController).
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


class _ClassificationWorker(QtCore.QObject):
    """Run the synchronous classification Model away from the GUI thread."""

    progressChanged = QtCore.pyqtSignal(int, str)
    resultReady = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal()

    def __init__(
        self,
        service: ClassificationService,
        data: HSIData,
        request: UnsupervisedClassificationRequest,
    ) -> None:
        super().__init__()
        self._service = service
        self._data = data
        self._request = request
        self._cancel_requested = Event()

    @QtCore.pyqtSlot()
    def run(self) -> None:
        """Execute in a QThread and report only signal-safe result objects."""

        try:
            result = self._service.classify_unsupervised(
                self._data,
                self._request,
                progress=self.progressChanged.emit,
                is_cancelled=self._cancel_requested.is_set,
            )
        except CancelledError:
            self.cancelled.emit()
        except ClassificationError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # Keep Qt's event loop alive on programming faults.
            LOGGER.exception("Unexpected K-means classification failure")
            self.failed.emit(f"An unexpected error occurred: {exc}")
        else:
            self.resultReady.emit(result)
        finally:
            self.finished.emit()

    def cancel(self) -> None:
        """Thread-safely request cancellation at the next Model checkpoint."""

        self._cancel_requested.set()


class _SupervisedClassificationWorker(QtCore.QObject):
    """Open a resolved training pair and run one-example classification."""

    progressChanged = QtCore.pyqtSignal(int, str)
    resultReady = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal()

    def __init__(
        self,
        service: ClassificationService,
        target_data: HSIData,
        training_pair: TrainingFilePair,
        request: SupervisedClassificationRequest,
    ) -> None:
        super().__init__()
        self._service = service
        self._target_data = target_data
        self._training_pair = training_pair
        self._request = request
        self._cancel_requested = Event()

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            training_data = HSIReader().open(self._training_pair.cube_path)
            result = self._service.classify_supervised(
                self._target_data,
                training_data,
                self._training_pair.mask_path,
                self._request,
                progress=self.progressChanged.emit,
                is_cancelled=self._cancel_requested.is_set,
            )
        except CancelledError:
            self.cancelled.emit()
        except HSIError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # Keep Qt's event loop alive on programming faults.
            LOGGER.exception("Unexpected supervised classification failure")
            self.failed.emit(f"An unexpected error occurred: {exc}")
        else:
            self.resultReady.emit(result)
        finally:
            self.finished.emit()

    def cancel(self) -> None:
        self._cancel_requested.set()


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
        self._super_resolution_service = SuperResolutionService()
        self._super_resolution_request = SuperResolutionRequest()
        self._super_res_worker: SuperResolutionWorker | None = None
        self._super_res_result: SuperResolutionResult | None = None
        self._super_res_error: str | None = None
        self._sr_view_scale = 1
        self._viz_view_scale = 1
        self._high_res_notice_shown = False
        self._close_after_sr = False
        self._classification_service = ClassificationService()
        self._training_pair_resolver = TrainingPairResolver()
        self._active_visualization_mode: VisualizationMode = VisualizationMode.RGB
        self._visualization_results: dict[VisualizationMode, VisualizationResult] = {}
        self._classification_result: Optional[
            UnsupervisedClassificationResult | SupervisedClassificationResult
        ] = None
        self._classification_rgb: Optional[NDArray[np.uint8]] = None
        self._classification_thread: Optional[QtCore.QThread] = None
        self._classification_worker: Optional[
            _ClassificationWorker | _SupervisedClassificationWorker
        ] = None
        self._active_classification_button: Optional[QtWidgets.QPushButton] = None
        self._supervised_training_pair: Optional[TrainingFilePair] = None
        self._close_after_classification = False
        self._crop_undo_stack: list[_CropSnapshot] = []
        self._crop_redo_stack: list[_CropSnapshot] = []
        self._hypercube_controller = HypercubeController(
            self.modeHyperCube,
            self.visualizationStack,
            self.hypercubeWidget,
            self.statusbar,
            self._visualization_service,
        )
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
        self.pushButton_2.clicked.connect(self._on_supervised_classify_clicked)
        self.unsupervisedClassifyButton.clicked.connect(
            self._on_unsupervised_classify_clicked
        )
        self.numOfClassesEdit.setValidator(QtGui.QIntValidator(2, 65535, self))
        self.maxIterationsEdit.setValidator(QtGui.QIntValidator(1, 10000, self))
        self.numOfClassesEdit.setText("5")
        self.maxIterationsEdit.setText("20")
        # Keep the View's existing selector driven by the Model enum.  Storing
        # the enum as item data means the Controller does not depend on the
        # user-visible label when it builds a classification request.
        self.comboBox.clear()
        for classifier in SupervisedClassifierType:
            self.comboBox.addItem(classifier.value, classifier)
        self.comboBox.setToolTip(
            "Choose the Spectral Python classifier used for one-example transfer"
        )
        self.unsupervisedClassifyButton.setToolTip(
            "Load an image, then group pixels by spectral similarity with K-means"
        )
        self.pushButton_2.setToolTip(
            "Classify the current image from the selected reference mask and cube"
        )
        self.pushButton.setText("Ground-truth Mask")
        self.pushButton.setToolTip(
            "Select a mask; its hyperspectral cube is paired automatically by name"
        )
        self.lineEdit.setPlaceholderText(
            "Select mask; matching cube is detected automatically"
        )
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
        self.runSuperResButton.clicked.connect(self._run_super_resolution)
        self._set_super_resolution_ready()
        self._update_super_resolution_view_state(self.highResButton.isChecked())
        self.tabWidget.currentChanged.connect(self._on_tab_changed)

        for viewer in self._all_viewers():
            viewer.cropRequested.connect(self._on_crop_requested)
            viewer.spectrumPlotRequested.connect(self._on_spectrum_plot)
            viewer.meanIndexRequested.connect(self._on_mean_index)
            viewer.pixel_value_provider = self._pixel_values_at
        self.superResViewer.pixel_value_provider = self._sr_pixel_values_at
        self.classificationViewer.pixel_value_provider = (
            self._classification_pixel_values_at
        )

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

    def _update_super_resolution_view_state(self, show_processed: bool) -> None:
        if self._super_res_worker is not None:
            return
        self._refresh_super_resolution_display()
        self.superResStatusStack.setCurrentWidget(self.superResIdlePage)
        if not self._hsi_data.is_loaded():
            status = "Load an image to compare the original and processed result"
        elif show_processed and self._super_res_result is None:
            status = "Processed result not generated — run Super-Resolution"
        else:
            data = self._display_data()
            label = "MSDformer 2×" if show_processed else "Original"
            status = f"{label}: {data.columns} × {data.rows} pixels, {data.bands} bands"
            if show_processed and self._super_res_result.tiled:
                status += " · tiled inference"
        self.superResStatusText.setText(status)
        # A result already exists: every tab must reflect the low/high choice,
        # not just the Super-Resolution tab's own comparison viewer.
        if self._super_res_result is not None:
            self._refresh_visualization_pipeline()

    def _display_data(self) -> HSIData:
        """Return the dataset every tab should currently render.

        Once Super-Resolution has produced a result, the high/low toggle
        chooses between it and the original for the whole application, not
        just the Super-Resolution tab's own comparison viewer.
        """
        if self.highResButton.isChecked() and self._super_res_result is not None:
            return self._super_res_result.data
        return self._hsi_data

    def _refresh_super_resolution_display(self) -> None:
        previous_size = self.superResViewer.photo_size()
        state = self.superResViewer.get_view_state()
        if self.highResButton.isChecked() and self._super_res_result is None:
            self.superResViewer.rgb = None
            self.superResViewer.mask_array = None
            self.superResViewer.set_photo()
            return
        data = self._display_data()
        if data.rgb_array is None:
            return
        new_scale = 2 if data is not self._hsi_data else 1
        self.superResViewer.rgb = data.rgb_array
        self.superResViewer.mask_array = data.mask_array
        pixmap = hsi_utils.numpy_to_qpixmap(data.rgb_array)
        self.superResViewer.set_photo(pixmap)
        factor = new_scale / self._sr_view_scale
        # Preserve comparison framing for LR/HR toggles, but refit after a
        # source crop/resize just like the other viewers (LEAF-153).
        if (
            state is not None
            and previous_size is not None
            and previous_size * factor == pixmap.size()
        ):
            self.superResViewer.queue_view_state((state[0] / factor, state[1] * factor))
        self._sr_view_scale = new_scale

    def _sr_pixel_values_at(self, row: int, column: int) -> Mapping[str, PixelValueEntry]:
        data = self._display_data()
        if self.highResButton.isChecked() and self._super_res_result is None:
            return {}
        rgb = data.rgb_array
        if rgb is None or not (0 <= row < rgb.shape[0] and 0 <= column < rgb.shape[1]):
            return {}
        color = tuple(int(value) for value in rgb[row, column])
        return {"RGB": PixelValueEntry(value=color, color=color)}

    def _run_super_resolution(self) -> None:
        if self._super_res_worker is not None:
            self._cancel_super_resolution()
            return
        if not self._hsi_data.is_loaded():
            return
        try:
            self._super_resolution_service.validate(self._hsi_data, self._super_resolution_request)
        except HSIError as exc:
            QMessageBox.critical(self, "Unable to run Super-Resolution", str(exc))
            self.superResStatusText.setText(str(exc))
            return
        self._super_res_error = None
        self.lowResButton.setEnabled(False)
        self.highResButton.setEnabled(False)
        self.actionLoadImage.setEnabled(False)
        self.runSuperResButton.setText("Cancel")
        self.runSuperResButton.setToolTip("Cancel after the current inference tile")
        self.superResProgressBar.setValue(0)
        self.superResStatusStack.setCurrentWidget(self.superResProgressPage)
        # Both features read the same SpyFile. Finish cancellation of any
        # hypercube read before handing the source to the SR worker.
        self._hypercube_controller.stop_and_wait()
        worker = SuperResolutionWorker(self._super_resolution_service, self._hsi_data,
                                       self._super_resolution_request, parent=self)
        self._super_res_worker = worker
        worker.progress.connect(self._on_super_resolution_progress)
        worker.result_ready.connect(self._on_super_resolution_result)
        worker.failed.connect(self._on_super_resolution_failed)
        worker.cancelled.connect(self._on_super_resolution_cancelled)
        worker.finished.connect(self._finish_super_resolution)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _cancel_super_resolution(self) -> None:
        if self._super_res_worker is not None:
            self._super_res_worker.requestInterruption()
            self.runSuperResButton.setEnabled(False)
            self.runSuperResButton.setText("Cancelling…")
            self.statusbar.showMessage("Cancelling after the current SR operation…")

    def _on_super_resolution_progress(self, value: int, message: str) -> None:
        self.superResProgressBar.setValue(value)
        self.superResProgressBar.setToolTip(message)
        self.statusbar.showMessage(message)

    def _on_super_resolution_result(self, result, display) -> None:
        if self._super_res_worker.isInterruptionRequested():
            self._on_super_resolution_cancelled()
            return
        result.data.rgb_array = display.display_rgb
        result.data.mask_array = np.zeros(display.display_rgb.shape[:2], dtype=np.uint8)
        self._super_res_result = result
        self.highResButton.setChecked(True)
        self._refresh_super_resolution_display()
        self.superResProgressBar.setValue(100)
        self.statusbar.showMessage("Super-Resolution complete", 5000)

    def _on_super_resolution_failed(self, message: str) -> None:
        self._super_res_error = f"SR failed: {message}"
        LOGGER.error("%s", self._super_res_error)
        if not self._close_after_sr:
            QMessageBox.critical(self, "Unable to run Super-Resolution", message)

    def _on_super_resolution_cancelled(self) -> None:
        self._super_res_error = "Super-Resolution cancelled; previous image retained"

    def _finish_super_resolution(self) -> None:
        self._super_res_worker = None
        self._set_super_resolution_ready()
        self._update_super_resolution_view_state(self.highResButton.isChecked())
        if self._super_res_error:
            self.superResStatusText.setText(self._super_res_error)
            self.statusbar.showMessage(self._super_res_error)
        if self._close_after_sr:
            QtCore.QTimer.singleShot(0, self.close)
        else:
            # A cube build cancelled to make room for SR must not remain stuck
            # at "Computing hypercube". The original source is still unchanged.
            self._hypercube_controller.resume(self._display_data())

    def _set_super_resolution_ready(self) -> None:
        self.actionLoadImage.setEnabled(True)
        self.lowResButton.setEnabled(True)
        self.highResButton.setEnabled(True)
        self.runSuperResButton.setEnabled(self._hsi_data.is_loaded())
        self.runSuperResButton.setText("Run Super-Resolution")
        self.runSuperResButton.setToolTip("Run the 480-band MSDformer model at 2× spatial resolution")

    def _reset_super_resolution(self) -> None:
        self._super_res_result = None
        self._super_res_error = None
        self.lowResButton.setChecked(True)
        self.superResProgressBar.setValue(0)
        self._set_super_resolution_ready()
        self._update_super_resolution_view_state(False)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._hypercube_controller.shutdown()
        if self._classification_is_running():
            assert self._classification_worker is not None
            self._classification_worker.cancel()
            self._close_after_classification = True
            if self._active_classification_button is not None:
                self._active_classification_button.setEnabled(False)
                self._active_classification_button.setText("Cancelling…")
            self.statusbar.showMessage(
                "Cancelling classification before closing the application…"
            )
            event.ignore()
            return
        # Never destroy a running QThread or block Qt waiting for inference.
        if self._super_res_worker is not None:
            self._close_after_sr = True
            self._cancel_super_resolution()
            event.ignore()
            return
        self._super_res_result = None
        super().closeEvent(event)

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
        if self._classification_is_running():
            QMessageBox.information(
                self,
                "Classification in progress",
                "Cancel the current classification before changing training data.",
            )
            return
        mask_path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open Ground-truth Mask",
            "",
            (
                "Ground-truth Masks (*.png *.tif *.tiff *.bmp *.jpg *.jpeg);;"
                "All Files (*)"
            ),
        )
        if not mask_path_str:
            return

        try:
            pair = self._training_pair_resolver.resolve(mask_path_str)
        except ClassificationError as exc:
            self._supervised_training_pair = None
            self.lineEdit.clear()
            self.lineEdit.setToolTip("")
            QMessageBox.critical(self, "Invalid training-file pair", str(exc))
            self.statusbar.showMessage("Training-file pairing failed", 8000)
            return

        self._supervised_training_pair = pair
        self.lineEdit.setText(str(pair.mask_path))
        self.lineEdit.setToolTip(
            f"Mask: {pair.mask_path}\nHyperspectral cube: {pair.cube_path}"
        )
        self.statusbar.showMessage(
            f"Training mask paired with {pair.cube_path.name}",
            8000,
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
        if self._super_res_worker is not None:
            return
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
        if self._super_res_worker is not None:
            self.statusbar.showMessage("Cancel or finish SR before loading another image")
            return
        if self._classification_is_running():
            QMessageBox.information(
                self,
                "Classification in progress",
                "Cancel the current classification before loading another image.",
            )
            return
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
        self._clear_classification_result()
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

    def _save_image(self) -> None:
        if not self._hsi_data.is_loaded():
            QMessageBox.information(self, "Nothing to save", "Load an image first.")
            return

        result = self._visualization_results.get(self._active_visualization_mode)
        display_rgb = result.display_rgb if result is not None else self._display_data().rgb_array
        if (
            self._active_viewer is self.classificationViewer
            and self._classification_rgb is not None
        ):
            display_rgb = self._classification_rgb
        if self.tabWidget.currentWidget() is self.SuperResolution:
            if self.highResButton.isChecked() and self._super_res_result is None:
                QMessageBox.information(self, "Nothing to save", "Run Super-Resolution first.")
                return
            display_rgb = self._display_data().rgb_array

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
        self.visualizationStack.setCurrentWidget(self.viewer)
        self._active_visualization_mode = mode
        if self._hsi_data.is_loaded():
            self._refresh_viewers_display()

    def _recompute_visualizations(self) -> None:
        self._hypercube_controller.stop_and_wait()
        self._visualization_results = {}
        data = self._display_data()
        for mode in _CACHED_VISUALIZATION_MODES:
            try:
                self._visualization_results[mode] = self._visualization_service.render(
                    data, VisualizationRequest(mode=mode)
                )
            except (VisualizationError, WavelengthError) as exc:
                LOGGER.info("Skipping %s visualization: %s", mode.value, exc)

        # Keep the active source on the same RGB stretch as Visualization.
        # After a crop, the previous RGB array has stale limits.
        rgb_result = self._visualization_results.get(VisualizationMode.RGB)
        if rgb_result is not None:
            data.rgb_array = rgb_result.display_rgb

    def _refresh_viewers_display(self) -> None:
        data = self._display_data()
        result = self._visualization_results.get(self._active_visualization_mode)
        display_rgb = result.display_rgb if result is not None else data.rgb_array
        pixmap = hsi_utils.numpy_to_qpixmap(display_rgb)
        new_scale = 2 if data is not self._hsi_data else 1
        factor = new_scale / self._viz_view_scale
        for viewer in self._all_viewers():
            if viewer is self.superResViewer:
                continue
            previous_size = viewer.photo_size()
            state = viewer.get_view_state()
            viewer_display = (
                self._classification_rgb
                if viewer is self.classificationViewer
                and self._classification_rgb is not None
                else display_rgb
            )
            viewer.rgb        = data.rgb_array
            viewer.mask_array = data.mask_array
            pixmap = hsi_utils.numpy_to_qpixmap(viewer_display)
            viewer.set_photo(pixmap)
            # Restore the previous pan/zoom, rescaled by `factor`, when the
            # image dimensions changed only because of a low/high-res swap
            # (factor==1 covers the unchanged-size case, e.g. switching
            # visualization mode). Otherwise (e.g. after a crop) let
            # set_photo's fresh fit_in_view() stand, so the view actually
            # rescales to the new image size.
            if state is not None and previous_size is not None and previous_size * factor == pixmap.size():
                viewer.queue_view_state((state[0] / factor, state[1] * factor))
        self._viz_view_scale = new_scale
        self._refresh_super_resolution_display()

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

    def _classification_pixel_values_at(
        self, row: int, column: int
    ) -> Mapping[str, object]:
        """Add the zero-based K-means class ID to classification hover data."""

        values = dict(self._pixel_values_at(row, column))
        result = self._classification_result
        if result is not None and (
            0 <= row < result.class_map.shape[0]
            and 0 <= column < result.class_map.shape[1]
        ):
            values["Class"] = int(result.class_map[row, column])
        return values

    # ------------------------------------------------------------------ #
    # Private: unsupervised classification                                #
    # ------------------------------------------------------------------ #

    def _on_unsupervised_classify_clicked(self) -> None:
        """Start K-means, or turn the same button into a cancellation action."""

        if self._classification_is_running():
            self._cancel_active_classification()
            return
        if not self._hsi_data.is_loaded():
            QMessageBox.information(self, "Nothing to classify", "Load an image first.")
            return

        try:
            request = UnsupervisedClassificationRequest(
                n_classes=int(self.numOfClassesEdit.text()),
                max_iterations=int(self.maxIterationsEdit.text()),
            )
        except ValueError:
            QMessageBox.critical(
                self,
                "Invalid classification settings",
                "Enter whole numbers for classes and maximum iterations.",
            )
            return
        except ClassificationError as exc:
            QMessageBox.critical(self, "Invalid classification settings", str(exc))
            return

        estimate = self._classification_service.estimate_kmeans_working_bytes(
            self._hsi_data, request
        )
        if estimate >= 1_000_000_000 and not self._confirm_large_classification(
            estimate
        ):
            return

        self._start_classification_worker(request)

    def _on_supervised_classify_clicked(self) -> None:
        """Train from the automatically paired example and classify the target."""

        if self._classification_is_running():
            self._cancel_active_classification()
            return
        if not self._hsi_data.is_loaded():
            QMessageBox.information(self, "Nothing to classify", "Load an image first.")
            return
        if self._supervised_training_pair is None:
            QMessageBox.critical(
                self,
                "Training mask required",
                (
                    "Select a ground-truth mask first. The program will look beside "
                    "it for a same-base hyperspectral cube or an "
                    "_hyperspectral cube pair."
                ),
            )
            return
        classifier = self.comboBox.currentData()
        if not isinstance(classifier, SupervisedClassifierType):
            QMessageBox.critical(
                self,
                "Unsupported supervised classifier",
                "Select a supported supervised classifier and try again.",
            )
            return
        request = SupervisedClassificationRequest(classifier)

        worker = _SupervisedClassificationWorker(
            self._classification_service,
            self._hsi_data,
            self._supervised_training_pair,
            request,
        )
        self._launch_classification_worker(
            worker,
            self.pushButton_2,
            f"Starting {request.classifier.value} classification…",
        )

    def _cancel_active_classification(self) -> None:
        assert self._classification_worker is not None
        self._classification_worker.cancel()
        if self._active_classification_button is not None:
            self._active_classification_button.setEnabled(False)
            self._active_classification_button.setText("Cancelling…")
        self.statusbar.showMessage(
            "Cancelling after the current classification stage…"
        )

    def _confirm_large_classification(self, estimated_bytes: int) -> bool:
        gibibytes = estimated_bytes / (1024 ** 3)
        answer = QMessageBox.question(
            self,
            "Large classification job",
            (
                f"K-means may require about {gibibytes:.1f} GiB of working memory. "
                "Cropping the image first is recommended. Continue anyway?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _start_classification_worker(
        self, request: UnsupervisedClassificationRequest
    ) -> None:
        worker = _ClassificationWorker(
            self._classification_service,
            self._hsi_data,
            request,
        )
        self._launch_classification_worker(
            worker,
            self.unsupervisedClassifyButton,
            "Starting K-means classification…",
        )

    def _launch_classification_worker(
        self,
        worker: _ClassificationWorker | _SupervisedClassificationWorker,
        active_button: QtWidgets.QPushButton,
        starting_message: str,
    ) -> None:
        """Create one worker/thread pair; all View updates stay on the GUI thread."""

        self._hypercube_controller.stop_and_wait()
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progressChanged.connect(self._on_classification_progress)
        worker.resultReady.connect(self._on_classification_result)
        worker.failed.connect(self._on_classification_failed)
        worker.cancelled.connect(self._on_classification_cancelled)
        worker.finished.connect(self._on_classification_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_classification_worker)

        self._classification_thread = thread
        self._classification_worker = worker
        self._active_classification_button = active_button
        self.numOfClassesEdit.setEnabled(False)
        self.maxIterationsEdit.setEnabled(False)
        self.pushButton.setEnabled(False)
        self.comboBox.setEnabled(False)
        self.actionLoadImage.setEnabled(False)
        self.unsupervisedClassifyButton.setEnabled(
            active_button is self.unsupervisedClassifyButton
        )
        self.pushButton_2.setEnabled(active_button is self.pushButton_2)
        active_button.setText("Cancel")
        active_button.setToolTip(
            "Request cancellation after the current classification stage"
        )
        self.statusbar.showMessage(starting_message)
        thread.start()

    @QtCore.pyqtSlot(int, str)
    def _on_classification_progress(self, value: int, message: str) -> None:
        self.statusbar.showMessage(f"Classification {value}% — {message}")

    @QtCore.pyqtSlot(object)
    def _on_classification_result(self, result: object) -> None:
        if not isinstance(
            result,
            (UnsupervisedClassificationResult, SupervisedClassificationResult),
        ):
            self._on_classification_failed("Worker returned an invalid result.")
            return
        self._classification_result = result
        class_ids = (
            result.class_ids
            if isinstance(result, SupervisedClassificationResult)
            else tuple(range(result.n_classes))
        )
        self._classification_rgb = self._colorize_class_map(
            result.class_map,
            class_ids,
        )
        self._show_classification_result()
        populated = int(np.count_nonzero(result.class_pixel_counts))
        operation = (
            result.classifier.value
            if isinstance(result, SupervisedClassificationResult)
            else "K-means"
        )
        self.statusbar.showMessage(
            f"{operation} complete: {populated}/{result.n_classes} populated classes",
            8000,
        )

    @QtCore.pyqtSlot(str)
    def _on_classification_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Classification failed", message)
        self.statusbar.showMessage("Classification failed", 8000)

    @QtCore.pyqtSlot()
    def _on_classification_cancelled(self) -> None:
        self.statusbar.showMessage("Classification cancelled", 5000)

    @QtCore.pyqtSlot()
    def _on_classification_finished(self) -> None:
        self.numOfClassesEdit.setEnabled(True)
        self.maxIterationsEdit.setEnabled(True)
        self.pushButton.setEnabled(True)
        self.comboBox.setEnabled(True)
        self.actionLoadImage.setEnabled(True)
        self.unsupervisedClassifyButton.setEnabled(self._hsi_data.is_loaded())
        self.pushButton_2.setEnabled(self._hsi_data.is_loaded())
        self.unsupervisedClassifyButton.setText("Classify")
        self.pushButton_2.setText("Classify")
        self.unsupervisedClassifyButton.setToolTip(
            "Group pixels by spectral similarity with K-means"
        )
        self.pushButton_2.setToolTip(
            "Classify from the selected reference mask and paired cube"
        )

    @QtCore.pyqtSlot()
    def _clear_classification_worker(self) -> None:
        self._classification_worker = None
        self._classification_thread = None
        self._active_classification_button = None
        if self._close_after_classification:
            self._close_after_classification = False
            QtCore.QTimer.singleShot(0, self.close)

    def _show_classification_result(self) -> None:
        if self._classification_rgb is None:
            return
        viewer = self.classificationViewer
        state = viewer.get_view_state()
        viewer.set_photo(hsi_utils.numpy_to_qpixmap(self._classification_rgb))
        if state is not None:
            viewer.queue_view_state(state)

    @staticmethod
    def _colorize_class_map(
        class_map: np.ndarray,
        class_ids: tuple[int, ...],
    ) -> NDArray[np.uint8]:
        """Map arbitrary class IDs to stable, evenly spaced display colors."""

        display_rgb = np.zeros((*class_map.shape, 3), dtype=np.uint8)
        for color_index, class_id in enumerate(class_ids):
            color = QtGui.QColor.fromHsvF(
                color_index / max(len(class_ids), 1),
                0.72,
                0.92,
            )
            display_rgb[class_map == class_id] = (
                color.red(),
                color.green(),
                color.blue(),
            )
        return display_rgb

    def _clear_classification_result(self) -> None:
        """Discard labels whenever the underlying cube geometry changes."""

        self._classification_result = None
        self._classification_rgb = None

    def _classification_is_running(self) -> bool:
        return (
            self._classification_thread is not None
            and self._classification_thread.isRunning()
        )

    # ------------------------------------------------------------------ #
    # Private: viewer signal handlers                                      #
    # ------------------------------------------------------------------ #

    def _on_spectrum_plot(self, pos: QPointF) -> None:
        if not self._hsi_data.is_loaded():
            return
        if self._classification_is_running():
            self.statusbar.showMessage(
                "Spectrum reads are unavailable while classification is running",
                5000,
            )
            return

        if self._super_res_worker is not None:
            self.statusbar.showMessage("Spectrum reads are paused during SR")
            return
        data = self._display_data()
        if self.sender() is self.superResViewer and not self.superResViewer.has_photo():
            return
        row, column = int(pos.y()), int(pos.x())
        if not (0 <= row < data.rows and 0 <= column < data.columns):
            return

        # A hypercube worker still in flight reads the same underlying
        # SpyFile handle; serialize this read behind it rather than risking
        # a concurrent read of a non-thread-safe file object.
        self._hypercube_controller.stop_and_wait()
        try:
            result = self._visualization_service.spectrum(data, row, column)
        except HSIError as exc:
            QMessageBox.critical(self, "Unable to plot spectrum", str(exc))
            return

        dialog = SpectrumDialog(result, parent=self)
        dialog.exec()

    def _on_mean_index(self, index_name: str) -> None:
        if self.sender() is self.superResViewer and self.highResButton.isChecked():
            self.statusbar.showMessage("Index means are available for the original image in Visualization", 5000)
            return
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
        dialog = IndexMeanDialog(
            index_name,
            mean_value,
            result.value_range,
            result.colormap,
            parent=self,
        )
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_crop_requested(self, rect: QtCore.QRectF) -> None:
        if self._super_res_worker is not None:
            self.statusbar.showMessage("Cancel or finish SR before cropping")
            return
        if self.highResButton.isChecked() and self._super_res_result is not None:
            # Every viewer displays the SR result at 2x; crop rectangles are
            # only valid against the Original the rest of the pipeline crops.
            self.statusbar.showMessage("Select Original before cropping, then rerun SR", 5000)
            return
        if not self._hsi_data.is_loaded():
            return
        if self._classification_is_running():
            self.statusbar.showMessage(
                "Cancel classification before changing the image crop",
                5000,
            )
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

        self._clear_classification_result()
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

    def _resolution_scale_for(self, viewer: HSIViewer) -> int:
        """Return 2 if `viewer` is currently showing the SR result, else 1.

        Visualization/Calibration/Classification track the same toggle as
        the Super-Resolution tab, so a tab switch can land on either side
        of a low/high-res swap independently of the Super-Resolution tab's
        own comparison view.
        """
        if viewer is self.superResViewer:
            return self._sr_view_scale
        return self._viz_view_scale

    def _on_tab_changed(self, index: int) -> None:
        new_viewer = self._viewer_for_tab(index)
        if new_viewer is None:
            return

        if self._active_viewer is not None and self._active_viewer is not new_viewer:
            state = self._active_viewer.get_view_state()
            if state is not None:
                source_scale = self._resolution_scale_for(self._active_viewer)
                target_scale = self._resolution_scale_for(new_viewer)
                factor = target_scale / source_scale
                new_viewer.queue_view_state((state[0] / factor, state[1] * factor))

        if (
            not self._high_res_notice_shown
            and self.tabWidget.widget(index) is self.Visualization
            and self.highResButton.isChecked()
            and self._super_res_result is not None
        ):
            self._high_res_notice_shown = True
            QMessageBox.information(
                self,
                "Viewing Super-Resolution image",
                "Visualization is now showing the Super-Resolution (high-res) "
                "result instead of the original image.",
            )

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
        self._clear_classification_result()
        self._push_image_to_viewers()

    def _undo_crop(self) -> None:
        if (
            self._super_res_worker is not None
            or self._classification_is_running()
            or not self._crop_undo_stack
        ):
            return
        self._crop_redo_stack.append(self._snapshot_current_state())
        self._restore_snapshot(self._crop_undo_stack.pop())
        self.statusbar.showMessage("Crop undone")

    def _redo_crop(self) -> None:
        if (
            self._super_res_worker is not None
            or self._classification_is_running()
            or not self._crop_redo_stack
        ):
            return
        self._crop_undo_stack.append(self._snapshot_current_state())
        self._restore_snapshot(self._crop_redo_stack.pop())
        self.statusbar.showMessage("Crop redone")

    def _push_image_to_viewers(self) -> None:
        self._reset_super_resolution()
        self._refresh_visualization_pipeline()

    def _refresh_visualization_pipeline(self) -> None:
        self._recompute_visualizations()
        self._refresh_viewers_display()
        self._hypercube_controller.refresh(self._display_data())
