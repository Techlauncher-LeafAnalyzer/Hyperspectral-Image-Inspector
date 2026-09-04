from __future__ import annotations

import logging
from threading import Event
from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import NDArray
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QFileDialog, QMessageBox

import core.hsi_utils as hsi_utils
from core import (
    CancelledError,
    ClassificationError,
    ClassificationLayerComposite,
    ClassificationLayerModel,
    ClassificationService,
    HSIData,
    HSIError,
    HSIReader,
    SupervisedClassificationRequest,
    SupervisedClassificationResult,
    SupervisedClassifierType,
    TrainingFilePair,
    TrainingPairResolver,
    UnsupervisedClassificationRequest,
    UnsupervisedClassificationResult,
)
from ui.classification_layer_panel import ClassificationLayerPanel
from ui.theme import VIEWER_SCENE_BACKGROUND
from ui.viewer import HSIViewer

_LAYER_COMPOSITE_BACKGROUND = (
    VIEWER_SCENE_BACKGROUND.red(),
    VIEWER_SCENE_BACKGROUND.green(),
    VIEWER_SCENE_BACKGROUND.blue(),
)
_OPACITY_REFRESH_INTERVAL_MS = 40

LOGGER = logging.getLogger(__name__)

_ClassificationResult = Union[UnsupervisedClassificationResult, SupervisedClassificationResult]


class _ClassificationWorker(QtCore.QObject):
    """Run the synchronous K-means classification Model away from the GUI thread."""

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


class ClassificationController(QObject):
    """Owns the Classification tab's worker lifecycle and result state.

    Wires the unsupervised/supervised controls, the ground-truth picker, and
    the classification viewer to background workers. Kept separate from
    ``MainWindowController`` for the same reason as ``HypercubeController``:
    all classification-specific state (workers, thread lifecycle, colorized
    result) lives in one place rather than mixed into the rest of the
    application's wiring.
    """

    readyToClose = QtCore.pyqtSignal()

    def __init__(
        self,
        display_data_provider: Callable[[], HSIData],
        is_super_resolution_active: Callable[[], bool],
        service: ClassificationService,
        training_pair_resolver: TrainingPairResolver,
        viewer: HSIViewer,
        layer_panel: ClassificationLayerPanel,
        statusbar: QtWidgets.QStatusBar,
        unsupervised_button: QtWidgets.QPushButton,
        supervised_button: QtWidgets.QPushButton,
        groundtruth_button: QtWidgets.QPushButton,
        classifier_combo: QtWidgets.QComboBox,
        num_classes_edit: QtWidgets.QLineEdit,
        max_iterations_edit: QtWidgets.QLineEdit,
        groundtruth_path_edit: QtWidgets.QLineEdit,
        load_image_action: QtGui.QAction,
        stop_hypercube: Callable[[], None],
        parent_widget: QtWidgets.QWidget,
    ) -> None:
        super().__init__()
        self._display_data_provider = display_data_provider
        self._is_super_resolution_active = is_super_resolution_active
        self._service = service
        self._training_pair_resolver = training_pair_resolver
        self._viewer = viewer
        self._layer_panel = layer_panel
        self._statusbar = statusbar
        self._unsupervised_button = unsupervised_button
        self._supervised_button = supervised_button
        self._groundtruth_button = groundtruth_button
        self._classifier_combo = classifier_combo
        self._num_classes_edit = num_classes_edit
        self._max_iterations_edit = max_iterations_edit
        self._groundtruth_path_edit = groundtruth_path_edit
        self._load_image_action = load_image_action
        self._stop_hypercube = stop_hypercube
        self._parent = parent_widget

        self._result: Optional[_ClassificationResult] = None
        self._rgb: Optional[NDArray[np.uint8]] = None
        self._layers: Optional[ClassificationLayerModel] = None
        self._active_data: Optional[HSIData] = None
        self._sr_notice_shown = False
        self._opacity_refresh_timer = QtCore.QTimer(self)
        self._opacity_refresh_timer.setSingleShot(True)
        self._opacity_refresh_timer.setInterval(_OPACITY_REFRESH_INTERVAL_MS)
        self._opacity_refresh_timer.timeout.connect(self._show_result)
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[
            Union[_ClassificationWorker, _SupervisedClassificationWorker]
        ] = None
        self._active_button: Optional[QtWidgets.QPushButton] = None
        self._training_pair: Optional[TrainingFilePair] = None
        self._close_after_classification = False

        self._configure_controls()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def rgb(self) -> Optional[NDArray[np.uint8]]:
        return self._rgb

    @property
    def display_data(self) -> Optional[HSIData]:
        """Return the data source the current result was classified against."""

        return self._active_data

    def composited_rgb(self) -> Optional[NDArray[np.uint8]]:
        """Return the current layer-composited display image, if any result exists.

        This is what the classification viewer actually shows -- it reflects
        per-class visibility/opacity, unlike :attr:`rgb`. Other tabs (saving
        the active viewer, refreshing after a Super-Resolution toggle) must
        read this instead of :attr:`rgb` to avoid showing a stale, flattened
        image.
        """

        composite = self._composite()
        return composite.display_rgb if composite is not None else None

    def class_id_at(self, row: int, column: int) -> Optional[int]:
        """Return the zero-based class ID at ``(row, column)``, if any."""

        result = self._result
        if result is None or not (
            0 <= row < result.class_map.shape[0]
            and 0 <= column < result.class_map.shape[1]
        ):
            return None
        return int(result.class_map[row, column])

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def set_image_loaded(self, loaded: bool) -> None:
        self._unsupervised_button.setEnabled(loaded)
        self._supervised_button.setEnabled(loaded)

    def clear_result(self) -> None:
        """Discard labels whenever the underlying cube geometry changes."""

        self._result = None
        self._rgb = None
        self._layers = None
        self._active_data = None
        self._layer_panel.clear()

    def request_close(self) -> bool:
        """Begin cancelling a running classification for an application close.

        Returns ``True`` if the caller must defer closing until
        ``readyToClose`` fires, or ``False`` if nothing was running.
        """

        if not self.is_running():
            return False
        assert self._worker is not None
        self._worker.cancel()
        self._close_after_classification = True
        if self._active_button is not None:
            self._active_button.setEnabled(False)
            self._active_button.setText("Cancelling…")
        self._statusbar.showMessage(
            "Cancelling classification before closing the application…"
        )
        return True

    # ------------------------------------------------------------------ #
    # Private: initial UI wiring                                          #
    # ------------------------------------------------------------------ #

    def _configure_controls(self) -> None:
        self._groundtruth_button.clicked.connect(self._on_select_groundtruth_clicked)
        self._supervised_button.clicked.connect(self._on_supervised_classify_clicked)
        self._unsupervised_button.clicked.connect(self._on_unsupervised_classify_clicked)
        self._layer_panel.visibilityChanged.connect(self._on_layer_visibility_changed)
        self._layer_panel.opacityChanged.connect(self._on_layer_opacity_changed)
        self._layer_panel.setAllVisibleRequested.connect(self._on_set_all_visible_requested)
        self._layer_panel.globalOpacityChanged.connect(self._on_global_opacity_changed)
        self._layer_panel.outlineModeChanged.connect(self._on_outline_mode_changed)

        self._num_classes_edit.setValidator(QtGui.QIntValidator(2, 65535, self))
        self._max_iterations_edit.setValidator(QtGui.QIntValidator(1, 10000, self))
        self._num_classes_edit.setText("5")
        self._max_iterations_edit.setText("20")

        # Keep the View's existing selector driven by the Model enum. Storing
        # the enum as item data means the Controller does not depend on the
        # user-visible label when it builds a classification request.
        self._classifier_combo.clear()
        for classifier in SupervisedClassifierType:
            self._classifier_combo.addItem(classifier.value, classifier)
        self._classifier_combo.setToolTip(
            "Choose the Spectral Python classifier used for one-example transfer"
        )
        self._unsupervised_button.setToolTip(
            "Load an image, then group pixels by spectral similarity with K-means"
        )
        self._supervised_button.setToolTip(
            "Classify the current image from the selected reference mask and cube"
        )
        self._groundtruth_button.setText("Ground-truth Mask")
        self._groundtruth_button.setToolTip(
            "Select a mask; its hyperspectral cube is paired automatically by name"
        )
        self._groundtruth_path_edit.setPlaceholderText(
            "Select mask; matching cube is detected automatically"
        )

    # ------------------------------------------------------------------ #
    # Private: ground-truth file selection                                #
    # ------------------------------------------------------------------ #

    def _on_select_groundtruth_clicked(self) -> None:
        if self.is_running():
            QMessageBox.information(
                self._parent,
                "Classification in progress",
                "Cancel the current classification before changing training data.",
            )
            return
        mask_path_str, _ = QFileDialog.getOpenFileName(
            self._parent,
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
            self._training_pair = None
            self._groundtruth_path_edit.clear()
            self._groundtruth_path_edit.setToolTip("")
            QMessageBox.critical(self._parent, "Invalid training-file pair", str(exc))
            self._statusbar.showMessage("Training-file pairing failed", 8000)
            return

        self._training_pair = pair
        self._groundtruth_path_edit.setText(str(pair.mask_path))
        self._groundtruth_path_edit.setToolTip(
            f"Mask: {pair.mask_path}\nHyperspectral cube: {pair.cube_path}"
        )
        self._statusbar.showMessage(
            f"Training mask paired with {pair.cube_path.name}",
            8000,
        )

    # ------------------------------------------------------------------ #
    # Private: unsupervised / supervised classification                   #
    # ------------------------------------------------------------------ #

    def _on_unsupervised_classify_clicked(self) -> None:
        """Start K-means, or turn the same button into a cancellation action."""

        if self.is_running():
            self._cancel_active()
            return
        data = self._display_data_provider()
        if not data.is_loaded():
            QMessageBox.information(self._parent, "Nothing to classify", "Load an image first.")
            return

        try:
            request = UnsupervisedClassificationRequest(
                n_classes=int(self._num_classes_edit.text()),
                max_iterations=int(self._max_iterations_edit.text()),
            )
        except ValueError:
            QMessageBox.critical(
                self._parent,
                "Invalid classification settings",
                "Enter whole numbers for classes and maximum iterations.",
            )
            return
        except ClassificationError as exc:
            QMessageBox.critical(self._parent, "Invalid classification settings", str(exc))
            return

        estimate = self._service.estimate_kmeans_working_bytes(data, request)
        if estimate >= 1_000_000_000 and not self._confirm_large_job(estimate):
            return

        self._notify_if_classifying_super_resolution()
        self._active_data = data
        worker = _ClassificationWorker(self._service, data, request)
        self._launch_worker(
            worker, self._unsupervised_button, "Starting K-means classification…"
        )

    def _on_supervised_classify_clicked(self) -> None:
        """Train from the automatically paired example and classify the target."""

        if self.is_running():
            self._cancel_active()
            return
        data = self._display_data_provider()
        if not data.is_loaded():
            QMessageBox.information(self._parent, "Nothing to classify", "Load an image first.")
            return
        if self._training_pair is None:
            QMessageBox.critical(
                self._parent,
                "Training mask required",
                (
                    "Select a ground-truth mask first. The program will look beside "
                    "it for a same-base hyperspectral cube or an "
                    "_hyperspectral cube pair."
                ),
            )
            return
        classifier = self._classifier_combo.currentData()
        if not isinstance(classifier, SupervisedClassifierType):
            QMessageBox.critical(
                self._parent,
                "Unsupported supervised classifier",
                "Select a supported supervised classifier and try again.",
            )
            return
        request = SupervisedClassificationRequest(classifier)

        self._notify_if_classifying_super_resolution()
        self._active_data = data
        worker = _SupervisedClassificationWorker(
            self._service,
            data,
            self._training_pair,
            request,
        )
        self._launch_worker(
            worker,
            self._supervised_button,
            f"Starting {request.classifier.value} classification…",
        )

    def _cancel_active(self) -> None:
        assert self._worker is not None
        self._worker.cancel()
        if self._active_button is not None:
            self._active_button.setEnabled(False)
            self._active_button.setText("Cancelling…")
        self._statusbar.showMessage(
            "Cancelling after the current classification stage…"
        )

    def _notify_if_classifying_super_resolution(self) -> None:
        """Tell the user, once per session, that classification targets the SR result.

        Mirrors ``MainWindowController``'s one-time "Viewing Super-Resolution
        image" notice for the Visualization tab.
        """

        if self._sr_notice_shown or not self._is_super_resolution_active():
            return
        self._sr_notice_shown = True
        QMessageBox.information(
            self._parent,
            "Classifying Super-Resolution image",
            "Classification is running on the Super-Resolution (high-res) "
            "result instead of the original image.",
        )

    def _confirm_large_job(self, estimated_bytes: int) -> bool:
        gibibytes = estimated_bytes / (1024 ** 3)
        answer = QMessageBox.question(
            self._parent,
            "Large classification job",
            (
                f"K-means may require about {gibibytes:.1f} GiB of working memory. "
                "Cropping the image first is recommended. Continue anyway?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _launch_worker(
        self,
        worker: Union[_ClassificationWorker, _SupervisedClassificationWorker],
        active_button: QtWidgets.QPushButton,
        starting_message: str,
    ) -> None:
        """Create one worker/thread pair; all View updates stay on the GUI thread."""

        self._stop_hypercube()
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progressChanged.connect(self._on_progress)
        worker.resultReady.connect(self._on_result)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        worker.finished.connect(self._on_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_worker)

        self._thread = thread
        self._worker = worker
        self._active_button = active_button
        self._num_classes_edit.setEnabled(False)
        self._max_iterations_edit.setEnabled(False)
        self._groundtruth_button.setEnabled(False)
        self._classifier_combo.setEnabled(False)
        self._load_image_action.setEnabled(False)
        self._unsupervised_button.setEnabled(active_button is self._unsupervised_button)
        self._supervised_button.setEnabled(active_button is self._supervised_button)
        active_button.setText("Cancel")
        active_button.setToolTip(
            "Request cancellation after the current classification stage"
        )
        self._statusbar.showMessage(starting_message)
        thread.start()

    @QtCore.pyqtSlot(int, str)
    def _on_progress(self, value: int, message: str) -> None:
        self._statusbar.showMessage(f"Classification {value}% — {message}")

    @QtCore.pyqtSlot(object)
    def _on_result(self, result: object) -> None:
        if not isinstance(
            result,
            (UnsupervisedClassificationResult, SupervisedClassificationResult),
        ):
            self._on_failed("Worker returned an invalid result.")
            return
        if self._active_data is None:
            self._active_data = self._display_data_provider()
        self._result = result
        class_ids = (
            result.class_ids
            if isinstance(result, SupervisedClassificationResult)
            else tuple(range(result.n_classes))
        )
        self._rgb = self._colorize_class_map(result.class_map, class_ids)
        self._layers = ClassificationLayerModel(result)
        self._layer_panel.set_layers(self._layers.layers)
        self._show_result()
        populated = int(np.count_nonzero(result.class_pixel_counts))
        operation = (
            result.classifier.value
            if isinstance(result, SupervisedClassificationResult)
            else "K-means"
        )
        self._statusbar.showMessage(
            f"{operation} complete: {populated}/{result.n_classes} populated classes",
            8000,
        )

    @QtCore.pyqtSlot(str)
    def _on_failed(self, message: str) -> None:
        QMessageBox.critical(self._parent, "Classification failed", message)
        self._statusbar.showMessage("Classification failed", 8000)

    @QtCore.pyqtSlot()
    def _on_cancelled(self) -> None:
        self._statusbar.showMessage("Classification cancelled", 5000)

    @QtCore.pyqtSlot()
    def _on_finished(self) -> None:
        self._num_classes_edit.setEnabled(True)
        self._max_iterations_edit.setEnabled(True)
        self._groundtruth_button.setEnabled(True)
        self._classifier_combo.setEnabled(True)
        self._load_image_action.setEnabled(True)
        loaded = self._display_data_provider().is_loaded()
        self._unsupervised_button.setEnabled(loaded)
        self._supervised_button.setEnabled(loaded)
        self._unsupervised_button.setText("Classify")
        self._supervised_button.setText("Classify")
        self._unsupervised_button.setToolTip(
            "Group pixels by spectral similarity with K-means"
        )
        self._supervised_button.setToolTip(
            "Classify from the selected reference mask and paired cube"
        )

    @QtCore.pyqtSlot()
    def _clear_worker(self) -> None:
        self._worker = None
        self._thread = None
        self._active_button = None
        if self._close_after_classification:
            self._close_after_classification = False
            QtCore.QTimer.singleShot(0, self.readyToClose.emit)

    @QtCore.pyqtSlot(int, bool)
    def _on_layer_visibility_changed(self, class_id: int, visible: bool) -> None:
        if self._layers is None:
            return
        try:
            self._layers.set_class_visible(class_id, visible)
        except ClassificationError as exc:
            self._statusbar.showMessage(str(exc), 8000)
            return
        self._show_result()

    @QtCore.pyqtSlot(int, float)
    def _on_layer_opacity_changed(self, class_id: int, opacity: float) -> None:
        if self._layers is None:
            return
        try:
            self._layers.set_class_opacity(class_id, opacity)
        except ClassificationError as exc:
            self._statusbar.showMessage(str(exc), 8000)
            return
        self._opacity_refresh_timer.start()

    @QtCore.pyqtSlot(bool)
    def _on_set_all_visible_requested(self, visible: bool) -> None:
        if self._layers is None:
            return
        self._layers.set_all_visible(visible)
        self._layer_panel.set_layers(self._layers.layers)
        self._show_result()

    @QtCore.pyqtSlot(float)
    def _on_global_opacity_changed(self, opacity: float) -> None:
        if self._layers is None:
            return
        try:
            self._layers.set_global_opacity(opacity)
        except ClassificationError as exc:
            self._statusbar.showMessage(str(exc), 8000)
            return
        self._opacity_refresh_timer.start()

    @QtCore.pyqtSlot(bool)
    def _on_outline_mode_changed(self, enabled: bool) -> None:
        if self._layers is None:
            return
        self._layers.set_outline_mode(enabled)
        self._show_result()

    def _composite(self) -> Optional[ClassificationLayerComposite]:
        if self._rgb is None or self._layers is None:
            return None
        base_rgb = self._active_data.rgb_array if self._active_data is not None else None
        if base_rgb is not None and base_rgb.shape == (*self._layers.image_shape, 3):
            return self._layers.compose_display(self._rgb, base_rgb=base_rgb)
        return self._layers.compose_display(
            self._rgb, background_color=_LAYER_COMPOSITE_BACKGROUND
        )

    def _show_result(self) -> None:
        composite_rgb = self.composited_rgb()
        if composite_rgb is None:
            return
        state = self._viewer.get_view_state()
        self._viewer.set_photo(hsi_utils.numpy_to_qpixmap(composite_rgb))
        if state is not None:
            self._viewer.queue_view_state(state)

    @staticmethod
    def _colorize_class_map(
        class_map: np.ndarray,
        class_ids: tuple,
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
