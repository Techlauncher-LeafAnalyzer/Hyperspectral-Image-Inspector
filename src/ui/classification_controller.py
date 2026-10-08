from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Event
from pathlib import Path
from typing import Callable, Mapping, Optional, Union

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
    HSI_FILE_FILTER,
    SupervisedClassificationRequest,
    SupervisedClassificationResult,
    SupervisedClassifierType,
    TrainingFilePair,
    TrainingPairResolver,
    UnsupervisedClassificationRequest,
    UnsupervisedClassificationResult,
    VisualizationMode,
    VisualizationResult,
)
from ui.classification_colors import classification_palette
from ui.classification_layer_panel import ClassificationLayerPanel
from ui.theme import VIEWER_SCENE_BACKGROUND
from ui.viewer import HSIViewer, PixelValueEntry

_LAYER_COMPOSITE_BACKGROUND = (
    VIEWER_SCENE_BACKGROUND.red(),
    VIEWER_SCENE_BACKGROUND.green(),
    VIEWER_SCENE_BACKGROUND.blue(),
)
_OPACITY_REFRESH_INTERVAL_MS = 40

LOGGER = logging.getLogger(__name__)

_ClassificationResult = Union[UnsupervisedClassificationResult, SupervisedClassificationResult]


@dataclass
class _ClassificationSlot:
    """One resolution level's independent classification state.

    The Controller keeps one slot for the original image and one for the
    Super-Resolution result, keyed by the same boolean
    ``is_super_resolution_active()`` used to pick a data source, so
    classifying at one resolution never overwrites -- or gets shown against
    the wrong-shaped base image of -- a result already produced at the
    other.
    """

    result: Optional[_ClassificationResult] = None
    rgb: Optional[NDArray[np.uint8]] = None
    layers: Optional[ClassificationLayerModel] = None
    active_data: Optional[HSIData] = None
    means: Optional[dict[int, dict[str, PixelValueEntry]]] = None


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
    runningChanged = QtCore.pyqtSignal(bool)

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
        hyperspectral_button: QtWidgets.QPushButton,
        classifier_combo: QtWidgets.QComboBox,
        num_classes_edit: QtWidgets.QLineEdit,
        max_iterations_edit: QtWidgets.QLineEdit,
        groundtruth_path_edit: QtWidgets.QLineEdit,
        hyperspectral_path_edit: QtWidgets.QLineEdit,
        load_image_action: QtGui.QAction,
        stop_hypercube: Callable[[], None],
        parent_widget: QtWidgets.QWidget,
        *,
        is_calibrated: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__()
        self._display_data_provider = display_data_provider
        self._is_super_resolution_active = is_super_resolution_active
        self._is_calibrated = is_calibrated or (lambda: False)
        self._service = service
        self._training_pair_resolver = training_pair_resolver
        self._viewer = viewer
        self._layer_panel = layer_panel
        self._statusbar = statusbar
        self._unsupervised_button = unsupervised_button
        self._supervised_button = supervised_button
        self._groundtruth_button = groundtruth_button
        self._hyperspectral_button = hyperspectral_button
        self._classifier_combo = classifier_combo
        self._num_classes_edit = num_classes_edit
        self._max_iterations_edit = max_iterations_edit
        self._groundtruth_path_edit = groundtruth_path_edit
        self._hyperspectral_path_edit = hyperspectral_path_edit
        self._load_image_action = load_image_action
        self._stop_hypercube = stop_hypercube
        self._parent = parent_widget
        self._visualization_results: Mapping[
            VisualizationMode, VisualizationResult
        ] = {}

        self._slots: dict[bool, _ClassificationSlot] = {
            False: _ClassificationSlot(),
            True: _ClassificationSlot(),
        }
        self._pending_slot_key = False
        self._calibrated_slots = {False: _ClassificationSlot(), True: _ClassificationSlot()}
        self._pending_calibrated = False
        self._background_mode = VisualizationMode.RGB.value
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
        self._training_mask_path: Optional[Path] = None
        self._training_cube_path: Optional[Path] = None
        self._manual_cube_path: Optional[Path] = None
        self._close_after_classification = False

        self._configure_controls()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def _current_slot(self) -> _ClassificationSlot:
        """Return the slot matching whichever resolution is now displayed."""

        slots = self._calibrated_slots if self._is_calibrated() else self._slots
        return slots[self._is_super_resolution_active()]

    @property
    def rgb(self) -> Optional[NDArray[np.uint8]]:
        return self._current_slot.rgb

    @property
    def display_data(self) -> Optional[HSIData]:
        """Return the data source the current result was classified against."""

        return self._current_slot.active_data

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

        result = self._current_slot.result
        if result is None or not (
            0 <= row < result.class_map.shape[0]
            and 0 <= column < result.class_map.shape[1]
        ):
            return None
        class_id = int(result.class_map[row, column])
        return class_id if class_id >= 0 else None

    def is_running(self) -> bool:
        return self._thread is not None

    def set_image_loaded(self, loaded: bool) -> None:
        self._unsupervised_button.setEnabled(loaded)
        self._supervised_button.setEnabled(loaded)

    def clear_result(self) -> None:
        """Discard both resolutions' labels when the cube geometry changes."""

        self._slots = {False: _ClassificationSlot(), True: _ClassificationSlot()}
        self._calibrated_slots = {False: _ClassificationSlot(), True: _ClassificationSlot()}
        self._layer_panel.clear()

    def set_visualization_results(
        self, results: Mapping[VisualizationMode, VisualizationResult]
    ) -> None:
        """Use ``results`` (rendered from the displayed data) for class means.

        Call whenever the visualizations are recomputed; cached means are
        discarded and rebuilt on the next layer-panel refresh.
        """

        self._visualization_results = results
        for slot in (*self._slots.values(), *self._calibrated_slots.values()):
            slot.means = None
        slot = self._current_slot
        if slot.layers is not None:
            self._populate_layer_panel(slot)
            self._show_result()

    def clear_super_resolution_result(self) -> None:
        """Discard only the Super-Resolution slot, e.g. when its SR image is discarded.

        Used when the high-res Super-Resolution result is thrown away (such as
        before re-calibrating), leaving the original-resolution slot unaffected
        and still valid.
        """

        self._slots[True] = _ClassificationSlot()
        self._calibrated_slots[True] = _ClassificationSlot()

    def refresh_display(self) -> None:
        """Show the layer panel for whichever classification result is active.

        Call this whenever the Super-Resolution low/high toggle changes.
        The classification pixmap itself needs no separate action here:
        ``composited_rgb()`` already reads the slot matching the *current*
        toggle state, so the caller's own viewer refresh
        (``MainWindowController._refresh_viewers_display``) picks up the
        right image on its own -- or falls back to the plain photo when
        this resolution has no classification result yet. Only the layer
        panel, which this Controller owns, needs this explicit sync.
        """

        slot = self._current_slot
        if slot.layers is not None:
            self._populate_layer_panel(slot)
        else:
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
        self._hyperspectral_button.clicked.connect(self._on_select_hyperspectral_clicked)
        self._supervised_button.clicked.connect(self._on_supervised_classify_clicked)
        self._unsupervised_button.clicked.connect(self._on_unsupervised_classify_clicked)
        self._layer_panel.visibilityChanged.connect(self._on_layer_visibility_changed)
        self._layer_panel.opacityChanged.connect(self._on_layer_opacity_changed)
        self._layer_panel.setAllVisibleRequested.connect(self._on_set_all_visible_requested)
        self._layer_panel.globalOpacityChanged.connect(self._on_global_opacity_changed)
        self._layer_panel.outlineModeChanged.connect(self._on_outline_mode_changed)
        self._layer_panel.backgroundChanged.connect(self._on_background_changed)

        self._num_classes_edit.setValidator(QtGui.QIntValidator(2, 65535, self))
        self._max_iterations_edit.setValidator(QtGui.QIntValidator(1, 10000, self))
        self._num_classes_edit.setText("5")
        self._max_iterations_edit.setText("20")

        # Keep the selector driven by the Model enum while using readable
        # labels in the UI. Classification requests still use the item data.
        self._classifier_combo.clear()
        classifier_labels = {
            SupervisedClassifierType.GAUSSIAN: "Gaussian",
            SupervisedClassifierType.MAHALANOBIS: "Mahalanobis distance",
        }
        for classifier in SupervisedClassifierType:
            self._classifier_combo.addItem(
                classifier_labels.get(classifier, classifier.value), classifier
            )
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
        self._hyperspectral_button.setToolTip(
            "Choose a training cube manually if automatic pairing is missing or wrong"
        )
        self._hyperspectral_path_edit.setPlaceholderText(
            "Paired automatically, or select a hyperspectral image"
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
            mask_path = self._training_pair_resolver.validate_mask_path(mask_path_str)
        except ClassificationError as exc:
            QMessageBox.critical(self._parent, "Invalid ground-truth mask", str(exc))
            return

        preselected_cube = self._manual_cube_path
        self._training_mask_path = mask_path
        self._groundtruth_path_edit.setText(str(mask_path))
        self._groundtruth_path_edit.setToolTip(str(mask_path))
        if preselected_cube is not None:
            try:
                pair = self._training_pair_resolver.resolve_manual(
                    mask_path, preselected_cube
                )
            except ClassificationError:
                self._set_training_cube(None)
            else:
                self._set_training_cube(pair.cube_path, pair=pair)
                self._statusbar.showMessage(
                    f"Using manually selected training cube {pair.cube_path.name}",
                    8000,
                )
                return
        try:
            pair = self._training_pair_resolver.resolve(mask_path)
        except ClassificationError:
            self._set_training_cube(None)
            self._statusbar.showMessage(
                "No matching hyperspectral image found; select one manually",
                8000,
            )
        else:
            self._set_training_cube(pair.cube_path, pair=pair)
            self._statusbar.showMessage(
                f"Training mask paired with {pair.cube_path.name}", 8000
            )

    def _on_select_hyperspectral_clicked(self) -> None:
        if self.is_running():
            QMessageBox.information(
                self._parent,
                "Classification in progress",
                "Cancel the current classification before changing training data.",
            )
            return
        cube_path_str, _ = QFileDialog.getOpenFileName(
            self._parent,
            "Open Training Hyperspectral Image",
            "",
            HSI_FILE_FILTER,
        )
        if not cube_path_str:
            return
        try:
            if self._training_mask_path is None:
                pair = None
                cube_path = self._training_pair_resolver.validate_cube_path(cube_path_str)
            else:
                pair = self._training_pair_resolver.resolve_manual(
                    self._training_mask_path, cube_path_str
                )
                cube_path = pair.cube_path
        except ClassificationError as exc:
            QMessageBox.critical(self._parent, "Invalid hyperspectral image", str(exc))
            return
        self._manual_cube_path = cube_path
        self._set_training_cube(cube_path, pair=pair)
        if pair is None:
            message = "Training cube selected; choose a ground-truth mask"
        else:
            message = f"Using manually selected training cube {cube_path.name}"
        self._statusbar.showMessage(message, 8000)

    def _set_training_cube(
        self, cube_path: Path | None, *, pair: TrainingFilePair | None = None
    ) -> None:
        self._training_cube_path = cube_path
        self._training_pair = pair
        self._hyperspectral_path_edit.setText(str(cube_path) if cube_path else "")
        self._hyperspectral_path_edit.setToolTip(str(cube_path) if cube_path else "")

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

        self._pending_slot_key = self._is_super_resolution_active()
        self._pending_calibrated = self._is_calibrated()
        self._current_slot.active_data = data
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
            if self._training_mask_path is None:
                title = "Training mask required"
                message = "Select a ground-truth mask first."
            else:
                title = "Training hyperspectral image required"
                message = "Select the hyperspectral image paired with this mask."
            QMessageBox.critical(self._parent, title, message)
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

        self._pending_slot_key = self._is_super_resolution_active()
        self._pending_calibrated = self._is_calibrated()
        self._current_slot.active_data = data
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
        self._hyperspectral_button.setEnabled(False)
        self._classifier_combo.setEnabled(False)
        self._load_image_action.setEnabled(False)
        self._unsupervised_button.setEnabled(active_button is self._unsupervised_button)
        self._supervised_button.setEnabled(active_button is self._supervised_button)
        active_button.setText("Cancel")
        active_button.setToolTip(
            "Request cancellation after the current classification stage"
        )
        self._statusbar.showMessage(starting_message)
        self.runningChanged.emit(True)
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
        # Target the slot captured when this job launched, not whichever
        # resolution happens to be on screen now -- the toggle may have
        # flipped while the worker was still running.
        slot_key = self._pending_slot_key
        slots = self._calibrated_slots if self._pending_calibrated else self._slots
        slot = slots[slot_key]
        if slot.active_data is None:
            slot.active_data = self._display_data_provider()
        slot.result = result
        class_ids = (
            result.class_ids
            if isinstance(result, SupervisedClassificationResult)
            else tuple(range(result.n_classes))
        )
        slot.rgb = self._colorize_class_map(result.class_map, class_ids)
        slot.layers = ClassificationLayerModel(result)
        if slot_key == self._is_super_resolution_active():
            self._populate_layer_panel(slot)
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
        self._hyperspectral_button.setEnabled(True)
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
            "Classify from the selected ground-truth mask and training cube"
        )

    @QtCore.pyqtSlot()
    def _clear_worker(self) -> None:
        self._worker = None
        self._thread = None
        self._active_button = None
        self.runningChanged.emit(False)
        if self._close_after_classification:
            self._close_after_classification = False
            QtCore.QTimer.singleShot(0, self.readyToClose.emit)

    @QtCore.pyqtSlot(int, bool)
    def _on_layer_visibility_changed(self, class_id: int, visible: bool) -> None:
        layers = self._current_slot.layers
        if layers is None:
            return
        try:
            layers.set_class_visible(class_id, visible)
        except ClassificationError as exc:
            self._statusbar.showMessage(str(exc), 8000)
            return
        self._show_result()

    @QtCore.pyqtSlot(int, float)
    def _on_layer_opacity_changed(self, class_id: int, opacity: float) -> None:
        layers = self._current_slot.layers
        if layers is None:
            return
        try:
            layers.set_class_opacity(class_id, opacity)
        except ClassificationError as exc:
            self._statusbar.showMessage(str(exc), 8000)
            return
        self._opacity_refresh_timer.start()

    @QtCore.pyqtSlot(bool)
    def _on_set_all_visible_requested(self, visible: bool) -> None:
        layers = self._current_slot.layers
        if layers is None:
            return
        layers.set_all_visible(visible)
        self._populate_layer_panel(self._current_slot)
        self._show_result()

    @QtCore.pyqtSlot(float)
    def _on_global_opacity_changed(self, opacity: float) -> None:
        layers = self._current_slot.layers
        if layers is None:
            return
        try:
            layers.set_global_opacity(opacity)
        except ClassificationError as exc:
            self._statusbar.showMessage(str(exc), 8000)
            return
        self._opacity_refresh_timer.start()

    @QtCore.pyqtSlot(bool)
    def _on_outline_mode_changed(self, enabled: bool) -> None:
        layers = self._current_slot.layers
        if layers is None:
            return
        layers.set_outline_mode(enabled)
        self._show_result()

    def _on_background_changed(self, mode: str) -> None:
        self._background_mode = mode
        self._show_result()

    def _background_names(self) -> tuple[str, ...]:
        names = [VisualizationMode.RGB.value]
        names += [
            mode.value
            for mode in self._visualization_results
            if mode is not VisualizationMode.RGB
        ]
        return tuple(names)

    def _background_rgb(self, slot: _ClassificationSlot) -> Optional[NDArray[np.uint8]]:
        """Return the selected visualization's image, else the plain RGB."""

        base_rgb = slot.active_data.rgb_array if slot.active_data is not None else None
        if self._background_mode != VisualizationMode.RGB.value:
            for mode, visualization in self._visualization_results.items():
                if (
                    mode.value == self._background_mode
                    and slot.layers is not None
                    and visualization.display_rgb.shape == (*slot.layers.image_shape, 3)
                ):
                    return visualization.display_rgb
        return base_rgb

    def _populate_layer_panel(self, slot: _ClassificationSlot) -> None:
        if slot.layers is None:
            return
        names = self._background_names()
        if self._background_mode not in names:
            self._background_mode = VisualizationMode.RGB.value
        self._layer_panel.set_backgrounds(names, self._background_mode)
        self._layer_panel.set_layers(
            slot.layers.layers,
            global_opacity=slot.layers.global_opacity,
            outline_mode=slot.layers.outline_mode,
            means=self._segment_means(slot),
        )

    def _segment_means(
        self, slot: _ClassificationSlot
    ) -> dict[int, dict[str, PixelValueEntry]]:
        """Return (and cache) each class's mean for every cached visualization."""

        if slot.means is not None:
            return slot.means
        means: dict[int, dict[str, PixelValueEntry]] = {}
        for mode, visualization in self._visualization_results.items():
            try:
                class_means = slot.layers.visualization_means(visualization)
            except ClassificationError as exc:
                LOGGER.info("Skipping %s class means: %s", mode.value, exc)
                continue
            for class_id, mean in class_means.items():
                means.setdefault(class_id, {})[mode.value] = PixelValueEntry(
                    value=mean.value, color=mean.color
                )
        slot.means = means
        return means

    def _composite(self) -> Optional[ClassificationLayerComposite]:
        slot = self._current_slot
        if slot.rgb is None or slot.layers is None:
            return None
        base_rgb = self._background_rgb(slot)
        if base_rgb is not None and base_rgb.shape == (*slot.layers.image_shape, 3):
            return slot.layers.compose_display(slot.rgb, base_rgb=base_rgb)
        return slot.layers.compose_display(
            slot.rgb, background_color=_LAYER_COMPOSITE_BACKGROUND
        )

    def _show_result(self) -> None:
        composite_rgb = self.composited_rgb()
        if composite_rgb is None:
            return
        state = self._viewer.get_view_state()
        active_data = self._current_slot.active_data
        self._viewer.set_photo(
            hsi_utils.numpy_to_qpixmap(
                composite_rgb,
                active_data.roi_mask if active_data is not None else None,
            )
        )
        if state is not None:
            self._viewer.queue_view_state(state)

    @staticmethod
    def _colorize_class_map(
        class_map: np.ndarray,
        class_ids: tuple,
    ) -> NDArray[np.uint8]:
        """Map arbitrary class IDs to stable, evenly spaced display colors."""

        display_rgb = np.zeros((*class_map.shape, 3), dtype=np.uint8)
        for class_id, color in classification_palette(class_ids).items():
            display_rgb[class_map == class_id] = color
        return display_rgb
