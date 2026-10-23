from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
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
    AnnotationError,
    AnnotationKind,
    AnnotationLayerModel,
    AnnotationStyle,
    CalibrationError,
    CalibrationFrameResolver,
    CalibrationResult,
    CalibrationService,
    CancelledError,
    ClassIndexStatistics,
    ClassificationError,
    ClassificationLayerComposite,
    ClassificationLayerModel,
    ClassificationService,
    ExportRegion,
    LeafLabel,
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
    export_class_summary_csv,
    draw_leaf_labels,
    measure_region,
    region_mask,
    place_leaf_labels,
    separate_leaves,
)
from ui.annotation_render import render_annotations
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
_REGION_REFRESH_INTERVAL_MS = 30

LOGGER = logging.getLogger(__name__)

_ClassificationResult = Union[UnsupervisedClassificationResult, SupervisedClassificationResult]


@dataclass(frozen=True)
class AnalysisRegion:
    """A class or annotation region offered to full-spectrum analysis."""

    label: str
    mask: np.ndarray
    color: tuple[int, int, int]
    selected: bool


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
    leaves_separated: bool = False
    fixed_colors: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    # Each separated leaf's number, drawn beneath it on the canvas and in
    # saved images; empty until leaves are separated.
    leaf_labels: tuple[LeafLabel, ...] = ()


@dataclass
class _AnnotationSlot:
    """Annotation stack for one displayed image.

    Annotations use image-pixel coordinates, so each resolution/calibration
    state keeps its own stack rather than sharing mismatched coordinates.
    """

    layers: AnnotationLayerModel = field(default_factory=AnnotationLayerModel)
    selected_id: Optional[int] = None


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
        *,
        target_is_calibrated: bool = False,
    ) -> None:
        super().__init__()
        self._service = service
        self._target_data = target_data
        self._training_pair = training_pair
        self._request = request
        self._target_is_calibrated = target_is_calibrated
        self._cancel_requested = Event()

    def _calibrate_training(
        self, training_data: HSIData
    ) -> tuple[HSIData, Optional[CalibrationResult], SupervisedClassificationRequest]:
        """Put the training cube in the target's calibrated units.

        Returns the cube to train on, the calibration result that owns its
        temporary files (``None`` when uncalibrated), and the request to use.
        Without a usable dark/bright pair the cube stays raw and the request
        switches to per-band standardization, which removes the unit gap.
        """

        fallback = replace(self._request, standardize_features=True)
        pair = CalibrationFrameResolver().resolve(training_data.source_path)
        if pair is None:
            self.progressChanged.emit(
                0, "No calibration frames for training image; standardizing features"
            )
            return training_data, None, fallback
        reader = HSIReader()
        dark = bright = None
        try:
            self.progressChanged.emit(0, "Calibrating training image")
            dark = reader.open(pair.dark_path)
            bright = reader.open(pair.bright_path)
            result = CalibrationService().calibrate(
                training_data,
                dark,
                bright,
                is_cancelled=self._cancel_requested.is_set,
            )
        except CalibrationError as exc:
            LOGGER.warning("Training calibration failed, standardizing instead: %s", exc)
            self.progressChanged.emit(
                0, "Training calibration failed; standardizing features"
            )
            return training_data, None, fallback
        finally:
            for frame in (dark, bright):
                if frame is not None:
                    frame.close()
        return result.data, result, self._request

    @QtCore.pyqtSlot()
    def run(self) -> None:
        training_data: Optional[HSIData] = None
        calibration: Optional[CalibrationResult] = None
        try:
            training_data = HSIReader().open(self._training_pair.cube_path)
            request = self._request
            cube = training_data
            if self._target_is_calibrated:
                cube, calibration, request = self._calibrate_training(training_data)
            result = self._service.classify_supervised(
                self._target_data,
                cube,
                self._training_pair.mask_path,
                request,
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
            if calibration is not None:
                calibration.cleanup()
            if training_data is not None:
                training_data.close()
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
        separate_leaves_button: QtWidgets.QPushButton | None = None,
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
        self._separate_leaves_button = separate_leaves_button
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
        self._annotation_slots: dict[tuple[bool, bool], _AnnotationSlot] = {}
        self._opacity_refresh_timer = QtCore.QTimer(self)
        self._opacity_refresh_timer.setSingleShot(True)
        self._opacity_refresh_timer.setInterval(_OPACITY_REFRESH_INTERVAL_MS)
        self._opacity_refresh_timer.timeout.connect(self._show_result)
        self._region_refresh_timer = QtCore.QTimer(self)
        self._region_refresh_timer.setSingleShot(True)
        self._region_refresh_timer.setInterval(_REGION_REFRESH_INTERVAL_MS)
        self._region_refresh_timer.timeout.connect(self._refresh_regions)
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
    def _annotation_slot(self) -> _AnnotationSlot:
        key = (self._is_calibrated(), self._is_super_resolution_active())
        return self._annotation_slots.setdefault(key, _AnnotationSlot())

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
        if composite is None:
            return None
        slot = self._current_slot
        if not slot.leaf_labels or slot.layers is None:
            return composite.display_rgb
        return draw_leaf_labels(
            composite.display_rgb, slot.leaf_labels, slot.layers.visible_class_ids
        )

    def with_annotations(self, rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Draw the displayed image's visible annotations onto ``rgb`` for export.

        Annotations are kept out of :meth:`composited_rgb` because the viewer
        draws them itself, as editable items; baking them in there would show
        them twice. Only an export should flatten them into pixels.
        """

        return render_annotations(rgb, self._annotation_slot.layers.layers)

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

    def analysis_regions(self) -> Optional[tuple[HSIData, dict[str, "AnalysisRegion"]]]:
        """Every class and measured annotation region, for spectral analysis.

        Returns ``(data, {key: AnalysisRegion})``; keys are ``"class:<id>"``
        for classification classes and ``"annotation:<id>"`` for measured,
        visible rectangle/ellipse/polygon annotations, so both are analysed as
        classes. ``selected`` mirrors each row's checkbox in the layer panel.
        ``None`` when there is neither a classification nor a region.
        """

        regions: dict[str, AnalysisRegion] = {}
        slot = self._current_slot
        data: Optional[HSIData] = None
        if slot.result is not None and slot.layers is not None and slot.active_data is not None:
            data = slot.active_data
            colors = classification_palette(slot.layers.class_ids, slot.fixed_colors)
            class_map = slot.result.class_map
            for layer in slot.layers.layers:
                regions[f"class:{layer.class_id}"] = AnalysisRegion(
                    layer.name, class_map == layer.class_id,
                    colors[layer.class_id], layer.visible,
                )
        if data is None:
            data = self._display_data_provider()
            if not data.is_loaded():
                return None
        shape = (data.rows, data.columns)
        for layer, measurement in self._measured_regions():
            if measurement.pixel_count == 0:
                continue
            regions[f"annotation:{layer.layer_id}"] = AnalysisRegion(
                layer.name, region_mask(layer, shape), tuple(layer.style.stroke[:3]),
                self._layer_panel.is_region_selected(layer.layer_id),
            )
        return (data, regions) if regions else None

    def is_running(self) -> bool:
        return self._thread is not None

    def set_image_loaded(self, loaded: bool) -> None:
        self._unsupervised_button.setEnabled(loaded)
        self._supervised_button.setEnabled(loaded)
        if loaded:
            self._update_separate_leaves_button()
        elif self._separate_leaves_button is not None:
            self._separate_leaves_button.setEnabled(False)

    def clear_result(self) -> None:
        """Discard both resolutions' labels when the cube geometry changes."""

        self._slots = {False: _ClassificationSlot(), True: _ClassificationSlot()}
        self._calibrated_slots = {False: _ClassificationSlot(), True: _ClassificationSlot()}
        self._annotation_slots = {}
        self._layer_panel.clear()
        self._sync_annotations()
        self._update_separate_leaves_button()

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
        self._refresh_regions()

    def clear_super_resolution_result(self) -> None:
        """Discard only the Super-Resolution slot, e.g. when its SR image is discarded.

        Used when the high-res Super-Resolution result is thrown away (such as
        before re-calibrating), leaving the original-resolution slot unaffected
        and still valid.
        """

        self._slots[True] = _ClassificationSlot()
        self._calibrated_slots[True] = _ClassificationSlot()
        self._update_separate_leaves_button()

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
        self._sync_annotations()
        self._update_separate_leaves_button()

    def sync_annotations(self) -> None:
        """Redraw annotations after another component replaced the viewer photo."""

        self._sync_annotations()

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
        if self._separate_leaves_button is not None:
            self._separate_leaves_button.clicked.connect(self._on_separate_leaves_clicked)
            self._separate_leaves_button.setToolTip(
                "Run a supervised classification first, then split clearly "
                "separate plant clusters into one class per leaf"
            )
        self._layer_panel.visibilityChanged.connect(self._on_layer_visibility_changed)
        self._layer_panel.opacityChanged.connect(self._on_layer_opacity_changed)
        self._layer_panel.renameRequested.connect(self._on_layer_rename_requested)
        self._layer_panel.regionRenameRequested.connect(
            self._on_annotation_rename_requested
        )
        self._layer_panel.setAllVisibleRequested.connect(self._on_set_all_visible_requested)
        self._layer_panel.globalOpacityChanged.connect(self._on_global_opacity_changed)
        self._layer_panel.outlineModeChanged.connect(self._on_outline_mode_changed)
        self._layer_panel.backgroundChanged.connect(self._on_background_changed)
        self._layer_panel.exportRequested.connect(self._on_export_requested)
        page = self._layer_panel.annotation_page
        page.toolChanged.connect(self._on_annotation_tool_changed)
        page.styleChanged.connect(self._on_annotation_style_changed)
        page.layerSelected.connect(self._on_annotation_selected)
        page.visibilityChanged.connect(self._on_annotation_visibility_changed)
        page.moveRequested.connect(self._on_annotation_move_requested)
        page.deleteRequested.connect(self._on_annotation_delete_requested)
        page.measureChanged.connect(self._on_annotation_measure_changed)
        self._viewer.annotationShapeDrawn.connect(self._on_annotation_shape_drawn)
        self._viewer.annotationTextRequested.connect(self._on_annotation_text_requested)
        self._viewer.annotationArrowDrawn.connect(self._on_annotation_arrow_drawn)
        self._viewer.annotationPolygonDrawn.connect(self._on_annotation_polygon_drawn)
        self._viewer.annotationGeometryChanged.connect(self._on_annotation_geometry_changed)
        self._viewer.annotationInteractionFinished.connect(
            lambda: self._annotation_slot.layers.break_coalescing()
        )
        self._viewer.annotationRotationChanged.connect(self._on_annotation_rotation_changed)
        self._viewer.annotationPicked.connect(self._on_annotation_picked)
        self._viewer.annotationDeselected.connect(self._on_annotation_deselected)
        self._layer_panel.modeChanged.connect(self._on_panel_mode_changed)

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
        self._groundtruth_button.setToolTip(
            "Select a mask; its hyperspectral cube is paired automatically by name"
        )
        self._hyperspectral_button.setToolTip(
            "Choose a training cube manually if automatic pairing is missing or wrong"
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
            target_is_calibrated=self._pending_calibrated,
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
        if self._separate_leaves_button is not None:
            self._separate_leaves_button.setEnabled(False)
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
        slot.means = None
        slot.leaves_separated = False
        slot.fixed_colors = {}
        slot.leaf_labels = ()
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

    def _update_separate_leaves_button(self) -> None:
        """Enable leaf separation only for an unseparated supervised result."""

        if self._separate_leaves_button is None:
            return
        slot = self._current_slot
        self._separate_leaves_button.setEnabled(
            not self.is_running()
            and isinstance(slot.result, SupervisedClassificationResult)
            and not slot.leaves_separated
        )

    def _on_separate_leaves_clicked(self) -> None:
        """Replace the plant class with one class per clearly separate leaf."""

        slot = self._current_slot
        if not isinstance(slot.result, SupervisedClassificationResult) or self.is_running():
            return
        try:
            separation = separate_leaves(slot.result)
        except ClassificationError as exc:
            QMessageBox.information(self._parent, "Cannot separate leaves", str(exc))
            return
        # Keep any names the user already gave the classes that survive.
        names = {}
        if slot.layers is not None:
            names = {layer.class_id: layer.name for layer in slot.layers.layers}
        names.update(separation.leaf_names)
        result = separation.result
        slot.result = result
        slot.fixed_colors = separation.fixed_colors
        slot.rgb = self._colorize_class_map(
            result.class_map, result.class_ids, slot.fixed_colors
        )
        slot.layers = ClassificationLayerModel(result, names=names)
        slot.means = None
        slot.leaves_separated = True
        slot.leaf_labels = place_leaf_labels(result.class_map, separation.leaf_numbers)
        self._populate_layer_panel(slot)
        self._show_result()
        self._update_separate_leaves_button()
        self._statusbar.showMessage(
            f"Separated the plant into {separation.leaf_count} leaf classes", 8000
        )

    @QtCore.pyqtSlot()
    def _clear_worker(self) -> None:
        self._worker = None
        self._thread = None
        self._active_button = None
        # Only now is the job no longer running, so the button can enable.
        self._update_separate_leaves_button()
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

    @QtCore.pyqtSlot(int, str)
    def _on_layer_rename_requested(self, class_id: int, name: str) -> None:
        layers = self._current_slot.layers
        if layers is None:
            return
        try:
            layers.rename_class(class_id, name)
        except ClassificationError as exc:
            self._statusbar.showMessage(str(exc), 8000)
            return
        self._populate_layer_panel(self._current_slot)

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

    @QtCore.pyqtSlot()
    def _on_export_requested(self) -> None:
        """Save per-class names, sizes, and visualization means as a CSV."""

        slot = self._current_slot
        measured = self._measured_regions()
        if slot.layers is None and not measured:
            return
        path, _ = QFileDialog.getSaveFileName(
            self._parent,
            "Export Class Summary",
            "class_summary.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        if not Path(path).suffix:
            path += ".csv"
        means: dict[int, dict[str, object]] = {}
        statistics: dict[int, dict[str, ClassIndexStatistics]] = {}
        class_layers = ()
        total_pixels = 0
        if slot.layers is not None:
            class_layers = slot.layers.layers
            means = {
                class_id: {name: entry.value for name, entry in entries.items()}
                for class_id, entries in self._segment_means(slot).items()
            }
            for mode, visualization in self._visualization_results.items():
                try:
                    per_class = slot.layers.visualization_statistics(visualization)
                except ClassificationError as exc:
                    LOGGER.info("Skipping %s class statistics: %s", mode.value, exc)
                    continue
                for class_id, stats in per_class.items():
                    statistics.setdefault(class_id, {})[mode.value] = stats
            rows, columns = slot.layers.image_shape
            total_pixels = rows * columns
        else:
            shape = next(
                (v.display_rgb.shape[:2] for v in self._visualization_results.values()),
                (0, 0),
            )
            total_pixels = int(shape[0]) * int(shape[1])
        regions = [
            ExportRegion(
                layer.name,
                measurement.pixel_count,
                {name: mean.value for name, mean in measurement.means.items()},
            )
            for layer, measurement in measured
        ]
        try:
            export_class_summary_csv(
                path, class_layers, means, total_pixels, statistics, regions
            )
        except ClassificationError as exc:
            QMessageBox.critical(self._parent, "Export failed", str(exc))
            return
        self._statusbar.showMessage(f"Class summary exported to {path}", 8000)

    @QtCore.pyqtSlot(bool)
    def _on_outline_mode_changed(self, enabled: bool) -> None:
        layers = self._current_slot.layers
        if layers is None:
            return
        layers.set_outline_mode(enabled)
        self._show_result()

    # ------------------------------------------------------------------ #
    # Private: annotations                                                #
    # ------------------------------------------------------------------ #

    def _sync_annotations(self) -> None:
        slot = self._annotation_slot
        self._viewer.set_annotations(slot.layers.layers, slot.selected_id)
        self._layer_panel.annotation_page.set_layers(
            slot.layers.layers, slot.selected_id
        )
        self._refresh_regions()

    def _refresh_regions(self) -> None:
        """Show each measured, visible annotation as a row on the Classes tab.

        Means are recomputed from the annotation's current geometry every time,
        so a region always reflects where its shape is now.
        """

        self._region_refresh_timer.stop()
        regions = []
        for layer, measurement in self._measured_regions():
            means = {
                name: PixelValueEntry(value=mean.value, color=mean.color)
                for name, mean in measurement.means.items()
            }
            regions.append(
                (layer.layer_id, layer.name, layer.style.stroke,
                 measurement.pixel_count, means)
            )
        self._layer_panel.set_regions(tuple(regions))

    def _measured_regions(self) -> list:
        """Return ``(layer, measurement)`` for each visible, measured annotation."""

        return [
            (layer, measure_region(layer, self._visualization_results))
            for layer in self._annotation_slot.layers.layers
            if layer.measure_index and layer.visible
        ]

    def _schedule_region_refresh(self) -> None:
        """Coalesce the many events of a drag into one recalculation."""

        if any(layer.measure_index for layer in self._annotation_slot.layers.layers):
            self._region_refresh_timer.start()

    def _on_annotation_rename_requested(self, layer_id: int, name: str) -> None:
        slot = self._annotation_slot
        try:
            slot.layers.rename(layer_id, name)
        except AnnotationError as exc:
            self._statusbar.showMessage(str(exc), 8000)
            return
        # Rebuilds the annotation list and the region rows with the new name.
        self._sync_annotations()

    def _on_annotation_measure_changed(self, layer_id: int, enabled: bool) -> None:
        slot = self._annotation_slot
        try:
            slot.layers.set_measure_index(layer_id, enabled)
        except AnnotationError as exc:
            self._statusbar.showMessage(str(exc), 8000)
            self._sync_annotations()
            return
        self._refresh_regions()

    def _on_panel_mode_changed(self, mode: str) -> None:
        # Leaving the annotation view must not leave a drawing tool armed,
        # otherwise clicks on the class view would silently draw.
        if mode != "annotations":
            self._layer_panel.annotation_page.set_tool(None)
            self._viewer.set_annotation_tool(None)

    def _on_annotation_tool_changed(self, kind: Optional[AnnotationKind]) -> None:
        self._viewer.set_annotation_tool(
            kind, self._layer_panel.annotation_page.current_style
        )

    def _on_annotation_style_changed(self, style: AnnotationStyle) -> None:
        """Apply style edits to the selected layer; they also seed new shapes."""

        slot = self._annotation_slot
        if slot.selected_id is None:
            return
        try:
            slot.layers.set_style(slot.selected_id, style)
        except AnnotationError as exc:
            self._statusbar.showMessage(str(exc), 8000)
            return
        self._sync_annotations()

    def _on_annotation_selected(self, layer_id: int) -> None:
        slot = self._annotation_slot
        layer = slot.layers.selected_or_none(layer_id)
        if layer is None:
            return
        slot.selected_id = layer_id
        slot.layers.break_coalescing()
        self._layer_panel.annotation_page.set_style(layer.style)
        self._viewer.set_annotations(slot.layers.layers, slot.selected_id)

    def _on_annotation_visibility_changed(self, layer_id: int, visible: bool) -> None:
        slot = self._annotation_slot
        try:
            slot.layers.set_visible(layer_id, visible)
        except AnnotationError as exc:
            self._statusbar.showMessage(str(exc), 8000)
            return
        self._viewer.set_annotations(slot.layers.layers, slot.selected_id)
        self._refresh_regions()

    def _on_annotation_move_requested(self, layer_id: int, delta: int) -> None:
        slot = self._annotation_slot
        try:
            slot.layers.move(layer_id, delta)
        except AnnotationError as exc:
            self._statusbar.showMessage(str(exc), 8000)
            return
        self._sync_annotations()

    def _on_annotation_delete_requested(self, layer_id: int) -> None:
        slot = self._annotation_slot
        try:
            slot.layers.remove(layer_id)
        except AnnotationError as exc:
            self._statusbar.showMessage(str(exc), 8000)
            return
        if slot.selected_id == layer_id:
            slot.selected_id = None
        self._sync_annotations()

    def _on_annotation_shape_drawn(self, kind: AnnotationKind, rect: QtCore.QRectF) -> None:
        slot = self._annotation_slot
        style = self._layer_panel.annotation_page.current_style()
        try:
            layer = slot.layers.add_shape(
                kind, rect.x(), rect.y(), rect.width(), rect.height(), style
            )
        except AnnotationError as exc:
            self._statusbar.showMessage(str(exc), 8000)
            return
        slot.selected_id = layer.layer_id
        self._sync_annotations()

    def _on_annotation_geometry_changed(
        self, layer_id: int, x: float, y: float, width: float, height: float
    ) -> None:
        slot = self._annotation_slot
        try:
            slot.layers.set_geometry(layer_id, x, y, width, height)
        except AnnotationError as exc:
            self._statusbar.showMessage(str(exc), 8000)
            return
        # Only the canvas needs redrawing; rebuilding the list mid-drag would
        # steal its selection state for no visible change.
        self._viewer.set_annotations(slot.layers.layers, slot.selected_id)
        self._schedule_region_refresh()

    def _on_annotation_rotation_changed(self, layer_id: int, degrees: float) -> None:
        slot = self._annotation_slot
        try:
            slot.layers.set_rotation(layer_id, degrees)
        except AnnotationError as exc:
            self._statusbar.showMessage(str(exc), 8000)
            return
        self._viewer.set_annotations(slot.layers.layers, slot.selected_id)
        self._schedule_region_refresh()

    def _on_annotation_deselected(self) -> None:
        slot = self._annotation_slot
        if slot.selected_id is None:
            return
        slot.selected_id = None
        slot.layers.break_coalescing()
        self._sync_annotations()

    def _on_annotation_picked(self, layer_id: int) -> None:
        slot = self._annotation_slot
        layer = slot.layers.selected_or_none(layer_id)
        if layer is None:
            return
        slot.selected_id = layer_id
        slot.layers.break_coalescing()
        self._layer_panel.annotation_page.set_style(layer.style)
        self._sync_annotations()

    def _on_annotation_arrow_drawn(
        self, tail: QtCore.QPointF, head: QtCore.QPointF
    ) -> None:
        slot = self._annotation_slot
        style = self._layer_panel.annotation_page.current_style()
        try:
            layer = slot.layers.add_arrow(
                tail.x(), tail.y(), head.x(), head.y(), style
            )
        except AnnotationError as exc:
            self._statusbar.showMessage(str(exc), 8000)
            return
        slot.selected_id = layer.layer_id
        self._sync_annotations()

    def _on_annotation_polygon_drawn(self, points: list) -> None:
        slot = self._annotation_slot
        style = self._layer_panel.annotation_page.current_style()
        try:
            layer = slot.layers.add_polygon(points, style)
        except AnnotationError as exc:
            self._statusbar.showMessage(str(exc), 8000)
            return
        slot.selected_id = layer.layer_id
        self._sync_annotations()

    def undo_annotation(self) -> bool:
        """Undo the last annotation change on the displayed image.

        Returns whether anything was undone, so the caller can fall back to
        its own undo (crop) when there was nothing to undo here.
        """

        return self._step_annotation_history(redo=False)

    def redo_annotation(self) -> bool:
        return self._step_annotation_history(redo=True)

    def _step_annotation_history(self, *, redo: bool) -> bool:
        slot = self._annotation_slot
        changed = slot.layers.redo() if redo else slot.layers.undo()
        if not changed:
            return False
        selected = slot.layers.selected_or_none(slot.selected_id)
        if selected is None:
            slot.selected_id = None
        else:
            self._layer_panel.annotation_page.set_style(selected.style)
        self._sync_annotations()
        self._statusbar.showMessage(
            "Annotation change redone" if redo else "Annotation change undone", 4000
        )
        return True

    def _on_annotation_text_requested(self, point: QtCore.QPointF) -> None:
        text, accepted = QtWidgets.QInputDialog.getText(
            self._parent, "Add text", "Annotation text:"
        )
        if not accepted:
            return
        slot = self._annotation_slot
        style = self._layer_panel.annotation_page.current_style()
        try:
            layer = slot.layers.add_text(point.x(), point.y(), text, style)
        except AnnotationError as exc:
            self._statusbar.showMessage(str(exc), 8000)
            return
        slot.selected_id = layer.layer_id
        self._sync_annotations()

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
            fixed_colors=slot.fixed_colors,
        )

    def _segment_means(
        self, slot: _ClassificationSlot
    ) -> dict[int, dict[str, PixelValueEntry]]:
        """Return (and cache) each class's mean for every cached visualization."""

        if slot.means is not None:
            return slot.means
        means: dict[int, dict[str, PixelValueEntry]] = {}
        if self._visualization_results:
            # Every class gets a (possibly empty) entry so its row still shows
            # the index means block, with dashes where a class has no pixels.
            for layer in slot.layers.layers:
                means[layer.class_id] = {}
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
        self._sync_annotations()

    @staticmethod
    def _colorize_class_map(
        class_map: np.ndarray,
        class_ids: tuple,
        fixed_colors: Optional[Mapping[int, tuple[int, int, int]]] = None,
    ) -> NDArray[np.uint8]:
        """Map arbitrary class IDs to stable, evenly spaced display colors."""

        display_rgb = np.zeros((*class_map.shape, 3), dtype=np.uint8)
        for class_id, color in classification_palette(class_ids, fixed_colors).items():
            display_rgb[class_map == class_id] = color
        return display_rgb
