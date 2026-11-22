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
    AnchorMode,
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
    LeafClusters,
    SupervisedClassificationRequest,
    SupervisedClassificationResult,
    SupervisedClassifierType,
    SuperResolutionService,
    TrainingFilePair,
    TrainingPairResolver,
    UnsupervisedClassificationRequest,
    UnsupervisedClassificationResult,
    VectorMask,
    VectorMaskModel,
    VectorTraceOptions,
    VisualizationMode,
    VisualizationResult,
    apply_vector_masks,
    resample_classification,
    scaled_layer,
    export_class_summary_csv,
    draw_leaf_labels,
    leaf_clusters,
    measure_region,
    region_mask,
    place_leaf_labels,
    separate_leaves,
    trace_mask,
)
from ui.annotation_panel import PEN_TOOL
from ui.annotation_render import render_annotations
from ui.classification_colors import classification_palette
from ui.classification_layer_panel import ClassificationLayerPanel
from ui.theme import VIEWER_SCENE_BACKGROUND
from ui.vector_mask_dialog import VectorMaskDialog
from ui.vector_pen import VectorOutline, VectorSelection
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
# Golden-ratio hue steps keep successive drawn classes far apart in colour.
_DRAWN_HUE_STEP = 0.618034


def _drawn_class_color(class_id: int) -> tuple[int, int, int]:
    color = QtGui.QColor.fromHsvF((class_id * _DRAWN_HUE_STEP) % 1.0, 0.72, 0.92)
    return color.red(), color.green(), color.blue()


@dataclass(frozen=True)
class AnalysisRegion:
    """A class or annotation region offered to full-spectrum analysis."""

    label: str
    mask: np.ndarray
    color: tuple[int, int, int]
    selected: bool


@dataclass
class _ClassificationSlot:
    """The one classification state, shared by every image version.

    ``base_result`` is stored once, at the size it was computed. Calibration
    and Super-Resolution only change what is displayed, so ``result`` (and
    the layers, colours and means derived from it) is rebuilt for the
    displayed image: ``base_result`` resampled to its size, with ``vectors``
    applied. Nothing is discarded when the displayed version changes.
    """

    # What the canvas shows: ``base_result`` at the displayed size, edited.
    result: Optional[_ClassificationResult] = None
    # The classification before any vector-mask edits, at its own size.
    base_result: Optional[_ClassificationResult] = None
    # Outlines in original-image pixels; scaled to the displayed size.
    vectors: VectorMaskModel = field(default_factory=VectorMaskModel)
    # The vector mask the pen edits and draws new paths into.
    vector_class: Optional[int] = None
    vector_selection: Optional[VectorSelection] = None
    rgb: Optional[NDArray[np.uint8]] = None
    layers: Optional[ClassificationLayerModel] = None
    active_data: Optional[HSIData] = None
    means: Optional[dict[int, dict[str, PixelValueEntry]]] = None
    leaves_separated: bool = False
    fixed_colors: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    # Each separated leaf's number, drawn beneath it on the canvas and in
    # saved images; empty until leaves are separated.
    leaf_labels: tuple[LeafLabel, ...] = ()
    clusters: Optional[LeafClusters] = None
    # Leaf class ID -> number, to re-place the labels after a vector edit.
    leaf_numbers: dict[int, int] = field(default_factory=dict)
    # Names of classes drawn from scratch with the pen. Kept so that redoing
    # an undone drawing brings the class back under the same name.
    drawn_names: dict[int, str] = field(default_factory=dict)
    # Snapshots of the class layer stack (order, deleted) for undo/redo.
    stack_undo: list = field(default_factory=list)
    stack_redo: list = field(default_factory=list)


@dataclass
class _AnnotationSlot:
    """The annotation stack, shared by every image version.

    Geometry is in original-image pixels; the Controller scales it to the
    displayed image (2x for Super-Resolution) and scales edits back.
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
    # The supervised result behind `brightness_clusters()` changed.
    segmentationChanged = QtCore.pyqtSignal()
    # The pixels, names or colours of classes or measured regions may have
    # changed: re-read analysis_regions() to follow them.
    regionsChanged = QtCore.pyqtSignal()

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
        background_rgb: Callable[[], Optional[NDArray[np.uint8]]] | None = None,
        refresh_canvas: Callable[[], None] | None = None,
        display_scale: Callable[[], float] | None = None,
    ) -> None:
        super().__init__()
        self._display_data_provider = display_data_provider
        self._is_super_resolution_active = is_super_resolution_active
        self._is_calibrated = is_calibrated or (lambda: False)
        # What the classes are composited over (the shown visualization).
        self._background_provider = background_rgb
        # Redraws the shared canvas; without one, set the viewer's photo here.
        self._refresh_canvas = refresh_canvas
        # Displayed image size over the original's, e.g. 2 for Super-Resolution.
        self._display_scale = display_scale or (
            lambda: SuperResolutionService.SCALE if is_super_resolution_active() else 1
        )
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

        self._slot = _ClassificationSlot()
        self._annotations = _AnnotationSlot()
        # One undo history across annotations, class outlines and the class
        # stack: which kind of change each step was, newest last.
        self._history: list[str] = []
        self._redo_history: list[str] = []
        self._hook_history()
        self._opacity_refresh_timer = QtCore.QTimer(self)
        self._opacity_refresh_timer.setSingleShot(True)
        self._opacity_refresh_timer.setInterval(_OPACITY_REFRESH_INTERVAL_MS)
        self._opacity_refresh_timer.timeout.connect(self._show_result)
        self._region_refresh_timer = QtCore.QTimer(self)
        self._region_refresh_timer.setSingleShot(True)
        self._region_refresh_timer.setInterval(_REGION_REFRESH_INTERVAL_MS)
        # Mid-drag refreshes update the rows only; spectra follow on release.
        self._region_refresh_timer.timeout.connect(
            lambda: self._refresh_regions(notify=False)
        )
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
        return self._slot

    @property
    def _annotation_slot(self) -> _AnnotationSlot:
        return self._annotations

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
            composite.display_rgb, slot.leaf_labels, slot.layers.visible_class_ids,
            classification_palette(slot.layers.class_ids, slot.fixed_colors),
        )

    def with_annotations(self, rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Draw the displayed image's visible annotations onto ``rgb`` for export.

        Annotations are kept out of :meth:`composited_rgb` because the viewer
        draws them itself, as editable items; baking them in there would show
        them twice. Only an export should flatten them into pixels.
        """

        return render_annotations(rgb, self._display_layers())

    def class_id_at(self, row: int, column: int) -> Optional[int]:
        """Return the zero-based class ID at ``(row, column)``, if any."""

        layers = self._current_slot.layers
        return None if layers is None else layers.class_at(row, column)

    def analysis_regions(self) -> Optional[tuple[HSIData, dict[str, "AnalysisRegion"]]]:
        """Every class and measured annotation region, for spectral analysis.

        Returns ``(data, {key: AnalysisRegion})``; keys are ``"class:<id>"``
        for classification classes and ``"annotation:<id>"`` for measured,
        visible rectangle/ellipse annotations, so both are analysed as
        classes. ``selected`` mirrors each row's checkbox in the layer panel.
        ``None`` when there is neither a classification nor a region.
        """

        regions: dict[str, AnalysisRegion] = {}
        slot = self._current_slot
        data: Optional[HSIData] = None
        if slot.result is not None and slot.layers is not None and slot.active_data is not None:
            data = slot.active_data
            colors = classification_palette(slot.layers.class_ids, slot.fixed_colors)
            for layer in slot.layers.layers:
                regions[f"class:{layer.class_id}"] = AnalysisRegion(
                    layer.name, slot.layers.mask_for_class(layer.class_id).astype(bool),
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
                layer.name, region_mask(self._display_layer(layer), shape),
                tuple(layer.style.stroke[:3]), layer.visible,
            )
        return (data, regions) if regions else None

    def brightness_clusters(self) -> Optional[LeafClusters]:
        """Leaf clusters of the displayed supervised result, if there is one.

        Leaves are split for this purpose even when the user has not pressed
        Separate Leaves; the layer panel is left untouched. Computed once per
        result.
        """

        slot = self._current_slot
        if not isinstance(slot.result, SupervisedClassificationResult):
            return None
        if slot.clusters is None:
            slot.clusters = leaf_clusters(slot.result, separated=slot.leaves_separated)
        return slot.clusters

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
        """Discard classes and annotations when the cube geometry changes."""

        self._slot = _ClassificationSlot()
        self._annotations = _AnnotationSlot()
        self._history.clear()
        self._redo_history.clear()
        self._hook_history()
        self._layer_panel.clear()
        self._sync_annotations()
        self._update_separate_leaves_button()
        self.segmentationChanged.emit()

    def set_visualization_results(
        self, results: Mapping[VisualizationMode, VisualizationResult]
    ) -> None:
        """Use ``results`` (rendered from the displayed data) for class means.

        Call whenever the visualizations are recomputed; cached means are
        discarded and rebuilt on the next layer-panel refresh.
        """

        self._visualization_results = results
        slot = self._current_slot
        slot.means = None
        self._follow_display(slot)
        if slot.layers is not None:
            self._populate_layer_panel(slot)
            self._show_result()
        self._refresh_regions()

    def refresh_display(self) -> None:
        """Follow a change of displayed image (calibration or Super-Resolution).

        Classes and annotations carry over: the classes are rebuilt at the
        displayed size and annotations are redrawn scaled to it.
        """

        slot = self._current_slot
        self._follow_display(slot)
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
        self._layer_panel.bordersOnlyChanged.connect(self._on_layer_borders_only_changed)
        self._layer_panel.renameRequested.connect(self._on_layer_rename_requested)
        self._layer_panel.regionRenameRequested.connect(
            self._on_annotation_rename_requested
        )
        self._layer_panel.setAllVisibleRequested.connect(self._on_set_all_visible_requested)
        self._layer_panel.globalOpacityChanged.connect(self._on_global_opacity_changed)
        self._layer_panel.outlineModeChanged.connect(self._on_outline_mode_changed)
        self._layer_panel.exportRequested.connect(self._on_export_requested)
        page = self._layer_panel.annotation_page
        page.toolChanged.connect(self._on_annotation_tool_changed)
        page.styleChanged.connect(self._on_annotation_style_changed)
        page.measureChanged.connect(self._on_annotation_measure_changed)
        page.vectorMaskSelected.connect(self._on_vector_mask_selected)
        page.anchorModeRequested.connect(self._on_anchor_mode_requested)
        page.deleteAnchorRequested.connect(self._on_delete_anchor_requested)
        page.vectorRevertRequested.connect(self._on_vector_revert_requested)
        self._layer_panel.vectorizeRequested.connect(self._on_vectorize_requested)
        self._viewer.vectorMaskEdited.connect(self._on_vector_mask_edited)
        self._viewer.vectorClassDrawn.connect(self._on_vector_class_drawn)
        self._viewer.vectorSelectionChanged.connect(self._on_vector_selection_changed)
        self._viewer.vectorEditFinished.connect(
            lambda: self._current_slot.vectors.break_coalescing()
        )
        self._viewer.annotationShapeDrawn.connect(self._on_annotation_shape_drawn)
        self._viewer.annotationTextRequested.connect(self._on_annotation_text_requested)
        self._viewer.annotationArrowDrawn.connect(self._on_annotation_arrow_drawn)
        self._viewer.annotationGeometryChanged.connect(self._on_annotation_geometry_changed)
        self._viewer.annotationInteractionFinished.connect(
            self._on_annotation_interaction_finished
        )
        self._viewer.annotationRotationChanged.connect(self._on_annotation_rotation_changed)
        self._viewer.annotationPicked.connect(self._on_annotation_picked)
        self._viewer.annotationDeselected.connect(self._on_annotation_deselected)
        self._layer_panel.layerSelected.connect(self._on_layer_selected)
        self._layer_panel.annotationVisibilityChanged.connect(
            self._on_annotation_visibility_changed
        )
        self._layer_panel.annotationMoveRequested.connect(self._on_annotation_move_requested)
        self._layer_panel.annotationDeleteRequested.connect(
            self._on_annotation_delete_requested
        )
        self._layer_panel.classMoveRequested.connect(self._on_class_move_requested)
        self._layer_panel.classDeleteRequested.connect(self._on_class_delete_requested)

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

        worker = _SupervisedClassificationWorker(
            self._service,
            data,
            self._training_pair,
            request,
            target_is_calibrated=self._is_calibrated(),
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
        # The result is kept at the size it was computed; if the displayed
        # image changed while the worker ran, it is resampled to fit.
        slot = self._current_slot
        self._reset_vectors(slot, result)
        slot.layers = ClassificationLayerModel(result)
        slot.leaves_separated = False
        slot.fixed_colors = {}
        slot.leaf_numbers = {}
        self._apply_vectors(slot)
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
        # Leaf classes replace the plant class, so outlines start afresh.
        self._reset_vectors(slot, result)
        slot.fixed_colors = separation.fixed_colors
        slot.layers = ClassificationLayerModel(result, names=names)
        slot.leaves_separated = True
        slot.leaf_numbers = dict(separation.leaf_numbers)
        self._apply_vectors(slot)
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
        slot = self._current_slot
        if class_id in slot.drawn_names:
            slot.drawn_names[class_id] = layers.layers[layers.class_ids.index(class_id)].name
        self._populate_layer_panel(slot)
        self.regionsChanged.emit()

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

    @QtCore.pyqtSlot(int, int)
    def _on_class_move_requested(self, class_id: int, delta: int) -> None:
        slot = self._current_slot
        if slot.layers is None:
            return
        snapshot = slot.layers.stack
        if not slot.layers.move_class(class_id, delta):
            return
        self._push_stack(slot, snapshot)
        self._populate_layer_panel(slot)
        self._show_result()
        self.regionsChanged.emit()

    @QtCore.pyqtSlot(int)
    def _on_class_delete_requested(self, class_id: int) -> None:
        """Delete a class layer; Ctrl+Z brings it back.

        A class drawn with the Pen is removed with its outline. A classified
        class leaves the layer stack, keeping its pixels (and any outline)
        so undoing restores it exactly.
        """

        slot = self._current_slot
        if slot.layers is None or class_id not in slot.layers.paint_order:
            return
        if class_id in slot.drawn_names:
            self._on_vector_revert_requested(class_id)
            self._layer_panel.set_selected_layer(None)
            return
        name = next(layer.name for layer in slot.layers.layers if layer.class_id == class_id)
        snapshot = slot.layers.stack
        slot.layers.remove_class(class_id)
        self._push_stack(slot, snapshot)
        if slot.vector_class == class_id:
            slot.vector_class = None
            slot.vector_selection = None
        self._layer_panel.set_selected_layer(None)
        self._populate_layer_panel(slot)
        self._show_result()
        self.regionsChanged.emit()
        self.segmentationChanged.emit()
        self._statusbar.showMessage(f"Deleted {name} · Ctrl+Z to undo", 5000)

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

    @QtCore.pyqtSlot(int, bool)
    def _on_layer_borders_only_changed(self, class_id: int, enabled: bool) -> None:
        layers = self._current_slot.layers
        if layers is None:
            return
        try:
            layers.set_class_borders_only(class_id, enabled)
        except ClassificationError as exc:
            self._statusbar.showMessage(str(exc), 8000)
            return
        self._show_result()

    # ------------------------------------------------------------------ #
    # Private: annotations                                                #
    # ------------------------------------------------------------------ #

    def _sync_annotations(self) -> None:
        slot = self._annotation_slot
        self._viewer.set_annotations(self._display_layers(), slot.selected_id)
        self._show_annotation_selection()
        self._refresh_regions()
        self._sync_vectors()

    def _show_annotation_selection(self) -> None:
        """Mirror the selected annotation in the layer list and tool options."""

        slot = self._annotation_slot
        selected = slot.layers.selected_or_none(slot.selected_id)
        self._layer_panel.annotation_page.set_selected_layer(selected)
        if selected is not None:
            self._layer_panel.set_selected_layer(f"annotation:{selected.layer_id}")
        elif (self._layer_panel.selected_layer or "").startswith("annotation:"):
            self._layer_panel.set_selected_layer(None)

    def _refresh_regions(self, *, notify: bool = True) -> None:
        """List every annotation as a layer, with measured regions' means.

        Means are recomputed from the annotation's current geometry every time,
        so a region always reflects where its shape is now. ``notify`` also
        emits :attr:`regionsChanged`, so plotted spectra follow.
        """

        self._region_refresh_timer.stop()
        measurements = {}
        for layer, measurement in self._measured_regions():
            means = {
                name: PixelValueEntry(value=mean.value, color=mean.color)
                for name, mean in measurement.means.items()
            }
            measurements[layer.layer_id] = (measurement.pixel_count, means)
        self._layer_panel.set_annotations(self._annotation_slot.layers.layers, measurements)
        if notify:
            self.regionsChanged.emit()

    def _on_annotation_interaction_finished(self) -> None:
        self._annotation_slot.layers.break_coalescing()
        # A shape drag only refreshed the rows; now the spectra catch up.
        self.regionsChanged.emit()

    def _measured_regions(self) -> list:
        """Return ``(layer, measurement)`` for each visible, measured annotation."""

        return [
            (layer, measure_region(self._display_layer(layer), self._visualization_results))
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

    def _on_layer_selected(self, key: str) -> None:
        """A layer row was clicked: select an annotation, or let go of one."""

        if key.startswith("annotation:"):
            self._on_annotation_selected(int(key.split(":", 1)[1]))
            return
        if self._annotation_slot.selected_id is not None:
            self._annotation_slot.selected_id = None
            self._annotation_slot.layers.break_coalescing()
            self._viewer.set_annotations(self._display_layers(), None)
            self._layer_panel.annotation_page.set_selected_layer(None)
        if key.startswith("class:") and self._viewer.pen_active:
            self._edit_class_outline(int(key.split(":", 1)[1]))

    def _edit_class_outline(self, class_id: int) -> None:
        """Aim the Pen at a class layer, tracing its outline the first time.

        A class's outline is its vector mask: once traced, the class region
        is whatever the outline encloses.
        """

        slot = self._current_slot
        if slot.layers is None or class_id not in slot.layers.class_ids:
            return
        if class_id not in slot.vectors:
            mask = trace_mask(
                slot.layers.mask_for_class(class_id).astype(bool), VectorTraceOptions()
            )
            if not mask.paths:
                self._statusbar.showMessage("This class has no region large enough to outline", 5000)
                return
            slot.vectors.set_mask(class_id, mask.scaled(1 / self._display_scale()))
            slot.vector_selection = (class_id, self._largest_path(mask), -1)
            slot.vector_class = class_id
            self._apply_vectors(slot)
            self._statusbar.showMessage(
                "Outlined the class; drag its points to reshape it", 5000
            )
            return
        self._on_vector_mask_selected(class_id)

    def _on_annotation_tool_changed(self, kind: Optional[AnnotationKind | str]) -> None:
        pen = kind == PEN_TOOL
        self._viewer.set_pen_tool(pen)
        self._viewer.set_annotation_tool(
            None if pen else kind, self._layer_panel.annotation_page.current_style
        )
        self._sync_vectors()
        if pen and not self._current_slot.vectors:
            self._statusbar.showMessage(
                "Click points on the image to draw a new class, or select a class "
                "layer to edit its outline",
                8000,
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
        self._viewer.set_annotations(self._display_layers(), slot.selected_id)
        self._show_annotation_selection()

    def _on_annotation_visibility_changed(self, layer_id: int, visible: bool) -> None:
        slot = self._annotation_slot
        try:
            slot.layers.set_visible(layer_id, visible)
        except AnnotationError as exc:
            self._statusbar.showMessage(str(exc), 8000)
            return
        self._viewer.set_annotations(self._display_layers(), slot.selected_id)
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
        scale = self._display_scale()
        try:
            layer = slot.layers.add_shape(
                kind, rect.x() / scale, rect.y() / scale,
                rect.width() / scale, rect.height() / scale, style,
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
        scale = self._display_scale()
        try:
            slot.layers.set_geometry(
                layer_id, x / scale, y / scale, width / scale, height / scale
            )
        except AnnotationError as exc:
            self._statusbar.showMessage(str(exc), 8000)
            return
        # Only the canvas needs redrawing; rebuilding the list mid-drag would
        # steal its selection state for no visible change.
        self._viewer.set_annotations(self._display_layers(), slot.selected_id)
        self._schedule_region_refresh()

    def _on_annotation_rotation_changed(self, layer_id: int, degrees: float) -> None:
        slot = self._annotation_slot
        try:
            slot.layers.set_rotation(layer_id, degrees)
        except AnnotationError as exc:
            self._statusbar.showMessage(str(exc), 8000)
            return
        self._viewer.set_annotations(self._display_layers(), slot.selected_id)
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
        self._sync_annotations()

    def _on_annotation_arrow_drawn(
        self, tail: QtCore.QPointF, head: QtCore.QPointF
    ) -> None:
        slot = self._annotation_slot
        style = self._layer_panel.annotation_page.current_style()
        scale = self._display_scale()
        try:
            layer = slot.layers.add_arrow(
                tail.x() / scale, tail.y() / scale, head.x() / scale, head.y() / scale, style
            )
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

    def _hook_history(self) -> None:
        """Log every new undo step of the current annotation and outline models."""

        self._annotations.layers.on_record = lambda: self._log("annotation")
        self._slot.vectors.on_record = lambda: self._log("outline")

    def _log(self, kind: str) -> None:
        self._history.append(kind)
        self._redo_history.clear()

    def _push_stack(self, slot: _ClassificationSlot, snapshot) -> None:
        """Record the class stack as it was before a reorder or delete."""

        slot.stack_undo.append(snapshot)
        slot.stack_redo.clear()
        self._log("stack")

    def _step_annotation_history(self, *, redo: bool) -> bool:
        """Undo (or redo) the newest change to annotations, class outlines or
        the class stack, whichever it was."""

        source, target = (
            (self._redo_history, self._history) if redo
            else (self._history, self._redo_history)
        )
        while source:
            kind = source.pop()
            if self._step_history(kind, redo=redo):
                target.append(kind)
                return True
        return False

    def _step_history(self, kind: str, *, redo: bool) -> bool:
        if kind == "outline":
            return self._step_vector_history(redo=redo)
        if kind == "stack":
            return self._step_stack_history(redo=redo)
        slot = self._annotation_slot
        changed = slot.layers.redo() if redo else slot.layers.undo()
        if not changed:
            return False
        if slot.layers.selected_or_none(slot.selected_id) is None:
            slot.selected_id = None
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
            scale = self._display_scale()
            layer = slot.layers.add_text(point.x() / scale, point.y() / scale, text, style)
        except AnnotationError as exc:
            self._statusbar.showMessage(str(exc), 8000)
            return
        slot.selected_id = layer.layer_id
        self._sync_annotations()

    def _display_layer(self, layer):
        """``layer`` as drawn over the displayed image (scaled for 2x)."""

        return scaled_layer(layer, self._display_scale())

    def _display_layers(self) -> tuple:
        return tuple(
            self._display_layer(layer) for layer in self._annotation_slot.layers.layers
        )

    def _background_rgb(self, slot: _ClassificationSlot) -> Optional[NDArray[np.uint8]]:
        """The image the classes are drawn over: the shown visualization."""

        if self._background_provider is not None:
            return self._background_provider()
        return slot.active_data.rgb_array if slot.active_data is not None else None

    def _populate_layer_panel(self, slot: _ClassificationSlot) -> None:
        if slot.layers is None:
            return
        self._layer_panel.set_layers(
            slot.layers.layers,
            global_opacity=slot.layers.global_opacity,
            outline_mode=slot.layers.outline_mode,
            means=self._segment_means(slot),
            fixed_colors=slot.fixed_colors,
            vector_class_ids=frozenset(slot.vectors.masks),
        )

    # ------------------------------------------------------------------ #
    # Private: vector masks                                               #
    # ------------------------------------------------------------------ #

    def _reset_vectors(self, slot: _ClassificationSlot, base: _ClassificationResult) -> None:
        """Start a fresh, unedited vector-mask and class-stack history over ``base``."""

        slot.base_result = base
        slot.vectors = VectorMaskModel()
        slot.vector_class = None
        slot.vector_selection = None
        slot.drawn_names = {}
        slot.stack_undo.clear()
        slot.stack_redo.clear()
        # Steps of the replaced classification can no longer be undone.
        for log in (self._history, self._redo_history):
            log[:] = [kind for kind in log if kind == "annotation"]
        self._hook_history()

    def _rebuild(self, slot: _ClassificationSlot) -> bool:
        """Derive the displayed classes from the stored result and outlines.

        Everything that reads class pixels -- the composite, index means,
        spectra, CSV export, saved images -- reads ``slot.result``, so this
        is the single place the stored classification and its outline edits
        become the class regions of the image now displayed.
        """

        if slot.base_result is None or slot.layers is None:
            return False
        data = self._display_data_provider()
        shape = (
            (data.rows, data.columns) if data.is_loaded()
            else slot.base_result.class_map.shape
        )
        scale = self._display_scale()
        try:
            base = resample_classification(slot.base_result, shape)
            result = apply_vector_masks(
                base,
                {class_id: mask.scaled(scale) for class_id, mask in slot.vectors.masks.items()},
            )
            layers = slot.layers.with_result(result)
            for class_id, name in slot.drawn_names.items():
                if class_id in layers.class_ids:
                    layers.rename_class(class_id, name)
        except ClassificationError as exc:
            self._statusbar.showMessage(str(exc), 8000)
            return False
        slot.active_data = data
        slot.result = result
        slot.layers = layers
        slot.rgb = self._colorize_class_map(
            result.class_map, layers.class_ids, slot.fixed_colors
        )
        slot.means = None
        slot.clusters = None
        slot.leaf_labels = (
            place_leaf_labels(result.class_map, slot.leaf_numbers)
            if slot.leaf_numbers else ()
        )
        return True

    def _follow_display(self, slot: _ClassificationSlot) -> None:
        """Rebuild the classes if the displayed image has changed under them."""

        if slot.base_result is not None and slot.active_data is not self._display_data_provider():
            self._rebuild(slot)

    def _apply_vectors(self, slot: _ClassificationSlot) -> None:
        """Rebuild the displayed classes after an edit, and show them."""

        if self._rebuild(slot):
            self._populate_layer_panel(slot)
            self._show_result()
            self.segmentationChanged.emit()

    def _sync_vectors(self) -> None:
        """Show the displayed result's vector masks on the canvas and panel."""

        slot = self._current_slot
        outlines: list[VectorOutline] = []
        entries = []
        if slot.layers is not None and slot.vectors:
            colors = classification_palette(slot.layers.class_ids, slot.fixed_colors)
            names = {layer.class_id: layer.name for layer in slot.layers.layers}
            scale = self._display_scale()
            for class_id, mask in slot.vectors.masks.items():
                if class_id not in names:
                    continue  # a deleted layer's outline, back from an undo
                outlines.append(VectorOutline(class_id, colors[class_id], mask.scaled(scale)))
                entries.append(
                    (class_id, names[class_id], colors[class_id],
                     len(mask.paths), mask.anchor_count)
                )
        if slot.vector_class not in slot.vectors:
            # No mask chosen: the pen's next path starts a new class.
            slot.vector_class = None
        self._viewer.set_vector_masks(
            tuple(outlines), slot.vector_selection, slot.vector_class
        )
        # The viewer drops a selection whose path or point no longer exists.
        slot.vector_selection = self._viewer.vector_selection
        page = self._layer_panel.annotation_page
        page.set_vector_masks(tuple(entries), slot.vector_class)
        page.set_anchor_state(self._selected_anchor(slot)[1])
        if self._viewer.pen_active:
            # The class whose outline the Pen edits is the selected layer.
            selected = self._layer_panel.selected_layer or ""
            if slot.vector_class is not None:
                self._layer_panel.set_selected_layer(f"class:{slot.vector_class}")
            elif selected.startswith("class:"):
                self._layer_panel.set_selected_layer(None)

    @staticmethod
    def _selected_anchor(
        slot: _ClassificationSlot,
    ) -> tuple[Optional[VectorMask], Optional[AnchorMode]]:
        """The selected point's mask and mode, or ``(None, None)``."""

        selection = slot.vector_selection
        if selection is None or selection[2] < 0:
            return None, None
        class_id, path_index, anchor_index = selection
        mask = slot.vectors.get(class_id)
        if mask is None or path_index >= len(mask.paths):
            return None, None
        path = mask.paths[path_index]
        if anchor_index >= len(path):
            return None, None
        return mask, path.anchors[anchor_index].mode

    @staticmethod
    def _largest_path(mask: VectorMask) -> int:
        if not mask.paths:
            return -1
        return max(range(len(mask.paths)), key=lambda index: mask.paths[index].area())

    def _on_vectorize_requested(self) -> None:
        """Trace the chosen classes into vector masks and start editing them."""

        slot = self._current_slot
        if slot.layers is None or slot.result is None:
            return
        colors = classification_palette(slot.layers.class_ids, slot.fixed_colors)
        classes = tuple(
            (layer.class_id, layer.name, colors[layer.class_id],
             layer.visible and layer.pixel_count > 0)
            for layer in slot.layers.layers
        )
        dialog = VectorMaskDialog(classes, self._parent)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        options = dialog.options()
        masks: dict[int, VectorMask] = {}
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            for class_id in dialog.selected_class_ids():
                mask = trace_mask(
                    slot.layers.mask_for_class(class_id).astype(bool), options
                )
                if mask.paths:
                    masks[class_id] = mask.scaled(1 / self._display_scale())
        except ClassificationError as exc:
            QMessageBox.critical(self._parent, "Vector mask failed", str(exc))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if not masks:
            QMessageBox.information(
                self._parent,
                "Nothing to trace",
                "The selected classes have no regions larger than "
                f"{options.min_area:g} pixels.",
            )
            return
        slot.vectors.set_masks(masks)
        first = next(iter(masks))
        slot.vector_class = first
        slot.vector_selection = (first, self._largest_path(masks[first]), -1)
        self._apply_vectors(slot)
        # Converting is the first step of editing, so go straight to the pen.
        self._layer_panel.annotation_page.set_tool(PEN_TOOL)
        self._layer_panel.update_tool_options()
        self._on_annotation_tool_changed(PEN_TOOL)
        paths = sum(len(mask.paths) for mask in masks.values())
        points = sum(mask.anchor_count for mask in masks.values())
        self._statusbar.showMessage(
            f"Converted {len(masks)} class{'es' if len(masks) != 1 else ''} into vector "
            f"masks ({paths} paths, {points} points); edit them with the Pen",
            8000,
        )

    @QtCore.pyqtSlot(int, object, bool)
    def _on_vector_mask_edited(self, class_id: int, mask: object, final: bool) -> None:
        slot = self._current_slot
        if slot.layers is None or class_id not in slot.layers.class_ids:
            return
        mask = mask.scaled(1 / self._display_scale())
        # The moves of one drag are one undo step; the viewer's
        # vectorEditFinished breaks the run when the drag ends.
        slot.vectors.set_mask(
            class_id, mask, coalesce_key=None if final else ("drag", class_id)
        )
        slot.vector_class = class_id
        if final:
            self._apply_vectors(slot)
        else:
            # Re-rasterizing on every move would lag; the outline alone
            # follows the cursor and the region updates on release.
            self._sync_vectors()

    @QtCore.pyqtSlot(object)
    def _on_vector_class_drawn(self, mask: object) -> None:
        """Make a new class from a path the pen closed with no mask chosen."""

        slot = self._current_slot
        if slot.layers is None and not self._start_blank_result(slot):
            return
        mask = mask.scaled(1 / self._display_scale())
        # Reuse the empty slot a removed drawn class leaves behind (K-means
        # class IDs stay contiguous), else take the next ID.
        free = [
            layer.class_id for layer in slot.layers.layers
            if layer.class_id in slot.drawn_names and layer.class_id not in slot.vectors
        ]
        class_id = free[0] if free else max(slot.layers.class_ids, default=-1) + 1
        number = 1 + sum(1 for drawn in slot.vectors.masks if drawn in slot.drawn_names)
        # Pin the current colours, or the palette would re-spread every hue
        # to make room for the new class.
        slot.fixed_colors = {
            **classification_palette(slot.layers.class_ids, slot.fixed_colors),
            class_id: _drawn_class_color(class_id),
        }
        slot.drawn_names[class_id] = f"Drawn class {number}"
        slot.vectors.set_mask(class_id, mask)
        slot.vector_class = class_id
        slot.vector_selection = (class_id, 0, -1)
        if not self._rebuild(slot):
            return
        # A drawn class starts as borders only, so the image shows through.
        slot.layers.set_class_borders_only(class_id, True)
        self._populate_layer_panel(slot)
        self._show_result()
        self.segmentationChanged.emit()
        self._statusbar.showMessage(
            f"Added {slot.drawn_names[class_id]}; keep drawing to add more paths "
            "to it, or press Esc to start another class",
            8000,
        )

    def _start_blank_result(self, slot: _ClassificationSlot) -> bool:
        """Give ``slot`` an empty classification for the pen to draw classes into."""

        data = self._display_data_provider()
        if not data.is_loaded():
            return False
        rows, columns = data.rows, data.columns
        result = UnsupervisedClassificationResult(
            class_map=np.full((rows, columns), -1, dtype=np.int32),
            one_hot_masks=np.zeros((0, rows, columns), dtype=np.uint8),
            cluster_centers=np.zeros((0, 0), dtype=np.float32),
            class_pixel_counts=np.zeros(0, dtype=np.int64),
            band_indices=(),
            band_wavelengths_nm=np.zeros(0),
            iterations_completed=0,
        )
        slot.active_data = data
        slot.result = result
        self._reset_vectors(slot, result)
        slot.layers = ClassificationLayerModel(result)
        slot.leaves_separated = False
        slot.leaf_numbers = {}
        slot.fixed_colors = {}
        self._update_separate_leaves_button()
        return True

    @QtCore.pyqtSlot(int, int, int)
    def _on_vector_selection_changed(
        self, class_id: int, path_index: int, anchor_index: int
    ) -> None:
        slot = self._current_slot
        if class_id < 0:
            # Deselecting (Esc) aims the pen at a new class again.
            slot.vector_selection = None
            slot.vector_class = None
        else:
            slot.vector_selection = (class_id, path_index, anchor_index)
            slot.vector_class = class_id
        self._sync_vectors()

    def _on_vector_mask_selected(self, class_id: int) -> None:
        """Aim the pen at ``class_id``'s mask, or at a new class when negative."""

        slot = self._current_slot
        if class_id < 0:
            slot.vector_class = None
            slot.vector_selection = None
            self._sync_vectors()
            return
        mask = slot.vectors.get(class_id)
        if mask is None:
            return
        slot.vector_class = class_id
        if slot.vector_selection is None or slot.vector_selection[0] != class_id:
            path_index = self._largest_path(mask)
            slot.vector_selection = (class_id, path_index, -1) if path_index >= 0 else None
        self._sync_vectors()

    def _on_anchor_mode_requested(self, mode: object) -> None:
        if not isinstance(mode, AnchorMode):
            return
        slot = self._current_slot
        mask, current = self._selected_anchor(slot)
        if mask is None or current is mode:
            return
        class_id, path_index, anchor_index = slot.vector_selection
        path = mask.paths[path_index].set_mode(anchor_index, mode)
        slot.vectors.set_mask(class_id, mask.with_path(path_index, path))
        self._apply_vectors(slot)

    def _on_delete_anchor_requested(self) -> None:
        slot = self._current_slot
        mask, _mode = self._selected_anchor(slot)
        if mask is None:
            return
        class_id, path_index, anchor_index = slot.vector_selection
        path = mask.paths[path_index].remove_anchor(anchor_index)
        slot.vectors.set_mask(class_id, mask.with_path(path_index, path))
        slot.vector_selection = None if path is None else (class_id, path_index, -1)
        self._apply_vectors(slot)

    def _on_vector_revert_requested(self, class_id: int) -> None:
        slot = self._current_slot
        if class_id not in slot.vectors:
            return
        slot.vectors.remove(class_id)
        if slot.vector_selection is not None and slot.vector_selection[0] == class_id:
            slot.vector_selection = None
        self._apply_vectors(slot)
        if class_id in slot.layers.class_ids:
            self._statusbar.showMessage("Vector mask removed; segmented pixels restored", 5000)
        else:
            self._statusbar.showMessage("Drawn class removed", 5000)

    def _step_stack_history(self, *, redo: bool) -> bool:
        slot = self._current_slot
        source, target = (
            (slot.stack_redo, slot.stack_undo) if redo else (slot.stack_undo, slot.stack_redo)
        )
        if not source or slot.layers is None:
            return False
        target.append(slot.layers.stack)
        slot.layers.set_stack(source.pop())
        self._populate_layer_panel(slot)
        self._show_result()
        self.regionsChanged.emit()
        self.segmentationChanged.emit()
        self._statusbar.showMessage(
            "Class layer change redone" if redo else "Class layer change undone", 4000
        )
        return True

    def _step_vector_history(self, *, redo: bool) -> bool:
        slot = self._current_slot
        changed = slot.vectors.redo() if redo else slot.vectors.undo()
        if not changed:
            return False
        self._apply_vectors(slot)
        self._statusbar.showMessage(
            "Vector mask change redone" if redo else "Vector mask change undone", 4000
        )
        return True

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
        # Overlapping layers paint bottom to top in the layer stack's order.
        paint = dict(
            colors=classification_palette(slot.layers.class_ids, slot.fixed_colors),
            paint_order=slot.layers.paint_order,
        )
        base_rgb = self._background_rgb(slot)
        if base_rgb is not None and base_rgb.shape == (*slot.layers.image_shape, 3):
            return slot.layers.compose_display(slot.rgb, base_rgb=base_rgb, **paint)
        return slot.layers.compose_display(
            slot.rgb, background_color=_LAYER_COMPOSITE_BACKGROUND, **paint
        )

    def _show_result(self) -> None:
        if self._refresh_canvas is not None:
            # The window composites classes over the shown visualization.
            self._refresh_canvas()
            return
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
