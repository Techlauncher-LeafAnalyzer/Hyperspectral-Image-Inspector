from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np
from numpy.typing import NDArray
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QFileDialog, QMessageBox

import core.hsi_utils as hsi_utils
from core.marker_crop import find_plus_crop
from core import (
    CalibrationFrameResolver,
    CalibrationService,
    ClassificationService,
    HSIData,
    HSIError,
    HSIReader,
    HSI_FILE_FILTER,
    LeafClusters,
    ON_DEMAND_VISUALIZATION_MODES,
    OPTIONAL_VISUALIZATION_MODES,
    SpectralAnalysisService,
    SuperResolutionRequest,
    SuperResolutionResult,
    SuperResolutionService,
    TrainingPairResolver,
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
from ui.batch_controller import BatchController
from ui.calibration_controller import CalibrationController
from ui.branding import APP_NAME, ICON_PATH
from ui.classification_controller import ClassificationController
from ui.custom_index_dialog import CustomIndexDialog, CustomIndexStore
from ui.generated.MainWindow import Ui_MainWindow
from ui.generated.WorkflowPanels import Ui_WorkflowPanels
from ui.index_mean_dialog import IndexMeanDialog
from ui.image_capabilities import ImageCapabilityController
from ui.resolution_toggle import ResolutionSwitchGroup
from ui.resource_usage import ResourceUsageWidget
from ui.hypercube_controller import HypercubeController
from ui.spectrum_controller import SpectrumController
from ui.super_resolution_worker import SuperResolutionWorker
from ui.tab_transition.handler import TabTransitionHandler
from ui.tool_windows import (
    STRIPE_GAP,
    StripeMenuButton,
    ToolSettingsPanel,
    ToolStripeButton,
    ToolWindow,
    ToolWindowStripe,
    ViewToggleButton,
    paint_batch_icon,
    paint_calibration_icon,
    paint_file_icon,
    paint_segmentation_icon,
    paint_spectrum_icon,
    paint_super_resolution_icon,
    paint_visualization_icon,
)
from ui.theme import CLASSIFIER_BORDER_QSS, CLASSIFIER_POPUP_QSS
from ui.viewer import VISUALIZATION_LABELS, HSIViewer, PixelValueEntry


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
    VisualizationMode.NDWI,
    VisualizationMode.NDMI,
)


LOGGER = logging.getLogger(__name__)

@dataclass
class _CropSnapshot:
    """A prior image state, kept around so a crop can be undone."""

    rgb_array:    NDArray[np.uint8]
    mask_array:   NDArray[np.uint8]
    spectral_obj: Optional[object]
    roi_mask:     Optional[NDArray[np.bool_]] = None


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
        self._setup_workflow_panels_form()
        self.separateLeavesButton = self.classificationLayerPanel.separate_leaves_button
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(QtGui.QIcon(str(ICON_PATH)))
        # Qt's app stylesheet misses the combo's left edge when its border changes.
        self.comboBox.setStyleSheet(CLASSIFIER_BORDER_QSS)
        # Qt renders the combo popup in a separate window, so style its view directly.
        self.comboBox.view().setStyleSheet(CLASSIFIER_POPUP_QSS)

        self._hsi_data = HSIData()
        self._hsi_reader = HSIReader()
        self._visualization_service = VisualizationService()
        self._visualization_export_service = VisualizationExportService()
        self._super_resolution_service = SuperResolutionService()
        self._super_resolution_request = SuperResolutionRequest()
        self._image_capabilities = ImageCapabilityController(
            self._visualization_service, self._super_resolution_service, self.runSuperResButton
        )
        self._super_res_worker: SuperResolutionWorker | None = None
        self._super_res_result: SuperResolutionResult | None = None
        self._super_res_error: str | None = None
        self._sr_view_scale = 1
        self._viz_view_scale = 1
        self._close_after_sr = False
        # Whether the user chose to view the Super-Resolution result (the left
        # stripe's toggle). Effective only once a result exists; see
        # _is_super_resolution_active.
        self._high_resolution_selected = False
        self._active_visualization_mode: VisualizationMode = VisualizationMode.RGB
        # True while the Visualization tab shows the spectral-similarity map
        # instead of a VisualizationMode render.
        self._similarity_view = False
        self._visualization_results: dict[VisualizationMode, VisualizationResult] = {}
        self._visualization_cache: dict[
            tuple[bool, bool], tuple[HSIData, dict[VisualizationMode, VisualizationResult]]
        ] = {}
        # CUSTOM and BRIGHTNESS are rendered on demand; see
        # `_ensure_on_demand_result`. The last applied formula persists.
        self._custom_index_store = CustomIndexStore()
        self._custom_formula: str | None = self._custom_index_store.load()
        self._custom_formula_name = self._saved_formula_name()
        self._custom_index_dialog_factory = CustomIndexDialog
        self._on_demand_errors: dict[VisualizationMode, str] = {}
        # Full-cube band sums keyed by id(data), validated against the
        # source handle and ROI so a crop or calibration never reuses one.
        self._band_sums: dict[int, tuple[HSIData, object, object, np.ndarray]] = {}
        self._checked_mode_button: QtWidgets.QAbstractButton | None = None
        self._segmentation_refresh = QtCore.QTimer(self)
        self._segmentation_refresh.setSingleShot(True)
        self._segmentation_refresh.timeout.connect(self._on_segmentation_settled)
        self._crop_undo_stack: list[_CropSnapshot] = []
        self._crop_redo_stack: list[_CropSnapshot] = []
        self._hypercube_controller = HypercubeController(
            self.modeHyperCube,
            self.visualizationStack,
            self.hypercubeWidget,
            self.statusbar,
            self._visualization_service,
        )
        self._classification_controller = ClassificationController(
            self._display_data,
            self._is_super_resolution_active,
            ClassificationService(),
            TrainingPairResolver(),
            self.classificationViewer,
            self.classificationLayerPanel,
            self.statusbar,
            self.unsupervisedClassifyButton,
            self.pushButton_2,
            self.pushButton,
            self.hyperspectralImageButton,
            self.comboBox,
            self.numOfClassesEdit,
            self.maxIterationsEdit,
            self.lineEdit,
            self.hyperspectralImageEdit,
            self.actionLoadImage,
            lambda: self._hypercube_controller.stop_and_wait(),
            self,
            is_calibrated=lambda: self._calibration_controller.display_result is not None,
            separate_leaves_button=self.classificationLayerPanel.separate_leaves_button,
        )
        self._calibration_controller = CalibrationController(
            self._hsi_data,
            service=CalibrationService(),
            frame_resolver=CalibrationFrameResolver(),
            viewer=self.calibrationViewer,
            statusbar=self.statusbar,
            dark_button=self.darkFileButton,
            dark_edit=self.darkFileEdit,
            bright_button=self.referenceFileButton,
            bright_edit=self.referenceFileEdit,
            calibrate_button=self.calibrateButton,
            load_image_action=self.actionLoadImage,
            stop_hypercube=lambda: self._hypercube_controller.stop_and_wait(),
            resume_hypercube=lambda: self._hypercube_controller.resume(
                self._display_data()
            ),
            refresh_source_views=self._push_image_to_viewers,
            refresh_current_views=self._refresh_viewers_display,
            fallback_pixel_values=self._pixel_values_at,
            parent_widget=self,
            display_data=self._display_data,
            is_high_resolution=self._is_super_resolution_active,
            super_resolution_source=self._super_resolution_data,
        )
        self._classification_controller.readyToClose.connect(self.close)
        self._calibration_controller.readyToClose.connect(self.close)
        self._classification_controller.runningChanged.connect(
            self._on_classification_running_changed
        )
        self._classification_controller.segmentationChanged.connect(
            self._on_segmentation_changed
        )
        self._calibration_controller.runningChanged.connect(
            self._on_calibration_running_changed
        )
        self._calibration_controller.resultReady.connect(
            self._on_calibration_result_ready
        )
        self._calibration_controller.referencesChanged.connect(
            self._on_calibration_references_changed
        )
        self._batch_controller = BatchController(
            file_list=self.batchFileList,
            results_list=self.batchResultsList,
            add_files_button=self.batchAddFilesButton,
            add_folder_button=self.batchAddFolderButton,
            remove_button=self.batchRemoveButton,
            clear_button=self.batchClearButton,
            calibrate_check=self.batchCalibrateCheck,
            super_resolve_check=self.batchSuperResolveCheck,
            classify_check=self.batchClassifyCheck,
            classify_mode_tabs=self.batchClassifyModeTabs,
            num_classes_edit=self.batchNumClassesEdit,
            max_iterations_edit=self.batchMaxIterationsEdit,
            groundtruth_button=self.batchGroundtruthButton,
            groundtruth_edit=self.batchGroundtruthEdit,
            training_cube_button=self.batchTrainingCubeButton,
            training_cube_edit=self.batchTrainingCubeEdit,
            classifier_combo=self.batchClassifierCombo,
            separate_leaves_check=self.batchSeparateLeavesCheck,
            output_dir_button=self.batchOutputDirButton,
            output_dir_edit=self.batchOutputDirEdit,
            format_combo=self.batchFormatCombo,
            overwrite_check=self.batchOverwriteCheck,
            export_csv_check=self.batchExportCsvCheck,
            run_button=self.batchRunButton,
            progress_bar=self.batchProgressBar,
            status_label=self.batchStatusLabel,
            parent_widget=self,
        )
        self._batch_controller.readyToClose.connect(self.close)
        self._spectrum_controller = SpectrumController(
            visualization_service=self._visualization_service,
            analysis_service=SpectralAnalysisService(),
            data_for_viewer=self._spectrum_data_for_viewer,
            display_data=self._display_data,
            blocked_reason=self._spectrum_blocked_reason,
            hypercube_busy=lambda: self._hypercube_controller.is_building(),
            stop_hypercube=lambda: self._hypercube_controller.stop_and_wait(),
            resume_hypercube=lambda: self._hypercube_controller.resume(
                self._display_data()
            ),
            analysis_regions=self._classification_controller.analysis_regions,
            viewer_names={
                self.viewer: "Visualization",
                self.superResViewer: "Super-Resolution",
                self.calibrationViewer: "Calibration",
                self.classificationViewer: "Classification",
            },
            statusbar=self.statusbar,
            parent_widget=self,
        )
        self._spectrum_controller.similarityReady.connect(self._on_similarity_ready)
        self.classificationLayerPanel.classSpectraRequested.connect(
            self._spectrum_controller.show_class_spectra
        )
        self.classificationLayerPanel.inspectRequested.connect(
            self._spectrum_controller.inspect_region
        )
        self._stripe_geometry_timer = QtCore.QTimer(self)
        self._stripe_geometry_timer.setSingleShot(True)
        self._stripe_geometry_timer.timeout.connect(self._sync_stripe_geometry)
        self._configure_tabs()
        self._configure_file_menu()
        self._configure_status_bar()
        self._configure_tool_windows()
        self._resolution_switches = ResolutionSwitchGroup(
            self.superResolutionSwitch,
            self.calibrationSwitch,
            self._select_canvas_resolution,
            self,
            on_calibration_clicked=self._select_canvas_calibration,
        )
        self._connect_signals()
        self._configure_workflow_panels()
        self._active_viewer = self._viewer_for_tab(self.tabWidget.currentIndex())
        self.tabWidget.installEventFilter(self)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched in (self.tabWidget, self.classificationLayerPanel) and event.type() in (
            QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.Move,
            QtCore.QEvent.Type.Show, QtCore.QEvent.Type.Hide,
            QtCore.QEvent.Type.LayoutRequest,
        ):
            self._stripe_geometry_timer.start(0)
        return super().eventFilter(watched, event)

    def _sync_stripe_geometry(self) -> None:
        """Fill the space either side of the image area with stripe buttons.

        Buttons span from the window edge to the image viewer with an even
        :data:`STRIPE_GAP` on both sides: the outer gap is inside the stripe,
        and the inner gap is measured to the image area itself, so the page
        pane's border/padding count toward it rather than adding to it.
        """
        page = self.tabWidget.currentWidget()
        if page is None:
            return
        # Distance from each stripe's inner edge to where page content starts
        # (just the pane border: pages have no side margins).
        # A docked panel sits flush against its stripe, so there the stripe
        # keeps its full inner gap.
        content_left = page.mapTo(self, QtCore.QPoint()).x()
        content_right = page.mapTo(self, QtCore.QPoint(page.width(), 0)).x()
        left = self.toolWindowStripe
        left_edge = left.mapTo(self, QtCore.QPoint()).x() + left.stripe_width()
        left.set_inner_inset(
            0 if self._stripe_has_dock_open(left) else max(content_left - left_edge, 0)
        )
        right = self.workflowStripe
        right_edge = right.mapTo(self, QtCore.QPoint(right.width(), 0)).x() - right.stripe_width()
        right.set_inner_inset(
            0 if self._stripe_has_dock_open(right) else max(right_edge - content_right, 0)
        )
        # Flyouts follow their buttons and make way for docked panels.
        for tool in right.tool_windows:
            tool.place_flyout()

    @staticmethod
    def _stripe_has_dock_open(stripe: ToolWindowStripe) -> bool:
        return any(
            not tool.compact and not tool.floating and tool.is_open()
            for tool in stripe.tool_windows
        )

    # ------------------------------------------------------------------ #
    # Private: signal wiring                                               #
    # ------------------------------------------------------------------ #

    def _configure_tool_windows(self) -> None:
        """Build the JetBrains-style left stripe.

        Top: the logo and the File menu. Then the Super-Resolution and
        Calibration view toggles, which switch every view between image
        versions (highlighted = High Res / After, gray = nothing to compare
        yet). Below a divider: the Spectrum tool window, highlighted while
        open (docked or popped out).
        """
        self.toolWindowStripe = ToolWindowStripe(self)
        self.toolWindowStripe.set_logo(str(ICON_PATH))
        self.addToolBar(self.toolWindowStripe.toolbar_area, self.toolWindowStripe)
        # The stripes span the old side margins up to the image area, and
        # each page's image starts at the pane's edges, so the gap between
        # a stripe's buttons and the image is the same in every view.
        self.horizontalLayout.setContentsMargins(0, STRIPE_GAP, 0, STRIPE_GAP)
        self.fileMenuButton = StripeMenuButton(
            paint_file_icon, "File", self._file_menu, self.toolWindowStripe
        )
        self.fileMenuButton.setObjectName("fileMenuButton")
        self.fileMenuButton.setAccessibleDescription(
            "Open the menu for loading and saving images"
        )
        self.fileMenuButton.setToolTip("File: open or save an image")
        self.toolWindowStripe.add_button(self.fileMenuButton)
        self.toolWindowStripe.add_divider()
        self.superResolutionSwitch = ViewToggleButton(
            paint_super_resolution_icon, "superResolutionSwitch",
            off_label="Low Res", on_label="High Res (Super-Resolution 2×)",
            unavailable_hint="Super-Resolution: run Super-Resolution to compare resolutions",
            parent=self.toolWindowStripe,
        )
        self.calibrationSwitch = ViewToggleButton(
            paint_calibration_icon, "calibrationSwitch",
            off_label="Before Calibration", on_label="After Calibration",
            unavailable_hint="Calibration: run Calibration to compare before and after",
            parent=self.toolWindowStripe,
        )
        for switch in (self.superResolutionSwitch, self.calibrationSwitch):
            switch.set_state(checked=False, available=False)
            self.toolWindowStripe.add_button(switch)
        self.toolWindowStripe.add_divider()
        self.spectrumTool = ToolWindow(
            self, self.toolWindowStripe,
            object_name="spectrum", title="Spectrum",
            icon_painter=paint_spectrum_icon,
            content=self._spectrum_controller.inspector,
            tooltip="Spectrum Inspector: live spectrum under the cursor "
                    "(Ctrl+Click a pixel to pin it)",
        )
        self._spectrum_controller.set_tool_window(self.spectrumTool)
        self.spectrumTool.floatingChanged.connect(
            lambda floating: self._spectrum_controller.inspector.set_compact(not floating)
        )
        self._spectrum_controller.inspector.set_compact(True)
        self.spectrumTool.set_available(
            False, "Spectrum Inspector: load an image first"
        )

    def _setup_workflow_panels_form(self) -> None:
        """Build the right-sidebar panels from ``qt/WorkflowPanels.ui``.

        Their widgets become window attributes, like ``Ui_MainWindow``'s, so
        controllers and tests address both forms the same way. Each panel is
        later moved into its own tool window by
        :meth:`_configure_workflow_panels`.
        """
        form = Ui_WorkflowPanels()
        holder = QtWidgets.QWidget()
        form.setupUi(holder)
        for name, value in vars(form).items():
            setattr(self, name, value)
        # The button group outlives the throwaway holder.
        self.modeButtons.setParent(self)
        holder.deleteLater()

    def _configure_workflow_panels(self) -> None:
        """Build the right stripe: one captioned pill per workflow panel.

        These replace the old main tabs. Visualization, Super-Resolution and
        Calibration need little room, so each pill opens a compact flyout over
        the image (shown on hover, pinned by a click) holding a panel from
        ``WorkflowPanels.ui``. Segment is a plain toggle for the Segmentation
        view's own inspector (see :meth:`_on_segmentation_pill_clicked`), and
        Batch gets a full-height dock; dividers set those two apart. The
        first four pills are bound to a view: turning to one shows that view
        on the canvas, and its pill carries an accent marker. Batch has no
        view of its own: it works on files, not the canvas.

        ``tabWidget`` is the canvas's page stack, with its tab bar hidden, so
        view switching keeps its pan/zoom hand-off and fade.
        """
        stripe = self.workflowStripe = ToolWindowStripe(
            self, edge=QtCore.Qt.Edge.RightEdge, title="Workflow panels",
            object_name="workflowStripe", show_logo=False,
        )
        self.addToolBar(stripe.toolbar_area, stripe)
        self.tabWidget.tabBar().hide()

        self.workflowTools: dict[str, ToolWindow] = {}
        # Stripe button → the canvas view it is bound to.
        self._view_pills: dict[ToolStripeButton, QtWidgets.QWidget] = {}
        flyouts = (
            # name, title, caption, icon, content, view page, tooltip
            ("visualization", "Visualization", "Visualize", paint_visualization_icon,
             self.visualizationPanel, self.Visualization,
             "Visualization: RGB, vegetation indices, HyperCube and spectral similarity"),
            ("superResolution", "Super-Resolution", "Super-Res", paint_super_resolution_icon,
             self.superResolutionPanel, self.SuperResolution,
             "Super-Resolution: upscale the image 2× with MSDformer"),
            ("calibration", "Calibration", "Calibrate", paint_calibration_icon,
             self.calibrationPanel, self.Calibration,
             "Calibration: correct the image with dark and white references"),
        )
        for name, title, caption, painter, content, page, tooltip in flyouts:
            tool = self._add_workflow_tool(
                name, title, caption, painter, content, tooltip, compact=True,
                popped_size=QtCore.QSize(380, 560),
            )
            self._view_pills[tool.button] = page
            tool.activated.connect(lambda page=page: self.tabWidget.setCurrentWidget(page))

        stripe.add_divider()
        self.segmentationPill = ToolStripeButton(
            paint_segmentation_icon, "Segmentation", stripe, caption="Segment"
        )
        self.segmentationPill.setObjectName("segmentationStripeButton")
        self.segmentationPill.setToolTip(
            "Segmentation: classify pixels into classes and annotate regions"
        )
        stripe.add_button(self.segmentationPill)
        self._view_pills[self.segmentationPill] = self.Classification
        self.segmentationPill.clicked.connect(self._on_segmentation_pill_clicked)
        # The classifier form joins the inspector beside the image.
        self.classificationLayerPanel.set_classifier_controls(self.segmentationPanel)
        self.classificationModeTabs.currentChanged.connect(self._fit_classification_mode_tabs)
        self._fit_classification_mode_tabs()

        stripe.add_divider()
        self._add_workflow_tool(
            "batch", "Batch", "Batch", paint_batch_icon, self.batchPanel,
            "Batch: run the pipeline over many files", compact=False,
            popped_size=QtCore.QSize(760, 640),
        )

        stripe.flyout_obstacles = lambda: [self.classificationLayerPanel]
        # Re-place flyouts when the view (and so the inspector) changes.
        self.classificationLayerPanel.installEventFilter(self)
        self.tabWidget.currentChanged.connect(lambda _index: self._stripe_geometry_timer.start(0))
        self.tabWidget.currentChanged.connect(self._sync_view_pills)
        self._sync_view_pills()

    def _add_workflow_tool(
        self, name: str, title: str, caption: str, painter, content: QtWidgets.QWidget,
        tooltip: str, *, compact: bool, popped_size: QtCore.QSize,
    ) -> ToolWindow:
        tool = ToolWindow(
            self, self.workflowStripe,
            object_name=name, title=title, caption=caption,
            icon_painter=painter, content=ToolSettingsPanel(content), tooltip=tooltip,
            popped_size=popped_size, compact=compact,
        )
        self.workflowTools[name] = tool
        tool.host.visibilityChanged.connect(lambda _visible: self._stripe_geometry_timer.start(0))
        tool.floatingChanged.connect(lambda _floating: self._stripe_geometry_timer.start(0))
        return tool

    def _segmentation_sidebar_shown(self) -> bool:
        return (
            self.tabWidget.currentWidget() is self.Classification
            and not self.classificationLayerPanel.isHidden()
        )

    def _on_segmentation_pill_clicked(self) -> None:
        """Show the Segmentation view with its inspector, or hide the inspector.

        Hiding keeps the view and its results on the canvas; showing again
        brings the inspector back exactly as it was (same mode, scroll, width).
        """
        if self._segmentation_sidebar_shown():
            self.classificationLayerPanel.hide()
        else:
            self.tabWidget.setCurrentWidget(self.Classification)
            self.classificationLayerPanel.show()
        self._sync_view_pills()

    def _fit_classification_mode_tabs(self, *_args) -> None:
        """Size the Segmentation method tabs to the shown page, not the tallest.

        QTabWidget's size hint covers every page, so cap its height instead;
        this keeps the inspector's class list as tall as possible.
        """
        tabs = self.classificationModeTabs
        page = tabs.currentWidget()
        if page is None:
            return
        tabs.setMaximumHeight(tabs.tabBar().sizeHint().height() + page.sizeHint().height())

    def _sync_view_pills(self, *_args) -> None:
        page = self.tabWidget.currentWidget()
        for button, button_page in self._view_pills.items():
            button.set_current(button_page is page)
        # The Segment pill is selected while its inspector is on screen.
        self.segmentationPill.setChecked(self._segmentation_sidebar_shown())
        self.segmentationPill.update()

    def _configure_status_bar(self) -> None:
        """Show the loaded file's name and the CPU/RAM monitor."""
        self.imageFilePath = QtWidgets.QLabel("No image loaded", self.statusbar)
        self.imageFilePath.setObjectName("imageFilePath")
        self.statusbar.addPermanentWidget(self.imageFilePath)
        self._resource_usage = ResourceUsageWidget(self.statusbar)
        self.statusbar.addPermanentWidget(self._resource_usage)

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
        """Build the File menu opened from the left stripe's File button."""
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
        # The menu opens from the left stripe's File button. Registering the
        # actions on the window keeps Ctrl+O / Ctrl+S working while it's closed.
        self.addActions((self.actionLoadImage, self.actionSaveImage))

    def _connect_signals(self) -> None:
        self.actionLoadImage.triggered.connect(self._load_image)
        self.actionSaveImage.triggered.connect(self._save_image)
        self.runSuperResButton.clicked.connect(self._run_super_resolution)
        self._set_super_resolution_ready()
        self._update_super_resolution_view_state(self._high_resolution_selected)
        self.tabWidget.currentChanged.connect(self._on_tab_changed)

        for viewer in self._all_viewers():
            viewer.cropRequested.connect(self._on_crop_requested)
            viewer.polygonCropRequested.connect(self._on_polygon_crop_requested)
            viewer.spectrumPlotRequested.connect(self._on_spectrum_plot)
            viewer.spectrumInspectorRequested.connect(self._spectrum_controller.show)
            viewer.imageMeanSpectrumRequested.connect(
                self._spectrum_controller.show_image_mean
            )
            viewer.classSpectraRequested.connect(
                self._spectrum_controller.show_class_spectra
            )
            viewer.meanIndexRequested.connect(self._on_mean_index)
            viewer.pixel_value_provider = self._pixel_values_at
        self.superResViewer.pixel_value_provider = self._sr_pixel_values_at
        self.calibrationViewer.pixel_value_provider = (
            self._calibration_controller.pixel_values_at
        )
        self.classificationViewer.pixel_value_provider = (
            self._classification_pixel_values_at
        )
        self.classificationViewer.class_spectra_available = True

        self.modeSimilarity.toggled.connect(self._on_similarity_mode_toggled)
        self.spectrumInspectorButton.clicked.connect(self._spectrum_controller.show)
        self._configure_on_demand_mode_buttons()

        mode_buttons = (
            (self.modeRGB, VisualizationMode.RGB),
            (self.modeNDVI, VisualizationMode.NDVI),
            (self.modeEVI, VisualizationMode.EVI),
            (self.modeMCARI, VisualizationMode.MCARI),
            (self.modeMTVI, VisualizationMode.MTVI),
            (self.modeOSAVI, VisualizationMode.OSAVI),
            (self.modePRI, VisualizationMode.PRI),
            (self.modeNDWI, VisualizationMode.NDWI),
            (self.modeNDMI, VisualizationMode.NDMI),
            (self.modeBrightness, VisualizationMode.BRIGHTNESS),
            (self.modeCustom, VisualizationMode.CUSTOM),
        )
        for button, mode in mode_buttons:
            button.toggled.connect(
                lambda checked, mode=mode: self._on_visualization_mode_toggled(mode, checked)
            )
        self._visualization_mode_buttons = mode_buttons
        self._image_capabilities.refresh_visualizations(self._hsi_data, mode_buttons)
        self.modeHyperCube.toggled.connect(self._on_hypercube_mode_toggled)
        self.modeRGB.setChecked(True)

        QtGui.QShortcut(
            QtGui.QKeySequence.StandardKey.Undo,
            self,
            self._undo_crop,
        )
        # Ctrl+Shift+Z is the platform redo on Linux/macOS but not Windows;
        # list it explicitly, without repeating a key the platform already
        # provides (a duplicate would make the shortcut ambiguous).
        redo_keys = QtGui.QKeySequence.keyBindings(QtGui.QKeySequence.StandardKey.Redo)
        extra = QtGui.QKeySequence("Ctrl+Shift+Z")
        if extra not in redo_keys:
            redo_keys.append(extra)
        redo_shortcut = QtGui.QShortcut(redo_keys[0], self, self._redo_crop)
        redo_shortcut.setKeys(redo_keys)

    def _configure_on_demand_mode_buttons(self) -> None:
        """Wire the Brightness/Custom Index controls and add the mode caption.

        The radio buttons and the formula button live in the Visualization
        panel (``WorkflowPanels.ui``); the caption sits under the image so it
        stays visible after the panel's flyout closes.
        """

        self.editCustomIndexButton.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.editCustomIndexButton.clicked.connect(self._on_edit_custom_index_clicked)
        self._set_custom_formula_name(self._custom_formula_name)

        # Says which on-demand mode is shown and how it was computed, since
        # Brightness silently changes meaning after a supervised result.
        self.visualizationModeCaption = QtWidgets.QLabel(self.Visualization)
        self.visualizationModeCaption.setObjectName("visualizationModeCaption")
        self.visualizationModeCaption.setWordWrap(True)
        self.visualizationModeCaption.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.visualizationModeCaption.hide()
        self.verticalLayout.addWidget(self.visualizationModeCaption)

    def _update_super_resolution_view_state(self, show_processed: bool) -> None:
        if self._super_res_worker is not None:
            return
        self._calibration_controller.resolution_changed()
        self._sync_image_switches()
        self._refresh_super_resolution_display()
        self.superResStatusStack.setCurrentWidget(self.superResIdlePage)
        if not self._hsi_data.is_loaded():
            status = "Load an image to compare the original and processed result"
        elif show_processed and self._super_res_result is None:
            status = "Processed result not generated — run Super-Resolution"
        else:
            data = self._display_data()
            if show_processed:
                label = (
                    "Calibrated MSDformer 2×"
                    if self._calibration_controller.display_result
                    else "MSDformer 2×"
                )
            else:
                label = (
                    "Calibrated"
                    if self._calibration_controller.display_result
                    else "Original"
                )
            status = f"{label}: {data.columns} × {data.rows} pixels, {data.bands} bands"
            if show_processed and self._super_res_result.tiled:
                status += " · tiled inference"
        self.superResStatusText.setText(status)
        # Both image-state choices update every tab, even without an SR result.
        if self._hsi_data.is_loaded() and not self._pipeline_busy():
            self._refresh_visualization_pipeline()
            self._classification_controller.refresh_display()

    def _display_data(self) -> HSIData:
        """Return the dataset every tab should currently render.

        Once Super-Resolution has produced a result, the high/low toggle
        chooses between it and the original for the whole application, not
        just the Super-Resolution tab's own comparison viewer.
        """
        calibration = self._calibration_controller.display_result
        if calibration is not None:
            return calibration.data
        if self._is_super_resolution_active():
            return self._super_res_result.data
        return self._hsi_data

    def _super_resolution_data(self) -> HSIData | None:
        """Raw SR cube (SR only ever runs on uncalibrated data), if present."""
        return None if self._super_res_result is None else self._super_res_result.data

    def _set_high_resolution_selected(self, high: bool) -> None:
        """Choose whether every view shows the Super-Resolution result."""

        if high == self._high_resolution_selected:
            return
        self._high_resolution_selected = high
        self._update_super_resolution_view_state(high)

    def _is_super_resolution_active(self) -> bool:
        """Return whether ``_display_data`` currently resolves to the SR result."""

        return self._high_resolution_selected and self._super_res_result is not None

    def _select_canvas_resolution(self, high_resolution: bool) -> None:
        if self._pipeline_busy():
            self._sync_image_switches()
            return
        if high_resolution:
            self._set_high_resolution_selected(True)
        else:
            self._set_high_resolution_selected(False)

    def _select_canvas_calibration(self, calibrated: bool) -> None:
        if self._pipeline_busy():
            self._sync_image_switches()
            return
        previous = self._display_data()
        self._calibration_controller.select_calibrated(calibrated)
        if self._display_data() is not previous:
            self._update_super_resolution_view_state(self._high_resolution_selected)
            label = "After Calibration" if calibrated else "Before Calibration"
            self.statusbar.showMessage(f"Viewing {label}", 3000)
        self._sync_image_switches()

    def _sync_image_switches(self) -> None:
        controller = self._calibration_controller
        enabled = not self._pipeline_busy()
        self._resolution_switches.sync(
            available=self._super_res_result is not None,
            high_resolution=self._is_super_resolution_active(), enabled=enabled,
        )
        self._resolution_switches.sync_calibration(
            available=any(controller.result_for_resolution(high) is not None for high in (False, True)),
            calibrated=controller.display_result is not None, enabled=enabled,
            current_available=controller.result is not None,
        )

    def _refresh_super_resolution_display(self) -> None:
        previous_size = self.superResViewer.photo_size()
        state = self.superResViewer.get_view_state()
        if self._high_resolution_selected and self._super_res_result is None:
            self.superResViewer.rgb = None
            self.superResViewer.mask_array = None
            self.superResViewer.set_photo()
            return
        data = self._display_data()
        if data.rgb_array is None:
            return
        new_scale = 2 if self._is_super_resolution_active() else 1
        self.superResViewer.rgb = data.rgb_array
        self.superResViewer.mask_array = data.mask_array
        pixmap = hsi_utils.numpy_to_qpixmap(data.rgb_array, data.roi_mask)
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
        if self._high_resolution_selected and self._super_res_result is None:
            return {}
        rgb = data.rgb_array
        if rgb is None or not (0 <= row < rgb.shape[0] and 0 <= column < rgb.shape[1]):
            return {}
        color = tuple(int(value) for value in rgb[row, column])
        return {"RGB": PixelValueEntry(value=color, color=color)}

    def _run_super_resolution(self) -> None:
        if self._calibration_controller.is_running():
            return
        if self._classification_controller.is_running():
            return
        if self._super_res_worker is not None:
            self._cancel_super_resolution()
            return
        if not self._hsi_data.is_loaded():
            return
        if self._calibration_controller.result_for_resolution(False) is not None:
            answer = QMessageBox.warning(
                self,
                "Super-Resolution needs uncalibrated data",
                "Super-Resolution only runs on the uncalibrated image. Continuing "
                "will revert the current calibration (and any classification made "
                "from it) before running. Calibrate again after Super-Resolution "
                "to calibrate both resolutions.",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Ok:
                return
            self._calibration_controller.revert_calibration()
        source_data = self._hsi_data
        try:
            self._super_resolution_service.validate(source_data, self._super_resolution_request)
        except HSIError as exc:
            QMessageBox.critical(self, "Unable to run Super-Resolution", str(exc))
            self.superResStatusText.setText(str(exc))
            return
        self._super_res_error = None
        self._resolution_switches.sync(
            available=self._super_res_result is not None,
            high_resolution=self._is_super_resolution_active(),
            enabled=False,
        )
        self.actionLoadImage.setEnabled(False)
        self._calibration_controller.set_external_running(True)
        self._set_classification_controls_available(False)
        self.runSuperResButton.setText("Cancel")
        self.runSuperResButton.setToolTip("Cancel after the current inference tile")
        self.superResProgressBar.setValue(0)
        self.superResStatusStack.setCurrentWidget(self.superResProgressPage)
        # Both features read the same SpyFile. Finish cancellation of any
        # hypercube read before handing the source to the SR worker.
        self._hypercube_controller.stop_and_wait()
        worker = SuperResolutionWorker(self._super_resolution_service, source_data,
                                       self._super_resolution_request, parent=self)
        self._super_res_worker = worker
        self._sync_image_switches()
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

    @staticmethod
    def _scaled_roi_mask(
        mask: Optional[NDArray[np.bool_]], shape: tuple[int, int]
    ) -> Optional[NDArray[np.bool_]]:
        """Resample a region-of-interest mask onto a differently sized frame.

        Super-Resolution rebuilds the cube at 2x into a fresh ``HSIData``, so
        an active polygon crop has to be carried over rather than inherited.
        Nearest-neighbour indexing keeps this exact for the integer 2x case
        and still correct if the scale factor ever changes.
        """
        if mask is None:
            return None
        rows, columns = int(shape[0]), int(shape[1])
        if rows < 1 or columns < 1:
            return None
        row_index = np.arange(rows) * mask.shape[0] // rows
        column_index = np.arange(columns) * mask.shape[1] // columns
        return mask[np.ix_(row_index, column_index)]

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
        # SR upscales the source's bounding box 2x into a brand-new HSIData,
        # which knows nothing of an active polygon crop. Carry the ROI across
        # at the same 2x, or the high-res view would silently reinstate the
        # pixels the user excluded.
        result.data.roi_mask = self._scaled_roi_mask(
            self._hsi_data.roi_mask, display.display_rgb.shape[:2]
        )
        self._super_res_result = result
        self._visualization_cache = {
            key: cached for key, cached in self._visualization_cache.items() if not key[0]
        }
        # Either operation changes the cube used for classification.
        self._classification_controller.clear_result()
        self._calibration_controller.clear_super_resolution_result()
        self._set_high_resolution_selected(True)
        self._refresh_super_resolution_display()
        self.superResProgressBar.setValue(100)
        self._crop_undo_stack.clear()
        self._crop_redo_stack.clear()
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
        self._calibration_controller.set_external_running(False)
        self._set_classification_controls_available(True)
        self._update_super_resolution_view_state(self._high_resolution_selected)
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
        self._image_capabilities.refresh_super_resolution(self._hsi_data)

    def _reset_super_resolution(self) -> None:
        self._super_res_result = None
        self._super_res_error = None
        self._set_high_resolution_selected(False)
        self.superResProgressBar.setValue(0)
        self._set_super_resolution_ready()
        self._update_super_resolution_view_state(False)

    def _set_classification_controls_available(self, available: bool) -> None:
        self.numOfClassesEdit.setEnabled(available)
        self.maxIterationsEdit.setEnabled(available)
        self.pushButton.setEnabled(available)
        self.comboBox.setEnabled(available)
        self._classification_controller.set_image_loaded(
            available and self._hsi_data.is_loaded()
        )

    @QtCore.pyqtSlot(bool)
    def _on_classification_running_changed(self, running: bool) -> None:
        self._calibration_controller.set_external_running(running)
        if running:
            self.runSuperResButton.setEnabled(False)
        else:
            self._set_super_resolution_ready()
        self._sync_image_switches()

    @QtCore.pyqtSlot(bool)
    def _on_calibration_running_changed(self, running: bool) -> None:
        if running:
            self.runSuperResButton.setEnabled(False)
            self._set_classification_controls_available(False)
        else:
            self._set_super_resolution_ready()
            self._set_classification_controls_available(True)
        # runningChanged(True) precedes creation of the first calibration worker.
        self._sync_image_switches()
        if running:
            self._resolution_switches.set_enabled(False)

    @QtCore.pyqtSlot(bool)
    def _on_calibration_result_ready(self, high: bool) -> None:
        self._crop_undo_stack.clear()
        self._crop_redo_stack.clear()
        self._classification_controller.clear_calibrated_result()
        self._clear_calibration_previews()
        self._update_super_resolution_view_state(self._high_resolution_selected)

    @QtCore.pyqtSlot(bool)
    def _on_calibration_references_changed(self, had_result: bool) -> None:
        if had_result:
            self._classification_controller.clear_calibrated_result()
        if self._hsi_data.is_loaded():
            self._clear_calibration_previews()
            self._update_super_resolution_view_state(self._high_resolution_selected)

    def _clear_calibration_previews(self) -> None:
        self._visualization_cache = {
            key: cached for key, cached in self._visualization_cache.items() if not key[1]
        }

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._hypercube_controller.shutdown()
        if self._calibration_controller.request_close():
            event.ignore()
            return
        if self._classification_controller.request_close():
            event.ignore()
            return
        if self._batch_controller.request_close():
            event.ignore()
            return
        # Never destroy a running QThread or block Qt waiting for inference.
        if self._super_res_worker is not None:
            self._close_after_sr = True
            self._cancel_super_resolution()
            event.ignore()
            return
        self._visualization_cache.clear()
        self._band_sums.clear()
        self._custom_index_store.save(self._custom_formula, self._custom_formula_name)
        self._super_res_result = None
        self._spectrum_controller.shutdown()
        for transition in self._tab_transitions:
            transition.stop()
        self._resolution_switches.stop()
        self._resource_usage.stop()
        super().closeEvent(event)

    # ------------------------------------------------------------------ #
    # Private: image I/O                                                   #
    # ------------------------------------------------------------------ #

    def _load_image(self) -> None:
        if self._super_res_worker is not None:
            return
        image_path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open Hyperspectral Image",
            "",
            HSI_FILE_FILTER,
        )
        if not image_path_str:
            return

        self.load_image_from_path(Path(image_path_str))

    def load_image_from_path(self, image_path: Path) -> None:
        if self._calibration_controller.is_running():
            self.statusbar.showMessage(
                "Cancel or finish calibration before loading another image"
            )
            return
        if self._super_res_worker is not None:
            self.statusbar.showMessage("Cancel or finish SR before loading another image")
            return
        if self._classification_controller.is_running():
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
        self._classification_controller.clear_result()
        self._spectrum_controller.image_changed()
        self.spectrumTool.set_available(True)
        calibration_status = self._calibration_controller.source_loaded(
            self._hsi_data.source_path
        )
        loaded_path = self._hsi_data.data_path

        # Status bar: the name; its tooltip and the window title carry more.
        self.imageFilePath.setText(loaded_path.name)
        self.imageFilePath.setToolTip(f"File loaded: {loaded_path}")
        self.setWindowTitle(f"{loaded_path.name} — {APP_NAME}")
        self._classification_controller.set_image_loaded(True)
        self._crop_undo_stack.clear()
        self._crop_redo_stack.clear()
        self._band_sums.clear()
        self._active_visualization_mode = VisualizationMode.RGB
        self._similarity_view = False
        self.modeRGB.setChecked(True)
        auto_cropped = self._auto_crop_marker_pattern()
        self._push_image_to_viewers()
        status = f"Loaded {loaded_path.name}. {calibration_status}"
        if auto_cropped:
            status += " Auto-cropped to Leaf2QR pattern (Ctrl+Z to undo)."
        self.statusbar.showMessage(status, 8000)

    def _auto_crop_marker_pattern(self) -> bool:
        """Polygon-crop to the inner borders of the four Leaf2QR corner tags.

        Silently does nothing when the pattern is absent. Goes through
        `_apply_crop` so the crop is undoable like a manual one.
        """
        rgb = self._hsi_data.rgb_array
        polygon = None if rgb is None else find_plus_crop(rgb)
        if polygon is None:
            return False
        self._apply_crop(
            lambda: self._hsi_data.crop_polygon(polygon), "Auto-cropped"
        )
        return bool(self._crop_undo_stack)

    def _save_image(self) -> None:
        if not self._hsi_data.is_loaded():
            QMessageBox.information(self, "Nothing to save", "Load an image first.")
            return

        display_rgb = self._display_data().rgb_array
        if self.tabWidget.currentWidget() is self.Visualization:
            visualization_rgb = self._visualization_display_rgb()
            if visualization_rgb is not None:
                display_rgb = visualization_rgb
        classification_rgb = (
            self._classification_controller.composited_rgb()
            if self._active_viewer is self.classificationViewer
            else None
        )
        if classification_rgb is not None:
            display_rgb = classification_rgb
        if self._active_viewer is self.classificationViewer:
            # The viewer draws annotations as live items over this image, so
            # they must be flattened into the saved pixels explicitly. This
            # also covers annotating the plain image before any classification.
            display_rgb = self._classification_controller.with_annotations(display_rgb)
        calibration_result = self._calibration_controller.display_result
        if self._active_viewer is self.calibrationViewer and calibration_result:
            display_rgb = calibration_result.data.rgb_array
        if self.tabWidget.currentWidget() is self.SuperResolution:
            if self._high_resolution_selected and self._super_res_result is None:
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
                # Keep a polygon crop's excluded pixels transparent rather
                # than exporting them as black, which would read as data.
                alpha_mask=self._display_data().roi_mask,
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
        previous_button = self._checked_mode_button
        self._checked_mode_button = self.sender()
        if (
            mode is VisualizationMode.CUSTOM
            and self._custom_formula is None
            and not self._edit_custom_index()
        ):
            # Nothing to show without a formula; return to the prior mode.
            self._checked_mode_button = previous_button
            (previous_button or self.modeRGB).setChecked(True)
            return
        self.visualizationStack.setCurrentWidget(self.viewer)
        self._active_visualization_mode = mode
        self._similarity_view = False
        if self._hsi_data.is_loaded():
            self._show_active_mode()
        else:
            self._update_mode_caption()
        # After `_show_active_mode`, so on-demand modes are rendered first.
        self._spectrum_controller.set_visualization(self._visualization_results.get(mode))

    def _on_hypercube_mode_toggled(self, checked: bool) -> None:
        if checked:
            self._checked_mode_button = self.modeHyperCube
        self._update_mode_caption()

    def _show_active_mode(self) -> None:
        """Render the active mode if it is on demand, then show it."""

        if self._ensure_on_demand_result(self._active_visualization_mode):
            self._hypercube_controller.resume(self._display_data())
        self._refresh_viewers_display()
        self._update_mode_caption()

    def _ensure_on_demand_result(self, mode: VisualizationMode) -> bool:
        """Render CUSTOM/BRIGHTNESS into the current results if missing.

        Failures are kept in `_on_demand_errors` for the mode caption rather
        than shown as dialogs. Returns whether the cube was read.
        """

        if (
            mode not in ON_DEMAND_VISUALIZATION_MODES
            or mode in self._visualization_results
            or not self._hsi_data.is_loaded()
            or self._pipeline_busy()
        ):
            return False
        if mode is VisualizationMode.CUSTOM and self._custom_formula is None:
            return False
        data = self._display_data()
        # Serialize behind any hypercube reader of the same SpyFile handle.
        self._hypercube_controller.stop_and_wait()
        self._on_demand_errors.pop(mode, None)
        try:
            if mode is VisualizationMode.CUSTOM:
                request = VisualizationRequest(mode, formula=self._custom_formula)
            else:
                clusters = self._brightness_clusters(data)
                request = VisualizationRequest(
                    mode,
                    segments=None if clusters is None else clusters.labels,
                    band_sum=self._band_sum_for(data),
                )
            result = self._visualization_service.render(data, request)
        except (VisualizationError, WavelengthError) as exc:
            self._on_demand_errors[mode] = str(exc)
            LOGGER.info("Unable to render %s: %s", mode.value, exc)
        else:
            self._visualization_results[mode] = result
            self._classification_controller.set_visualization_results(
                self._visualization_results
            )
        self._update_on_demand_availability()
        return True

    def _band_sum_for(self, data: HSIData) -> np.ndarray:
        cached = self._band_sums.get(id(data))
        if (
            cached is not None
            and cached[0] is data
            and cached[1] is data.spectral_obj
            and cached[2] is data.roi_mask
        ):
            return cached[3]
        self.statusbar.showMessage(f"Summing all {data.bands} bands for Brightness…")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            total = self._visualization_service.band_sum(data)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.statusbar.clearMessage()
        self._band_sums[id(data)] = (data, data.spectral_obj, data.roi_mask, total)
        return total

    def _brightness_clusters(self, data: HSIData) -> LeafClusters | None:
        """Leaf clusters from a supervised result matching `data`, if any."""

        clusters = self._classification_controller.brightness_clusters()
        if clusters is None or clusters.labels.shape != (data.rows, data.columns):
            return None
        return clusters if clusters.count else None

    def _update_on_demand_availability(self) -> None:
        names = {VisualizationMode.BRIGHTNESS.value} if self._hsi_data.is_loaded() else set()
        if VisualizationMode.CUSTOM in self._visualization_results:
            names.add(VisualizationMode.CUSTOM.value)
        for viewer in self._all_viewers():
            viewer.available_on_demand_indices = frozenset(names)

    def _discard_on_demand_results(self, mode: VisualizationMode) -> None:
        self._visualization_results.pop(mode, None)
        for _data, results in self._visualization_cache.values():
            results.pop(mode, None)
        self._on_demand_errors.pop(mode, None)

    def _edit_custom_index(self) -> bool:
        """Open the formula editor; apply and persist an accepted formula."""

        data = self._display_data()
        dialog = self._custom_index_dialog_factory(
            data.wavelengths_nm if data.is_loaded() else None,
            self._custom_formula,
            parent=self,
            store=self._custom_index_store,
        )
        accepted = dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted and dialog.formula
        if not accepted:
            # The library may have been edited even though nothing was applied.
            self._set_custom_formula_name(
                self._custom_index_store.name_for(self._custom_formula)
                if self._custom_formula else None
            )
            self._update_mode_caption()
            return False
        self._custom_formula = dialog.formula
        self._set_custom_formula_name(dialog.formula_name)
        self._custom_index_store.save(self._custom_formula, self._custom_formula_name)
        self._discard_on_demand_results(VisualizationMode.CUSTOM)
        self._classification_controller.set_visualization_results(
            self._visualization_results
        )
        label = self._custom_formula_name or self._custom_formula
        self.statusbar.showMessage(f"Custom index set to {label}", 5000)
        return True

    def _saved_formula_name(self) -> str | None:
        """The library name of the applied formula, if it is still saved."""

        store = self._custom_index_store
        if self._custom_formula is None:
            return None
        entry = store.find(store.active_name() or "")
        if entry is not None and entry.formula == self._custom_formula:
            return entry.name
        return store.name_for(self._custom_formula)

    def _set_custom_formula_name(self, name: str | None) -> None:
        self._custom_formula_name = name
        if self._custom_formula is None:
            tooltip = "Your own index formula built from any bands; use ✎ Formula to define it"
        else:
            title = f"{name}: " if name else ""
            tooltip = f"Custom index — {title}{self._custom_formula}"
        self.modeCustom.setToolTip(tooltip)

    def _on_edit_custom_index_clicked(self) -> None:
        if not self._edit_custom_index():
            return
        if self.modeCustom.isChecked():
            if self._hsi_data.is_loaded():
                self._show_active_mode()
        else:
            self.modeCustom.setChecked(True)

    def _on_segmentation_changed(self) -> None:
        # Brightness switches between whole-image and per-cluster scaling.
        self._discard_on_demand_results(VisualizationMode.BRIGHTNESS)
        # Defer: a crop or load emits this before the pipeline recomputes.
        self._segmentation_refresh.start(0)

    def _on_segmentation_settled(self) -> None:
        if (
            self._active_visualization_mode is VisualizationMode.BRIGHTNESS
            and self._hsi_data.is_loaded()
            and not self._pipeline_busy()
        ):
            self._show_active_mode()
            clusters = self._brightness_clusters(self._display_data())
            self.statusbar.showMessage(
                "Brightness now scaled per leaf cluster"
                if clusters is not None
                else "Brightness now scaled over the whole image",
                5000,
            )
        else:
            self._update_mode_caption()

    def _update_mode_caption(self) -> None:
        """Explain the shown on-demand mode, or hide the caption."""

        caption = self.visualizationModeCaption
        mode = self._active_visualization_mode
        if (
            mode not in ON_DEMAND_VISUALIZATION_MODES
            or self.modeHyperCube.isChecked()
            or self.modeSimilarity.isChecked()
            or not self._hsi_data.is_loaded()
        ):
            caption.hide()
            self._update_brightness_tooltip()
            return
        result = self._visualization_results.get(mode)
        error = self._on_demand_errors.get(mode)
        if mode is VisualizationMode.CUSTOM:
            if result is None:
                text = (
                    "<b>Custom index unavailable</b> — "
                    f"{_escape(error or 'no formula defined')}. "
                    "Click <b>✎ Formula</b> in the Visualization panel to change it."
                )
            else:
                bands = ", ".join(
                    f"{token} → {wavelength:.1f} nm"
                    for token, wavelength in result.band_wavelengths_nm.items()
                )
                name = (
                    f" <b>{_escape(self._custom_formula_name)}</b> ·"
                    if self._custom_formula_name else ""
                )
                text = (
                    f"<b>Custom index</b> ·{name} <code>{_escape(self._custom_formula or '')}</code>"
                    f" · {bands} · colours stretched to the 2–98th percentile"
                )
        else:
            data = self._display_data()
            clusters = self._brightness_clusters(data)
            if result is None:
                text = (
                    "<b>Brightness unavailable</b> — "
                    f"{_escape(error or 'not computed')}."
                )
            elif clusters is not None:
                plural = "s" if clusters.count != 1 else ""
                text = (
                    "<b>Brightness · Per leaf cluster</b> — the sum of all "
                    f"{data.bands} bands is scaled 0–1 separately within each of "
                    f"{clusters.count} cluster{plural} from the supervised "
                    "classification, between its 2nd and 98th percentiles so a "
                    "few glints cannot darken a leaf; background is left blank. "
                    "Compare pixels within a leaf, not between leaves."
                )
            else:
                text = (
                    "<b>Brightness · Whole image</b> — the sum of all "
                    f"{data.bands} bands is scaled 0–1 across the whole image, "
                    "between its 2nd and 98th percentiles. After a supervised "
                    "classification it is scaled per leaf cluster instead."
                )
        caption.setText(text)
        caption.setProperty("state", "warning" if result is None else "")
        caption.style().unpolish(caption)
        caption.style().polish(caption)
        caption.show()
        self._update_brightness_tooltip()

    def _update_brightness_tooltip(self) -> None:
        clusters = (
            self._brightness_clusters(self._display_data())
            if self._hsi_data.is_loaded()
            else None
        )
        scope = (
            "Currently scaled per leaf cluster (supervised classification found)."
            if clusters is not None
            else "Currently scaled over the whole image; a supervised "
            "classification switches it to per leaf cluster."
        )
        self.modeBrightness.setToolTip(
            "Spectral brightness: every pixel's spectrum summed over all bands "
            f"and scaled 0–1. {scope}"
        )

    def _on_similarity_mode_toggled(self, checked: bool) -> None:
        self._similarity_view = checked
        if not checked:
            return
        self._checked_mode_button = self.modeSimilarity
        self._update_mode_caption()
        self.visualizationStack.setCurrentWidget(self.viewer)
        self._spectrum_controller.set_visualization(None)
        if self._spectrum_controller.similarity_for(self._display_data()) is None:
            self._spectrum_controller.show()
            self.statusbar.showMessage(
                "Pin a spectrum or plot class spectra, then choose Map similarity", 8000
            )
        if self._hsi_data.is_loaded():
            self._refresh_viewers_display()

    def _on_similarity_ready(self) -> None:
        if self.modeSimilarity.isChecked():
            self._refresh_viewers_display()
        else:
            self.modeSimilarity.setChecked(True)
        self.tabWidget.setCurrentWidget(self.Visualization)

    def _visualization_display_rgb(self) -> Optional[NDArray[np.uint8]]:
        """Return what the Visualization viewer shows, if not the plain RGB."""

        if self._similarity_view:
            similarity = self._spectrum_controller.similarity_for(self._display_data())
            return similarity.display_rgb if similarity is not None else None
        result = self._visualization_results.get(self._active_visualization_mode)
        return result.display_rgb if result is not None else None

    def _recompute_visualizations(self) -> None:
        self._hypercube_controller.stop_and_wait()
        data = self._display_data()
        reasons = self._image_capabilities.refresh_visualizations(
            data, self._visualization_mode_buttons
        )
        self._update_brightness_tooltip()
        self._set_custom_formula_name(self._custom_formula_name)
        available = frozenset(
            mode.value for mode in OPTIONAL_VISUALIZATION_MODES if reasons[mode] is None
        )
        for viewer in self._all_viewers():
            viewer.available_optional_indices = available
        key = (self._is_super_resolution_active(), self._calibration_controller.display_result is not None)
        cached = self._visualization_cache.get(key)
        if cached is not None and cached[0] is data:
            self._visualization_results = cached[1]
        else:
            self._visualization_results = {}
            for mode in _CACHED_VISUALIZATION_MODES:
                if reasons[mode] is not None:
                    continue
                try:
                    self._visualization_results[mode] = self._visualization_service.render(
                        data, VisualizationRequest(mode=mode)
                    )
                except (VisualizationError, WavelengthError) as exc:
                    LOGGER.info("Skipping %s visualization: %s", mode.value, exc)
            self._visualization_cache[key] = (data, self._visualization_results)
        # The custom index is cheap (a few bands), so keep it alongside the
        # built-in indices for hover values and class means; Brightness reads
        # every band, so it is only rendered while selected.
        self._ensure_on_demand_result(VisualizationMode.CUSTOM)
        self._ensure_on_demand_result(self._active_visualization_mode)
        self._update_on_demand_availability()

        # Keep the active source on the same RGB stretch as Visualization.
        # After a crop, the previous RGB array has stale limits.
        rgb_result = self._visualization_results.get(VisualizationMode.RGB)
        if rgb_result is not None:
            data.rgb_array = rgb_result.display_rgb
            self.modeRGB.setToolTip(rgb_result.title)
        self._classification_controller.set_visualization_results(
            self._visualization_results
        )
        if not self._similarity_view:
            self._spectrum_controller.set_visualization(
                self._visualization_results.get(self._active_visualization_mode)
            )

    def _refresh_viewers_display(self) -> None:
        data = self._display_data()
        self._sync_image_switches()
        display_rgb = self._visualization_display_rgb()
        if display_rgb is None:
            display_rgb = data.rgb_array
        rgb_display = data.rgb_array
        new_scale = 2 if self._is_super_resolution_active() else 1
        factor = new_scale / self._viz_view_scale
        for viewer in self._all_viewers():
            if viewer is self.superResViewer:
                continue
            previous_size = viewer.photo_size()
            state = viewer.get_view_state()
            classification_rgb = (
                self._classification_controller.composited_rgb()
                if viewer is self.classificationViewer
                else None
            )
            calibration_result = self._calibration_controller.display_result
            use_calibration = viewer is self.calibrationViewer and calibration_result
            use_classification = classification_rgb is not None
            if use_calibration:
                viewer_display = calibration_result.data.rgb_array
            elif use_classification:
                viewer_display = classification_rgb
            elif viewer is self.viewer:
                viewer_display = display_rgb
            else:
                viewer_display = rgb_display
            viewer_data = (
                calibration_result.data
                if use_calibration
                else self._classification_controller.display_data
                if use_classification
                else data
            )
            viewer.rgb        = viewer_data.rgb_array
            viewer.mask_array = viewer_data.mask_array
            pixmap = hsi_utils.numpy_to_qpixmap(
                viewer_display, viewer_data.roi_mask
            )
            viewer.set_photo(pixmap)
            # Restore the previous pan/zoom, rescaled by `factor`, when the
            # image dimensions changed only because of a low/high-res swap
            # (factor==1 covers the unchanged-size case, e.g. switching
            # visualization mode). Otherwise (e.g. after a crop) let
            # set_photo's fresh fit_in_view() stand, so the view actually
            # rescales to the new image size.
            viewer_factor = factor
            if (
                state is not None
                and previous_size is not None
                and previous_size * viewer_factor == pixmap.size()
            ):
                viewer.queue_view_state(
                    (state[0] / viewer_factor, state[1] * viewer_factor)
                )
        self._viz_view_scale = new_scale
        self._classification_controller.sync_annotations()
        self._refresh_super_resolution_display()
        self._spectrum_controller.refresh_markers()

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
        class_id = self._classification_controller.class_id_at(row, column)
        if class_id is not None:
            values["Class"] = class_id
        return values

    # ------------------------------------------------------------------ #
    # Private: viewer signal handlers                                      #
    # ------------------------------------------------------------------ #

    def _spectrum_blocked_reason(self) -> Optional[str]:
        """Why the cube can't be read for a spectrum right now, if it can't."""

        if self._calibration_controller.is_running():
            return "Spectrum reads are unavailable while calibration is running"
        if self._classification_controller.is_running():
            return "Spectrum reads are unavailable while classification is running"
        if self._super_res_worker is not None:
            return "Spectrum reads are paused during SR"
        return None

    def _spectrum_data_for_viewer(self, viewer: HSIViewer) -> Optional[HSIData]:
        """Return the cube whose pixels ``viewer`` currently shows."""

        if not self._hsi_data.is_loaded():
            return None
        if viewer is self.superResViewer and not viewer.has_photo():
            return None
        calibration_result = self._calibration_controller.display_result
        if viewer is self.calibrationViewer and calibration_result:
            return calibration_result.data
        return self._display_data()

    def _on_spectrum_plot(self, pos: QPointF) -> None:
        viewer = self.sender()
        if not isinstance(viewer, HSIViewer):
            return
        self._spectrum_controller.pin(viewer, int(pos.y()), int(pos.x()))

    def _on_mean_index(self, index_name: str) -> None:
        if self.sender() is self.superResViewer and self._high_resolution_selected:
            self.statusbar.showMessage("Index means are available for the original image in Visualization", 5000)
            return
        if not self._hsi_data.is_loaded():
            return

        try:
            mode = VisualizationMode(index_name)
        except ValueError:
            return

        if self._ensure_on_demand_result(mode):
            self._hypercube_controller.resume(self._display_data())
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
            VISUALIZATION_LABELS.get(index_name, index_name),
            mean_value,
            result.value_range,
            result.colormap,
            parent=self,
        )
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _pipeline_busy(self) -> bool:
        """Report whether a background worker holds exclusive access to the cube.

        Shared by `_crop_is_blocked` and crop undo/redo: all three must not
        run while SR, calibration, or classification is mutating/reading the
        same `_hsi_data`/SpyFile.
        """
        return (
            self._super_res_worker is not None
            or self._calibration_controller.is_running()
            or self._classification_controller.is_running()
        )

    def _crop_is_blocked(self) -> bool:
        """Report whether the pipeline can accept a crop right now.

        Shared by the rectangle and polygon paths; both invalidate the same
        downstream state, so both are subject to the same guards.
        """
        if self._calibration_controller.is_running():
            self.statusbar.showMessage("Cancel or finish calibration before cropping")
            return True
        if self._super_res_worker is not None:
            self.statusbar.showMessage("Cancel or finish SR before cropping")
            return True
        if self._high_resolution_selected and self._super_res_result is not None:
            # Every viewer displays the SR result at 2x; crop selections are
            # only valid against the Original the rest of the pipeline crops.
            self.statusbar.showMessage("Select Original before cropping, then rerun SR", 5000)
            return True
        if not self._hsi_data.is_loaded():
            return True
        if self._classification_controller.is_running():
            self.statusbar.showMessage(
                "Cancel classification before changing the image crop",
                5000,
            )
            return True
        return False

    def _invalidate_for_crop_change(self) -> None:
        """Discard classification/calibration state tied to the old crop geometry.

        Shared by applying a new crop and restoring a crop snapshot (undo/redo);
        both leave `_hsi_data` with a geometry that the prior results were not
        computed against.
        """
        self._classification_controller.clear_result()
        self._calibration_controller.source_geometry_changed()
        self._spectrum_controller.geometry_changed()

    def _apply_crop(self, crop: Callable[[], tuple[int, int] | None], label: str) -> None:
        """Snapshot, apply a crop operation, and refresh, or roll back."""
        self._crop_undo_stack.append(self._snapshot_current_state())
        self._crop_redo_stack.clear()

        cropped_size = crop()
        if cropped_size is None:
            self._crop_undo_stack.pop()
            return

        self._invalidate_for_crop_change()
        self.statusbar.showMessage(
            f"{label} to {cropped_size[0]}x{cropped_size[1]}"
        )

    def _on_crop_requested(self, rect: QtCore.QRectF) -> None:
        if self._crop_is_blocked():
            return
        self._apply_crop(
            lambda: self._hsi_data.crop(
                rect.left(), rect.top(), rect.right(), rect.bottom()
            ),
            "Cropped",
        )

    def _on_polygon_crop_requested(self, vertices: list) -> None:
        """Apply an irregular crop: bounding box plus a region-of-interest mask.

        The cube stays rectangular (SPy has no other representation), so this
        crops to the polygon's bounding box and records the polygon itself on
        ``HSIData.roi_mask``. Visualization stretches, index means, and
        classification then read through ``HSIData.masked`` and skip the
        excluded pixels.
        """
        if self._crop_is_blocked():
            return
        self._apply_crop(
            lambda: self._hsi_data.crop_polygon(vertices),
            "Cropped to polygon within",
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
        if viewer is self.calibrationViewer and self._calibration_controller.display_result:
            return 2 if self._is_super_resolution_active() else 1
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

        self._active_viewer = new_viewer

    def _snapshot_current_state(self) -> _CropSnapshot:
        return _CropSnapshot(
            rgb_array=self._hsi_data.rgb_array,
            mask_array=self._hsi_data.mask_array,
            spectral_obj=self._hsi_data.spectral_obj,
            roi_mask=self._hsi_data.roi_mask,
        )

    def _restore_snapshot(self, snapshot: _CropSnapshot) -> None:
        self._hsi_data.rgb_array    = snapshot.rgb_array
        self._hsi_data.mask_array   = snapshot.mask_array
        self._hsi_data.spectral_obj = snapshot.spectral_obj
        self._hsi_data.roi_mask     = snapshot.roi_mask
        self._invalidate_for_crop_change()

    def _annotation_history_active(self) -> bool:
        """Whether Ctrl+Z should target annotations rather than the crop."""

        return self._active_viewer is self.classificationViewer

    def _undo_crop(self) -> None:
        if self._annotation_history_active() and (
            self._classification_controller.undo_annotation()
        ):
            return
        if self._pipeline_busy() or not self._crop_undo_stack:
            return
        self._crop_redo_stack.append(self._snapshot_current_state())
        self._restore_snapshot(self._crop_undo_stack.pop())
        self.statusbar.showMessage("Crop undone")

    def _redo_crop(self) -> None:
        if self._annotation_history_active() and (
            self._classification_controller.redo_annotation()
        ):
            return
        if self._pipeline_busy() or not self._crop_redo_stack:
            return
        self._crop_undo_stack.append(self._snapshot_current_state())
        self._restore_snapshot(self._crop_redo_stack.pop())
        self.statusbar.showMessage("Crop redone")

    def _push_image_to_viewers(self) -> None:
        self._visualization_cache.clear()
        self._reset_super_resolution()
        self._refresh_visualization_pipeline()

    def _refresh_visualization_pipeline(self) -> None:
        self._recompute_visualizations()
        self._refresh_viewers_display()
        self._update_mode_caption()
        self._hypercube_controller.refresh(self._display_data(), reuse=True)


def _escape(text: str) -> str:
    """Escape plain text for a rich-text QLabel."""

    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
