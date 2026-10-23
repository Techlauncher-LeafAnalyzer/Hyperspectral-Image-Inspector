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
from ui.generated.MainWindow import Ui_MainWindow
from ui.index_mean_dialog import IndexMeanDialog
from ui.image_capabilities import ImageCapabilityController
from ui.resolution_toggle import ResolutionSwitchGroup
from ui.resource_usage import ResourceUsageWidget
from ui.hypercube_controller import HypercubeController
from ui.spectrum_controller import SpectrumController
from ui.super_resolution_worker import SuperResolutionWorker
from ui.tab_transition.handler import TabTransitionHandler
from ui.tool_windows import (
    ToolWindow,
    ToolWindowStripe,
    ViewToggleButton,
    paint_calibration_icon,
    paint_spectrum_icon,
    paint_super_resolution_icon,
)
from ui.theme import CLASSIFIER_BORDER_QSS, CLASSIFIER_POPUP_QSS
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
        self._active_visualization_mode: VisualizationMode = VisualizationMode.RGB
        # True while the Visualization tab shows the spectral-similarity map
        # instead of a VisualizationMode render.
        self._similarity_view = False
        self._visualization_results: dict[VisualizationMode, VisualizationResult] = {}
        self._visualization_cache: dict[
            tuple[bool, bool], tuple[HSIData, dict[VisualizationMode, VisualizationResult]]
        ] = {}
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
            separate_leaves_button=self.separateLeavesButton,
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
        self._configure_tabs()
        self._configure_file_menu()
        self._resource_usage = ResourceUsageWidget(self.tabWidget)
        self.tabWidget.setCornerWidget(
            self._resource_usage, QtCore.Qt.Corner.TopRightCorner
        )
        # Keep the Designer radio buttons as the shared state source; the visible
        # choice is the Super-Resolution toggle in the left tool-window stripe.
        self.lowResButton.hide()
        self.highResButton.hide()
        self.superResFlowArrow.hide()
        self.superResComparisonLabel.setText("Processing")
        self.superResGrid.addWidget(
            self.runSuperResButton, 0, 1, 1, 3, QtCore.Qt.AlignmentFlag.AlignLeft
        )
        self._configure_tool_windows()
        self._resolution_switches = ResolutionSwitchGroup(
            self.superResolutionSwitch,
            self.calibrationSwitch,
            self._select_canvas_resolution,
            self,
            on_calibration_clicked=self._select_canvas_calibration,
        )
        self._connect_signals()
        self._active_viewer = self._viewer_for_tab(self.tabWidget.currentIndex())
        self._stripe_geometry_timer = QtCore.QTimer(self)
        self._stripe_geometry_timer.setSingleShot(True)
        self._stripe_geometry_timer.timeout.connect(self._sync_stripe_geometry)
        self.tabWidget.installEventFilter(self)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is self.tabWidget and event.type() in (
            QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.Move,
            QtCore.QEvent.Type.Show, QtCore.QEvent.Type.LayoutRequest,
        ):
            self._stripe_geometry_timer.start(0)
        return super().eventFilter(watched, event)

    def _sync_stripe_geometry(self) -> None:
        """Fill the space left of the image area with buttons, and align the logo.

        Buttons span from the window edge to the image viewer with an even
        :data:`STRIPE_GAP` on both sides: the left gap is inside the stripe,
        and the right gap is measured to the image area itself, so the tab
        pane's border/padding count toward it rather than adding to it. The
        logo centres on the main tab bar's midpoint.
        """
        stripe = self.toolWindowStripe
        tab_bar = self.tabWidget.tabBar()
        page = self.tabWidget.currentWidget()
        if not tab_bar.count() or page is None:
            return
        # Distance from the stripe's right edge to where page content starts
        # (just the pane border: pages have no left margin).
        content_left = page.mapTo(self, QtCore.QPoint()).x()
        stripe_left = stripe.mapTo(self, QtCore.QPoint()).x()
        pane_inset = content_left - (stripe_left + stripe.stripe_width())
        stripe.set_right_inset(max(pane_inset, 0))
        # Siblings, not ancestors: map through global coordinates.
        mid = stripe.mapFromGlobal(tab_bar.mapToGlobal(tab_bar.tabRect(0).center())).y()
        stripe.align_logo_to(mid)

    # ------------------------------------------------------------------ #
    # Private: signal wiring                                               #
    # ------------------------------------------------------------------ #

    def _configure_tool_windows(self) -> None:
        """Build the JetBrains-style left stripe.

        Top: the Super-Resolution and Calibration view toggles, which switch
        every tab between image versions (highlighted = High Res / After,
        gray = nothing to compare yet). Below a divider: the Spectrum tool
        window, highlighted while open (docked or popped out).
        """
        self.toolWindowStripe = ToolWindowStripe(self)
        self.toolWindowStripe.set_logo(str(ICON_PATH))
        self.addToolBar(QtCore.Qt.ToolBarArea.LeftToolBarArea, self.toolWindowStripe)
        # The stripe now spans the old left margin up to the image area, and
        # each page's image starts at the pane's left edge, so the gap between
        # the stripe's buttons and the image is the same on every tab.
        for layout in (self.horizontalLayout, *(
            page.layout() for page in (
                self.Visualization, self.SuperResolution, self.Calibration,
                self.Classification, self.Batch,
            ) if page.layout() is not None
        )):
            margins = layout.contentsMargins()
            layout.setContentsMargins(0, margins.top(), margins.right(), margins.bottom())
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
        self.highResButton.toggled.connect(
            self._update_super_resolution_view_state
        )
        self.lowResButton.setToolTip(
            "View the original file before Super-Resolution processing"
        )
        self.highResButton.setToolTip(
            "View the processed result after Super-Resolution"
        )
        self.runSuperResButton.clicked.connect(self._run_super_resolution)
        self._set_super_resolution_ready()
        self._update_super_resolution_view_state(self.highResButton.isChecked())
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

        # SWIR water indices live outside the generated form: they are the
        # applicable modes for cameras whose range excludes the visible bands.
        self.modeNDWI = QtWidgets.QRadioButton("NDWI", self.modeSelect)
        self.modeNDWI.setObjectName("modeNDWI")
        self.modeNDMI = QtWidgets.QRadioButton("NDMI", self.modeSelect)
        self.modeNDMI.setObjectName("modeNDMI")
        for column, button in enumerate((self.modeNDWI, self.modeNDMI)):
            self.modeButtons.addButton(button)
            self.gridLayout.addWidget(button, 2, column, 1, 1)
        # Full-spectrum tools: a Spectral Angle Mapper view, and the live
        # Spectrum Inspector that drives it.
        self.modeSimilarity = QtWidgets.QRadioButton("Similarity", self.modeSelect)
        self.modeSimilarity.setObjectName("modeSimilarity")
        self.modeSimilarity.setToolTip(
            "Spectral angle to a reference spectrum across every band "
            "(bright = similar). Choose the reference in the Spectrum Inspector."
        )
        self.modeButtons.addButton(self.modeSimilarity)
        self.gridLayout.addWidget(self.modeSimilarity, 2, 2, 1, 1)
        self.modeSimilarity.toggled.connect(self._on_similarity_mode_toggled)
        self.spectrumInspectorButton = QtWidgets.QPushButton(
            "Spectrum Inspector", self.modeSelect
        )
        self.spectrumInspectorButton.setObjectName("spectrumInspectorButton")
        self.spectrumInspectorButton.setToolTip(
            "Open a live spectrum plot that follows the cursor over any image.\n"
            "Ctrl+Click a pixel (or right-click → Spectrum Analyzer) to pin it."
        )
        self.spectrumInspectorButton.clicked.connect(self._spectrum_controller.show)
        self.gridLayout.addWidget(self.spectrumInspectorButton, 2, 3, 1, 1)
        self.modeSelect.setMaximumHeight(170)

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
        )
        for button, mode in mode_buttons:
            button.toggled.connect(
                lambda checked, mode=mode: self._on_visualization_mode_toggled(mode, checked)
            )
        self._visualization_mode_buttons = mode_buttons
        self._image_capabilities.refresh_visualizations(self._hsi_data, mode_buttons)
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

    def _is_super_resolution_active(self) -> bool:
        """Return whether ``_display_data`` currently resolves to the SR result."""

        return self.highResButton.isChecked() and self._super_res_result is not None

    def _select_canvas_resolution(self, high_resolution: bool) -> None:
        if self._pipeline_busy():
            self._sync_image_switches()
            return
        if high_resolution:
            self.highResButton.setChecked(True)
        else:
            self.lowResButton.setChecked(True)

    def _select_canvas_calibration(self, calibrated: bool) -> None:
        if self._pipeline_busy():
            self._sync_image_switches()
            return
        previous = self._display_data()
        self._calibration_controller.select_calibrated(calibrated)
        if self._display_data() is not previous:
            self._update_super_resolution_view_state(self.highResButton.isChecked())
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
        if self.highResButton.isChecked() and self._super_res_result is None:
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
        if self.highResButton.isChecked() and self._super_res_result is None:
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
        self.lowResButton.setEnabled(False)
        self.highResButton.setEnabled(False)
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
        self.highResButton.setChecked(True)
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
        self._image_capabilities.refresh_super_resolution(self._hsi_data)

    def _reset_super_resolution(self) -> None:
        self._super_res_result = None
        self._super_res_error = None
        self.lowResButton.setChecked(True)
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
        self.lowResButton.setEnabled(not running)
        self.highResButton.setEnabled(not running)
        # runningChanged(True) precedes creation of the first calibration worker.
        self._sync_image_switches()
        if running:
            self._resolution_switches.set_enabled(False)

    @QtCore.pyqtSlot(bool)
    def _on_calibration_result_ready(self, high: bool) -> None:
        self._crop_undo_stack.clear()
        self._crop_redo_stack.clear()
        self._classification_controller.clear_result()
        self._clear_calibration_previews()
        self._update_super_resolution_view_state(self.highResButton.isChecked())

    @QtCore.pyqtSlot(bool)
    def _on_calibration_references_changed(self, had_result: bool) -> None:
        if had_result:
            self._classification_controller.clear_result()
        if self._hsi_data.is_loaded():
            self._clear_calibration_previews()
            self._update_super_resolution_view_state(self.highResButton.isChecked())

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

        loaded_file_text = f"File Loaded: {loaded_path}"
        self.imageFilePath.setText(loaded_file_text)
        self.imageFilePath.setToolTip(str(loaded_path))
        self.superResFilePath.setText(loaded_file_text)
        self.superResFilePath.setToolTip(str(loaded_path))
        self.classificationFilePath.setText(loaded_file_text)
        self.classificationFilePath.setToolTip(str(loaded_path))
        self._classification_controller.set_image_loaded(True)
        self._crop_undo_stack.clear()
        self._crop_redo_stack.clear()
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
        self.visualizationStack.setCurrentWidget(self.viewer)
        self._active_visualization_mode = mode
        self._similarity_view = False
        self._spectrum_controller.set_visualization(self._visualization_results.get(mode))
        if self._hsi_data.is_loaded():
            self._refresh_viewers_display()

    def _on_similarity_mode_toggled(self, checked: bool) -> None:
        self._similarity_view = checked
        if not checked:
            return
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
        if self.highResButton.isChecked() and self._super_res_result is not None:
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
        self._hypercube_controller.refresh(self._display_data(), reuse=True)
