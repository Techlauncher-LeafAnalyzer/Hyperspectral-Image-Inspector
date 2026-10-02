"""Calibration-tab state and GUI-thread orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QFileDialog, QMessageBox

import core.hsi_utils as hsi_utils
from core import (
    CalibrationFrameResolver,
    CalibrationResult,
    CalibrationService,
    HSIData,
    HSIError,
    VisualizationResult,
    order_calibration_frame_paths,
)
from ui.calibration_worker import CalibrationWorker
from ui.viewer import HSIViewer, PixelValueEntry


LOGGER = logging.getLogger(__name__)


class CalibrationController(QtCore.QObject):
    """Own calibration controls, paths, worker lifecycle, and rendered result."""

    readyToClose = QtCore.pyqtSignal()
    runningChanged = QtCore.pyqtSignal(bool)
    resultReady = QtCore.pyqtSignal(bool)
    referencesChanged = QtCore.pyqtSignal(bool)

    def __init__(
        self,
        source_data: HSIData,
        service: CalibrationService,
        frame_resolver: CalibrationFrameResolver,
        viewer: HSIViewer,
        statusbar: QtWidgets.QStatusBar,
        dark_button: QtWidgets.QPushButton,
        dark_edit: QtWidgets.QLineEdit,
        bright_button: QtWidgets.QPushButton,
        bright_edit: QtWidgets.QLineEdit,
        calibrate_button: QtWidgets.QPushButton,
        load_image_action: QtGui.QAction,
        stop_hypercube: Callable[[], None],
        resume_hypercube: Callable[[], None],
        refresh_source_views: Callable[[], None],
        refresh_current_views: Callable[[], None],
        fallback_pixel_values: Callable[
            [int, int], Mapping[str, PixelValueEntry]
        ],
        parent_widget: QtWidgets.QWidget,
        display_data: Callable[[], HSIData] | None = None,
        is_high_resolution: Callable[[], bool] | None = None,
        super_resolution_source: Callable[[], HSIData | None] | None = None,
    ) -> None:
        super().__init__(parent_widget)
        self._source_data = source_data
        self._service = service
        self._frame_resolver = frame_resolver
        self._viewer = viewer
        self._statusbar = statusbar
        self._dark_button = dark_button
        self._dark_edit = dark_edit
        self._bright_button = bright_button
        self._bright_edit = bright_edit
        self._calibrate_button = calibrate_button
        self._load_image_action = load_image_action
        self._stop_hypercube = stop_hypercube
        self._resume_hypercube = resume_hypercube
        self._refresh_source_views = refresh_source_views
        self._refresh_current_views = refresh_current_views
        self._fallback_pixel_values = fallback_pixel_values
        self._parent = parent_widget
        self._display_data = display_data or (lambda: self._source_data)
        self._is_high_resolution = is_high_resolution or (lambda: False)
        # Returns the raw 2x SR cube, or None when no SR image exists.
        self._super_resolution_source = super_resolution_source or (lambda: None)

        self._worker: CalibrationWorker | None = None
        self._results: dict[bool, CalibrationResult | None] = {False: None, True: None}
        self._working_resolution = False
        self._completed_resolution: bool | None = None
        self._pending_jobs: list = []
        self._error: str | None = None
        self._dark_path: Path | None = None
        self._bright_path: Path | None = None
        self._tracks_source: dict[bool, bool] = {False: False, True: False}
        self._external_running = False
        self._close_after_calibration = False

        self._dark_button.clicked.connect(self._select_dark_file)
        self._bright_button.clicked.connect(self._select_bright_file)
        self._calibrate_button.clicked.connect(self._run_or_cancel)
        self._dark_button.setToolTip("Select an ENVI/PSI dark-reference cube")
        self._bright_button.setToolTip("Select an ENVI/PSI bright-reference cube")
        self._viewer.pixel_value_provider = self.pixel_values_at
        self._update_ready()

    @property
    def service(self) -> CalibrationService:
        """Expose the injected service for diagnostics and test substitution."""

        return self._service

    @property
    def result(self) -> CalibrationResult | None:
        return self.result_for_resolution(self._is_high_resolution())

    def result_for_resolution(self, high: bool) -> CalibrationResult | None:
        return self._results[high]

    @property
    def last_error(self) -> str | None:
        return self._error

    @property
    def dark_path(self) -> Path | None:
        return self._dark_path

    @property
    def bright_path(self) -> Path | None:
        return self._bright_path

    def is_running(self) -> bool:
        # Keep source changes blocked until Qt has delivered the result and
        # finished signals, even if the OS thread has already exited.
        return self._worker is not None

    def set_external_running(self, running: bool) -> None:
        """Disable calibration controls while SR or classification owns the cube."""

        self._external_running = running
        self._update_ready()

    def resolution_changed(self) -> None:
        """Refresh controls when the shared Original/2× choice changes."""

        self._update_ready()

    def source_loaded(self, source_path: Path) -> str:
        """Reset stale state and optionally discover a nearby calibration pair."""

        self._tracks_source = {False: False, True: False}
        self._stop_hypercube()
        self.clear_result()
        pair = self._frame_resolver.resolve(source_path)
        if pair is None:
            self._set_paths(None, None)
            status = "No matching earlier calibration pair found; select files manually"
        else:
            self._set_paths(pair.dark_path, pair.bright_path)
            status = (
                f"Auto-selected calibration frames {pair.dark_path.name} and "
                f"{pair.bright_path.name} ({pair.frame_interval_seconds:g}s apart)"
            )
        self._update_ready()
        return status

    def source_geometry_changed(self) -> None:
        """Rebuild an existing result after crop, undo, or redo."""

        recalibrate = self._tracks_source[False]
        self._stop_hypercube()
        self.clear_result()
        self._refresh_source_views()
        if recalibrate:
            self._run_or_cancel()

    def clear_result(self) -> None:
        for high in (False, True):
            self._clear_resolution(high)
        self._error = None

    def clear_super_resolution_result(self) -> None:
        self._clear_resolution(True)

    def _clear_resolution(self, high: bool) -> None:
        previous = self._results[high]
        self._results[high] = None
        self._tracks_source[high] = False
        if previous is not None:
            previous.cleanup()

    def request_close(self) -> bool:
        """Cancel active work and tell the caller whether close must be deferred."""

        if not self.is_running():
            self.clear_result()
            return False
        self._close_after_calibration = True
        self._cancel()
        return True

    def pixel_values_at(
        self, row: int, column: int
    ) -> Mapping[str, PixelValueEntry]:
        result = self.result
        if result is None or result.data.rgb_array is None:
            return self._fallback_pixel_values(row, column)
        rgb = result.data.rgb_array
        if not (0 <= row < rgb.shape[0] and 0 <= column < rgb.shape[1]):
            return {}
        color = tuple(int(value) for value in rgb[row, column])
        return {"Calibrated RGB": PixelValueEntry(value=color, color=color)}

    def _select_dark_file(self) -> None:
        selected = self._select_file("Open Dark Calibration File")
        if selected is not None:
            self._dark_path = selected
            self._dark_edit.setText(str(selected))
            self._dark_edit.setToolTip(str(selected))
            self._reference_changed()

    def _select_bright_file(self) -> None:
        selected = self._select_file("Open Bright Calibration File")
        if selected is not None:
            self._bright_path = selected
            self._bright_edit.setText(str(selected))
            self._bright_edit.setToolTip(str(selected))
            self._reference_changed()

    def _select_file(self, title: str) -> Path | None:
        if self.is_running() or self._external_running:
            return None
        selected, _ = QFileDialog.getOpenFileName(
            self._parent,
            title,
            "",
            (
                "Hyperspectral Images (*.hdr *.bil *.bip *.bsq *.dat *.img *.raw);;"
                "All Files (*)"
            ),
        )
        if not selected:
            return None
        path = Path(selected).expanduser().resolve()
        self._statusbar.showMessage(f"Selected {path.name}")
        return path

    def revert_calibration(self) -> None:
        """Discard every calibration result (e.g. before running SR)."""

        self._reference_changed()

    def _reference_changed(self) -> None:
        had_result = self._results[False] is not None
        self._tracks_source = {False: False, True: False}
        self._stop_hypercube()
        self.clear_result()
        if self._source_data.is_loaded():
            self._refresh_current_views()
        self._update_ready()
        self.referencesChanged.emit(had_result)

    def _set_paths(self, dark: Path | None, bright: Path | None) -> None:
        self._dark_path = dark
        self._bright_path = bright
        for edit, path in ((self._dark_edit, dark), (self._bright_edit, bright)):
            if path is None:
                edit.clear()
                edit.setToolTip("")
            else:
                edit.setText(str(path))
                edit.setToolTip(str(path))

    def _run_or_cancel(self) -> None:
        if self._worker is not None:
            self._cancel()
            return
        if self._external_running:
            return
        if not self._source_data.is_loaded():
            QMessageBox.information(
                self._parent, "Nothing to calibrate", "Load an image first."
            )
            return
        if self._dark_path is None or self._bright_path is None:
            QMessageBox.critical(
                self._parent,
                "Calibration references required",
                "Select both a dark-reference cube and a bright-reference cube.",
            )
            return

        try:
            dark, bright = order_calibration_frame_paths(
                self._dark_path, self._bright_path
            )
        except HSIError as exc:
            QMessageBox.critical(self._parent, "Invalid calibration frames", str(exc))
            return
        if (dark, bright) != (self._dark_path, self._bright_path):
            self._set_paths(dark, bright)
        # Calibrate the original and, when an SR image exists, the SR image
        # in the same operation, so neither result is stale. The
        # SR cube is 2x the original; its references are interpolated using
        # the original as the geometry reference.
        jobs = [(False, self._source_data, None, dark, bright)]
        super_resolution = self._super_resolution_source()
        if super_resolution is not None:
            jobs.append((True, super_resolution, self._source_data, dark, bright))

        self._error = None
        self._stop_hypercube()
        self._pending_jobs = jobs[1:]
        self._completed_resolution = None
        self._load_image_action.setEnabled(False)
        self._dark_button.setEnabled(False)
        self._bright_button.setEnabled(False)
        self._calibrate_button.setText("Cancel")
        self._calibrate_button.setToolTip(
            "Cancel after the current calibration read completes"
        )
        self._statusbar.showMessage("Starting radiometric calibration…")
        self.runningChanged.emit(True)
        self._launch(jobs[0])

    def _launch(self, job) -> None:
        high, source_data, reference_source, dark, bright = job
        worker = CalibrationWorker(
            self._service,
            source_data,
            dark,
            bright,
            parent=self,
            reference_source=reference_source,
        )
        self._working_resolution = high
        self._worker = worker
        worker.progress.connect(self._on_progress)
        worker.result_ready.connect(self._on_result)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        worker.finished.connect(self._on_finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _cancel(self) -> None:
        if self._worker is None:
            return
        self._worker.requestInterruption()
        self._calibrate_button.setEnabled(False)
        self._calibrate_button.setText("Cancelling…")
        self._statusbar.showMessage("Cancelling after the current calibration read…")

    @QtCore.pyqtSlot(int, str)
    def _on_progress(self, value: int, message: str) -> None:
        self._statusbar.showMessage(f"Calibration {value}% — {message}")

    @QtCore.pyqtSlot(object, object)
    def _on_result(self, result: object, display: object) -> None:
        if self._worker is None:
            if isinstance(result, CalibrationResult):
                result.cleanup()
            return
        if self._worker.isInterruptionRequested():
            if isinstance(result, CalibrationResult):
                result.cleanup()
            self._on_cancelled()
            return
        if not isinstance(result, CalibrationResult) or not isinstance(
            display, VisualizationResult
        ):
            if isinstance(result, CalibrationResult):
                result.cleanup()
            self._on_failed("Worker returned an invalid result.")
            return
        result.data.rgb_array = display.display_rgb
        result.data.mask_array = np.zeros(display.display_rgb.shape[:2], dtype=np.uint8)
        self._clear_resolution(self._working_resolution)
        self._results[self._working_resolution] = result
        self._tracks_source[self._working_resolution] = True
        self._completed_resolution = self._working_resolution
        if self._is_high_resolution() == self._working_resolution:
            self._show_result()
        minimum, maximum = result.reflectance_range
        resolution_note = (
            " (2× interpolated references)" if self._working_resolution else ""
        )
        self._statusbar.showMessage(
            f"Calibration complete{resolution_note}: reflectance range "
            f"{minimum:.4g} to {maximum:.4g}",
            8000,
        )

    @QtCore.pyqtSlot(str)
    def _on_failed(self, message: str) -> None:
        self._pending_jobs = []
        self._error = f"Calibration failed: {message}"
        LOGGER.error("%s", self._error)
        if not self._close_after_calibration:
            QMessageBox.critical(self._parent, "Calibration failed", message)

    @QtCore.pyqtSlot()
    def _on_cancelled(self) -> None:
        self._pending_jobs = []
        self._error = "Calibration cancelled"

    @QtCore.pyqtSlot()
    def _on_finished(self) -> None:
        self._worker = None
        if self._pending_jobs and not self._error and not self._close_after_calibration:
            self._launch(self._pending_jobs.pop(0))
            return
        self._load_image_action.setEnabled(True)
        self._update_ready()
        self.runningChanged.emit(False)
        if self._completed_resolution is not None:
            self.resultReady.emit(self._completed_resolution)
            self._completed_resolution = None
        if self._error:
            self._statusbar.showMessage(self._error, 8000)
        if self._close_after_calibration:
            self._close_after_calibration = False
            QtCore.QTimer.singleShot(0, self.readyToClose.emit)
        else:
            self._resume_hypercube()

    def _update_ready(self) -> None:
        running = self._worker is not None
        controls_available = not running and not self._external_running
        self._dark_button.setEnabled(controls_available)
        self._bright_button.setEnabled(controls_available)
        if running:
            return
        ready = (
            controls_available
            and self._source_data.is_loaded()
            and self._dark_path is not None
            and self._bright_path is not None
        )
        self._calibrate_button.setEnabled(ready)
        self._calibrate_button.setText("Calibrate")
        if ready:
            tooltip = "Apply dark/bright radiometric calibration"
        elif not self._source_data.is_loaded():
            tooltip = "Load a source image before calibration"
        elif self._external_running:
            tooltip = "Wait for the current cube-processing task to finish"
        else:
            tooltip = "Select both dark and bright reference cubes"
        self._calibrate_button.setToolTip(tooltip)

    def _show_result(self) -> None:
        result = self.result
        if result is None or result.data.rgb_array is None:
            return
        state = self._viewer.get_view_state()
        self._viewer.rgb = result.data.rgb_array
        self._viewer.mask_array = result.data.mask_array
        self._viewer.set_photo(
            hsi_utils.numpy_to_qpixmap(result.data.rgb_array, result.data.roi_mask)
        )
        if state is not None:
            self._viewer.queue_view_state(state)
