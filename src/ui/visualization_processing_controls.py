"""Visualization-only raw/calibrated switch and its selection state."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6 import QtCore, QtWidgets

from core import CalibrationResult, HSIData
from ui.resolution_toggle import ResolutionToggle


class VisualizationCalibrationController(QtCore.QObject):
    """Keep the calibrated preview independent from the global SR selection."""

    def __init__(
        self,
        canvas: QtWidgets.QWidget,
        calibration_result: Callable[[], CalibrationResult | None],
        is_high_resolution: Callable[[], bool],
        select_low_resolution: Callable[[], None],
        refresh: Callable[[], None],
        parent: QtCore.QObject,
    ) -> None:
        super().__init__(parent)
        self._calibration_result = calibration_result
        self._is_high_resolution = is_high_resolution
        self._select_low_resolution = select_low_resolution
        self._refresh = refresh
        self._selected = False
        self.toggle = ResolutionToggle(
            canvas,
            "visualizationCalibrationSwitch",
            low_label="RAW",
            high_label="CALIBRATED",
            low_symbol="0",
            high_symbol="✓",
            width=160,
        )
        self.toggle.clicked.connect(self._on_clicked)

    def _on_clicked(self, calibrated: bool) -> None:
        self._selected = calibrated
        if calibrated and self._is_high_resolution():
            self._select_low_resolution()
        else:
            self._refresh()

    def data(self, fallback: HSIData) -> HSIData:
        if not self._selected or self._is_high_resolution():
            return fallback
        result = self._calibration_result()
        return result.data if result is not None else fallback

    def reset(self) -> None:
        self._selected = False

    def calibration_ready(self) -> None:
        self._selected = True

    def sync(
        self,
        *,
        resolution_available: bool,
        enabled: bool,
    ) -> None:
        available = self._calibration_result() is not None
        self.toggle.move(self.toggle.INSET, 56 if resolution_available else 12)
        self.toggle.setChecked(self._selected and not self._is_high_resolution())
        self.toggle.setEnabled(enabled)
        self.toggle.setVisible(available)
        self.raise_control()

    def raise_control(self) -> None:
        if not self.toggle.isHidden():
            self.toggle.raise_()

    def schedule_raise(self) -> None:
        QtCore.QTimer.singleShot(0, self.raise_control)
