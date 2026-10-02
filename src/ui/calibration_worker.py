"""Qt worker boundary for radiometric calibration and preview rendering."""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6 import QtCore

from core import (
    CalibrationResult,
    CalibrationService,
    CancelledError,
    HSIData,
    HSIError,
    HSIReader,
    VisualizationMode,
    VisualizationRequest,
    VisualizationService,
)


LOGGER = logging.getLogger(__name__)


class CalibrationWorker(QtCore.QThread):
    """Load references, calibrate the source, and render RGB off the GUI thread."""

    progress = QtCore.pyqtSignal(int, str)
    result_ready = QtCore.pyqtSignal(object, object)
    failed = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal()

    def __init__(
        self,
        service: CalibrationService,
        source_data: HSIData,
        dark_path: Path,
        bright_path: Path,
        parent=None,
        *,
        reference_source: HSIData | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._source_data = source_data
        self._dark_path = Path(dark_path)
        self._bright_path = Path(bright_path)
        self._reference_source = reference_source

    def run(self) -> None:
        dark: HSIData | None = None
        bright: HSIData | None = None
        result: CalibrationResult | None = None
        result_delivered = False
        try:
            self.progress.emit(0, "Opening calibration references")
            reader = HSIReader()
            dark = reader.open(self._dark_path)
            bright = reader.open(self._bright_path)
            result = self._service.calibrate(
                self._source_data,
                dark,
                bright,
                progress=lambda value, message: self.progress.emit(
                    int(value * 0.9), message
                ),
                is_cancelled=self.isInterruptionRequested,
                reference_source=self._reference_source,
            )
            self.progress.emit(95, "Rendering calibrated RGB preview")
            display = VisualizationService().render(
                result.data,
                VisualizationRequest(VisualizationMode.RGB),
                is_cancelled=self.isInterruptionRequested,
            )
            self.result_ready.emit(result, display)
            result_delivered = True
        except CancelledError:
            self.cancelled.emit()
        except HSIError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # Keep Qt's event loop alive on programming faults.
            LOGGER.exception("Unexpected calibration worker failure")
            self.failed.emit(f"An unexpected error occurred: {exc}")
        finally:
            if result is not None and not result_delivered:
                result.cleanup()
            if dark is not None:
                dark.close()
            if bright is not None:
                bright.close()
