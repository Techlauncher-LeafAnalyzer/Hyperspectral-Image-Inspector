"""Qt boundary for synchronous SR/visualization Models."""

import logging

from PyQt6 import QtCore

from core import (
    CancelledError, HSIError, SuperResolutionRequest,
    VisualizationRequest, VisualizationService,
)


LOGGER = logging.getLogger(__name__)


class SuperResolutionWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, str)
    result_ready = QtCore.pyqtSignal(object, object)
    failed = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal()

    def __init__(self, service, data, request=SuperResolutionRequest(), parent=None):
        super().__init__(parent)
        self._service = service
        self._data = data
        self._request = request

    def run(self):
        try:
            result = self._service.run(
                self._data, self._request,
                progress=lambda value, message: self.progress.emit(int(value * .95), message),
                is_cancelled=self.isInterruptionRequested,
            )
            self.progress.emit(97, "Rendering SR preview")
            display = VisualizationService().render(
                result.data, VisualizationRequest("RGB"),
                is_cancelled=self.isInterruptionRequested,
            )
            self.result_ready.emit(result, display)
        except CancelledError:
            self.cancelled.emit()
        except HSIError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            LOGGER.exception("Unexpected SR worker failure")
            self.failed.emit(f"Unexpected Super-Resolution failure: {exc}")
