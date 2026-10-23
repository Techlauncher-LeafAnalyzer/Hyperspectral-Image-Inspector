"""Qt worker boundary for running a batch job off the GUI thread."""

from __future__ import annotations

import logging

from PyQt6 import QtCore

from core import BatchError, BatchRequest, BatchResult, BatchService, CancelledError, HSIError


LOGGER = logging.getLogger(__name__)


class BatchWorker(QtCore.QThread):
    """Run ``BatchService.run`` on a background thread, reporting progress."""

    progress = QtCore.pyqtSignal(int, str)
    result_ready = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        service: BatchService,
        request: BatchRequest,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._request = request

    def run(self) -> None:
        try:
            result: BatchResult = self._service.run(
                self._request,
                progress=lambda value, message: self.progress.emit(value, message),
                is_cancelled=self.isInterruptionRequested,
            )
            self.result_ready.emit(result)
        except CancelledError:
            self.result_ready.emit(BatchResult(file_results=(), cancelled=True))
        except (BatchError, HSIError) as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # Keep Qt's event loop alive on programming faults.
            LOGGER.exception("Unexpected batch worker failure")
            self.failed.emit(f"An unexpected error occurred: {exc}")
