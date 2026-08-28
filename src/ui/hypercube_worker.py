from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from core import HSIData, VisualizationError, WavelengthError
from core.errors import CancelledError


class _HypercubeThread(QThread):
    def __init__(self, worker: HypercubeWorker) -> None:
        super().__init__(worker)
        self._worker = worker

    def run(self) -> None:
        self._worker._run()


class HypercubeWorker(QObject):
    """Computes ``HypercubeViewData`` on a background ``QThread``.

    Construct on the GUI thread with the visualization service and the
    ``HSIData`` to read, then call :meth:`start`. Exactly one of
    ``finished``/``failed`` fires per run; a ``cancel()`` that lands before
    completion fires neither, matching ``CancelledError``'s documented
    "neutral outcome" contract in ``core.errors``.
    """

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)  # HypercubeViewData
    failed = pyqtSignal(str)

    def __init__(self, service: Any, data: HSIData) -> None:
        super().__init__()
        self._service = service
        self._data = data
        self._cancelled = False

        # Keep the public QObject and its cleanup on the GUI thread. Only the
        # computation runs in the child thread, as in SuperResolutionWorker.
        self._thread = _HypercubeThread(self)
        self._thread.finished.connect(self._prepare_for_deletion)

    def start(self) -> None:
        """Begin computing on the background thread."""
        self._thread.start()

    def cancel(self) -> None:
        """Ask the in-flight computation to stop at its next checkpoint."""
        self._cancelled = True

    def _run(self) -> None:
        try:
            result = self._service.prepare_hypercube_view(
                self._data,
                progress=lambda value, message: self.progress.emit(value, message),
                is_cancelled=lambda: self._cancelled,
            )
        except CancelledError:
            pass
        except (VisualizationError, WavelengthError) as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"Unable to prepare hypercube: {exc}")
        else:
            self.finished.emit(result)

    @pyqtSlot()
    def _prepare_for_deletion(self) -> None:
        """Release QObjects on the GUI thread after OS-thread teardown."""
        self._thread.wait()
        self._thread.finished.disconnect(self._prepare_for_deletion)
        self._thread._worker = None
        self.deleteLater()
