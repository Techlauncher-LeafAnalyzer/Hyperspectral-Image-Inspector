from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QCoreApplication, QObject, QThread, pyqtSignal

from core import HSIData, VisualizationError, WavelengthError
from core.errors import CancelledError


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

        self._thread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._run)
        self._thread.finished.connect(self._thread.deleteLater)
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
        finally:
            # A disk or decoder failure must not strand the QThread: callers
            # serialize SPy reads by waiting for this thread to stop.
            self._thread.quit()

    def _prepare_for_deletion(self) -> None:
        """Finalize this run once the background thread is winding down.

        ``_thread.finished`` fires on the background thread just as it
        exits, invoking this method directly (same-thread connection). Two
        things need care here:

        1. ``self._thread`` and ``self`` hold references to each other (this
           object stores its own thread, and the ``started``/``finished``
           connections above keep bound methods of ``self`` alive on the
           thread's side), forming a reference cycle. Left alone, that cycle
           would only be broken by Python's periodic cyclic garbage
           collector, whose timing is unpredictable relative to the
           background thread's own OS-level teardown -- collecting these
           QObjects mid-teardown can crash. Disconnecting here drops the
           thread's references back to ``self``, letting ordinary
           (deterministic) reference counting reclaim both objects as soon
           as the caller drops its own reference.
        2. ``self.deleteLater()`` on an object still affined to the
           (about-to-vanish) background thread posts to that thread's own
           queue, which is a similarly fragile race. Reparenting to the GUI
           thread first makes it a normal cross-thread queued call instead.
        """
        self._thread.started.disconnect(self._run)
        self._thread.finished.disconnect(self._prepare_for_deletion)

        app = QCoreApplication.instance()
        if app is not None:
            self.moveToThread(app.thread())
        self.deleteLater()
