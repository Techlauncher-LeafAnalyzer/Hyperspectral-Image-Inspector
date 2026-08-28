"""Tests for :class:`HypercubeWorker`'s real background-thread behaviour.

These tests drive an actual ``QThread``: the work runs off the GUI thread and
``progress``/``finished``/``failed`` arrive over genuine queued (cross-thread)
connections. They deliberately do **not** use ``qtbot.waitSignal``.

Why not ``qtbot.waitSignal``
---------------------------
``HypercubeWorker`` self-destructs: ``_thread.finished`` is wired to both
``_thread.deleteLater()`` and ``_prepare_for_deletion()`` (which calls
``self.deleteLater()``). Those ``DeferredDelete`` events sit on the GUI
thread's queue and are dispatched the moment the GUI thread runs an event
loop.

``qtbot.waitSignal`` runs a nested ``QEventLoop.exec()`` on the GUI thread for
the whole duration of the run. So the deletions get dispatched *while the
background thread is still winding down*, which is a use-after-free race on
two fronts:

1. ``~QThread`` runs while ``QThreadPrivate::finish()`` has emitted
   ``finished`` but not yet cleared ``running`` -- Qt's own
   "Destroyed while thread is still running" abort, or a deadlock on the
   thread's internal mutex.
2. pytest-qt's ``SignalBlocker._cleanup`` then calls ``_silent_disconnect``
   on signals whose sender C++ object has just been freed.

Reproduced here at 10/10 fresh ``pytest`` invocations with the ``waitSignal``
formulation: 7 native crashes (``SIGABRT``/``SIGSEGV``, the faulthandler trace
bottoming out in ``pytestqt/wait_signal.py`` ``_silent_disconnect`` -> sip)
and 3 indefinite hangs.

What makes these tests reliable instead
---------------------------------------
``_run_worker_to_completion`` enforces a strict ordering:

1. ``worker.start()``.
2. ``thread.wait(...)`` -- a *blocking* wait that runs **no** event loop, so
   nothing can be deleted while the thread is alive. When it returns, the OS
   thread has fully exited, making both teardown races above impossible by
   construction.
3. Only then pump the GUI thread, delivering the already-queued signal
   payloads and finally the ``DeferredDelete`` events, at a point where
   deleting the worker and its thread is unambiguously safe.

Signals are captured by a plain-Python ``_Recorder`` connected before
``start()``, so no pytest-qt machinery ever holds a connection to an object
that is about to be destroyed. These tests use the ``qapp`` fixture rather
than ``qtbot`` for the same reason.

Measured: 20/20 fresh ``pytest`` invocations of this file green, plus repeated
full ``ui_tests`` suite runs. Removing the pump in step 3 makes all three
delivery assertions fail (nothing arrives), which confirms the payloads really
do travel over the cross-thread event queue rather than a direct call.
"""

from __future__ import annotations

from PyQt6.QtCore import QCoreApplication, QEvent, QEventLoop, QThread

from core import HSIReader, VisualizationError, VisualizationService
from core.errors import CancelledError
from ui.hypercube_worker import HypercubeWorker

_TIMEOUT_MS = 10_000


class _Recorder:
    """Collects a worker's signals into plain Python lists.

    Connected before ``start()`` so every emission is queued to the GUI
    thread and delivered by :func:`_run_worker_to_completion`'s pump.
    """

    def __init__(self, worker: HypercubeWorker) -> None:
        self.progress: list[tuple[int, str]] = []
        self.finished: list[object] = []
        self.failed: list[str] = []
        worker.progress.connect(lambda value, message: self.progress.append((value, message)))
        worker.finished.connect(self.finished.append)
        worker.failed.connect(self.failed.append)


def _run_worker_to_completion(worker: HypercubeWorker) -> None:
    """Run ``worker`` on its real thread and settle it, crash-free.

    The ordering is load-bearing -- see this module's docstring. In short:
    block on ``QThread.wait()`` (which runs no event loop, so the worker's own
    ``deleteLater()`` calls cannot fire mid-teardown), and only pump the GUI
    thread afterwards.
    """
    thread = worker._thread
    worker.start()

    assert thread.wait(_TIMEOUT_MS), "background thread did not exit within timeout"

    # The thread is gone; delivering its queued emissions is now safe.
    QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents)
    # Perform the worker's/thread's self-deletion deterministically here rather
    # than leaving DeferredDelete events pending for a later test's event loop.
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


class _StubService:
    """Visualization service double that records the thread it ran on."""

    def __init__(self, outcome: str) -> None:
        self._outcome = outcome
        self.ran_on_thread_id: int | None = None

    def prepare_hypercube_view(self, data, *, progress=None, is_cancelled=None):
        self.ran_on_thread_id = int(QThread.currentThreadId())
        if progress is not None:
            progress(50, "halfway")
        if self._outcome == "cancel":
            assert is_cancelled is not None and is_cancelled()
            raise CancelledError("cancelled")
        if self._outcome == "unexpected":
            raise OSError("disk read failed")
        raise VisualizationError("boom")


def test_worker_can_be_constructed(qapp, synthetic_cube_path):
    """Construction wires up the background QThread without starting it."""
    data = HSIReader().open(synthetic_cube_path)
    worker = HypercubeWorker(VisualizationService(), data)

    assert worker._thread is not None
    assert not worker._thread.isRunning()


def test_worker_emits_finished_with_hypercube_view_data(qapp, synthetic_cube_path):
    data = HSIReader().open(synthetic_cube_path)
    worker = HypercubeWorker(VisualizationService(), data)
    recorder = _Recorder(worker)

    _run_worker_to_completion(worker)

    assert recorder.failed == []
    assert len(recorder.finished) == 1
    assert recorder.finished[0].top_rgb.shape == (8, 8, 3)


def test_worker_reports_progress(qapp, synthetic_cube_path):
    data = HSIReader().open(synthetic_cube_path)
    worker = HypercubeWorker(VisualizationService(), data)
    recorder = _Recorder(worker)

    _run_worker_to_completion(worker)

    assert recorder.progress != []
    assert all(0 <= value <= 100 for value, _ in recorder.progress)


def test_worker_emits_failed_on_visualization_error(qapp, synthetic_cube_path):
    data = HSIReader().open(synthetic_cube_path)
    service = _StubService("fail")
    worker = HypercubeWorker(service, data)
    recorder = _Recorder(worker)

    _run_worker_to_completion(worker)

    assert recorder.failed == ["boom"]
    assert recorder.finished == []
    # The work really did happen off the GUI thread.
    assert service.ran_on_thread_id != int(QThread.currentThreadId())


def test_worker_stops_after_an_unexpected_read_error(qapp, synthetic_cube_path):
    data = HSIReader().open(synthetic_cube_path)
    worker = HypercubeWorker(_StubService("unexpected"), data)
    recorder = _Recorder(worker)

    _run_worker_to_completion(worker)

    assert recorder.failed == ["Unable to prepare hypercube: disk read failed"]
    assert recorder.finished == []


def test_worker_cancel_suppresses_finished_and_failed(qapp, synthetic_cube_path):
    data = HSIReader().open(synthetic_cube_path)
    service = _StubService("cancel")
    worker = HypercubeWorker(service, data)
    recorder = _Recorder(worker)

    worker.cancel()
    _run_worker_to_completion(worker)

    assert recorder.finished == []
    assert recorder.failed == []
    assert service.ran_on_thread_id != int(QThread.currentThreadId())
