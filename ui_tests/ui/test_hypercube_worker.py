"""Tests for :class:`HypercubeWorker`'s real background-thread behaviour.

The service runs on a real QThread and signals return to the GUI. The helper
waits for completion before draining signals for deterministic assertions;
the lifecycle regression additionally pumps a live event loop throughout
the run and cleanup, as the application does during SR/tab interactions.
"""

from __future__ import annotations

from PyQt6 import sip
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
    """Run the real worker, then drain its queued signals and cleanup."""
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


def test_worker_cleanup_with_live_event_loop_stays_on_gui_thread(qapp, qtbot, synthetic_cube_path):
    data = HSIReader().open(synthetic_cube_path)
    gui_thread_id = int(QThread.currentThreadId())
    for _ in range(10):
        service = _StubService("fail")
        worker = HypercubeWorker(service, data)
        thread = worker._thread
        recorder = _Recorder(worker)
        destroyed_on = []
        worker.destroyed.connect(lambda: destroyed_on.append(int(QThread.currentThreadId())))
        assert worker.thread() is qapp.thread()
        worker.start()
        # Exercise normal event dispatch during teardown, not a blocking wait
        # that would hide the former background-thread QObject deletion race.
        qtbot.waitUntil(lambda: sip.isdeleted(worker), timeout=_TIMEOUT_MS)
        assert sip.isdeleted(thread)
        assert destroyed_on == [gui_thread_id]
        assert service.ran_on_thread_id != gui_thread_id
        assert recorder.failed == ["boom"]
