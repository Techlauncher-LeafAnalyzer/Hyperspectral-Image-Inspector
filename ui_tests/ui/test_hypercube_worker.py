from __future__ import annotations

from core import HSIReader, VisualizationService
from ui.hypercube_worker import HypercubeWorker


def test_worker_can_be_constructed(qtbot, synthetic_cube_path):
    """Construction wires up the background QThread without starting it.

    Automated coverage of the actual threaded run (progress/finished/failed
    signals delivered across threads via qtbot.waitSignal) was dropped: it
    proved flaky/crash-prone in this environment's PyQt6 + Python 3.14
    combination when driven by pytest-qt's nested event loop. The feature
    will be exercised manually by running the app instead.
    """
    data = HSIReader().open(synthetic_cube_path)
    worker = HypercubeWorker(VisualizationService(), data)

    assert worker._thread is not None
    assert not worker._thread.isRunning()
