from __future__ import annotations

from pathlib import Path
import threading

import numpy as np
from spectral.io import envi

from core import CalibrationService, CancelledError
from ui.calibration_worker import CalibrationWorker

from conftest import SYNTHETIC_COLUMNS, SYNTHETIC_WAVELENGTHS_NM


def _reference(directory: Path, name: str, value: float, columns: int) -> Path:
    path = directory / f"{name}.hdr"
    envi.save_image(
        str(path),
        np.full(
            (21, columns, len(SYNTHETIC_WAVELENGTHS_NM)),
            value,
            dtype=np.float32,
        ),
        ext=".bip",
        interleave="bip",
        metadata={"wavelength": SYNTHETIC_WAVELENGTHS_NM},
    )
    return path


def _start(qtbot, worker: CalibrationWorker):
    """Start the worker; ``.wait()`` on the result blocks until it has finished.

    Polling ``isRunning()`` races the worker's queued signals: the thread can
    stop before ``result_ready``/``failed`` reach the GUI thread. ``finished``
    is queued after them, so once it arrives they have been delivered.
    """
    blocker = qtbot.waitSignal(worker.finished, timeout=30_000)
    worker.start()
    return blocker


def test_worker_calibrates_and_renders_preview(
    loaded_window, tmp_path, qtbot
):
    source = loaded_window._hsi_data
    raw = source.read_bands(range(source.bands)).copy()
    dark = _reference(tmp_path, "dark", 0.0, SYNTHETIC_COLUMNS)
    bright = _reference(tmp_path, "bright", 1.0, SYNTHETIC_COLUMNS)
    worker = CalibrationWorker(CalibrationService(), source, dark, bright)
    results = []
    failures = []
    worker.result_ready.connect(lambda result, display: results.append((result, display)))
    worker.failed.connect(failures.append)

    _start(qtbot, worker).wait()

    assert failures == []
    assert len(results) == 1
    result, display = results[0]
    np.testing.assert_allclose(
        result.data.read_bands(range(result.data.bands)), raw, atol=1e-6
    )
    assert display.display_rgb.shape == (source.rows, source.columns, 3)
    result.cleanup()


def test_worker_reports_model_validation_failure(
    loaded_window, tmp_path, qtbot
):
    source = loaded_window._hsi_data
    dark = _reference(tmp_path, "dark", 0.0, SYNTHETIC_COLUMNS)
    bright = _reference(tmp_path, "bright", 1.0, SYNTHETIC_COLUMNS - 1)
    worker = CalibrationWorker(CalibrationService(), source, dark, bright)
    results = []
    failures = []
    worker.result_ready.connect(lambda *items: results.append(items))
    worker.failed.connect(failures.append)

    _start(qtbot, worker).wait()

    assert results == []
    assert len(failures) == 1
    assert "columns" in failures[0]


def test_worker_honours_interruption_during_calibration(
    loaded_window, tmp_path, qtbot
):
    source = loaded_window._hsi_data
    dark = _reference(tmp_path, "dark", 0.0, SYNTHETIC_COLUMNS)
    bright = _reference(tmp_path, "bright", 1.0, SYNTHETIC_COLUMNS)
    started = threading.Event()

    class WaitingService:
        def calibrate(self, *args, is_cancelled, **kwargs):
            started.set()
            while not is_cancelled():
                threading.Event().wait(0.005)
            raise CancelledError("Calibration cancelled.")

    worker = CalibrationWorker(WaitingService(), source, dark, bright)
    cancelled = []
    results = []
    worker.cancelled.connect(lambda: cancelled.append(True))
    worker.result_ready.connect(lambda *items: results.append(items))

    done = _start(qtbot, worker)
    qtbot.waitUntil(started.is_set)
    worker.requestInterruption()
    done.wait()

    assert cancelled == [True]
    assert results == []
