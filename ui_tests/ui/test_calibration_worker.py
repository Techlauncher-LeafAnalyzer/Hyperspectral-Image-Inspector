from __future__ import annotations

import os
from pathlib import Path
import threading

import numpy as np
import pytest
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


def _wait(qtbot, worker: CalibrationWorker) -> None:
    qtbot.waitUntil(lambda: not worker.isRunning(), timeout=30_000)


@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS", "").casefold() == "true",
    reason="Calibration preview worker test is unreliable on GitHub Actions; enabled locally.",
)
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

    worker.start()
    _wait(qtbot, worker)

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

    worker.start()
    _wait(qtbot, worker)

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

    worker.start()
    qtbot.waitUntil(started.is_set)
    worker.requestInterruption()
    _wait(qtbot, worker)

    assert cancelled == [True]
    assert results == []
