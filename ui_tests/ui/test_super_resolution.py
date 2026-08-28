import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from PyQt6 import QtCore
from spectral.io import envi

from core import CancelledError, HSIReader, SuperResolutionError, SuperResolutionResult
from core.super_resolution_model import DEFAULT_CHECKPOINT


@pytest.fixture
def stub_sr(loaded_window, monkeypatch):
    """Controllable worker service for deterministic UI lifecycle tests."""
    control = SimpleNamespace(started=threading.Event(), release=threading.Event(), error=None)

    def run(data, request, *, progress, is_cancelled):
        control.started.set()
        progress(10, "Running test inference")
        while not control.release.wait(.005):
            if is_cancelled():
                raise CancelledError()
        if control.error:
            raise SuperResolutionError(control.error)
        storage = tempfile.TemporaryDirectory(prefix="sr-ui-test-")
        path = Path(storage.name) / "result.hdr"
        values = data.read_bands(range(data.bands)).repeat(2, 0).repeat(2, 1)
        envi.save_image(str(path), values, ext=".bip", interleave="bip",
                        metadata={"wavelength": data.wavelengths})
        return SuperResolutionResult(HSIReader().open(path), data.shape,
                                     Path("test.pth"), "cpu", False, _storage=storage)

    service = loaded_window._super_resolution_service
    monkeypatch.setattr(service, "validate", lambda *args: None)
    monkeypatch.setattr(service, "run", run)
    yield control
    control.release.set()


def finish(qtbot, window):
    qtbot.waitUntil(lambda: window._super_res_worker is None, timeout=30000)


def test_run_requires_input_and_rejects_wrong_bands(window, synthetic_cube_path, dialogs):
    assert not window.runSuperResButton.isEnabled()
    window.load_image_from_path(synthetic_cube_path)
    window.runSuperResButton.click()
    assert "exactly 480" in dialogs.critical[-1][-1]
    assert window._super_res_worker is None


def test_processed_selection_without_result_does_not_show_original(loaded_window):
    loaded_window.highResButton.setChecked(True)
    assert not loaded_window.superResViewer.has_photo()
    loaded_window.lowResButton.setChecked(True)
    assert loaded_window.superResViewer.has_photo()


def test_background_run_comparison_and_export(loaded_window, stub_sr, qtbot, file_dialog, tmp_path):
    window = loaded_window
    original = window._hsi_data.read_bands(range(8)).copy()
    ticks = []
    timer = QtCore.QTimer(window)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(5)
    window.runSuperResButton.click()
    qtbot.waitUntil(stub_sr.started.is_set)
    assert window.runSuperResButton.text() == "Cancel"
    assert not window.actionLoadImage.isEnabled()
    window.tabWidget.setCurrentIndex(2)
    qtbot.waitUntil(lambda: len(ticks) >= 3)
    assert window._super_res_worker is not None
    stub_sr.release.set()
    finish(qtbot, window)
    timer.stop()
    assert window.highResButton.isChecked()
    assert window.superResProgressBar.value() == 100
    assert window.superResViewer.rgb.shape == (16, 16, 3)
    assert window._super_res_result.data.shape == (16, 16, 8)
    assert window.actionLoadImage.isEnabled()
    np.testing.assert_array_equal(window._hsi_data.read_bands(range(8)), original)
    for viewer in (window.viewer, window.calibrationViewer, window.classificationViewer):
        assert viewer.rgb.shape == (8, 8, 3)
    window.lowResButton.setChecked(True)
    assert window.superResViewer.rgb.shape == (8, 8, 3)
    window.highResButton.setChecked(True)
    window.modeNDVI.setChecked(True)
    assert window.superResViewer.rgb.shape == (16, 16, 3)
    window.tabWidget.setCurrentWidget(window.SuperResolution)
    path = tmp_path / "sr.png"
    file_dialog.save_return = (str(path), "")
    window.actionSaveImage.trigger()
    with Image.open(path) as image:
        assert image.size == (16, 16)


def test_cancel_preserves_previous_result_and_blocks_source_changes(
    loaded_window, stub_sr, qtbot, synthetic_cube_path
):
    window = loaded_window
    stub_sr.release.set()
    window.runSuperResButton.click()
    finish(qtbot, window)
    previous = window._super_res_result
    stub_sr.release.clear()
    stub_sr.started.clear()
    window.runSuperResButton.click()
    qtbot.waitUntil(stub_sr.started.is_set)
    window.load_image_from_path(synthetic_cube_path)
    window._on_crop_requested(QtCore.QRectF(0, 0, 3, 3))
    assert window._hsi_data.shape == (8, 8, 8)
    assert window._super_res_result is previous
    window.runSuperResButton.click()
    finish(qtbot, window)
    assert window._super_res_result is previous
    assert "cancelled" in window.superResStatusText.text()
    assert window.runSuperResButton.isEnabled()


def test_error_restores_controls_and_retains_previous_result(loaded_window, stub_sr, qtbot, dialogs):
    window = loaded_window
    stub_sr.release.set()
    window.runSuperResButton.click()
    finish(qtbot, window)
    previous = window._super_res_result
    stub_sr.error = "Invalid checkpoint"
    window.runSuperResButton.click()
    finish(qtbot, window)
    assert "Invalid checkpoint" in dialogs.critical[-1][-1]
    assert window._super_res_result is previous
    assert window.lowResButton.isEnabled() and window.runSuperResButton.isEnabled()
    assert "failed" in window.superResStatusText.text()


def test_crop_and_new_load_invalidate_sr(loaded_window, stub_sr, qtbot, synthetic_cube_path):
    window = loaded_window
    stub_sr.release.set()
    window.runSuperResButton.click()
    finish(qtbot, window)
    window.superResViewer.cropRequested.emit(QtCore.QRectF(0, 0, 3, 3))
    assert window._hsi_data.shape == (8, 8, 8)  # HR coordinates cannot crop LR.
    window.lowResButton.setChecked(True)
    window._on_crop_requested(QtCore.QRectF(0, 0, 3, 3))
    assert window._super_res_result is None
    window.runSuperResButton.click()
    finish(qtbot, window)
    window.load_image_from_path(synthetic_cube_path)
    assert window._super_res_result is None and window.lowResButton.isChecked()


def test_close_cancels_without_destroying_running_thread(loaded_window, stub_sr, qtbot):
    loaded_window.show()
    loaded_window.runSuperResButton.click()
    qtbot.waitUntil(stub_sr.started.is_set)
    loaded_window.close()
    finish(qtbot, loaded_window)
    qtbot.waitUntil(lambda: not loaded_window.isVisible())


def test_sr_spectrum_uses_hr_coordinates(loaded_window, stub_sr, qtbot, monkeypatch):
    import ui.main_window as controller
    stub_sr.release.set()
    loaded_window.runSuperResButton.click()
    finish(qtbot, loaded_window)
    received = []

    def dialog(result, parent):
        received.append(result)
        return SimpleNamespace(exec=lambda: None)

    monkeypatch.setattr(controller, "SpectrumDialog", dialog)
    loaded_window.superResViewer.spectrumPlotRequested.emit(QtCore.QPointF(15, 14))
    assert len(received) == 1
    np.testing.assert_array_equal(received[0].values, loaded_window._super_res_result.data.read_pixel(14, 15))


def test_sr_tab_transfers_view_coordinates_at_correct_scale(loaded_window, stub_sr, qtbot, monkeypatch):
    stub_sr.release.set()
    loaded_window.runSuperResButton.click()
    finish(qtbot, loaded_window)
    window = loaded_window
    window._active_viewer = window.viewer
    monkeypatch.setattr(window.viewer, "get_view_state", lambda: (4.0, QtCore.QPointF(3, 5)))
    received = []
    monkeypatch.setattr(window.superResViewer, "queue_view_state", received.append)
    window._on_tab_changed(1)
    assert received[-1] == (2.0, QtCore.QPointF(6, 10))


@pytest.mark.sr_model
def test_real_model_runs_from_button_without_blocking_qt(window, sr_source, qtbot, dialogs):
    pytest.importorskip("torch")
    pytest.importorskip("scipy")
    if not DEFAULT_CHECKPOINT.is_file():
        pytest.skip("Supply model/fin_msdformer.pth to test actual inference")
    window.load_image_from_path(sr_source[0].source_path)
    window.tabWidget.setCurrentWidget(window.SuperResolution)
    ticks = []
    timer = QtCore.QTimer(window)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(5)
    window.runSuperResButton.click()
    finish(qtbot, window)
    timer.stop()
    assert ticks and not dialogs.critical
    assert window._super_res_result.data.shape == (14, 18, 480)
    assert window.superResViewer.has_photo()
    assert window.superResViewer.rgb.shape == (14, 18, 3)
    assert window.superResProgressBar.value() == 100
