from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from PyQt6 import QtCore
from spectral.io import envi

from core import CancelledError
from ui.viewer import PixelValueEntry

from conftest import SYNTHETIC_COLUMNS, SYNTHETIC_WAVELENGTHS_NM


def _write_reference(
    directory: Path,
    name: str,
    values: np.ndarray,
) -> Path:
    path = directory / f"{name}.hdr"
    envi.save_image(
        str(path),
        np.asarray(values, dtype=np.float32),
        ext=".bip",
        interleave="bip",
        metadata={"wavelength": SYNTHETIC_WAVELENGTHS_NM},
    )
    return path


def _select_references(window, file_dialog, dark_path: Path, white_path: Path):
    file_dialog.open_return = (str(dark_path), "")
    window.darkFileButton.click()
    file_dialog.open_return = (str(white_path), "")
    window.referenceFileButton.click()


def _wait_for_calibration(qtbot, window):
    qtbot.waitUntil(
        lambda: not window._calibration_controller.is_running(), timeout=30000
    )


def test_loading_timestamped_source_auto_selects_nearest_calibration_pair(
    window, tmp_path
):
    bands = len(SYNTHETIC_WAVELENGTHS_NM)
    source = _write_reference(
        tmp_path,
        "2025-04-17--12-51-23_round-0_cam-1_tray-1",
        np.full((8, SYNTHETIC_COLUMNS, bands), 0.5, dtype=np.float32),
    )
    dark = _write_reference(
        tmp_path,
        "2025-04-17--12-49-24_round-0_cam-1_calibFrame",
        np.zeros((21, SYNTHETIC_COLUMNS, bands), dtype=np.float32),
    )
    bright = _write_reference(
        tmp_path,
        "2025-04-17--12-49-29_round-0_cam-1_calibFrame",
        np.ones((21, SYNTHETIC_COLUMNS, bands), dtype=np.float32),
    )

    window.load_image_from_path(source)

    assert window._calibration_controller.dark_path == dark.resolve()
    assert window._calibration_controller.bright_path == bright.resolve()
    assert window.darkFileEdit.text() == str(dark.resolve())
    assert window.referenceFileEdit.text() == str(bright.resolve())
    assert window.calibrateButton.isEnabled()
    assert "Auto-selected calibration frames" in window.statusbar.currentMessage()


def test_loading_source_without_matching_pair_leaves_manual_selection_available(
    loaded_window, synthetic_cube_path, file_dialog, tmp_path
):
    window = loaded_window
    bands = len(SYNTHETIC_WAVELENGTHS_NM)
    dark = _write_reference(
        tmp_path,
        "manual_dark",
        np.zeros((21, SYNTHETIC_COLUMNS, bands), dtype=np.float32),
    )
    bright = _write_reference(
        tmp_path,
        "manual_bright",
        np.ones((21, SYNTHETIC_COLUMNS, bands), dtype=np.float32),
    )
    _select_references(window, file_dialog, dark, bright)
    assert window.calibrateButton.isEnabled()

    window.load_image_from_path(synthetic_cube_path)

    assert window._calibration_controller.dark_path is None
    assert window._calibration_controller.bright_path is None
    assert window.darkFileEdit.text() == ""
    assert window.referenceFileEdit.text() == ""
    assert not window.calibrateButton.isEnabled()
    assert "select files manually" in window.statusbar.currentMessage()


def test_manual_calibration_selection_preserves_existing_sr_result(
    loaded_window, file_dialog, tmp_path
):
    window = loaded_window
    previous_sr_result = object()
    window._super_res_result = previous_sr_result
    bands = len(SYNTHETIC_WAVELENGTHS_NM)
    dark = _write_reference(
        tmp_path,
        "manual_dark",
        np.zeros((21, SYNTHETIC_COLUMNS, bands), dtype=np.float32),
    )

    file_dialog.open_return = (str(dark), "")
    window.darkFileButton.click()

    assert window._calibration_controller.dark_path == dark.resolve()
    assert window._super_res_result is previous_sr_result


def test_reference_selection_enables_background_calibration_and_renders_result(
    loaded_window, file_dialog, tmp_path, qtbot, dialogs
):
    window = loaded_window
    raw = window._hsi_data.read_bands(range(window._hsi_data.bands)).copy()
    shape = (21, SYNTHETIC_COLUMNS, len(SYNTHETIC_WAVELENGTHS_NM))
    dark_values = np.full(shape, 0.15, dtype=np.float32)
    white_values = np.full(shape, 1.15, dtype=np.float32)
    dark_path = _write_reference(tmp_path, "dark", dark_values)
    white_path = _write_reference(tmp_path, "white", white_values)

    assert not window.calibrateButton.isEnabled()
    _select_references(window, file_dialog, dark_path, white_path.with_suffix(".bip"))

    assert window.calibrateButton.isEnabled()
    assert window._calibration_controller.dark_path == dark_path.resolve()
    assert window._calibration_controller.bright_path == white_path.with_suffix(
        ".bip"
    ).resolve()
    window.calibrateButton.click()
    assert window.calibrateButton.text() == "Cancel"
    assert not window.actionLoadImage.isEnabled()
    _wait_for_calibration(qtbot, window)

    assert not dialogs.critical
    result = window._calibration_controller.result
    assert result is not None
    expected = raw - 0.15
    np.testing.assert_allclose(
        result.data.read_bands(range(result.data.bands)), expected, atol=1e-6
    )
    np.testing.assert_array_equal(
        window._hsi_data.read_bands(range(window._hsi_data.bands)), raw
    )
    assert window.calibrationViewer.has_photo()
    np.testing.assert_array_equal(
        window.calibrationViewer.rgb, result.data.rgb_array
    )
    assert window.calibrateButton.isEnabled()
    assert window.calibrateButton.text() == "Calibrate"
    assert window.actionLoadImage.isEnabled()


def test_timestamped_calibration_frames_are_reordered_in_the_gui(
    loaded_window, file_dialog, tmp_path, qtbot, dialogs
):
    window = loaded_window
    raw = window._hsi_data.read_bands(range(window._hsi_data.bands)).copy()
    shape = (21, SYNTHETIC_COLUMNS, len(SYNTHETIC_WAVELENGTHS_NM))
    earlier_dark = _write_reference(
        tmp_path,
        "2025-04-17--12-49-24_round-0_cam-1_calibFrame",
        np.zeros(shape, dtype=np.float32),
    )
    later_bright = _write_reference(
        tmp_path,
        "2025-04-17--12-49-29_round-0_cam-1_calibFrame",
        np.ones(shape, dtype=np.float32),
    )
    _select_references(window, file_dialog, later_bright, earlier_dark)

    window.calibrateButton.click()
    _wait_for_calibration(qtbot, window)

    assert not dialogs.critical
    assert window._calibration_controller.dark_path == earlier_dark.resolve()
    assert window._calibration_controller.bright_path == later_bright.resolve()
    assert window.darkFileEdit.text() == str(earlier_dark.resolve())
    assert window.referenceFileEdit.text() == str(later_bright.resolve())
    result = window._calibration_controller.result
    assert result is not None
    np.testing.assert_allclose(
        result.data.read_bands(range(result.data.bands)), raw, atol=1e-6
    )


def test_invalid_reference_pair_fails_with_actionable_message(
    loaded_window, file_dialog, tmp_path, qtbot, dialogs
):
    bands = len(SYNTHETIC_WAVELENGTHS_NM)
    dark = _write_reference(
        tmp_path,
        "dark",
        np.zeros((21, SYNTHETIC_COLUMNS, bands), dtype=np.float32),
    )
    white = _write_reference(
        tmp_path,
        "white_wrong_width",
        np.ones((21, SYNTHETIC_COLUMNS - 1, bands), dtype=np.float32),
    )
    _select_references(loaded_window, file_dialog, dark, white)

    loaded_window.calibrateButton.click()
    _wait_for_calibration(qtbot, loaded_window)

    assert loaded_window._calibration_controller.result is None
    assert dialogs.critical
    message = dialogs.critical[-1][-1]
    assert "columns" in message and "must match" in message
    assert loaded_window.calibrateButton.isEnabled()


def test_calibration_can_be_cancelled_without_replacing_the_view(
    loaded_window, file_dialog, tmp_path, qtbot, monkeypatch
):
    window = loaded_window
    bands = len(SYNTHETIC_WAVELENGTHS_NM)
    dark = _write_reference(
        tmp_path,
        "dark",
        np.zeros((21, SYNTHETIC_COLUMNS, bands), dtype=np.float32),
    )
    white = _write_reference(
        tmp_path,
        "white",
        np.ones((21, SYNTHETIC_COLUMNS, bands), dtype=np.float32),
    )
    _select_references(window, file_dialog, dark, white)
    started = threading.Event()

    def wait_for_cancel(*args, progress, is_cancelled, **kwargs):
        started.set()
        progress(20, "Waiting for cancellation")
        while not is_cancelled():
            threading.Event().wait(0.005)
        raise CancelledError("Calibration cancelled.")

    monkeypatch.setattr(
        window._calibration_controller.service, "calibrate", wait_for_cancel
    )
    original_rgb = window.calibrationViewer.rgb.copy()
    window.calibrateButton.click()
    qtbot.waitUntil(started.is_set)

    window.calibrateButton.click()
    _wait_for_calibration(qtbot, window)

    assert window._calibration_controller.result is None
    np.testing.assert_array_equal(window.calibrationViewer.rgb, original_rgb)
    assert "cancelled" in window._calibration_controller.last_error.lower()


def test_calibrated_spectrum_hover_and_export_use_calibrated_data(
    loaded_window, file_dialog, tmp_path, qtbot, monkeypatch
):
    window = loaded_window
    bands = len(SYNTHETIC_WAVELENGTHS_NM)
    dark = _write_reference(
        tmp_path,
        "dark",
        np.zeros((21, SYNTHETIC_COLUMNS, bands), dtype=np.float32),
    )
    white = _write_reference(
        tmp_path,
        "white",
        np.full((21, SYNTHETIC_COLUMNS, bands), 2, dtype=np.float32),
    )
    _select_references(window, file_dialog, dark, white)
    window.calibrateButton.click()
    _wait_for_calibration(qtbot, window)
    result = window._calibration_controller.result
    assert result is not None

    rgb = tuple(int(value) for value in result.data.rgb_array[1, 2])
    assert window.calibrationViewer.pixel_value_provider(1, 2) == {
        "Calibrated RGB": PixelValueEntry(value=rgb, color=rgb)
    }

    received = []
    import ui.main_window as controller

    monkeypatch.setattr(
        controller,
        "SpectrumDialog",
        lambda spectrum, parent: SimpleNamespace(
            exec=lambda: received.append(spectrum)
        ),
    )
    window.calibrationViewer.spectrumPlotRequested.emit(QtCore.QPointF(2, 1))
    assert len(received) == 1
    np.testing.assert_allclose(
        received[0].values, result.data.read_pixel(1, 2), atol=1e-6
    )

    output = tmp_path / "calibrated.png"
    file_dialog.save_return = (str(output), "")
    window.tabWidget.setCurrentWidget(window.Calibration)
    window.actionSaveImage.trigger()
    with Image.open(output) as image:
        np.testing.assert_array_equal(np.asarray(image), result.data.rgb_array)

    window.tabWidget.setCurrentWidget(window.Visualization)
    assert window._display_data() is result.data
    assert window.visualizationStack.findChild(QtCore.QObject, "visualizationCalibrationSwitch") is None
    window.viewer.spectrumPlotRequested.emit(QtCore.QPointF(2, 1))
    np.testing.assert_allclose(
        received[-1].values, result.data.read_pixel(1, 2), atol=1e-6
    )
    file_dialog.save_return = (str(tmp_path / "visualized_calibrated.png"), "")
    window.actionSaveImage.trigger()
    with Image.open(tmp_path / "visualized_calibrated.png") as image:
        np.testing.assert_array_equal(np.asarray(image), window.viewer.rgb)


def test_calibration_tracks_crop_from_any_tab_and_clears_history(
    loaded_window, file_dialog, tmp_path, qtbot
):
    window = loaded_window
    raw = window._hsi_data.read_bands(range(window._hsi_data.bands)).copy()
    bands = len(SYNTHETIC_WAVELENGTHS_NM)
    dark = _write_reference(
        tmp_path,
        "dark",
        np.zeros((21, SYNTHETIC_COLUMNS, bands), dtype=np.float32),
    )
    white = _write_reference(
        tmp_path,
        "white",
        np.ones((21, SYNTHETIC_COLUMNS, bands), dtype=np.float32),
    )
    _select_references(window, file_dialog, dark, white)
    window.calibrateButton.click()
    _wait_for_calibration(qtbot, window)
    initial_result = window._calibration_controller.result
    assert initial_result is not None

    # Crop from the Calibration tab itself.
    window.calibrationViewer.cropRequested.emit(QtCore.QRectF(0, 0, 6, 6))
    _wait_for_calibration(qtbot, window)
    calibrated = window._calibration_controller.result
    assert calibrated is not None and calibrated is not initial_result
    assert calibrated.data.shape == (6, 6, bands)
    assert calibrated.source_spatial_bounds == ((0, 6), (0, 6))
    np.testing.assert_allclose(
        calibrated.data.read_bands(range(bands)), raw[:6, :6, :], atol=1e-6
    )
    assert not window._crop_undo_stack
    assert not window._crop_redo_stack

    window._undo_crop()
    assert window._calibration_controller.result is calibrated
    assert calibrated.data.shape == (6, 6, bands)

    # Cropping from another page also rebuilds the reference geometry.
    window.viewer.cropRequested.emit(QtCore.QRectF(1, 1, 4, 4))
    _wait_for_calibration(qtbot, window)
    calibrated = window._calibration_controller.result
    assert calibrated is not None
    assert calibrated.data.shape == (4, 4, bands)
    assert calibrated.source_spatial_bounds == ((1, 5), (1, 5))
    np.testing.assert_allclose(
        calibrated.data.read_bands(range(bands)), raw[1:5, 1:5, :], atol=1e-6
    )
    assert not window._crop_undo_stack
    assert not window._crop_redo_stack
    window._undo_crop()
    window._redo_crop()
    assert window._calibration_controller.result is calibrated
    assert window._hsi_data.shape == (4, 4, bands)
    assert window.calibrationViewer.rgb.shape == (4, 4, 3)


def test_calibration_clears_obsolete_classification(
    loaded_window, file_dialog, tmp_path, qtbot
):
    window = loaded_window
    window.numOfClassesEdit.setText("2")
    window.maxIterationsEdit.setText("3")
    window.unsupervisedClassifyButton.click()
    qtbot.waitUntil(lambda: window._classification_controller._thread is None)
    assert window._classification_controller._slots[False].result is not None

    shape = (21, SYNTHETIC_COLUMNS, len(SYNTHETIC_WAVELENGTHS_NM))
    dark = _write_reference(tmp_path, "dark", np.zeros(shape, dtype=np.float32))
    bright = _write_reference(tmp_path, "bright", np.full(shape, 2, dtype=np.float32))
    _select_references(window, file_dialog, dark, bright)
    assert window._classification_controller._slots[False].result is not None
    qtbot.waitUntil(lambda: window.calibrateButton.isEnabled())
    window.calibrateButton.click()
    _wait_for_calibration(qtbot, window)
    assert window._calibration_controller.result is not None
    assert all(slot.result is None for slot in window._classification_controller._slots.values())
