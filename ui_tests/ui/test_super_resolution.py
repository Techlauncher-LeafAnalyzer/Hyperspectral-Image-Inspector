import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from PyQt6 import QtCore, QtWidgets
from spectral.io import envi

from core import CancelledError, HSIReader, SuperResolutionError, SuperResolutionResult, VisualizationMode
from core.super_resolution_model import DEFAULT_CHECKPOINT
from ui.index_mean_dialog import IndexMeanDialog
from ui.viewer import PixelValueEntry, format_pixel_values_html


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


def test_canvas_resolution_switches_follow_the_toggle(loaded_window, stub_sr, qtbot):
    window = loaded_window
    switches = window._resolution_switches.switches
    assert len(switches) == 3
    assert all(switch.isHidden() for switch in switches)

    # Checking high-res before a result exists changes nothing: there is
    # still no Super-Resolution data to view.
    window.highResButton.setChecked(True)
    assert all(switch.isHidden() for switch in switches)
    window.lowResButton.setChecked(True)

    stub_sr.release.set()
    window.runSuperResButton.click()
    finish(qtbot, window)
    assert window.highResButton.isChecked()
    assert all(not switch.isHidden() and switch.isChecked() for switch in switches)

    switches[2].click()
    assert window.lowResButton.isChecked()
    assert all(not switch.isChecked() for switch in switches)
    switches[0].click()
    assert window.highResButton.isChecked()
    assert all(switch.isChecked() for switch in switches)


def test_super_resolution_on_calibrated_image_warns_and_reverts_calibration(
    loaded_window, stub_sr, qtbot, file_dialog, tmp_path, monkeypatch
):
    window = loaded_window
    badges = window._resolution_switches.calibration_badges
    source = window._hsi_data
    shape = (21, source.original_shape[1], source.bands)
    _set_references(window, file_dialog, tmp_path, np.zeros(shape), np.full(shape, 2.0))
    window.calibrateButton.click()
    qtbot.waitUntil(lambda: not window._calibration_controller.is_running())
    assert window._calibration_controller.result is not None
    assert all(not badge.isHidden() for badge in badges)
    ok, cancel = QtWidgets.QMessageBox.StandardButton.Ok, QtWidgets.QMessageBox.StandardButton.Cancel
    answers, prompts = [cancel, ok], []

    def warn(*args):
        prompts.append(args[1:3])
        return answers.pop(0)

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(warn))

    # Cancelling keeps the calibration and does not start SR.
    window.runSuperResButton.click()
    assert window._super_res_worker is None and window._super_res_result is None
    assert window._calibration_controller.result_for_resolution(False) is not None
    assert all(not badge.isHidden() for badge in badges)

    # Accepting reverts calibration and runs SR on the raw (uncalibrated) cube.
    stub_sr.release.set()
    window.runSuperResButton.click()
    finish(qtbot, window)
    assert len(prompts) == 2 and "uncalibrated" in prompts[0][0] + prompts[0][1]
    assert window._calibration_controller.result_for_resolution(False) is None
    assert all(badge.isHidden() for badge in badges)
    assert window._display_data() is window._super_res_result.data
    np.testing.assert_allclose(
        window._super_res_result.data.read_bands(range(source.bands)),
        source.read_bands(range(source.bands)).repeat(2, 0).repeat(2, 1),
    )


def _set_references(window, file_dialog, tmp_path, dark, bright):
    source = window._hsi_data
    for name, values in (("dark", dark), ("bright", bright)):
        path = tmp_path / f"{name}.hdr"
        envi.save_image(
            str(path), values.astype(np.float32), ext=".bip", interleave="bip",
            metadata={"wavelength": source.wavelengths},
        )
        file_dialog.open_return = (str(path), "")
        (window.darkFileButton if name == "dark" else window.referenceFileButton).click()


def test_calibrating_with_raw_sr_calibrates_both_resolutions_without_prompt(
    loaded_window, stub_sr, qtbot, file_dialog, tmp_path, monkeypatch
):
    window = loaded_window
    source = window._hsi_data
    shape = (21, source.original_shape[1], source.bands)
    _set_references(window, file_dialog, tmp_path, np.zeros(shape), np.full(shape, 2.0))
    stub_sr.release.set()
    window.runSuperResButton.click()
    finish(qtbot, window)
    previous_sr = window._super_res_result
    badges = window._resolution_switches.calibration_badges
    assert all(badge.isHidden() for badge in badges)
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "question",
        staticmethod(lambda *a: pytest.fail("calibration must not ask to discard SR")),
    )

    for view in (window.lowResButton, window.highResButton):
        view.setChecked(True)
        window.calibrateButton.click()
        qtbot.waitUntil(lambda: not window._calibration_controller.is_running())
        controller = window._calibration_controller
        low, high = controller.result_for_resolution(False), controller.result_for_resolution(True)
        assert low is not None and high is not None
        assert all(not badge.isHidden() for badge in badges)
        assert window._super_res_result is previous_sr
        bands = range(source.bands)
        np.testing.assert_allclose(low.data.read_bands(bands), source.read_bands(bands) / 2)
        np.testing.assert_allclose(high.data.read_bands(bands),
                                   previous_sr.data.read_bands(bands) / 2)


def test_calibrating_super_resolution_image_interpolates_references(
    loaded_window, stub_sr, qtbot, file_dialog, tmp_path, dialogs
):
    window = loaded_window
    source = window._hsi_data
    columns, bands = source.original_shape[1], source.bands
    bright_row = (1.0 + np.arange(columns))[:, None] * np.ones((1, bands))
    _set_references(
        window, file_dialog, tmp_path,
        np.zeros((21, columns, bands)), np.broadcast_to(bright_row, (21, columns, bands)),
    )
    stub_sr.release.set()
    window.runSuperResButton.click()
    finish(qtbot, window)
    sr = window._super_res_result
    assert window.highResButton.isChecked()

    window.calibrateButton.click()
    qtbot.waitUntil(lambda: not window._calibration_controller.is_running())

    assert not dialogs.critical
    result = window._calibration_controller.result
    assert result is not None and result is window._calibration_controller.result_for_resolution(True)
    assert result.data.shape == sr.data.shape
    assert window._super_res_result is sr
    low = window._calibration_controller.result_for_resolution(False)
    assert low is not None and low.data.shape == source.shape
    assert window._display_data() is result.data
    positions = (np.arange(2 * columns) + 0.5) / 2 - 0.5
    reference = np.interp(positions, np.arange(columns), bright_row[:, 0])
    expected = sr.data.read_bands(range(bands)) / reference[None, :, None]
    np.testing.assert_allclose(result.data.read_bands(range(bands)), expected, rtol=1e-5)


def test_super_resolution_clears_crop_history(loaded_window, stub_sr, qtbot):
    window = loaded_window
    window.viewer.cropRequested.emit(QtCore.QRectF(0, 0, 6, 6))
    assert window._crop_undo_stack
    stub_sr.release.set()
    window.runSuperResButton.click()
    finish(qtbot, window)
    assert not window._crop_undo_stack
    assert not window._crop_redo_stack
    window.lowResButton.setChecked(True)
    window._undo_crop()
    assert window._hsi_data.shape[:2] == (6, 6)


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
    # Visualization/Calibration/Classification follow the high-res selection
    # once a result exists.
    for viewer in (window.viewer, window.calibrationViewer, window.classificationViewer):
        assert viewer.rgb.shape == (16, 16, 3)
    assert window._visualization_results[VisualizationMode.RGB].display_rgb.shape == (16, 16, 3)
    window.lowResButton.setChecked(True)
    assert window.superResViewer.rgb.shape == (8, 8, 3)
    for viewer in (window.viewer, window.calibrationViewer, window.classificationViewer):
        assert viewer.rgb.shape == (8, 8, 3)
    window.highResButton.setChecked(True)
    window.modeNDVI.setChecked(True)
    assert window.superResViewer.rgb.shape == (16, 16, 3)
    assert window.viewer.rgb.shape == (16, 16, 3)
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


def test_close_cancels_without_destroying_running_thread(loaded_window, stub_sr, qtbot, monkeypatch):
    monkeypatch.setattr(
        loaded_window._hypercube_controller, "resume",
        lambda data: pytest.fail("Closing must not restart a hypercube worker"),
    )
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


def test_sr_tab_transfers_view_coordinates_unscaled_when_resolutions_match(
    loaded_window, stub_sr, qtbot, monkeypatch
):
    """Visualization and the SR tab share one high/low toggle, so they always
    display the same resolution; switching between them must not rescale."""
    stub_sr.release.set()
    loaded_window.runSuperResButton.click()
    finish(qtbot, loaded_window)
    window = loaded_window
    window._active_viewer = window.viewer
    monkeypatch.setattr(window.viewer, "get_view_state", lambda: (4.0, QtCore.QPointF(3, 5)))
    received = []
    monkeypatch.setattr(window.superResViewer, "queue_view_state", received.append)

    # Both tabs show the high-res result by default once SR completes.
    window._on_tab_changed(1)
    assert received[-1] == (4.0, QtCore.QPointF(3, 5))

    # Switching back to Original affects both tabs identically, so the
    # transfer still needs no rescale.
    window.lowResButton.setChecked(True)
    window._active_viewer = window.viewer
    window._on_tab_changed(1)
    assert received[-1] == (4.0, QtCore.QPointF(3, 5))


def test_sr_pixel_tiles_use_the_displayed_image(loaded_window, stub_sr, qtbot):
    window = loaded_window
    stub_sr.release.set()
    window.runSuperResButton.click()
    finish(qtbot, window)
    for button, row, column in ((window.highResButton, 14, 15), (window.lowResButton, 6, 7)):
        button.setChecked(True)
        entries = window.superResViewer.pixel_value_provider(row, column)
        color = tuple(int(value) for value in window.superResViewer.rgb[row, column])
        assert entries == {"RGB": PixelValueEntry(value=color, color=color)}
        html = format_pixel_values_html(entries)
        assert 'bgcolor="#{:02x}{:02x}{:02x}"'.format(*color) in html
        assert "RGB: ({}, {}, {})".format(*color) in html
    assert window.superResViewer.pixel_value_provider(14, 15) == {}


def test_sr_comparison_preserves_framing_when_toggling_resolution(loaded_window, stub_sr, qtbot):
    window = loaded_window
    stub_sr.release.set()
    window.runSuperResButton.click()
    finish(qtbot, window)
    window.tabWidget.setCurrentWidget(window.SuperResolution)
    window.show()
    qtbot.waitExposed(window)
    qtbot.wait(50)
    window.superResViewer.set_view_state((4.0, QtCore.QPointF(8, 8)))
    window.lowResButton.setChecked(True)
    qtbot.wait(50)
    assert window.superResViewer.get_view_state()[0] == pytest.approx(8.0)
    window.highResButton.setChecked(True)
    qtbot.wait(50)
    assert window.superResViewer.get_view_state()[0] == pytest.approx(4.0)


def test_visualization_view_maps_framing_between_resolutions(
    loaded_window, stub_sr, qtbot, monkeypatch
):
    window = loaded_window
    stub_sr.release.set()
    window.runSuperResButton.click()
    finish(qtbot, window)
    window.tabWidget.setCurrentWidget(window.Visualization)
    window.show()
    qtbot.waitExposed(window)
    qtbot.wait(50)
    captured = []
    original_queue = window.viewer.queue_view_state

    def record_view_state(state):
        captured.append(state)
        original_queue(state)

    monkeypatch.setattr(window.viewer, "queue_view_state", record_view_state)
    window.viewer.set_view_state((2.5, QtCore.QPointF(8, 8)))
    high_state = window.viewer.get_view_state()
    window.lowResButton.setChecked(True)
    qtbot.wait(50)
    assert window.viewer.get_view_state()[0] == pytest.approx(5.0)
    assert captured[-1][0] == pytest.approx(5.0)
    assert captured[-1][1].x() == pytest.approx(high_state[1].x() / 2)
    assert captured[-1][1].y() == pytest.approx(high_state[1].y() / 2)
    window.viewer.set_view_state((4.0, QtCore.QPointF(4, 4)))
    low_state = window.viewer.get_view_state()
    window.highResButton.setChecked(True)
    qtbot.wait(50)
    assert window.viewer.get_view_state()[0] == pytest.approx(2.0)
    assert captured[-1][0] == pytest.approx(2.0)
    assert captured[-1][1].x() == pytest.approx(low_state[1].x() * 2)
    assert captured[-1][1].y() == pytest.approx(low_state[1].y() * 2)
    window.lowResButton.setChecked(True)
    qtbot.wait(50)
    assert window.viewer.get_view_state()[0] == pytest.approx(4.0)


def test_canvas_switches_stay_fixed_and_above_each_page(
    loaded_window, stub_sr, qtbot
):
    window = loaded_window
    stub_sr.release.set()
    window.runSuperResButton.click()
    finish(qtbot, window)
    window.show()
    qtbot.waitExposed(window)

    cases = (
        (window.Visualization, window.visualizationStack, 0),
        (window.Calibration, window.calibrationViewer, 1),
        (window.Classification, window.classificationViewer, 2),
    )
    for page, canvas, index in cases:
        window.tabWidget.setCurrentWidget(page)
        qtbot.wait(30)
        switch = window._resolution_switches.switches[index]
        assert switch.parentWidget() is canvas
        assert switch.pos() == QtCore.QPoint(12, 12)
        assert switch.isVisible()
        assert canvas.childAt(switch.geometry().center()) is switch

    window.resize(1100, 820)
    window._refresh_viewers_display()
    qtbot.wait(30)
    for page, canvas, index in cases:
        window.tabWidget.setCurrentWidget(page)
        qtbot.wait(30)
        switch = window._resolution_switches.switches[index]
        assert switch.pos() == QtCore.QPoint(12, 12)
        assert canvas.childAt(switch.geometry().center()) is switch

    window.tabWidget.setCurrentWidget(window.Visualization)
    window.modeHyperCube.setChecked(True)
    qtbot.wait(30)
    switch = window._resolution_switches.switches[0]
    assert switch.pos() == QtCore.QPoint(12, 12)
    assert window.visualizationStack.childAt(switch.geometry().center()) is switch


def test_high_res_switch_shown_without_notice_when_switching_to_visualization(loaded_window, stub_sr, qtbot, dialogs):
    window = loaded_window
    stub_sr.release.set()
    window.tabWidget.setCurrentWidget(window.SuperResolution)
    window.runSuperResButton.click()
    finish(qtbot, window)
    assert window.highResButton.isChecked()
    assert not dialogs.information

    window.tabWidget.setCurrentWidget(window.Visualization)
    assert not dialogs.information
    assert not window._resolution_switches.switches[0].isHidden()

    window.tabWidget.setCurrentWidget(window.SuperResolution)
    window.tabWidget.setCurrentWidget(window.Visualization)
    assert not dialogs.information


def test_high_res_notice_not_shown_for_low_res_selection(loaded_window, qtbot, dialogs):
    window = loaded_window
    window.tabWidget.setCurrentWidget(window.Calibration)
    window.tabWidget.setCurrentWidget(window.Visualization)
    assert not dialogs.information


def test_super_resolution_clears_prior_classification_results(
    loaded_window, stub_sr, qtbot, dialogs
):
    """A new SR source invalidates classifications at both resolutions."""

    window = loaded_window
    window.numOfClassesEdit.setText("2")
    window.maxIterationsEdit.setText("3")

    window.unsupervisedClassifyButton.click()
    qtbot.waitUntil(lambda: not window._classification_controller.is_running(), timeout=5000)
    assert len(window.classificationLayerPanel._rows) == 2
    window.classificationLayerPanel._rows[1]._toggle.setChecked(False)
    low_res_image = window.classificationViewer._photo.pixmap().toImage().copy()

    stub_sr.release.set()
    qtbot.waitUntil(lambda: window.runSuperResButton.isEnabled())
    window.runSuperResButton.click()
    finish(qtbot, window)

    # High-res has never been classified: the layer panel is empty and the
    # viewer falls back to the plain Super-Resolution image, exactly like
    # Visualization's own low/high toggle falls back to the plain photo.
    assert window.highResButton.isChecked()
    assert len(window.classificationLayerPanel._rows) == 0
    assert window.classificationViewer.rgb.shape == (16, 16, 3)

    window.unsupervisedClassifyButton.click()
    qtbot.waitUntil(lambda: not window._classification_controller.is_running(), timeout=5000)
    assert len(window.classificationLayerPanel._rows) == 2
    assert not dialogs.information
    high_res_image = window.classificationViewer._photo.pixmap().toImage().copy()
    assert high_res_image != low_res_image

    # The pre-SR low-res classification is obsolete, while the classification
    # made after SR remains available on the high-res result.
    window.lowResButton.setChecked(True)
    assert len(window.classificationLayerPanel._rows) == 0

    window.highResButton.setChecked(True)
    assert len(window.classificationLayerPanel._rows) == 2
    assert window.classificationLayerPanel._rows[1]._toggle.isChecked()
    assert window.classificationViewer._photo.pixmap().toImage() == high_res_image


def test_supervised_high_res_classification_has_no_resolution_popup(
    loaded_window, synthetic_cube_path, stub_sr, file_dialog, qtbot, dialogs
):
    window = loaded_window
    labels = np.ones((8, 8), dtype=np.uint8)
    labels[4:] = 2
    mask_path = synthetic_cube_path.with_name("synthetic_mask.png")
    Image.fromarray(labels).save(mask_path)
    file_dialog.open_return = (str(mask_path), "")
    window.pushButton.click()

    stub_sr.release.set()
    window.runSuperResButton.click()
    finish(qtbot, window)
    assert window.highResButton.isChecked()

    window.pushButton_2.click()
    qtbot.waitUntil(
        lambda: window._classification_controller._thread is None,
        timeout=10000,
    )
    assert not dialogs.information
    assert not dialogs.critical
    assert window._classification_controller._current_slot.result is not None


def test_rerunning_super_resolution_discards_the_stale_high_res_classification(
    loaded_window, stub_sr, qtbot
):
    window = loaded_window
    window.numOfClassesEdit.setText("2")
    window.maxIterationsEdit.setText("3")

    stub_sr.release.set()
    window.runSuperResButton.click()
    finish(qtbot, window)
    assert window.highResButton.isChecked()

    window.unsupervisedClassifyButton.click()
    qtbot.waitUntil(lambda: not window._classification_controller.is_running(), timeout=5000)
    assert len(window.classificationLayerPanel._rows) == 2

    window.lowResButton.setChecked(True)
    stub_sr.release.clear()
    stub_sr.started.clear()
    window.runSuperResButton.click()
    qtbot.waitUntil(stub_sr.started.is_set)
    stub_sr.release.set()
    finish(qtbot, window)

    # Rerunning SR must discard the classification made against the old SR
    # result: `highResButton` flips back on completion, revealing an empty
    # (not stale) layer panel for the new high-res data.
    assert window.highResButton.isChecked()
    assert len(window.classificationLayerPanel._rows) == 0


def test_index_mean_dialog_remains_available_for_original_after_sr(loaded_window, stub_sr, qtbot):
    window = loaded_window
    stub_sr.release.set()
    window.runSuperResButton.click()
    finish(qtbot, window)
    # HR indices have not been computed, so keep the existing clear message.
    window.superResViewer.meanIndexRequested.emit("NDVI")
    assert not window.findChildren(IndexMeanDialog)
    assert "original image" in window.statusbar.currentMessage()
    window.lowResButton.setChecked(True)
    window.superResViewer.meanIndexRequested.emit("NDVI")
    dialog = window.findChild(IndexMeanDialog)
    assert dialog is not None and dialog.isVisible()
    value = np.nanmean(window._visualization_results[VisualizationMode.NDVI].values)
    assert dialog.findChild(QtWidgets.QLabel, "indexMeanValue").text() == f"{value:.4f}"
    dialog.close()


def test_sr_serializes_cube_reads_and_resumes_interrupted_hypercube(
    loaded_window, stub_sr, qtbot, monkeypatch
):
    window = loaded_window
    controller = window._hypercube_controller
    controller.stop_and_wait()
    active = threading.Event()
    stopped = threading.Event()
    service = window._visualization_service
    prepare = service.prepare_hypercube_view

    def interrupted_build(data, *, progress, is_cancelled):
        active.set()
        try:
            while not stopped.wait(.005):
                if is_cancelled():
                    raise CancelledError()
        finally:
            active.clear()
            stopped.set()

    monkeypatch.setattr(service, "prepare_hypercube_view", interrupted_build)
    controller.refresh(window._hsi_data)
    qtbot.waitUntil(active.is_set)
    generation = controller._generation
    run = window._super_resolution_service.run

    def guarded_run(*args, **kwargs):
        assert stopped.is_set() and not active.is_set(), "Cube readers must not overlap"
        return run(*args, **kwargs)

    monkeypatch.setattr(window._super_resolution_service, "run", guarded_run)
    window.runSuperResButton.click()
    qtbot.waitUntil(stub_sr.started.is_set)
    assert controller._worker is None
    # A queued callback from the cancelled job must not overwrite SR status.
    controller._on_progress(generation, 10, "stale cube read")
    assert "stale cube read" not in window.statusbar.currentMessage()
    monkeypatch.setattr(service, "prepare_hypercube_view", prepare)
    stub_sr.release.set()
    finish(qtbot, window)
    qtbot.waitUntil(lambda: controller._view_data is not None)
    window.modeHyperCube.setChecked(True)
    assert window.visualizationStack.currentWidget() is window.hypercubeWidget
    assert window.hypercubeWidget._view_data is controller._view_data
    assert window._super_res_result.data.shape == (16, 16, 8)


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
