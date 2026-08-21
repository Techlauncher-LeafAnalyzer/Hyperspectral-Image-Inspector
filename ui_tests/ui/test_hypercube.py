from __future__ import annotations

from PyQt6.QtCore import QCoreApplication, QEvent, QEventLoop, QRectF

from core import VisualizationService


def test_hypercube_button_is_enabled_before_and_after_load(window, synthetic_cube_path):
    assert window.modeHyperCube.isEnabled()
    window.load_image_from_path(synthetic_cube_path)
    assert window.modeHyperCube.isEnabled()


def test_loading_image_defers_hypercube_work_until_its_mode_is_selected(loaded_window):
    assert loaded_window._hypercube_worker is None

    loaded_window.modeHyperCube.click()

    assert loaded_window._hypercube_worker is not None


def test_hypercube_worker_completion_releases_controller_reference(loaded_window):
    loaded_window.modeHyperCube.click()
    worker = loaded_window._hypercube_worker
    assert worker is not None

    assert worker._thread.wait(10_000)
    QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    assert loaded_window._hypercube_worker is None
    assert loaded_window._hypercube_view_data is not None


def test_selecting_hypercube_mode_switches_stack_page(loaded_window):
    loaded_window.modeHyperCube.click()

    assert loaded_window.visualizationStack.currentWidget() is loaded_window.hypercubeWidget


def test_leaving_hypercube_mode_restores_viewer_page(loaded_window):
    loaded_window.modeHyperCube.click()

    loaded_window.modeRGB.click()

    assert loaded_window.visualizationStack.currentWidget() is loaded_window.viewer


def test_cropping_starts_a_new_hypercube_generation(loaded_window):
    loaded_window.modeHyperCube.click()
    first_generation = loaded_window._hypercube_generation

    # left=1, top=1, right=5, bottom=5 -> a 4x4 crop out of the 8x8 fixture,
    # emitted the same way ui_tests/ui/test_crop.py already exercises crop.
    loaded_window.viewer.cropRequested.emit(QRectF(1, 1, 4, 4))

    assert loaded_window._hypercube_generation > first_generation


def test_hypercube_finished_callback_updates_state_and_widget(loaded_window):
    loaded_window.modeHyperCube.click()
    result = VisualizationService().prepare_hypercube_view(loaded_window._hsi_data)

    loaded_window._on_hypercube_finished(loaded_window._hypercube_generation, result)

    assert loaded_window._hypercube_view_data is result
    assert loaded_window._hypercube_error is None
    assert loaded_window.hypercubeWidget._view_data is result


def test_stale_hypercube_generation_is_ignored(loaded_window):
    result = VisualizationService().prepare_hypercube_view(loaded_window._hsi_data)
    stale_generation = loaded_window._hypercube_generation - 1

    loaded_window._on_hypercube_finished(stale_generation, result)

    assert loaded_window._hypercube_view_data is not result


def test_hypercube_failure_is_recorded_and_shown(loaded_window):
    loaded_window.modeHyperCube.click()

    loaded_window._on_hypercube_failed(loaded_window._hypercube_generation, "no wavelengths")

    assert loaded_window._hypercube_error == "no wavelengths"
    assert loaded_window.hypercubeWidget._view_data is None
