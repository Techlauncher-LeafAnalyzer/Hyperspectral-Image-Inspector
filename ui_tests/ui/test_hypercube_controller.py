from __future__ import annotations

from PyQt6.QtWidgets import QRadioButton, QStackedWidget, QStatusBar

from core import HSIData, HSIReader, VisualizationService
from ui.hypercube_controller import HypercubeController
from ui.hypercube_widget import HypercubeWidget


def _make_controller(qtbot):
    """Build a HypercubeController against real, minimal Qt widgets.

    No MainWindow involved -- this exercises the controller's own public
    surface directly, per this feature's "keep the wiring in a controller"
    design (see src/ui/hypercube_controller.py).
    """
    mode_button = QRadioButton()
    qtbot.addWidget(mode_button)
    placeholder = QRadioButton()  # stands in for the 2D viewer's stack page
    qtbot.addWidget(placeholder)
    widget = HypercubeWidget()
    qtbot.addWidget(widget)
    stack = QStackedWidget()
    qtbot.addWidget(stack)
    stack.addWidget(placeholder)
    stack.addWidget(widget)
    statusbar = QStatusBar()
    qtbot.addWidget(statusbar)

    controller = HypercubeController(
        mode_button, stack, widget, statusbar, VisualizationService()
    )
    return controller, mode_button, stack, widget, placeholder


def test_toggling_mode_button_switches_stack_page(qtbot):
    controller, mode_button, stack, widget, placeholder = _make_controller(qtbot)
    stack.setCurrentWidget(placeholder)

    mode_button.setChecked(True)

    assert stack.currentWidget() is widget


def test_untoggling_mode_button_does_not_touch_the_stack(qtbot):
    controller, mode_button, stack, widget, placeholder = _make_controller(qtbot)
    mode_button.setChecked(True)
    stack.setCurrentWidget(placeholder)

    mode_button.setChecked(False)

    assert stack.currentWidget() is placeholder


def test_refresh_with_unloaded_data_shows_placeholder_and_starts_no_worker(qtbot):
    controller, mode_button, stack, widget, placeholder = _make_controller(qtbot)
    mode_button.setChecked(True)

    controller.refresh(HSIData())

    assert controller._worker is None
    assert widget._status_label.text() == "Load an image to view its hypercube."


def test_refresh_with_loaded_data_starts_a_worker_and_bumps_generation(
    qtbot, synthetic_cube_path
):
    controller, mode_button, stack, widget, placeholder = _make_controller(qtbot)
    data = HSIReader().open(synthetic_cube_path)
    first_generation = controller._generation

    controller.refresh(data)

    assert controller._worker is not None
    assert controller._generation > first_generation
    controller.stop_and_wait()


def test_finished_callback_updates_state_and_widget(qtbot, synthetic_cube_path):
    controller, mode_button, stack, widget, placeholder = _make_controller(qtbot)
    data = HSIReader().open(synthetic_cube_path)
    mode_button.setChecked(True)
    result = VisualizationService().prepare_hypercube_view(data)

    controller._on_finished(controller._generation, result)

    assert controller._view_data is result
    assert controller._error is None
    assert widget._view_data is result


def test_stale_generation_is_ignored(qtbot, synthetic_cube_path):
    controller, mode_button, stack, widget, placeholder = _make_controller(qtbot)
    data = HSIReader().open(synthetic_cube_path)
    result = VisualizationService().prepare_hypercube_view(data)
    stale_generation = controller._generation - 1

    controller._on_finished(stale_generation, result)

    assert controller._view_data is not result


def test_failure_is_recorded_and_shown(qtbot):
    controller, mode_button, stack, widget, placeholder = _make_controller(qtbot)
    mode_button.setChecked(True)

    controller._on_failed(controller._generation, "no wavelengths")

    assert controller._error == "no wavelengths"
    assert widget._view_data is None
    assert widget._status_label.text() == "no wavelengths"


def test_stop_and_wait_is_a_no_op_without_a_worker(qtbot):
    controller, *_ = _make_controller(qtbot)

    controller.stop_and_wait()  # must not raise

    assert controller._worker is None


# ---------------------------------------------------------------------- #
# End-to-end wiring through MainWindowController                          #
# ---------------------------------------------------------------------- #


def test_main_window_wires_hypercube_button_to_the_stack(loaded_window):
    loaded_window.modeHyperCube.click()

    assert loaded_window.visualizationStack.currentWidget() is loaded_window.hypercubeWidget

    loaded_window.modeRGB.click()

    assert loaded_window.visualizationStack.currentWidget() is loaded_window.viewer


def test_loading_an_image_starts_a_hypercube_worker(loaded_window):
    controller = loaded_window._hypercube_controller

    assert controller._worker is not None
    assert controller._generation >= 1
    controller.stop_and_wait()
