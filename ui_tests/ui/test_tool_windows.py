"""JetBrains-style left stripe: highlight, pop-out and disabled states."""

from __future__ import annotations

import pytest
from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: F401

from ui.theme import (
    TOOL_STRIPE_DISABLED_COLOR,
    TOOL_STRIPE_ICON_COLOR,
    TOOL_STRIPE_SELECTED_COLOR,
    TOOL_STRIPE_SELECTED_ICON_COLOR,
)


@pytest.fixture
def shown_window(loaded_window, qtbot):
    loaded_window.resize(1200, 800)
    loaded_window.show()
    qtbot.waitExposed(loaded_window)
    return loaded_window


def _color(value):
    return QtGui.QColor(value).name()


def test_stripe_sits_on_the_left_edge_with_both_tools(shown_window):
    window = shown_window
    stripe = window.toolWindowStripe
    assert window.toolBarArea(stripe) == QtCore.Qt.ToolBarArea.LeftToolBarArea
    assert stripe.mapTo(window, QtCore.QPoint()).x() == 0
    assert [tool.button.text() for tool in stripe.tool_windows] == ["Spectrum"]
    # Logo, File, a divider, the view toggles, a divider, then tool windows
    # — top to bottom.
    def top(widget):
        return widget.mapTo(window, QtCore.QPoint()).y()

    file_divider, tools_divider = stripe.findChildren(
        QtWidgets.QFrame, "toolWindowStripeDivider"
    )
    order = [stripe.logo, window.fileMenuButton, file_divider,
             window.superResolutionSwitch, window.calibrationSwitch,
             tools_divider, window.spectrumTool.button]
    assert [top(w) for w in order] == sorted(top(w) for w in order)
    assert len({top(w) for w in order}) == len(order)
    assert window.findChild(QtWidgets.QDockWidget, "imageViewDock") is None
    for tool in stripe.tool_windows:
        assert window.dockWidgetArea(tool.dock) == QtCore.Qt.DockWidgetArea.LeftDockWidgetArea


def test_spectrum_button_states_follow_the_dock(shown_window, qtbot):
    tool = shown_window.spectrumTool
    button = tool.button
    # Closed: enabled, no background, plain icon.
    assert button.isEnabled() and not tool.is_open()
    assert button.background_color() is None
    assert button.icon_color().name() == _color(TOOL_STRIPE_ICON_COLOR)

    # Selected (docked): highlight background and accent icon.
    button.click()
    assert tool.is_open() and button.isChecked()
    assert not tool.floating and tool.dock.isVisible()
    assert button.background_color().name() == _color(TOOL_STRIPE_SELECTED_COLOR)
    assert button.icon_color().name() == _color(TOOL_STRIPE_SELECTED_ICON_COLOR)

    # Popped out: a normal top-level window; still selected/highlighted.
    tool.title_bar.float_button.click()
    assert tool.floating and tool.window.isVisible() and tool.window.isWindow()
    assert tool.dock.isHidden()
    assert tool.panel.window() is tool.window
    assert button.isChecked()
    assert button.background_color().name() == _color(TOOL_STRIPE_SELECTED_COLOR)
    assert tool.title_bar.float_button.toolTip() == "Dock to the sidebar"

    # Closing the popped-out window (system ✕) hides the tool, which stays
    # popped out; the stripe button brings it back in its window.
    tool.window.close()
    assert not tool.is_open() and not button.isChecked() and tool.floating
    button.click()
    assert tool.window.isVisible() and button.isChecked()

    # Docking back returns the panel to the sidebar, still open; the
    # title-bar hide clears the highlight.
    tool.title_bar.float_button.click()
    assert not tool.floating and tool.window.isHidden()
    assert tool.panel.window() is shown_window and tool.dock.isVisible()
    assert button.isChecked()
    tool.title_bar.hide_button.click()
    assert not tool.is_open() and not button.isChecked()
    assert button.background_color() is None

    # Clicking the selected button closes the window again.
    button.click()
    button.click()
    assert not tool.is_open()


def test_disabled_tool_is_gray_and_cannot_open(window):
    tool = window.spectrumTool  # nothing loaded yet
    assert not tool.button.isEnabled()
    assert tool.button.icon_color().name() == _color(TOOL_STRIPE_DISABLED_COLOR)
    assert tool.button.background_color() is None
    assert "load an image" in tool.button.toolTip().lower()
    tool.show()
    assert not tool.is_open()


def test_view_toggles_gray_until_a_comparison_exists(loaded_window, file_dialog, tmp_path, qtbot):
    import numpy as np
    from spectral.io import envi

    window = loaded_window
    calibration, resolution = window.calibrationSwitch, window.superResolutionSwitch
    assert not calibration.isEnabled() and not resolution.isEnabled()
    assert calibration.icon_color().name() == _color(TOOL_STRIPE_DISABLED_COLOR)
    source = window._hsi_data
    for name, value in (("dark", 0), ("bright", 2)):
        path = tmp_path / f"{name}.hdr"
        envi.save_image(str(path), np.full((50, source.columns, source.bands), value,
                                          dtype=np.float32), ext=".bip", interleave="bip",
                        metadata={"wavelength": source.wavelengths})
        file_dialog.open_return = (str(path), "")
        (window.darkFileButton if name == "dark" else window.referenceFileButton).click()
    window.calibrateButton.click()
    qtbot.waitUntil(lambda: not window._calibration_controller.is_running())
    # Highlighted: After Calibration is shown on every tab.
    assert calibration.isEnabled() and calibration.isChecked()
    assert calibration.background_color().name() == _color(TOOL_STRIPE_SELECTED_COLOR)
    calibration.click()
    assert window._display_data() is window._hsi_data
    assert not calibration.isChecked() and calibration.background_color() is None
    calibration.click()
    assert window._display_data() is window._calibration_controller.result.data
    window.referenceFileButton.click()  # discards the comparison
    assert not calibration.isEnabled()
    assert not resolution.isEnabled()  # still no SR result


def test_spectrum_plot_opens_the_docked_tool(shown_window):
    shown_window.viewer.spectrumPlotRequested.emit(QtCore.QPointF(5, 3))
    tool = shown_window.spectrumTool
    assert tool.is_open() and tool.button.isChecked()
    assert shown_window._spectrum_controller.inspector.isVisible()


def test_docks_cannot_be_dragged_out_and_title_bars_are_not_drag_handles(shown_window):
    # Dragging a dock out needs a client-side mouse grab, which Wayland
    # refuses ("grabbing the mouse only for popup windows").
    from PyQt6.QtWidgets import QDockWidget

    for tool in shown_window.toolWindowStripe.tool_windows:
        features = tool.dock.features()
        assert not features & QDockWidget.DockWidgetFeature.DockWidgetMovable
        assert not features & QDockWidget.DockWidgetFeature.DockWidgetFloatable
        assert tool.title_bar.parentWidget() is tool.panel


def test_pop_out_round_trip_keeps_inspector_state(shown_window, qtbot):
    window = shown_window
    window.viewer.spectrumPlotRequested.emit(QtCore.QPointF(5, 3))
    tool = window.spectrumTool
    inspector = window._spectrum_controller.inspector
    assert inspector._plot_splitter.orientation() == QtCore.Qt.Orientation.Vertical
    for _ in range(3):
        tool.set_floating(True)
        assert inspector.isVisible() and inspector.window() is tool.window
        assert inspector._plot_splitter.orientation() == QtCore.Qt.Orientation.Horizontal
        tool.set_floating(False)
        assert inspector.isVisible() and inspector.window() is window
        assert inspector._plot_splitter.orientation() == QtCore.Qt.Orientation.Vertical
    assert len(inspector.pinned_spectra()) == 1
    # Hover keeps feeding the live trace after the round trips (once the
    # background hypercube build, during which hover skips reads, is done).
    qtbot.waitUntil(lambda: not window._hypercube_controller.is_building(), timeout=10000)
    window.viewer.pixelHovered.emit(1, 2)
    assert (inspector._live.row, inspector._live.column) == (1, 2)


def test_popped_out_window_remembers_its_size(shown_window, qtbot):
    tool = shown_window.spectrumTool
    tool.set_floating(True)
    tool.window.resize(700, 500)
    qtbot.wait(10)
    tool.set_floating(False)
    tool.set_floating(True)
    assert tool.window.size() == QtCore.QSize(700, 500)


def test_stripe_fills_space_left_of_the_image(shown_window, qtbot):
    from ui.tool_windows import STRIPE_GAP

    window = shown_window
    stripe = window.toolWindowStripe
    qtbot.wait(50)  # let the deferred stripe sync and relayout run
    window._sync_stripe_geometry()
    qtbot.wait(20)

    # Logo at the top, level with the top of the image pane.
    logo = stripe.logo
    assert not logo.pixmap().isNull()
    logo_top = logo.mapTo(window, QtCore.QPoint()).y()
    assert logo_top == STRIPE_GAP
    assert window.tabWidget.mapTo(window, QtCore.QPoint()).y() == STRIPE_GAP

    # Buttons: an even gap to the window edge and to the image area.
    button = window.superResolutionSwitch
    button_left = button.mapTo(window, QtCore.QPoint()).x()
    button_right = button_left + button.width()
    image_left = window.viewer.mapTo(window, QtCore.QPoint()).x()
    assert button_left == STRIPE_GAP
    assert image_left - button_right == STRIPE_GAP
    # Toggles sit below the logo row, larger than the old 36px buttons.
    assert button.mapTo(window, QtCore.QPoint()).y() > logo_top + logo.height()
    assert button.width() == button.height() > 36

    # Same painted ground colour as the window chrome above the image (with
    # the app theme; the test app runs unstyled by default).
    from PyQt6.QtWidgets import QApplication
    from ui.theme import apply_theme

    app = QApplication.instance()
    previous = app.styleSheet()
    apply_theme(app)
    qtbot.wait(20)
    image = window.grab().toImage()
    app.setStyleSheet(previous)
    stripe_pixel = image.pixelColor(2, window.height() - 60).name()
    chrome_pixel = image.pixelColor(image_left + 40, 1).name()
    assert stripe_pixel == chrome_pixel == "#edf2f1"
