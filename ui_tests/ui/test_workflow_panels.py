"""Right workflow stripe: pill tool windows that replace the main tabs."""

from __future__ import annotations

import pytest
from PyQt6 import QtCore, QtWidgets


@pytest.fixture
def shown_window(loaded_window, qtbot):
    loaded_window.resize(1400, 860)
    loaded_window.show()
    qtbot.waitExposed(loaded_window)
    return loaded_window


def _tools(window):
    return window.workflowTools


def test_tabs_are_replaced_by_pills_on_the_right_edge(shown_window):
    window = shown_window
    stripe = window.workflowStripe
    assert window.toolBarArea(stripe) == QtCore.Qt.ToolBarArea.RightToolBarArea
    assert stripe.mapTo(window, QtCore.QPoint(stripe.width(), 0)).x() == window.width()
    assert not stripe.logo.isVisible()
    # The main tab row is gone; the page stack stays for the canvas views.
    assert window.tabWidget.tabBar().isHidden()
    assert [window.tabWidget.widget(i) for i in range(window.tabWidget.count())] == [
        window.Visualization, window.SuperResolution, window.Calibration,
        window.Classification,
    ]
    assert [tool.button.caption for tool in stripe.tool_windows] == [
        "Visualize", "Super-Res", "Calibrate", "Segment", "Batch",
    ]
    for tool in stripe.tool_windows:
        # Pills: taller than wide squares, captioned under the icon.
        assert tool.button.height() > window.toolWindowStripe.button_side()
        assert window.dockWidgetArea(tool.dock) == QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        assert tool.button.isEnabled()


def test_visualization_opens_docked_by_default_and_is_marked_current(shown_window):
    tools = _tools(shown_window)
    assert tools["visualization"].is_open() and not tools["visualization"].floating
    assert tools["visualization"].button.isChecked()
    assert [name for name, tool in tools.items() if tool.button.is_current()] == ["visualization"]
    assert shown_window.tabWidget.currentWidget() is shown_window.Visualization


@pytest.mark.parametrize(
    "name, page_name",
    [
        ("visualization", "Visualization"),
        ("superResolution", "SuperResolution"),
        ("calibration", "Calibration"),
        ("segmentation", "Classification"),
    ],
)
def test_opening_a_pill_shows_its_view_and_moves_the_marker(shown_window, name, page_name):
    window = shown_window
    tools = _tools(window)
    window.tabWidget.setCurrentWidget(window.Calibration if name != "calibration" else window.Visualization)
    tools[name].hide()  # Visualization starts open; clicking would close it.
    tools[name].button.click()
    assert tools[name].is_open() and tools[name].dock.isVisible()
    assert window.tabWidget.currentWidget() is getattr(window, page_name)
    assert [n for n, tool in tools.items() if tool.button.is_current()] == [name]
    # One docked panel per side, like an IDE.
    assert [n for n, tool in tools.items() if tool.is_open()] == [name]
    # The left stripe's tools are independent of the right side.
    window.spectrumTool.show()
    assert tools[name].is_open() and window.spectrumTool.is_open()


def test_batch_panel_keeps_the_current_view(shown_window):
    window = shown_window
    tools = _tools(window)
    tools["segmentation"].show()
    tools["batch"].button.click()
    assert tools["batch"].is_open() and not tools["segmentation"].is_open()
    assert window.tabWidget.currentWidget() is window.Classification
    assert tools["segmentation"].button.is_current()
    assert not tools["batch"].button.is_current()
    assert tools["batch"].panel.isAncestorOf(window.batchRunButton)


def test_bottom_bars_moved_into_their_panels(shown_window):
    window = shown_window
    tools = _tools(window)
    moved = {
        "visualization": (window.modeSelect, window.modeRGB, window.modeHyperCube,
                          window.modeSimilarity, window.spectrumInspectorButton),
        "superResolution": (window.runSuperResButton, window.superResStatusStack),
        "calibration": (window.darkFileButton, window.referenceFileEdit, window.calibrateButton),
        "segmentation": (window.classificationModeTabs, window.comboBox,
                         window.unsupervisedClassifyButton),
        "batch": (window.batchFileList, window.batchRunButton),
    }
    pages = (window.Visualization, window.SuperResolution, window.Calibration,
             window.Classification)
    for name, widgets in moved.items():
        for widget in widgets:
            assert tools[name].panel.isAncestorOf(widget), (name, widget.objectName())
            assert not any(page.isAncestorOf(widget) for page in pages)
    # Each view page now holds just its canvas (plus Segmentation's inspector).
    def page_items(page):
        layout = page.layout()
        return [layout.itemAt(i).widget() for i in range(layout.count())]

    assert page_items(window.Visualization) == [window.visualizationStack]
    assert page_items(window.SuperResolution) == [window.superResViewer]
    assert page_items(window.Calibration) == [window.calibrationViewer]
    assert page_items(window.Classification) == [
        window.classificationViewer, window.classificationLayerPanel,
    ]


def test_panel_content_is_never_clipped(shown_window):
    from ui.tool_windows import ToolSettingsPanel

    for tool in _tools(shown_window).values():
        tool.show()
        QtWidgets.QApplication.processEvents()
        panel = tool.panel.findChild(ToolSettingsPanel)
        assert panel.viewport().width() >= panel.body.minimumSizeHint().width()
        assert panel.body.window() is tool.panel.window()


def test_popped_out_panel_drives_the_view_when_activated(shown_window):
    window = shown_window
    tools = _tools(window)
    segmentation = tools["segmentation"]
    segmentation.set_floating(True)
    assert segmentation.window.isVisible() and window.tabWidget.currentWidget() is window.Classification
    # Docking another panel leaves the popped-out one open.
    tools["calibration"].show()
    assert segmentation.is_open()
    assert window.tabWidget.currentWidget() is window.Calibration
    segmentation.window.activated.emit()
    assert window.tabWidget.currentWidget() is window.Classification
    assert segmentation.button.is_current() and not tools["calibration"].button.is_current()


def test_focus_into_a_panel_switches_the_view(shown_window):
    window = shown_window
    tools = _tools(window)
    tools["visualization"].set_floating(True)
    tools["segmentation"].show()
    assert window.tabWidget.currentWidget() is window.Classification
    tools["visualization"]._on_focus_changed(None, window.modeNDVI)
    assert window.tabWidget.currentWidget() is window.Visualization
    # Focus elsewhere (the canvas) changes nothing.
    tools["segmentation"]._on_focus_changed(None, window.viewer)
    assert window.tabWidget.currentWidget() is window.Visualization


def test_right_stripe_gap_matches_the_left(shown_window, qtbot):
    from ui.tool_windows import STRIPE_GAP

    window = shown_window
    _tools(window)["visualization"].hide()
    qtbot.wait(50)
    window._sync_stripe_geometry()
    qtbot.wait(20)
    button = _tools(window)["visualization"].button
    button_left = button.mapTo(window, QtCore.QPoint()).x()
    image_right = window.viewer.mapTo(window, QtCore.QPoint(window.viewer.width(), 0)).x()
    assert button_left - image_right == STRIPE_GAP
    assert window.width() - (button_left + button.width()) == STRIPE_GAP


def test_file_menu_lives_on_the_left_stripe(shown_window):
    window = shown_window
    button = window.fileMenuButton
    assert window.toolWindowStripe.isAncestorOf(button)
    assert not button.isCheckable()
    assert button.menu.actions() == [window.actionLoadImage, window.actionSaveImage]
    # Registered on the window too, so Ctrl+O / Ctrl+S work with the menu closed.
    assert window.actionLoadImage in window.actions()
    assert window.actionSaveImage in window.actions()
    button.menu.aboutToShow.emit()
    assert button.is_menu_open() and button.background_color() is not None
    button.menu.aboutToHide.emit()
    assert not button.is_menu_open() and button.background_color() is None


def test_loaded_file_shows_in_status_bar_and_title(shown_window, synthetic_cube_path):
    window = shown_window
    assert window.statusbar.isAncestorOf(window.imageFilePath)
    assert window.imageFilePath.isVisible()
    assert window.imageFilePath.text() == window._hsi_data.data_path.name
    assert str(window._hsi_data.data_path) in window.imageFilePath.toolTip()
    assert window.windowTitle().startswith(window._hsi_data.data_path.name)
