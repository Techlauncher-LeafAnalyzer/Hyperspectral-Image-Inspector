"""Right workflow stripe: pill tool windows over the one image canvas."""

from __future__ import annotations

import pytest
from PyQt6 import QtCore, QtWidgets

from ui.tool_windows import ToolStripeButton
from ui.viewer import HSIViewer


@pytest.fixture
def shown_window(loaded_window, qtbot):
    loaded_window.resize(1400, 860)
    loaded_window.show()
    qtbot.waitExposed(loaded_window)
    return loaded_window


def _tools(window):
    return window.workflowTools


def test_tabs_are_replaced_by_pills_on_the_right_edge(shown_window, qtbot):
    window = shown_window
    stripe = window.workflowStripe
    qtbot.wait(50)  # let the deferred stripe sync and relayout run
    assert window.toolBarArea(stripe) == QtCore.Qt.ToolBarArea.RightToolBarArea
    assert stripe.mapTo(window, QtCore.QPoint(stripe.width(), 0)).x() == window.width()
    assert not stripe.logo.isVisible()
    # No tabs or per-workflow pages: one viewer shows the image.
    assert not hasattr(window, "tabWidget")
    assert window.findChildren(HSIViewer) == [window.viewer]
    def top(widget):
        return widget.mapTo(window, QtCore.QPoint()).y()

    pills = sorted(
        (b for b in stripe.findChildren(ToolStripeButton) if b.caption),
        key=top,
    )
    assert [b.caption for b in pills] == [
        "Visualize", "Super-Res", "Calibrate", "Segment", "Batch",
    ]
    for tool in stripe.tool_windows:
        # Pills: taller than wide squares, captioned under the icon.
        assert tool.button.height() > window.toolWindowStripe.button_side()
        assert tool.button.isEnabled()
    tools = _tools(window)
    # Small panels are flyouts; Batch gets a full-height dock; Segment is a
    # plain toggle for the inspector beside the image, not a tool window.
    assert [n for n, tool in tools.items() if tool.compact] == [
        "visualization", "superResolution", "calibration",
    ]
    assert "segmentation" not in tools
    assert window.dockWidgetArea(tools["batch"].dock) == QtCore.Qt.DockWidgetArea.RightDockWidgetArea
    for name in ("visualization", "superResolution", "calibration"):
        assert tools[name].dock is None
        assert tools[name].host.parentWidget() is window


def test_dividers_set_apart_segment_and_batch(shown_window):
    window = shown_window
    tools = _tools(window)

    def top(widget):
        return widget.mapTo(window, QtCore.QPoint()).y()

    dividers = window.workflowStripe.findChildren(QtWidgets.QFrame, "toolWindowStripeDivider")
    assert len(dividers) == 2
    first, second = sorted(dividers, key=top)
    assert top(tools["calibration"].button) < top(first) < top(window.segmentationPill)
    assert top(window.segmentationPill) < top(second) < top(tools["batch"].button)


def test_nothing_open_at_start(shown_window):
    window = shown_window
    tools = _tools(window)
    assert not any(tool.is_open() or tool.is_peeking() for tool in tools.values())
    # No pill is bound to a view, so none carries the current-view marker.
    assert not any(tool.button.is_current() for tool in tools.values())
    assert window.classificationLayerPanel.isHidden() and window.toolPalette.isHidden()


@pytest.mark.parametrize("name", ["visualization", "superResolution", "calibration"])
def test_opening_a_pill_overlays_its_panel_on_the_canvas(shown_window, name):
    window = shown_window
    tools = _tools(window)
    window.segmentationPill.click()
    tools[name].button.click()
    assert tools[name].is_open() and tools[name].host.isVisible()
    assert [n for n, tool in tools.items() if tool.is_open()] == [name]
    # The panel floats over the canvas; the inspector beside it stays.
    assert tools[name].host.parentWidget() is window
    assert window.classificationLayerPanel.isVisible() and window.viewer.isVisible()
    # The left stripe's tools are independent of the right side.
    window.spectrumTool.show()
    assert tools[name].is_open() and window.spectrumTool.is_open()


def test_batch_panel_keeps_the_current_view(shown_window):
    window = shown_window
    tools = _tools(window)
    window.segmentationPill.click()
    tools["batch"].button.click()
    # The Batch dock and the Segmentation inspector coexist.
    assert tools["batch"].is_open() and window.classificationLayerPanel.isVisible()
    assert window.segmentationPill.isChecked()
    assert tools["batch"].panel.isAncestorOf(window.batchRunButton)


def test_bottom_bars_moved_into_their_panels(shown_window):
    window = shown_window
    tools = _tools(window)
    moved = {
        "visualization": (window.modeSelect, window.modeRGB, window.modeHyperCube,
                          window.modeSimilarity, window.spectrumInspectorButton),
        "superResolution": (window.runSuperResButton, window.superResStatusStack),
        "calibration": (window.darkFileButton, window.referenceFileEdit, window.calibrateButton),
        "batch": (window.batchFileList, window.batchRunButton),
    }
    for name, widgets in moved.items():
        for widget in widgets:
            assert tools[name].panel.isAncestorOf(widget), (name, widget.objectName())
            assert not window.canvas.isAncestorOf(widget)
    # The classifier form joined the Segmentation inspector.
    for widget in (window.classificationModeTabs, window.comboBox,
                   window.unsupervisedClassifyButton, window.pushButton_2):
        assert window.classificationLayerPanel.isAncestorOf(widget)
    # The canvas holds the toolbox, the image (and its mode caption) and the
    # inspector.
    layout = window.canvas.layout()
    items = [layout.itemAt(i) for i in range(layout.count())]
    assert [item.widget() for item in items] == [
        window.toolPalette, None, window.classificationLayerPanel,
    ]
    image = items[1].layout()
    assert [image.itemAt(i).widget() for i in range(image.count())] == [
        window.visualizationStack, window.visualizationModeCaption,
    ]


def test_panel_content_is_never_clipped(shown_window):
    from ui.tool_windows import ToolSettingsPanel

    for tool in _tools(shown_window).values():
        tool.show()
        QtWidgets.QApplication.processEvents()
        panel = tool.panel.findChild(ToolSettingsPanel)
        assert panel.viewport().width() >= panel.body.minimumSizeHint().width()
        assert panel.body.window() is tool.panel.window()


def test_popped_out_panel_stays_open_beside_another(shown_window):
    tools = _tools(shown_window)
    calibration = tools["calibration"]
    calibration.set_floating(True)
    assert calibration.window.isVisible()
    # Opening another panel leaves the popped-out one open.
    tools["visualization"].show()
    assert calibration.is_open() and tools["visualization"].is_open()


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


def _leave(tool, qtbot):
    """Pointer leaves the button and flyout (the test cursor is elsewhere)."""
    tool.button.hoverChanged.emit(False)
    tool.host.hoverChanged.emit(False)
    qtbot.wait(FLYOUT_HIDE_DELAY_MS + 100)


from ui.tool_windows import FLYOUT_HIDE_DELAY_MS  # noqa: E402


@pytest.mark.parametrize("name", ["visualization", "superResolution", "calibration"])
def test_hover_peeks_a_compact_panel_until_the_pointer_leaves(shown_window, qtbot, name):
    window = shown_window
    tool = _tools(window)[name]
    tool.button.hoverChanged.emit(True)
    assert tool.is_peeking() and tool.host.isVisible()
    # A peek is not "open": no highlight.
    assert not tool.is_open() and not tool.button.isChecked()
    # Small: sized to its content, not the full column.
    assert tool.host.height() < window.canvas.height() * 0.6
    # Moving onto the flyout keeps it.
    tool.button.hoverChanged.emit(False)
    tool.host.hoverChanged.emit(True)
    qtbot.wait(FLYOUT_HIDE_DELAY_MS + 100)
    assert tool.is_peeking()
    _leave(tool, qtbot)
    assert not tool.is_peeking() and tool.host.isHidden()


def test_clicking_the_pill_pins_the_flyout(shown_window, qtbot):
    window = shown_window
    tool = _tools(window)["calibration"]
    tool.button.hoverChanged.emit(True)
    tool.button.click()
    assert tool.is_open() and not tool.is_peeking() and tool.button.isChecked()
    _leave(tool, qtbot)
    assert tool.host.isVisible()
    # Clicking again unpins and closes it.
    tool.button.click()
    assert tool.host.isHidden() and not tool.button.isChecked()


def test_clicking_inside_a_peeking_flyout_pins_it(shown_window, qtbot):
    window = shown_window
    tools = _tools(window)
    tools["visualization"].button.hoverChanged.emit(True)
    tools["visualization"].host.pressed.emit()
    assert tools["visualization"].is_open()
    tools["superResolution"].button.hoverChanged.emit(True)
    tools["superResolution"]._on_focus_changed(None, window.runSuperResButton)
    assert tools["superResolution"].is_open()
    _leave(tools["superResolution"], qtbot)
    assert tools["superResolution"].host.isVisible()
    # One pinned flyout at a time.
    assert not tools["visualization"].is_open()


def test_one_peek_at_a_time(shown_window):
    tools = _tools(shown_window)
    tools["visualization"].button.hoverChanged.emit(True)
    tools["calibration"].button.hoverChanged.emit(True)
    assert tools["calibration"].is_peeking()
    assert not tools["visualization"].is_peeking() and tools["visualization"].host.isHidden()


def test_flyouts_clear_the_batch_dock(shown_window, qtbot):
    window = shown_window
    tools = _tools(window)
    tools["batch"].show()
    qtbot.wait(50)
    dock_left = tools["batch"].dock.mapTo(window, QtCore.QPoint()).x()
    # Peeking or pinned, a flyout sits left of the dock, level with its pill.
    peek = tools["superResolution"]
    peek.button.hoverChanged.emit(True)
    assert peek.host.geometry().right() < dock_left
    assert peek.host.y() == peek.button.mapTo(window, QtCore.QPoint()).y()
    peek.hide()
    pinned = tools["visualization"]
    pinned.show()
    window._sync_stripe_geometry()
    assert pinned.host.geometry().right() < dock_left
    # Hiding the dock lets the pinned flyout return beside the stripe.
    tools["batch"].hide()
    qtbot.wait(50)
    window._sync_stripe_geometry()
    assert pinned.host.geometry().right() > dock_left


def test_hovering_another_pill_never_overlaps_a_pinned_flyout(shown_window, qtbot):
    tools = _tools(shown_window)
    pinned, other = tools["visualization"], tools["calibration"]
    pinned.show()
    other.button.hoverChanged.emit(True)
    # The pinned flyout steps aside for the peek but stays pinned.
    assert other.is_peeking() and pinned.host.isHidden()
    assert pinned.is_open() and pinned.button.isChecked()
    # Peek to peek: the pinned one stays out of the way.
    tools["superResolution"].button.hoverChanged.emit(True)
    assert tools["superResolution"].is_peeking() and not other.is_peeking()
    assert pinned.host.isHidden()
    _leave(tools["superResolution"], qtbot)
    assert pinned.host.isVisible() and pinned.is_open()
    # Back on the pinned tool's own pill: it simply reappears.
    other.button.hoverChanged.emit(True)
    pinned.button.hoverChanged.emit(True)
    assert not other.is_peeking() and pinned.host.isVisible()
    visible = [t.host.geometry() for t in tools.values() if t.compact and t.host.isVisible()]
    assert len(visible) == 1


def test_flyouts_clear_the_class_layer_inspector(shown_window):
    window = shown_window
    window.segmentationPill.click()
    tool = _tools(window)["superResolution"]
    tool.button.hoverChanged.emit(True)
    window._sync_stripe_geometry()
    inspector_left = window.classificationLayerPanel.mapTo(window, QtCore.QPoint()).x()
    assert tool.host.geometry().right() < inspector_left


def test_segment_pill_toggles_the_inspector_and_keeps_results(shown_window):
    window = shown_window
    pill, inspector = window.segmentationPill, window.classificationLayerPanel
    palette = window.toolPalette
    assert not pill.isChecked() and inspector.isHidden() and palette.isHidden()
    # Not a hover pill: hovering shows nothing.
    assert not pill.receivers(pill.hoverChanged)
    pill.click()
    assert inspector.isVisible() and palette.isVisible() and pill.isChecked()
    # The toolbox sits at the image's left edge, the inspector at its right.
    left = window.viewer.mapTo(window, QtCore.QPoint()).x()
    assert palette.mapTo(window, QtCore.QPoint(palette.width(), 0)).x() <= left
    assert inspector.mapTo(window, QtCore.QPoint()).x() >= left + window.viewer.width()
    inspector.set_mode("annotations")
    inspector.set_panel_width(360)
    window.classificationModeTabs.setCurrentWidget(window.unsupervisedClassificationTab)
    # Second click: toolbox and inspector gone, the results stay on the canvas.
    pill.click()
    assert inspector.isHidden() and palette.isHidden() and not pill.isChecked()
    # Third click: back exactly as it was.
    pill.click()
    assert inspector.isVisible() and pill.isChecked()
    assert inspector.mode == "annotations" and inspector.panel_width == 360
    assert window.classificationModeTabs.currentWidget() is window.unsupervisedClassificationTab
    # Other panels open over the canvas without hiding the inspector.
    _tools(window)["calibration"].show()
    assert pill.isChecked() and inspector.isVisible()


def test_hiding_the_toolbox_turns_off_the_drawing_tool(shown_window):
    from core import AnnotationKind

    window = shown_window
    window.segmentationPill.click()
    page = window.classificationLayerPanel.annotation_page
    page._tool_buttons[AnnotationKind.RECTANGLE].click()
    assert window.viewer._annotation_tool is AnnotationKind.RECTANGLE
    window.segmentationPill.click()
    assert window.viewer._annotation_tool is None
    assert not any(button.isChecked() for button in page._tool_buttons.values())


def test_inspector_fits_the_shown_classifier_method(shown_window, qtbot):
    window = shown_window
    window.segmentationPill.click()
    tabs = window.classificationModeTabs
    tabs.setCurrentWidget(window.supervisedClassificationTab)
    qtbot.wait(50)
    supervised = tabs.height()
    tabs.setCurrentWidget(window.unsupervisedClassificationTab)
    qtbot.wait(50)
    assert tabs.height() < supervised - 40


def test_separate_leaves_lives_in_the_class_layer_inspector(shown_window):
    window = shown_window
    assert window.separateLeavesButton is window.classificationLayerPanel.separate_leaves_button
    assert window.classificationLayerPanel.isAncestorOf(window.separateLeavesButton)
    assert not window.segmentationPanel.isAncestorOf(window.separateLeavesButton)


def test_compact_flyout_pops_out_and_back(shown_window):
    tool = _tools(shown_window)["calibration"]
    tool.show()
    tool.set_floating(True)
    assert tool.window.isVisible() and tool.host.isHidden()
    assert tool.panel.window() is tool.window
    tool.set_floating(False)
    assert tool.host.isVisible() and tool.is_open()
    assert tool.panel.window() is shown_window
