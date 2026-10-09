"""Geometry and animation checks for the shared canvas resolution switch."""

import pytest
from PyQt6 import QtCore, QtWidgets, sip

from ui.resolution_toggle import CalibrationToggle, ImageStatePanel, ResolutionSwitchGroup, ResolutionToggle


def test_resolution_toggle_is_compact_and_animates_between_states(qtbot):
    canvas = QtWidgets.QWidget()
    qtbot.addWidget(canvas)
    canvas.resize(320, 200)
    switch = ResolutionToggle(canvas, "testResolutionSwitch")
    switch.show()
    canvas.show()
    qtbot.waitExposed(canvas)

    assert switch.size() == QtCore.QSize(208, 40)
    assert switch.pos() == QtCore.QPoint(12, 12)
    assert switch.progress == pytest.approx(0.0)

    with qtbot.waitSignal(switch._animation.finished):
        switch.click()
        qtbot.waitUntil(lambda: 0.0 < switch.progress < 1.0)
    assert switch.progress == pytest.approx(1.0)

    with qtbot.waitSignal(switch._animation.finished):
        switch.click()
        qtbot.waitUntil(lambda: 0.0 < switch.progress < 1.0)
    assert switch.progress == pytest.approx(0.0)


def test_resolution_toggle_does_not_animate_its_initial_state(qtbot):
    canvas = QtWidgets.QWidget()
    qtbot.addWidget(canvas)
    switch = ResolutionToggle(canvas, "testResolutionSwitch")

    assert switch._animation.parent() is switch
    assert switch._animation.state() == QtCore.QAbstractAnimation.State.Stopped


def test_resolution_toggle_reverses_the_existing_animation(qtbot):
    canvas = QtWidgets.QWidget()
    qtbot.addWidget(canvas)
    switch = ResolutionToggle(canvas, "testResolutionSwitch")
    animation = switch._animation
    switch.setChecked(True)
    animation.setCurrentTime(animation.duration() // 2)
    intermediate = switch.progress

    switch.setChecked(False)

    assert switch._animation is animation
    assert animation.startValue() == pytest.approx(intermediate)
    with qtbot.waitSignal(animation.finished):
        pass
    assert switch.progress == pytest.approx(0.0)


def test_hiding_resolution_toggle_stops_animation_and_keeps_selection(qtbot):
    canvas = QtWidgets.QWidget()
    qtbot.addWidget(canvas)
    switch = ResolutionToggle(canvas, "testResolutionSwitch")
    switch.show()
    canvas.show()
    switch.setChecked(True)
    assert switch._animation.state() == QtCore.QAbstractAnimation.State.Running

    canvas.hide()

    assert switch._animation.state() == QtCore.QAbstractAnimation.State.Stopped
    assert switch.progress == pytest.approx(1.0)
    canvas.show()
    switch.setChecked(False)
    with qtbot.waitSignal(switch._animation.finished):
        pass
    assert switch.progress == pytest.approx(0.0)


def test_deleting_canvas_during_toggle_animation_deletes_animation(qtbot):
    # Delete explicitly, so qtbot must not also close this widget at teardown.
    canvas = QtWidgets.QWidget()
    switch = ResolutionToggle(canvas, "testResolutionSwitch")
    animation = switch._animation
    switch.setChecked(True)
    assert animation.state() == QtCore.QAbstractAnimation.State.Running

    with qtbot.waitSignal(canvas.destroyed):
        canvas.deleteLater()

    assert sip.isdeleted(switch)
    assert sip.isdeleted(animation)


def test_image_state_panel_stacks_equal_pills_and_reclaims_space_when_empty(qtbot):
    canvas = QtWidgets.QWidget()
    qtbot.addWidget(canvas)
    canvas.resize(640, 400)
    panel = ImageStatePanel(canvas, "testResolutionSwitch")
    row = QtWidgets.QHBoxLayout(canvas)
    row.addWidget(panel)
    image = QtWidgets.QWidget(canvas)
    row.addWidget(image, 1)
    group = ResolutionSwitchGroup(
        (panel,), lambda high: None, canvas, on_calibration_clicked=lambda calibrated: None,
    )
    canvas.show()
    assert panel.isHidden()
    group.sync(available=True, high_resolution=False, enabled=True)
    group.sync_calibration(available=True, calibrated=True, enabled=True, current_available=True)
    qtbot.wait(20)
    resolution, calibration = group.switches[0], group.calibration_switches[0]
    assert panel.width() == 236
    assert resolution.size() == calibration.size()
    assert resolution.height() == 40
    assert resolution.width() <= panel.contentsRect().width() - 24
    for switch in (resolution, calibration):
        assert switch.geometry().right() < switch.parentWidget().width()
    origin = QtCore.QPoint()
    resolution_pos = resolution.mapTo(panel, origin)
    calibration_pos = calibration.mapTo(panel, origin)
    assert resolution_pos.x() == calibration_pos.x() == panel.contentsRect().left() + 12
    assert calibration_pos.y() > resolution_pos.y() + resolution.height()
    assert not panel.geometry().intersects(image.geometry())
    canvas.resize(520, 400)
    qtbot.wait(20)
    assert calibration.mapTo(panel, origin) == calibration_pos
    assert not panel.geometry().intersects(image.geometry())
    group.sync_calibration(available=True, calibrated=False, enabled=True, current_available=False)
    assert not calibration.isEnabled() and not calibration.isChecked()
    group.sync(available=False, high_resolution=False, enabled=True)
    assert panel.resolution_section.isHidden()
    assert not panel.isHidden()
    group.sync_calibration(available=False, calibrated=False, enabled=True, current_available=False)
    assert panel.isHidden()


def test_collapsed_image_panels_preserve_choices_across_pages_and_updates(qtbot):
    stack = QtWidgets.QStackedWidget()
    qtbot.addWidget(stack)
    stack.resize(640, 400)
    panels, images = [], []
    for index in range(2):
        page = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(page)
        panel = ImageStatePanel(page, f"page{index}ResolutionSwitch")
        image = QtWidgets.QWidget(page)
        row.addWidget(panel)
        row.addWidget(image, 1)
        panels.append(panel)
        images.append(image)
        stack.addWidget(page)
    requests = []
    group = ResolutionSwitchGroup(
        panels, requests.append, stack, on_calibration_clicked=requests.append,
    )
    group.sync(available=True, high_resolution=True, enabled=True)
    group.sync_calibration(available=True, calibrated=True, enabled=True, current_available=True)
    stack.show()
    qtbot.waitExposed(stack)
    expanded_image_width = images[0].width()
    qtbot.mouseClick(panels[0]._collapse_button, QtCore.Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: panels[0].width() == 48)
    assert all(panel.is_collapsed and panel.width() == 48 for panel in panels)
    assert images[0].width() > expanded_image_width
    assert all(panel.resolution_switch.isChecked() and panel.calibration_switch.isChecked()
               for panel in panels)
    assert requests == []

    stack.setCurrentIndex(1)
    group.sync(available=False, high_resolution=False, enabled=True)
    group.sync_calibration(available=True, calibrated=False, enabled=True, current_available=False)
    assert panels[1].isVisible() and panels[1]._body.isHidden()
    assert panels[1]._collapse_button.isVisible()
    assert panels[1]._collapse_button.toolTip() == "Expand image view"
    qtbot.mouseClick(panels[1]._collapse_button, QtCore.Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: panels[1]._body.isVisible())
    assert all(not panel.is_collapsed and panel.width() == 236 for panel in panels)
    assert panels[1].resolution_section.isHidden()
    assert panels[1].calibration_section.isVisible()
    assert not panels[1].calibration_switch.isChecked()
    assert not panels[1].calibration_switch.isEnabled()
    assert not panels[1].geometry().intersects(images[1].geometry())
    assert requests == []


def test_image_panel_reverses_collapse_and_settles_when_hidden_or_stopped(qtbot):
    canvas = QtWidgets.QWidget()
    qtbot.addWidget(canvas)
    panel = ImageStatePanel(canvas, "testResolutionSwitch")
    row = QtWidgets.QHBoxLayout(canvas)
    row.addWidget(panel)
    row.addWidget(QtWidgets.QWidget(canvas), 1)
    group = ResolutionSwitchGroup(
        (panel,), lambda high: None, canvas, on_calibration_clicked=lambda calibrated: None,
    )
    group.sync(available=True, high_resolution=True, enabled=True)
    canvas.show()
    qtbot.waitExposed(canvas)
    panel.toggle_collapsed()
    qtbot.waitUntil(lambda: 48 < panel.width() < 236)
    panel.toggle_collapsed()
    qtbot.waitUntil(lambda: panel.width() == 236 and panel._body.isVisible())
    panel.toggle_collapsed()
    canvas.hide()
    assert panel.width() == 48 and panel.is_collapsed
    assert panel._width_animation.state() == QtCore.QAbstractAnimation.State.Stopped
    canvas.show()
    panel.toggle_collapsed()
    group.stop()
    assert panel.width() == 236 and panel._body.isVisible()
    assert panel._width_animation.state() == QtCore.QAbstractAnimation.State.Stopped
    assert panel.resolution_switch.isChecked()


@pytest.mark.parametrize("toggle_type", [ResolutionToggle, CalibrationToggle])
def test_segments_select_directly_and_support_keyboard(qtbot, toggle_type):
    canvas = QtWidgets.QWidget()
    qtbot.addWidget(canvas)
    switch = toggle_type(canvas, "testSwitch")
    switch.show()
    canvas.show()
    switch.setChecked(True)
    qtbot.mouseClick(switch, QtCore.Qt.MouseButton.LeftButton,
                     pos=QtCore.QPoint(switch.width() // 4, switch.height() // 2))
    assert not switch.isChecked()
    # Selecting the already selected segment does not toggle to the other side.
    qtbot.mouseClick(switch, QtCore.Qt.MouseButton.LeftButton,
                     pos=QtCore.QPoint(switch.width() // 4, switch.height() // 2))
    assert not switch.isChecked()
    qtbot.keyClick(switch, QtCore.Qt.Key.Key_Right)
    assert switch.isChecked()
    assert switch._high_label in switch.accessibleDescription()
    qtbot.keyClick(switch, QtCore.Qt.Key.Key_Left)
    assert not switch.isChecked()
    qtbot.keyClick(switch, QtCore.Qt.Key.Key_Space)
    assert switch.isChecked()
    switch.setEnabled(False)
    qtbot.mouseClick(switch, QtCore.Qt.MouseButton.LeftButton,
                     pos=QtCore.QPoint(switch.width() // 4, switch.height() // 2))
    assert switch.isChecked()
