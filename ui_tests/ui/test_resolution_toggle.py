"""Geometry and animation checks for the shared canvas resolution switch."""

import pytest
from PyQt6 import QtCore, QtWidgets, sip

from ui.resolution_toggle import CalibrationToggle, ResolutionSwitchGroup, ResolutionToggle


def test_resolution_toggle_is_compact_and_animates_between_states(qtbot):
    canvas = QtWidgets.QWidget()
    qtbot.addWidget(canvas)
    canvas.resize(320, 200)
    switch = ResolutionToggle(canvas, "testResolutionSwitch")
    switch.show()
    canvas.show()
    qtbot.waitExposed(canvas)

    assert switch.size() == QtCore.QSize(208, 36)
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


def test_calibration_switches_align_and_wrap_on_narrow_canvases(qtbot):
    canvases = [QtWidgets.QWidget() for _ in range(4)]
    for canvas in canvases:
        qtbot.addWidget(canvas)
        canvas.resize(640, 200)
        canvas.show()
    group = ResolutionSwitchGroup(
        ((canvas, f"page{index}ResolutionSwitch") for index, canvas in enumerate(canvases[:3])),
        lambda high: None, canvases[0], on_calibration_clicked=lambda calibrated: None,
        calibration_only_canvases=((canvases[3], "superResolutionCalibrationSwitch"),),
    )
    assert len(group.switches) == 3 and len(group.calibration_switches) == 4
    assert all(switch.isHidden() for switch in group.calibration_switches)
    group.sync(available=True, high_resolution=False, enabled=True)
    group.sync_calibration(available=True, calibrated=True, enabled=True, current_available=True)
    for switch in group.calibration_switches[:3]:
        assert switch.pos() == QtCore.QPoint(228, 12)
        assert switch.isChecked()
        assert switch.height() == group.switches[0].height()
    assert group.calibration_switches[3].pos() == QtCore.QPoint(12, 12)
    for canvas in canvases:
        canvas.resize(400, 380)
    assert group.calibration_switches[0].pos() == QtCore.QPoint(12, 56)
    group.sync(available=False, high_resolution=False, enabled=True)
    assert all(switch.pos() == QtCore.QPoint(12, 12) for switch in group.calibration_switches)
    group.sync_calibration(available=True, calibrated=False, enabled=True, current_available=False)
    assert all(not switch.isEnabled() and not switch.isChecked() for switch in group.calibration_switches)
    group.sync_calibration(available=False, calibrated=False, enabled=True, current_available=False)
    assert all(switch.isHidden() for switch in group.calibration_switches)


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
