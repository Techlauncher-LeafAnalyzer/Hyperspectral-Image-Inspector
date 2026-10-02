"""Geometry and animation checks for the shared canvas resolution switch."""

import pytest
from PyQt6 import QtCore, QtWidgets, sip

from ui.resolution_toggle import ResolutionSwitchGroup, ResolutionToggle


def test_resolution_toggle_is_compact_and_animates_between_states(qtbot):
    canvas = QtWidgets.QWidget()
    qtbot.addWidget(canvas)
    canvas.resize(320, 200)
    switch = ResolutionToggle(canvas, "testResolutionSwitch")
    switch.show()
    canvas.show()
    qtbot.waitExposed(canvas)

    assert switch.size() == QtCore.QSize(116, 36)
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


def test_calibration_tags_stay_next_to_switch_or_at_fixed_canvas_inset(qtbot):
    canvases = [QtWidgets.QWidget() for _ in range(4)]
    for canvas in canvases:
        qtbot.addWidget(canvas)
        canvas.resize(320, 200)
        canvas.show()
    group = ResolutionSwitchGroup(
        ((canvas, f"page{index}ResolutionSwitch") for index, canvas in enumerate(canvases[:3])),
        lambda high: None, canvases[0],
        badge_only_canvases=((canvases[3], "superResolutionCalibrationBadge"),),
    )
    assert len(group.switches) == 3 and len(group.calibration_badges) == 4
    assert all(badge.isHidden() for badge in group.calibration_badges)
    group.sync(available=True, high_resolution=False, enabled=True)
    group.set_calibrated(True)
    assert all(not badge.isHidden() for badge in group.calibration_badges)
    for badge in group.calibration_badges[:3]:
        assert badge.pos() == QtCore.QPoint(136, 16)
        assert badge.text() == "Calibrated"
        assert badge.testAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert group.calibration_badges[3].pos() == QtCore.QPoint(12, 16)
    for canvas in canvases:
        canvas.resize(540, 380)
    group.raise_switches()
    assert group.calibration_badges[0].pos() == QtCore.QPoint(136, 16)
    group.sync(available=False, high_resolution=False, enabled=True)
    assert all(badge.pos() == QtCore.QPoint(12, 16) for badge in group.calibration_badges)
    group.set_calibrated(False)
    assert all(badge.isHidden() for badge in group.calibration_badges)
