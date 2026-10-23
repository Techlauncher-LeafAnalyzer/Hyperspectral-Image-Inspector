"""Super-Resolution and Calibration view toggles in the left tool-window stripe."""

import pytest
from PyQt6 import QtCore, QtWidgets

from ui.resolution_toggle import ResolutionSwitchGroup
from ui.theme import (
    TOOL_STRIPE_DISABLED_COLOR,
    TOOL_STRIPE_ICON_COLOR,
    TOOL_STRIPE_SELECTED_COLOR,
)
from ui.tool_windows import ViewToggleButton, paint_calibration_icon, paint_super_resolution_icon


@pytest.fixture
def group(qtbot):
    host = QtWidgets.QWidget()
    qtbot.addWidget(host)
    resolution = ViewToggleButton(
        paint_super_resolution_icon, "testResolutionSwitch",
        off_label="Low Res", on_label="High Res", unavailable_hint="Run SR first", parent=host,
    )
    calibration = ViewToggleButton(
        paint_calibration_icon, "testCalibrationSwitch",
        off_label="Before Calibration", on_label="After Calibration",
        unavailable_hint="Run Calibration first", parent=host,
    )
    requests = []
    group = ResolutionSwitchGroup(
        resolution, calibration, lambda high: requests.append(("res", high)), host,
        on_calibration_clicked=lambda cal: requests.append(("cal", cal)),
    )
    group.requests = requests
    group.host = host  # keep the buttons' parent alive for the test
    host.show()
    return group


def _name(color):
    from PyQt6.QtGui import QColor
    return QColor(color).name()


def test_toggles_are_gray_until_a_comparison_exists(group):
    group.sync(available=False, high_resolution=False, enabled=True)
    group.sync_calibration(available=False, calibrated=False, enabled=True, current_available=False)
    for switch, hint in ((group.switches[0], "Run SR first"),
                         (group.calibration_switches[0], "Run Calibration first")):
        assert not switch.isEnabled() and not switch.isChecked()
        assert switch.icon_color().name() == _name(TOOL_STRIPE_DISABLED_COLOR)
        assert switch.background_color() is None
        assert switch.toolTip() == hint
        assert not switch.isHidden()  # stripe icons gray out, never vanish


def test_highlight_marks_the_second_version_and_clicks_request_a_switch(group, qtbot):
    resolution = group.switches[0]
    group.sync(available=True, high_resolution=False, enabled=True)
    assert resolution.isEnabled() and not resolution.isChecked()
    assert resolution.background_color() is None
    assert resolution.icon_color().name() == _name(TOOL_STRIPE_ICON_COLOR)
    assert "Viewing Low Res" in resolution.toolTip()

    qtbot.mouseClick(resolution, QtCore.Qt.MouseButton.LeftButton)
    assert group.requests == [("res", True)]

    group.sync(available=True, high_resolution=True, enabled=True)
    assert resolution.isChecked()
    assert resolution.background_color().name() == _name(TOOL_STRIPE_SELECTED_COLOR)
    assert "Viewing High Res" in resolution.toolTip()


def test_sync_never_emits_clicks(group):
    group.sync(available=True, high_resolution=True, enabled=True)
    group.sync_calibration(available=True, calibrated=True, enabled=True, current_available=True)
    group.sync(available=True, high_resolution=False, enabled=True)
    assert group.requests == []


def test_calibration_missing_at_this_resolution_explains_itself(group):
    calibration = group.calibration_switches[0]
    group.sync_calibration(available=True, calibrated=False, enabled=True, current_available=False)
    assert not calibration.isEnabled() and not calibration.isChecked()
    assert "No calibrated result at this resolution" in calibration.toolTip()
    group.sync_calibration(available=True, calibrated=True, enabled=True, current_available=True)
    assert calibration.isEnabled() and calibration.isChecked()
    assert "Viewing After Calibration" in calibration.toolTip()


def test_busy_pipeline_locks_both_toggles(group):
    group.sync(available=True, high_resolution=True, enabled=True)
    group.sync_calibration(available=True, calibrated=True, enabled=True, current_available=True)
    group.set_enabled(False)
    assert not group.switches[0].isEnabled() and not group.calibration_switches[0].isEnabled()
    # The current choice is still shown while locked.
    assert group.switches[0].isChecked() and group.calibration_switches[0].isChecked()


def test_keyboard_space_requests_a_switch(group, qtbot):
    calibration = group.calibration_switches[0]
    group.sync_calibration(available=True, calibrated=True, enabled=True, current_available=True)
    calibration.setFocus()
    qtbot.keyClick(calibration, QtCore.Qt.Key.Key_Space)
    assert group.requests == [("cal", False)]
