"""Geometry and animation checks for the shared canvas resolution switch."""

import pytest
from PyQt6 import QtCore, QtWidgets

from ui.resolution_toggle import ResolutionToggle


def test_resolution_toggle_is_compact_and_animates_between_states(qtbot):
    canvas = QtWidgets.QWidget()
    qtbot.addWidget(canvas)
    canvas.resize(320, 200)
    switch = ResolutionToggle(canvas, "testResolutionSwitch")
    switch.show()
    canvas.show()
    qtbot.waitExposed(canvas)

    assert switch.size() == QtCore.QSize(138, 36)
    assert switch.pos() == QtCore.QPoint(12, 12)
    assert switch.progress == pytest.approx(0.0)

    switch.click()
    qtbot.wait(75)
    assert 0.0 < switch.progress < 1.0
    qtbot.wait(180)
    assert switch.progress == pytest.approx(1.0)

    switch.click()
    qtbot.wait(75)
    assert 0.0 < switch.progress < 1.0
    qtbot.wait(180)
    assert switch.progress == pytest.approx(0.0)
