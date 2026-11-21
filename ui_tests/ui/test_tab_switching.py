from __future__ import annotations

import pytest
from PyQt6 import QtCore, QtWidgets, sip

from ui.tab_transition.handler import TabTransitionHandler


def test_close_stops_active_transitions_and_resolution_animations(window):
    window._resolution_switches.sync(
        available=True, high_resolution=True, enabled=True
    )
    tabs = window.classificationModeTabs
    tabs.setCurrentIndex(1 - tabs.currentIndex())
    animations = window.findChildren(QtCore.QAbstractAnimation)
    assert any(a.state() == a.State.Running for a in animations)

    window.close()

    assert all(a.state() == a.State.Stopped for a in animations)
    assert all(not t._sync_timer.isActive() for t in window._tab_transitions)


@pytest.mark.parametrize("remove_page", [False, True], ids=["replace-effect", "delete-page"])
def test_fade_animation_dies_with_its_effect(qtbot, remove_page):
    tabs = QtWidgets.QTabWidget()
    qtbot.addWidget(tabs)
    tabs.addTab(QtWidgets.QWidget(), "First")
    page = QtWidgets.QWidget()
    tabs.addTab(page, "Second")
    handler = TabTransitionHandler(tabs)
    tabs.setCurrentIndex(1)
    effect = page.graphicsEffect()
    animation = next(
        a for a in tabs.findChildren(QtCore.QPropertyAnimation)
        if a.targetObject() is effect
    )
    assert animation.state() == animation.State.Running

    if remove_page:
        tabs.removeTab(1)
        with qtbot.waitSignal(page.destroyed):
            page.deleteLater()
    else:
        page.setGraphicsEffect(None)

    assert sip.isdeleted(effect)
    assert sip.isdeleted(animation)
    if not remove_page:
        # Replacing an effect must not leave a stale entry on the next switch.
        tabs.setCurrentIndex(0)
        tabs.setCurrentIndex(1)
        replacement = page.graphicsEffect()
        animation = next(
            a for a in tabs.findChildren(QtCore.QPropertyAnimation)
            if a.targetObject() is replacement
        )
        with qtbot.waitSignal(animation.finished):
            pass
        assert replacement.opacity() == pytest.approx(1.0)
        assert not replacement.isEnabled()
    handler.stop()
