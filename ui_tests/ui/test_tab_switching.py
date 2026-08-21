from __future__ import annotations

import pytest
from PyQt6.QtCore import QPointF


def test_on_tab_changed_forwards_pan_zoom_to_the_new_viewer(loaded_window, monkeypatch):
    previous_viewer = loaded_window.viewer
    next_viewer = loaded_window._viewer_for_tab(1)
    loaded_window._active_viewer = previous_viewer

    fixed_state = (1.5, QPointF(2.0, 2.0))
    monkeypatch.setattr(previous_viewer, "get_view_state", lambda: fixed_state)
    received = {}
    monkeypatch.setattr(
        next_viewer, "queue_view_state", lambda state: received.setdefault("state", state)
    )

    loaded_window._on_tab_changed(1)

    assert received["state"] == fixed_state
    assert loaded_window._active_viewer is next_viewer


def test_on_tab_changed_skips_forwarding_when_previous_viewer_has_no_state(
    loaded_window, monkeypatch
):
    previous_viewer = loaded_window.viewer
    next_viewer = loaded_window._viewer_for_tab(1)
    loaded_window._active_viewer = previous_viewer

    monkeypatch.setattr(previous_viewer, "get_view_state", lambda: None)
    calls = []
    monkeypatch.setattr(next_viewer, "queue_view_state", lambda state: calls.append(state))

    loaded_window._on_tab_changed(1)

    assert calls == []
    assert loaded_window._active_viewer is next_viewer


def test_switching_tabs_carries_real_pan_zoom_to_the_newly_active_viewer(loaded_window, qtbot):
    loaded_window.show()
    qtbot.waitExposed(loaded_window)

    loaded_window.viewer.set_view_state((2.0, QPointF(3.0, 4.0)))
    expected_scale, expected_center = loaded_window.viewer.get_view_state()

    loaded_window.tabWidget.setCurrentIndex(1)
    qtbot.wait(50)  # let the queued view-state / resize-settling timers fire

    new_viewer = loaded_window._viewer_for_tab(1)
    got_scale, got_center = new_viewer.get_view_state()
    assert got_scale == pytest.approx(expected_scale)
    assert got_center.x() == pytest.approx(expected_center.x(), abs=1.0)
    assert got_center.y() == pytest.approx(expected_center.y(), abs=1.0)
    assert loaded_window._active_viewer is new_viewer


def test_repeatedly_switching_tabs_does_not_drift_the_view(loaded_window, qtbot):
    """Bouncing between two tabs must not creep the pan by ~1px each hop."""
    loaded_window.show()
    qtbot.waitExposed(loaded_window)

    loaded_window.viewer.set_view_state((10.0, QPointF(4.0, 4.0)))
    qtbot.wait(50)
    _, initial_center = loaded_window.viewer.get_view_state()

    for i in range(8):
        target_index = 1 if loaded_window.tabWidget.currentIndex() == 0 else 0
        loaded_window.tabWidget.setCurrentIndex(target_index)
        qtbot.wait(50)

    loaded_window.tabWidget.setCurrentIndex(0)
    qtbot.wait(50)
    _, final_center = loaded_window.viewer.get_view_state()

    assert final_center.x() == pytest.approx(initial_center.x(), abs=0.1)
    assert final_center.y() == pytest.approx(initial_center.y(), abs=0.1)
