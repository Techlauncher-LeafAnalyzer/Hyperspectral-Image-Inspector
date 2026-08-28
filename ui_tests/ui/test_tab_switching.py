from __future__ import annotations

import pytest
import numpy as np
from PyQt6.QtCore import QPointF, QRectF

from core import VisualizationMode


def test_cropped_original_keeps_identical_pixels_across_tabs(loaded_window, qtbot, monkeypatch):
    window = loaded_window
    window.show()
    qtbot.waitExposed(window)
    window._on_crop_requested(QRectF(1, 1, 6, 6))

    # Cropping recomputes display limits. SR must not retain the old full-image
    # stretch while Visualization/Calibration/Classification use the new one.
    rgb = window._visualization_results[VisualizationMode.RGB].display_rgb
    np.testing.assert_array_equal(window._hsi_data.rgb_array, rgb)
    expected = window.viewer._photo.pixmap().toImage()
    cube_before = window._hsi_data.read_bands(range(window._hsi_data.bands)).copy()

    def unexpected_processing(*args, **kwargs):
        pytest.fail("Switching tabs must not re-render or process the Original")

    monkeypatch.setattr(window._visualization_service, "render", unexpected_processing)
    monkeypatch.setattr(window._super_resolution_service, "run", unexpected_processing)
    for index in (1, 2, 1, 3, 1, 0, 1, 0):
        window.tabWidget.setCurrentIndex(index)
        qtbot.wait(200)  # Allow the existing tab fade to finish.
        assert window._viewer_for_tab(index)._photo.pixmap().toImage() == expected
        assert window._hsi_data.rgb_array is rgb
    np.testing.assert_array_equal(
        window._hsi_data.read_bands(range(window._hsi_data.bands)), cube_before
    )


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
    # HSIViewer.queue_view_state keeps recentring on every resize for 300ms
    # (see viewer.py's `_pending_view_state`) to absorb late layout settling;
    # each hop must clear that window before the next one, otherwise an
    # in-flight recentre from hop N can land after hop N+1 has already begun.
    SETTLE_MS = 350

    loaded_window.show()
    qtbot.waitExposed(loaded_window)

    loaded_window.viewer.set_view_state((10.0, QPointF(4.0, 4.0)))
    qtbot.wait(SETTLE_MS)
    _, initial_center = loaded_window.viewer.get_view_state()

    for i in range(8):
        target_index = 1 if loaded_window.tabWidget.currentIndex() == 0 else 0
        loaded_window.tabWidget.setCurrentIndex(target_index)
        qtbot.wait(SETTLE_MS)

    loaded_window.tabWidget.setCurrentIndex(0)
    qtbot.wait(SETTLE_MS)
    _, final_center = loaded_window.viewer.get_view_state()

    assert final_center.x() == pytest.approx(initial_center.x(), abs=1e-6)
    assert final_center.y() == pytest.approx(initial_center.y(), abs=1e-6)
