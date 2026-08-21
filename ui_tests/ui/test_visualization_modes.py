from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QPointF

from core import VisualizationMode

MODE_BUTTONS = (
    ("modeRGB", VisualizationMode.RGB),
    ("modeNDVI", VisualizationMode.NDVI),
    ("modeEVI", VisualizationMode.EVI),
    ("modeMCARI", VisualizationMode.MCARI),
    ("modeMTVI", VisualizationMode.MTVI),
    ("modeOSAVI", VisualizationMode.OSAVI),
    ("modePRI", VisualizationMode.PRI),
)


@pytest.mark.parametrize("button_name, mode", MODE_BUTTONS)
def test_selecting_mode_renders_and_activates_it(loaded_window, button_name, mode):
    getattr(loaded_window, button_name).click()

    assert loaded_window._active_visualization_mode is mode
    result = loaded_window._visualization_results[mode]
    assert result.mode is mode
    assert result.display_rgb.shape == (8, 8, 3)
    assert result.display_rgb.dtype == np.uint8


@pytest.mark.parametrize("button_name, mode", MODE_BUTTONS)
def test_selecting_mode_updates_viewer_pixmap(loaded_window, button_name, mode):
    getattr(loaded_window, button_name).click()

    assert loaded_window.viewer.has_photo()
    # Every viewer tab receives the same display, not just the active one.
    for viewer in loaded_window._all_viewers():
        assert viewer.has_photo()


def test_switching_between_modes_changes_displayed_pixels(loaded_window):
    loaded_window.modeRGB.click()
    rgb_pixels = loaded_window._visualization_results[VisualizationMode.RGB].display_rgb.copy()

    loaded_window.modeNDVI.click()
    ndvi_pixels = loaded_window._visualization_results[VisualizationMode.NDVI].display_rgb

    assert not np.array_equal(rgb_pixels, ndvi_pixels)


def test_pixel_values_are_reported_for_every_cached_mode(loaded_window):
    values = loaded_window._pixel_values_at(0, 0)

    assert set(values) == {mode.value for _, mode in MODE_BUTTONS}


def test_switching_visualization_mode_preserves_pan_and_zoom(loaded_window, qtbot):
    loaded_window.show()
    qtbot.waitExposed(loaded_window)

    loaded_window.viewer.set_view_state((2.0, QPointF(3.0, 4.0)))
    expected_scale, expected_center = loaded_window.viewer.get_view_state()

    loaded_window.modeNDVI.click()
    qtbot.wait(50)  # let the queued view-state / resize-settling timers fire

    got_scale, got_center = loaded_window.viewer.get_view_state()
    assert got_scale == pytest.approx(expected_scale)
    assert got_center.x() == pytest.approx(expected_center.x(), abs=1.0)
    assert got_center.y() == pytest.approx(expected_center.y(), abs=1.0)
