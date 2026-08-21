from __future__ import annotations

import numpy as np
from PIL import Image

from core import VisualizationMode


def test_save_with_no_image_loaded_shows_info_and_writes_nothing(window, dialogs, file_dialog):
    window.actionSaveImage.trigger()

    assert len(dialogs.information) == 1
    assert file_dialog.save_return == ("", "")  # dialog never consulted


def test_save_writes_png_matching_active_visualization(loaded_window, tmp_path, file_dialog):
    target = tmp_path / "out.png"
    file_dialog.save_return = (str(target), "")

    loaded_window.actionSaveImage.trigger()

    assert target.exists()
    saved = np.array(Image.open(target).convert("RGB"))
    expected = loaded_window._visualization_results[VisualizationMode.RGB].display_rgb
    assert np.array_equal(saved, expected)


def test_save_after_switching_mode_saves_active_mode_pixels(loaded_window, tmp_path, file_dialog):
    loaded_window.modeNDVI.click()
    target = tmp_path / "ndvi.png"
    file_dialog.save_return = (str(target), "")

    loaded_window.actionSaveImage.trigger()

    saved = np.array(Image.open(target).convert("RGB"))
    expected = loaded_window._visualization_results[VisualizationMode.NDVI].display_rgb
    assert np.array_equal(saved, expected)


def test_save_cancelled_dialog_writes_nothing(loaded_window, tmp_path, file_dialog):
    target = tmp_path / "never-written.png"
    file_dialog.save_return = ("", "")

    loaded_window.actionSaveImage.trigger()

    assert not target.exists()
