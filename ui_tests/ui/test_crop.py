from __future__ import annotations

import numpy as np
from PIL import Image
from PyQt6.QtCore import QRectF

from core import VisualizationMode, VisualizationService

# left=1, top=1, right=5, bottom=5 -> a 4x4 crop out of the 8x8 fixture.
CROP_RECT = QRectF(1, 1, 4, 4)


def test_crop_shrinks_image_and_recomputes_visualizations(loaded_window):
    loaded_window.viewer.cropRequested.emit(CROP_RECT)

    assert loaded_window._hsi_data.rgb_array.shape[:2] == (4, 4)
    assert loaded_window._hsi_data.mask_array.shape == (4, 4)
    assert loaded_window._visualization_results[VisualizationMode.RGB].display_rgb.shape == (
        4,
        4,
        3,
    )
    assert len(loaded_window._crop_undo_stack) == 1
    assert loaded_window._crop_redo_stack == []
    assert "Cropped" in loaded_window.statusbar.currentMessage()


def test_crop_undo_restores_original_image(loaded_window):
    original_rgb = loaded_window._hsi_data.rgb_array.copy()

    loaded_window.viewer.cropRequested.emit(CROP_RECT)
    loaded_window._undo_crop()

    assert loaded_window._hsi_data.rgb_array.shape[:2] == (8, 8)
    assert np.array_equal(loaded_window._hsi_data.rgb_array, original_rgb)
    assert loaded_window._crop_undo_stack == []
    assert len(loaded_window._crop_redo_stack) == 1


def test_crop_redo_reapplies_the_crop(loaded_window):
    loaded_window.viewer.cropRequested.emit(CROP_RECT)
    cropped_rgb = loaded_window._hsi_data.rgb_array.copy()
    loaded_window._undo_crop()

    loaded_window._redo_crop()

    assert loaded_window._hsi_data.rgb_array.shape[:2] == (4, 4)
    assert np.array_equal(loaded_window._hsi_data.rgb_array, cropped_rgb)
    assert loaded_window._crop_redo_stack == []
    assert len(loaded_window._crop_undo_stack) == 1


def test_degenerate_crop_rect_is_a_noop(loaded_window):
    loaded_window.viewer.cropRequested.emit(QRectF(0, 0, 0, 0))

    assert loaded_window._hsi_data.rgb_array.shape[:2] == (8, 8)
    assert loaded_window._crop_undo_stack == []


def test_crop_without_a_loaded_image_is_a_noop(window):
    window.viewer.cropRequested.emit(CROP_RECT)

    assert not window._hsi_data.is_loaded()
    assert window._crop_undo_stack == []


def test_undo_and_redo_with_empty_stacks_are_noops(loaded_window):
    original_rgb = loaded_window._hsi_data.rgb_array.copy()

    loaded_window._undo_crop()
    loaded_window._redo_crop()

    assert np.array_equal(loaded_window._hsi_data.rgb_array, original_rgb)


def test_save_after_crop_reflects_cropped_dimensions(loaded_window, tmp_path, file_dialog):
    loaded_window.viewer.cropRequested.emit(CROP_RECT)
    target = tmp_path / "cropped.png"
    file_dialog.save_return = (str(target), "")

    loaded_window.actionSaveImage.trigger()

    assert Image.open(target).size == (4, 4)


def test_visualization_mode_switch_after_crop_uses_cropped_data(loaded_window):
    loaded_window.viewer.cropRequested.emit(CROP_RECT)

    loaded_window.modeNDVI.click()

    assert loaded_window._visualization_results[VisualizationMode.NDVI].display_rgb.shape == (
        4,
        4,
        3,
    )


def test_hypercube_view_after_crop_reads_the_cropped_region(loaded_window):
    # Regression test: SPy 0.25's default SubImage.read_subimage calls
    # array.array(rows) with no type code and always raises TypeError.
    # _CroppedSpyFile.read_subimage overrides this; prepare_hypercube_view
    # is the only caller that reads a cropped image via read_subimage
    # instead of read_band/read_bands, so it's the path that surfaced this.
    loaded_window.viewer.cropRequested.emit(CROP_RECT)

    result = VisualizationService().prepare_hypercube_view(loaded_window._hsi_data)

    assert result.surface_cube.shape[:2] == (4, 4)
