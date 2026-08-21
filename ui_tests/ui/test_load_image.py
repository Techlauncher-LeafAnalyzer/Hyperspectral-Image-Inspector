from __future__ import annotations

from core import VisualizationMode


def test_load_image_from_path_populates_state(window, synthetic_cube_path, synthetic_shape):
    window.load_image_from_path(synthetic_cube_path)

    assert window._hsi_data.is_loaded()
    assert window._hsi_data.shape == synthetic_shape
    assert window._active_visualization_mode is VisualizationMode.RGB
    assert window.viewer.has_photo()
    assert synthetic_cube_path.stem in window.imageFilePath.text()
    assert window.unsupervisedClassifyButton.isEnabled()
    assert window.pushButton_2.isEnabled()


def test_load_image_via_menu_action_uses_file_dialog(window, synthetic_cube_path, file_dialog):
    file_dialog.open_return = (str(synthetic_cube_path), "")

    window.actionLoadImage.trigger()

    assert window._hsi_data.is_loaded()


def test_load_image_cancelled_dialog_is_noop(window, file_dialog):
    file_dialog.open_return = ("", "")

    window.actionLoadImage.trigger()

    assert not window._hsi_data.is_loaded()
    assert not window.viewer.has_photo()


def test_load_image_missing_data_file_shows_error(window, tmp_path, dialogs):
    orphan_header = tmp_path / "orphan.hdr"
    orphan_header.write_text("ENVI\nsamples = 1\nlines = 1\nbands = 1\n")

    window.load_image_from_path(orphan_header)

    assert not window._hsi_data.is_loaded()
    assert len(dialogs.critical) == 1


def test_loading_a_second_image_resets_crop_history(window, synthetic_cube_path):
    window.load_image_from_path(synthetic_cube_path)
    window._crop_undo_stack.append(window._snapshot_current_state())

    window.load_image_from_path(synthetic_cube_path)

    assert window._crop_undo_stack == []
    assert window._crop_redo_stack == []
