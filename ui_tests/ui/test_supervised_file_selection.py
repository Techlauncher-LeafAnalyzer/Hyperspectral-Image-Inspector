"""Automatic and manual supervised training-cube selection."""

from __future__ import annotations

import numpy as np
from PIL import Image
from spectral.io import envi

def _mask(path):
    labels = np.ones((8, 8), dtype=np.uint8)
    labels[4:] = 2
    Image.fromarray(labels).save(path)
    return path


def _choose_mask(window, file_dialog, path):
    file_dialog.open_return = (str(path), "")
    window.pushButton.click()


def _choose_cube(window, file_dialog, path):
    file_dialog.open_return = (str(path), "")
    window.hyperspectralImageButton.click()


def test_mask_selection_shows_auto_paired_cube(loaded_window, synthetic_cube_path, file_dialog):
    mask = _mask(synthetic_cube_path.with_name("synthetic_mask.png"))

    _choose_mask(loaded_window, file_dialog, mask)

    expected_cube = synthetic_cube_path.with_suffix(".img").resolve()
    assert loaded_window.lineEdit.text() == str(mask.resolve())
    assert loaded_window.hyperspectralImageEdit.text() == str(expected_cube)
    pair = loaded_window._classification_controller._training_pair
    assert pair is not None and pair.cube_path == expected_cube


def test_unpaired_mask_can_use_manual_cube(
    loaded_window, synthetic_cube_path, file_dialog, tmp_path, dialogs, qtbot
):
    mask = _mask(tmp_path / "unpaired_mask.png")
    _choose_mask(loaded_window, file_dialog, mask)

    assert loaded_window.lineEdit.text() == str(mask.resolve())
    assert loaded_window.hyperspectralImageEdit.text() == ""
    assert loaded_window._classification_controller._training_pair is None
    assert not dialogs.critical

    cube = synthetic_cube_path.with_suffix(".img")
    _choose_cube(loaded_window, file_dialog, cube)
    assert loaded_window.hyperspectralImageEdit.text() == str(cube.resolve())
    pair = loaded_window._classification_controller._training_pair
    assert pair is not None and pair.naming_convention == "manual selection"

    loaded_window.pushButton_2.click()
    qtbot.waitUntil(
        lambda: loaded_window._classification_controller._thread is None,
        timeout=10000,
    )
    result = loaded_window._classification_controller._current_slot.result
    assert not dialogs.critical
    assert result is not None and result.training_cube_path == cube.resolve()


def test_manual_cube_overrides_auto_pair_and_header_selection_is_normalized(
    loaded_window, synthetic_cube_path, file_dialog, tmp_path
):
    mask = _mask(synthetic_cube_path.with_name("synthetic_mask.png"))
    _choose_mask(loaded_window, file_dialog, mask)
    auto_cube = loaded_window._classification_controller._training_pair.cube_path

    values = loaded_window._hsi_data.read_bands(range(loaded_window._hsi_data.bands))
    manual_header = tmp_path / "other_training.hdr"
    envi.save_image(
        str(manual_header),
        values,
        ext=".bip",
        interleave="bip",
        metadata={"wavelength": loaded_window._hsi_data.wavelengths_nm.tolist()},
    )
    _choose_cube(loaded_window, file_dialog, manual_header)

    manual_cube = manual_header.with_suffix(".bip").resolve()
    pair = loaded_window._classification_controller._training_pair
    assert pair is not None and pair.cube_path == manual_cube
    assert pair.mask_path == mask.resolve()
    assert pair.cube_path != auto_cube
    assert loaded_window.hyperspectralImageEdit.text() == str(manual_cube)


def test_manual_cube_selected_before_mask_is_retained(
    loaded_window, synthetic_cube_path, file_dialog
):
    cube = synthetic_cube_path.with_suffix(".img")
    _choose_cube(loaded_window, file_dialog, cube)
    assert loaded_window._classification_controller._training_pair is None

    mask = _mask(synthetic_cube_path.with_name("unpaired_mask.png"))
    _choose_mask(loaded_window, file_dialog, mask)

    pair = loaded_window._classification_controller._training_pair
    assert pair is not None and pair.naming_convention == "manual selection"
    assert pair.mask_path == mask.resolve()
    assert pair.cube_path == cube.resolve()
    assert loaded_window.hyperspectralImageEdit.text() == str(cube.resolve())


def test_invalid_manual_cube_keeps_previous_auto_pair(
    loaded_window, synthetic_cube_path, file_dialog, tmp_path, dialogs
):
    mask = _mask(synthetic_cube_path.with_name("synthetic_mask.png"))
    _choose_mask(loaded_window, file_dialog, mask)
    previous_pair = loaded_window._classification_controller._training_pair

    invalid = tmp_path / "missing_pair.bil"
    invalid.write_bytes(b"not a complete cube")
    _choose_cube(loaded_window, file_dialog, invalid)

    assert dialogs.critical
    assert ".hdr" in dialogs.critical[-1][2].lower()
    assert loaded_window._classification_controller._training_pair is previous_pair
    assert loaded_window.hyperspectralImageEdit.text() == str(previous_pair.cube_path)
