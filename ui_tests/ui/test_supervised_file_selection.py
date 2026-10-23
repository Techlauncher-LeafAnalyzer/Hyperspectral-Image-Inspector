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



def test_separate_leaves_enabled_after_supervised_run(
    loaded_window, synthetic_cube_path, file_dialog, qtbot, dialogs
):
    button = loaded_window.separateLeavesButton
    assert not button.isEnabled()
    mask = _mask(synthetic_cube_path.with_name("synthetic_mask.png"))
    _choose_mask(loaded_window, file_dialog, mask)

    loaded_window.pushButton_2.click()
    assert not button.isEnabled()
    qtbot.waitUntil(
        lambda: loaded_window._classification_controller._thread is None,
        timeout=10000,
    )

    assert not dialogs.critical
    assert button.isEnabled()


def test_separate_leaves_splits_plant_class_into_leaf_layers(
    loaded_window, dialogs, monkeypatch
):
    from functools import partial
    from pathlib import Path

    import ui.classification_controller as classification_controller
    from core import separate_leaves

    from core.classification_model import (
        SupervisedClassificationResult,
        SupervisedClassifierType,
    )

    # The synthetic image is 8x8, far below the default noise floor.
    monkeypatch.setattr(
        classification_controller,
        "separate_leaves",
        partial(separate_leaves, min_area_pixels=1),
    )
    controller = loaded_window._classification_controller
    button = loaded_window.separateLeavesButton

    class_map = np.full((8, 8), 2, dtype=np.int32)
    class_map[0:3, 0:3] = 1
    class_map[5:8, 5:8] = 1
    class_ids = (1, 2)
    masks = (class_map[None] == np.asarray(class_ids)[:, None, None]).astype(np.uint8)
    result = SupervisedClassificationResult(
        class_map=class_map,
        one_hot_masks=masks,
        class_ids=class_ids,
        class_pixel_counts=masks.reshape(2, -1).sum(axis=1),
        training_sample_counts=np.asarray([1, 1]),
        classifier=SupervisedClassifierType.GAUSSIAN,
        training_band_indices=(0,),
        target_band_indices=(0,),
        band_wavelengths_nm=np.asarray([550.0]),
        training_mask_path=Path("mask.png"),
        training_cube_path=Path("cube.img"),
    )
    controller._pending_slot_key = False
    controller._pending_calibrated = False
    controller._on_result(result)
    controller.refresh_display()
    assert button.isEnabled()

    controller._current_slot.layers.rename_class(2, "Soil")
    button.click()

    layers = controller._current_slot.layers.layers
    assert [layer.name for layer in layers] == ["Leaf 1", "Leaf 2", "Soil"]
    assert [layer.pixel_count for layer in layers] == [9, 9, 46]
    rgb = controller._current_slot.rgb
    assert rgb[4, 0].tolist() == [0, 0, 0]
    assert rgb[0, 0].any() and rgb[7, 7].any()
    assert not button.isEnabled()
    assert not dialogs.critical

    # Each leaf's number is burned in beneath it, and follows visibility.
    slot = controller._current_slot
    assert [label.text for label in slot.leaf_labels] == ["1", "2"]
    composited = controller.composited_rgb()
    flat = controller._composite().display_rgb
    assert not np.array_equal(composited, flat)
    for layer in slot.layers.layers:
        slot.layers.set_class_visible(layer.class_id, layer.name == "Soil")
    np.testing.assert_array_equal(
        controller.composited_rgb(), controller._composite().display_rgb
    )
