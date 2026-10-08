from __future__ import annotations

import json
import numpy as np
import pytest
from PIL import Image
from spectral.io import envi

from core import VisualizationMode


INDEX_BUTTONS = ("modeNDVI", "modeEVI", "modeMCARI", "modeMTVI", "modeOSAVI", "modePRI")


@pytest.mark.parametrize("camera, start, end, indices_available", [
    ("X10", 398.25, 1001.17, True), ("X17", 935.613, 1720.233, False),
])
def test_loading_new_camera_updates_preview_indices_and_sr(
    window, camera_cube_factory, dialogs, camera, start, end, indices_available
):
    data, _ = camera_cube_factory(np.linspace(start, end, 224), name=camera)
    window.load_image_from_path(data.source_path)
    assert not dialogs.critical
    assert window.viewer.has_photo() and window.modeRGB.isEnabled()
    assert window.modeHyperCube.isEnabled()
    assert window._hsi_data.bands == 224
    for name in INDEX_BUTTONS:
        button = getattr(window, name)
        assert button.isEnabled() == indices_available
        if not indices_available:
            assert "unavailable" in button.toolTip()
            button.click()
            assert window._active_visualization_mode is VisualizationMode.RGB
    assert not window.runSuperResButton.isEnabled()
    assert "incompatible" in window.runSuperResButton.text()
    window.runSuperResButton.click()
    assert window._super_res_worker is None and not dialogs.critical
    if not indices_available:
        assert "False-colour RGB" in window.modeRGB.toolTip()


def test_camera_switch_resets_and_restores_individual_visualization_controls(
    window, camera_cube_factory
):
    full, _ = camera_cube_factory([470, 531, 550, 570, 660, 670, 700, 800], name="full")
    sparse, _ = camera_cube_factory([400, 550, 670, 700, 800, 1000], name="sparse")
    window.load_image_from_path(full.source_path)
    window.modeEVI.click()
    window.load_image_from_path(sparse.source_path)
    assert window.modeRGB.isChecked()
    assert window.modeNDVI.isEnabled() and window.modeMCARI.isEnabled()
    assert window.modeMTVI.isEnabled() and window.modeOSAVI.isEnabled()
    assert not window.modeEVI.isEnabled() and not window.modePRI.isEnabled()
    window.load_image_from_path(full.source_path)
    assert all(getattr(window, name).isEnabled() for name in INDEX_BUTTONS)


def test_sr_compatibility_controls_restore_on_load_and_stay_disabled_after_crop(
    window, camera_cube_factory
):
    compatible, _ = camera_cube_factory(np.linspace(352.49, 898.81, 480), name="original")
    incompatible, _ = camera_cube_factory(np.linspace(935.613, 1720.233, 480), name="other480")
    window.load_image_from_path(compatible.source_path)
    assert window.runSuperResButton.isEnabled()
    window.load_image_from_path(incompatible.source_path)
    assert not window.runSuperResButton.isEnabled()
    assert "coverage" in window.runSuperResButton.toolTip()
    window._hsi_data.crop(1, 1, 7, 5)
    window._push_image_to_viewers()
    window._set_super_resolution_ready()
    assert not window.runSuperResButton.isEnabled()
    assert not window.modeNDVI.isEnabled()
    window.load_image_from_path(compatible.source_path)
    assert window.runSuperResButton.isEnabled()
    assert window.runSuperResButton.text() == "Run Super-Resolution"


def test_false_colour_preview_is_used_on_all_pages_and_can_be_saved(
    window, camera_cube_factory, file_dialog, tmp_path
):
    data, _ = camera_cube_factory(np.linspace(935.613, 1720.233, 224))
    window.load_image_from_path(data.source_path)
    result = window._visualization_results[VisualizationMode.RGB]
    assert dict(result.band_indices) == {"red": 115, "green": 30, "blue": 223}
    for viewer in window._all_viewers():
        assert viewer.has_photo()
        np.testing.assert_array_equal(viewer.rgb, result.display_rgb)
    path = tmp_path / "false-colour.png"
    file_dialog.save_return = (str(path), "")
    window.actionSaveImage.trigger()
    with Image.open(path) as saved:
        np.testing.assert_array_equal(np.asarray(saved.convert("RGB")), result.display_rgb)


def test_loading_json_swir_metadata_uses_fallback_and_disables_unsupported_tools(
    window, camera_cube_factory, dialogs, file_dialog
):
    data, _ = camera_cube_factory(np.linspace(935.613, 1720.233, 224), name="X17_json")
    header = data.header_path
    metadata = header.with_suffix(".json")
    metadata.write_text(json.dumps({"bil_hdr": envi.read_envi_header(str(header))}),
                        encoding="utf-8")
    data.close()
    header.unlink()
    file_dialog.open_return = (str(metadata), "")
    window.actionLoadImage.trigger()
    assert not dialogs.critical
    assert window._hsi_data.header_format == "JSON"
    assert window.viewer.has_photo()
    assert "False-colour RGB" in window.modeRGB.toolTip()
    assert not any(getattr(window, name).isEnabled() for name in INDEX_BUTTONS)
    assert not window.runSuperResButton.isEnabled()


def test_json_without_pixels_keeps_previously_loaded_image(
    loaded_window, tmp_path, dialogs
):
    previous = loaded_window._hsi_data.source_path
    metadata = tmp_path / "metadata_only.json"
    metadata.write_text('{"bil_hdr": {}, "frames": []}', encoding="utf-8")
    loaded_window.load_image_from_path(metadata)
    assert loaded_window._hsi_data.source_path == previous
    assert "copy the corresponding binary cube" in dialogs.critical[-1][2]


def test_mismatched_json_keeps_previous_cube_and_points_to_matching_hdr(
    loaded_window, camera_cube_factory, dialogs
):
    previous = loaded_window._hsi_data.source_path
    data, _ = camera_cube_factory(np.linspace(398.25, 1001.17, 224), name="X10_conflict")
    metadata = envi.read_envi_header(str(data.header_path))
    metadata["data type"] = 12  # The binary cube is float32, like the real X10 capture.
    path = data.header_path.with_suffix(".json")
    path.write_text(json.dumps({"bil_hdr": metadata}), encoding="utf-8")
    loaded_window.load_image_from_path(path)
    assert loaded_window._hsi_data.source_path == previous
    assert "conflicts" in dialogs.critical[-1][2]
    assert "X10_conflict.hdr" in dialogs.critical[-1][2]


def test_water_index_buttons_only_appear_for_swir_images(window, camera_cube_factory):
    assert window.modeNDWI.isHidden() and window.modeNDMI.isHidden()
    swir, _ = camera_cube_factory(np.linspace(935.613, 1720.233, 224), name="swir")
    window.load_image_from_path(swir.source_path)
    assert not window.modeNDWI.isHidden() and window.modeNDWI.isEnabled()
    assert not window.modeNDMI.isHidden() and window.modeNDMI.isEnabled()
    assert window.viewer.available_optional_indices == {"NDWI", "NDMI"}
    visible, _ = camera_cube_factory(np.linspace(398.25, 1001.17, 224), name="visible")
    window.load_image_from_path(visible.source_path)
    assert window.modeNDWI.isHidden() and window.modeNDMI.isHidden()
    assert not window.modeNDVI.isHidden() and window.modeNDVI.isEnabled()
    assert window.viewer.available_optional_indices == frozenset()
