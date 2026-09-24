from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from PyQt6.QtCore import QPointF

from core import ClassificationService, VisualizationMode
from core.classification_model import UnsupervisedClassificationRequest

# A pentagon over the 8x8 fixture: full width at the top, with both lower
# corners chamfered. Its bounding box is the whole image, so any shrinkage in
# these tests would come from the ROI, not from the box.
PENTAGON = [(0.0, 0.0), (8.0, 0.0), (8.0, 5.0), (4.0, 8.0), (0.0, 5.0)]


def test_polygon_crop_keeps_the_bounding_box_and_records_the_roi(loaded_window):
    loaded_window.viewer.polygonCropRequested.emit(PENTAGON)

    data = loaded_window._hsi_data
    # The cube stays rectangular; only the ROI narrows.
    assert data.rgb_array.shape[:2] == (8, 8)
    assert data.roi_mask is not None
    assert data.roi_mask.shape == (8, 8)
    assert data.roi_mask[0, 0]        # inside the flat top edge
    assert not data.roi_mask[7, 0]    # chamfered lower-left corner
    assert not data.roi_mask[7, 7]    # chamfered lower-right corner
    assert 0 < data.roi_mask.sum() < 64
    assert "polygon" in loaded_window.statusbar.currentMessage()


def test_polygon_crop_shrinks_to_the_bounding_box_when_smaller(loaded_window):
    triangle = [(2.0, 2.0), (6.0, 2.0), (2.0, 6.0)]

    loaded_window.viewer.polygonCropRequested.emit(triangle)

    data = loaded_window._hsi_data
    assert data.rgb_array.shape[:2] == (4, 4)
    assert data.roi_mask.shape == (4, 4)
    assert data.roi_mask[0, 0]
    assert not data.roi_mask[3, 3]


def test_excluded_pixels_are_not_data_in_index_values(loaded_window):
    loaded_window.viewer.polygonCropRequested.emit(PENTAGON)

    values = loaded_window._visualization_results[VisualizationMode.NDVI].values
    mask = loaded_window._hsi_data.roi_mask

    assert np.isnan(values[~mask]).all()
    assert np.isfinite(values[mask]).all()


def test_index_mean_ignores_excluded_pixels(loaded_window):
    """The polygon must change the reported mean, not just the picture."""
    before = float(
        np.nanmean(loaded_window._visualization_results[VisualizationMode.NDVI].values)
    )

    loaded_window.viewer.polygonCropRequested.emit(PENTAGON)
    after = float(
        np.nanmean(loaded_window._visualization_results[VisualizationMode.NDVI].values)
    )

    assert not np.isnan(after)
    assert after != pytest.approx(before)


def test_rgb_stretch_is_computed_over_the_region_only(loaded_window):
    """`get_rgb`'s whole-rectangle percentile would ignore the ROI entirely."""
    before = loaded_window._visualization_results[VisualizationMode.RGB].display_rgb.copy()

    loaded_window.viewer.polygonCropRequested.emit(PENTAGON)
    after = loaded_window._visualization_results[VisualizationMode.RGB].display_rgb

    assert after.shape == before.shape
    mask = loaded_window._hsi_data.roi_mask
    # Same pixels, different stretch limits, so the retained region rescales.
    assert not np.array_equal(after[mask], before[mask])


def test_polygon_crop_is_undoable(loaded_window):
    original = loaded_window._hsi_data.rgb_array.copy()

    loaded_window.viewer.polygonCropRequested.emit(PENTAGON)
    assert loaded_window._hsi_data.roi_mask is not None

    loaded_window._undo_crop()

    assert loaded_window._hsi_data.roi_mask is None
    assert np.array_equal(loaded_window._hsi_data.rgb_array, original)


def test_polygon_crop_redo_restores_the_roi(loaded_window):
    loaded_window.viewer.polygonCropRequested.emit(PENTAGON)
    cropped_mask = loaded_window._hsi_data.roi_mask.copy()
    loaded_window._undo_crop()

    loaded_window._redo_crop()

    assert np.array_equal(loaded_window._hsi_data.roi_mask, cropped_mask)


def test_successive_polygon_crops_intersect(loaded_window):
    loaded_window.viewer.polygonCropRequested.emit(PENTAGON)
    first = loaded_window._hsi_data.roi_mask.copy()

    # Chamfer the top-left corner too; the surviving region is the overlap.
    loaded_window.viewer.polygonCropRequested.emit(
        [(3.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0), (0.0, 3.0)]
    )
    second = loaded_window._hsi_data.roi_mask

    assert second.sum() < first.sum()
    assert not second[0, 0]


def test_degenerate_polygon_is_a_noop(loaded_window):
    loaded_window.viewer.polygonCropRequested.emit([(0.0, 0.0), (4.0, 4.0)])

    assert loaded_window._hsi_data.roi_mask is None
    assert loaded_window._hsi_data.rgb_array.shape[:2] == (8, 8)
    assert loaded_window._crop_undo_stack == []


def test_polygon_crop_without_a_loaded_image_is_a_noop(window):
    window.viewer.polygonCropRequested.emit(PENTAGON)

    assert not window._hsi_data.is_loaded()
    assert window._crop_undo_stack == []


def test_kmeans_excludes_the_region_from_clustering(loaded_window):
    """Excluded pixels must not form their own cluster or shift the centres."""
    loaded_window.viewer.polygonCropRequested.emit(PENTAGON)
    data = loaded_window._hsi_data

    result = ClassificationService().classify_unsupervised(
        data, UnsupervisedClassificationRequest(n_classes=3, max_iterations=10)
    )

    mask = data.roi_mask
    assert result.class_map.shape == (8, 8)
    # -1 marks "outside the region"; no one-hot layer may claim those pixels.
    assert (result.class_map[~mask] == -1).all()
    assert set(np.unique(result.class_map[mask])) <= {0, 1, 2}
    assert result.one_hot_masks[:, ~mask].sum() == 0
    assert result.class_pixel_counts.sum() == int(mask.sum())


def test_viewer_polygon_tool_collects_vertices(loaded_window, qtbot):
    """The context-menu tool must emit the vertices it collected."""
    viewer = loaded_window.viewer
    viewer._begin_polygon_crop_mode()
    for x, y in [(1.0, 1.0), (6.0, 1.0), (6.0, 6.0)]:
        viewer._add_polygon_vertex(QPointF(x, y))

    with qtbot.waitSignal(viewer.polygonCropRequested) as blocker:
        viewer._finish_polygon_crop()

    assert blocker.args[0] == [(1.0, 1.0), (6.0, 1.0), (6.0, 6.0)]
    assert not viewer._polygon_cropping
    assert viewer._polygon_points == []


def test_viewer_polygon_tool_needs_three_vertices(loaded_window):
    viewer = loaded_window.viewer
    emitted: list = []
    viewer.polygonCropRequested.connect(emitted.append)
    viewer._begin_polygon_crop_mode()
    viewer._add_polygon_vertex(QPointF(1.0, 1.0))

    viewer._finish_polygon_crop()

    assert emitted == []
    assert not viewer._polygon_cropping


def test_roi_mask_is_carried_onto_the_super_resolution_frame(loaded_window):
    """SR rebuilds the cube at 2x in a fresh HSIData, which would otherwise
    reinstate the pixels the polygon excluded."""
    loaded_window.viewer.polygonCropRequested.emit(PENTAGON)
    source = loaded_window._hsi_data.roi_mask

    scaled = loaded_window._scaled_roi_mask(source, (16, 16))

    assert scaled.shape == (16, 16)
    # Every source pixel maps to the 2x2 block that replaced it.
    assert np.array_equal(scaled, np.repeat(np.repeat(source, 2, 0), 2, 1))
    assert scaled.sum() == source.sum() * 4


def test_scaled_roi_mask_passes_through_an_absent_region(loaded_window):
    assert loaded_window._scaled_roi_mask(None, (16, 16)) is None


def test_viewer_renders_excluded_pixels_transparent_not_black(loaded_window):
    """Black is a legitimate reflectance reading, so it must never stand in
    for "outside the region"."""
    loaded_window.viewer.polygonCropRequested.emit(PENTAGON)
    mask = loaded_window._hsi_data.roi_mask

    image = loaded_window.viewer._photo.pixmap().toImage()

    assert image.hasAlphaChannel()
    corner = image.pixelColor(0, 7)     # chamfered lower-left, outside
    inside = image.pixelColor(0, 0)
    assert not mask[7, 0] and corner.alpha() == 0
    assert mask[0, 0] and inside.alpha() == 255


def test_uncropped_viewer_image_stays_fully_opaque(loaded_window):
    image = loaded_window.viewer._photo.pixmap().toImage()

    assert image.pixelColor(0, 0).alpha() == 255


def test_saving_a_polygon_crop_writes_transparent_pixels(
    loaded_window, tmp_path, file_dialog
):
    loaded_window.viewer.polygonCropRequested.emit(PENTAGON)
    target = tmp_path / "pentagon.png"
    file_dialog.save_return = (str(target), "")

    loaded_window.actionSaveImage.trigger()

    saved = Image.open(target)
    assert saved.mode == "RGBA"
    assert saved.size == (8, 8)
    assert saved.getpixel((0, 7))[3] == 0      # outside the polygon
    assert saved.getpixel((0, 0))[3] == 255    # inside


def test_saving_a_polygon_crop_to_jpeg_is_refused(
    loaded_window, tmp_path, file_dialog, dialogs
):
    """JPEG has no alpha; inventing a border colour would read as data."""
    loaded_window.viewer.polygonCropRequested.emit(PENTAGON)
    target = tmp_path / "pentagon.jpg"
    file_dialog.save_return = (str(target), "")

    loaded_window.actionSaveImage.trigger()

    assert not target.exists()
    assert dialogs.critical
    assert "alpha" in dialogs.critical[-1][2].lower()
