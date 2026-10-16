from __future__ import annotations

import numpy as np
import pytest

from core import (
    AnnotationError,
    AnnotationKind,
    AnnotationLayerModel,
    AnnotationStyle,
    VisualizationMode,
    VisualizationResult,
    measure_region,
    region_mask,
)

STYLE = AnnotationStyle()


def _layer(kind, x, y, w, h, rotation=0.0):
    model = AnnotationLayerModel()
    layer = model.add_shape(kind, x, y, w, h, STYLE)
    if rotation:
        model.set_rotation(layer.layer_id, rotation)
    return model.get(layer.layer_id)


def _index(values):
    return VisualizationResult(
        mode=VisualizationMode.NDVI,
        display_rgb=np.zeros((*values.shape, 3), dtype=np.uint8),
        values=values.astype(np.float32),
        title="NDVI",
        value_range=(0.0, 1.0),
        band_indices={},
        band_wavelengths_nm={},
        colormap="gray",
        display_limits=(0.0, 1.0),
    )


def test_rectangle_mask_covers_whole_pixels():
    mask = region_mask(_layer(AnnotationKind.RECTANGLE, 2, 3, 4, 2), (10, 10))
    assert mask.sum() == 8
    assert mask[3:5, 2:6].all()


def test_ellipse_mask_is_inscribed_in_its_box():
    box = region_mask(_layer(AnnotationKind.RECTANGLE, 0, 0, 10, 10), (10, 10))
    circle = region_mask(_layer(AnnotationKind.ELLIPSE, 0, 0, 10, 10), (10, 10))
    assert 0 < circle.sum() < box.sum()
    assert not circle[0, 0] and circle[5, 5]


def test_rotated_rectangle_keeps_its_area_roughly():
    flat = region_mask(_layer(AnnotationKind.RECTANGLE, 20, 20, 30, 10), (80, 80))
    turned = region_mask(_layer(AnnotationKind.RECTANGLE, 20, 20, 30, 10, 90), (80, 80))
    assert abs(int(turned.sum()) - int(flat.sum())) <= 20
    rows, cols = np.nonzero(turned)
    assert np.ptp(rows) > np.ptp(cols)  # a 90 degree turn makes it tall


def test_shape_partly_or_wholly_outside_is_clipped():
    assert region_mask(_layer(AnnotationKind.RECTANGLE, -5, -5, 10, 10), (20, 20)).sum() == 25
    assert region_mask(_layer(AnnotationKind.RECTANGLE, 50, 50, 5, 5), (20, 20)).sum() == 0


def test_text_and_arrow_cannot_be_measured():
    model = AnnotationLayerModel()
    arrow = model.add_arrow(0, 0, 5, 5, STYLE)
    with pytest.raises(AnnotationError):
        region_mask(arrow, (10, 10))
    with pytest.raises(AnnotationError):
        model.set_measure_index(arrow.layer_id, True)


def test_measure_region_means_only_enclosed_pixels():
    values = np.zeros((10, 10))
    values[2:4, 2:4] = 0.8
    measurement = measure_region(
        _layer(AnnotationKind.RECTANGLE, 2, 2, 2, 2), {VisualizationMode.NDVI: _index(values)}
    )
    assert measurement.pixel_count == 4
    assert measurement.means["NDVI"].value == pytest.approx(0.8)


def test_measure_region_follows_the_shape_when_it_moves():
    values = np.zeros((10, 10))
    values[:, 5:] = 1.0
    model = AnnotationLayerModel()
    layer = model.add_shape(AnnotationKind.RECTANGLE, 0, 0, 3, 3, STYLE)
    results = {VisualizationMode.NDVI: _index(values)}
    assert measure_region(layer, results).means["NDVI"].value == 0.0
    model.set_geometry(layer.layer_id, 6, 0, 3, 3)
    assert measure_region(model.get(layer.layer_id), results).means["NDVI"].value == 1.0


def test_measure_region_ignores_nan_and_omits_all_nan_region():
    values = np.full((6, 6), np.nan)
    values[0, 0] = 0.5
    results = {VisualizationMode.NDVI: _index(values)}
    assert measure_region(
        _layer(AnnotationKind.RECTANGLE, 0, 0, 3, 3), results
    ).means["NDVI"].value == pytest.approx(0.5)
    assert measure_region(_layer(AnnotationKind.RECTANGLE, 3, 3, 3, 3), results).means == {}


def test_measure_flag_is_undoable_and_defaults_off():
    model = AnnotationLayerModel()
    layer = model.add_shape(AnnotationKind.ELLIPSE, 0, 0, 5, 5, STYLE)
    assert not layer.measure_index
    model.set_measure_index(layer.layer_id, True)
    assert model.get(layer.layer_id).measure_index
    model.undo()
    assert not model.get(layer.layer_id).measure_index


def test_csv_export_appends_region_rows_after_classes(tmp_path):
    import csv

    from core import ClassificationLayer, ExportRegion, export_class_summary_csv

    layer = ClassificationLayer(class_id=0, name="Class 0", pixel_count=10, visible=True, opacity=1.0)
    path = export_class_summary_csv(
        tmp_path / "a.csv",
        [layer],
        {0: {"NDVI": 0.5}},
        100,
        None,
        [ExportRegion("Patch", 4, {"NDVI": 0.9, "RGB": (1.0, 2.0, 3.0)})],
    )
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["class_name"] for r in rows] == ["Class 0", "Patch"]
    assert rows[1]["class_id"] == "" and rows[1]["image_fraction"] == "0.04"
    assert rows[1]["NDVI_mean"] == "0.9" and rows[1]["RGB_B_mean"] == "3.0"
    assert rows[0]["RGB_R_mean"] == ""  # classes without that mean stay empty
