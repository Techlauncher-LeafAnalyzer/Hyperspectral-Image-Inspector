from __future__ import annotations

import math

import numpy as np
import pytest

from core import (
    Anchor,
    AnchorMode,
    ClassificationError,
    ClassificationLayerModel,
    VectorMask,
    VectorMaskModel,
    VectorPath,
    VectorTraceOptions,
    apply_vector_masks,
    trace_mask,
)
from core.classification_model import UnsupervisedClassificationResult

EXACT = VectorTraceOptions(tolerance=0.01, corner_angle=0.0, min_area=0.0)


def _square(x: float, y: float, size: float) -> VectorPath:
    return VectorPath((
        Anchor(x, y), Anchor(x + size, y), Anchor(x + size, y + size), Anchor(x, y + size)
    ))


def _result(class_map: np.ndarray) -> UnsupervisedClassificationResult:
    n_classes = int(class_map.max()) + 1
    masks = np.stack([(class_map == c).astype(np.uint8) for c in range(n_classes)])
    return UnsupervisedClassificationResult(
        class_map=class_map.astype(np.int32),
        one_hot_masks=masks,
        cluster_centers=np.zeros((n_classes, 2), dtype=np.float32),
        class_pixel_counts=masks.reshape(n_classes, -1).sum(axis=1),
        band_indices=(0, 1),
        band_wavelengths_nm=np.array([500.0, 600.0]),
        iterations_completed=1,
    )


def test_exact_trace_round_trips_pixels_including_holes_and_edges():
    mask = np.zeros((40, 50), dtype=bool)
    mask[5:30, 8:40] = True
    mask[12:20, 15:22] = False  # a hole
    mask[0:4, 0:4] = True  # touches the image corner
    mask[34:40, 44:50] = True
    traced = trace_mask(mask, EXACT)
    assert len(traced.paths) == 4
    np.testing.assert_array_equal(traced.rasterize(mask.shape), mask)


def test_trace_drops_specks_and_fills_small_holes():
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:35, 5:35] = True
    mask[20, 20] = False  # one-pixel hole
    mask[1, 38] = True  # one-pixel speck
    traced = trace_mask(mask, VectorTraceOptions(min_area=4.0))
    assert len(traced.paths) == 1
    raster = traced.rasterize(mask.shape)
    assert raster[20, 20] and not raster[1, 38]


def test_trace_marks_sharp_turns_as_corners_and_gentle_ones_smooth():
    square = np.zeros((30, 30), dtype=bool)
    square[5:25, 5:25] = True
    (path,) = trace_mask(square).paths
    assert len(path) == 4
    assert all(anchor.mode is AnchorMode.CORNER for anchor in path.anchors)
    yy, xx = np.mgrid[:80, :80]
    disc = (xx - 40) ** 2 + (yy - 40) ** 2 < 30**2
    (circle,) = trace_mask(disc).paths
    assert all(anchor.mode is AnchorMode.SMOOTH for anchor in circle.anchors)
    # Smooth handles make the outline hug the disc closely.
    raster = VectorMask((circle,)).rasterize(disc.shape)
    assert np.count_nonzero(raster != disc) < 0.03 * disc.sum()


def test_empty_mask_traces_to_nothing():
    assert trace_mask(np.zeros((5, 5), dtype=bool)).paths == ()


def test_rasterize_uses_even_odd_for_nested_paths():
    mask = VectorMask((_square(0, 0, 10), _square(3, 3, 4)))
    raster = mask.rasterize((10, 10))
    assert raster[1, 1] and not raster[5, 5]
    assert raster.sum() == 100 - 16


def test_smooth_handle_drag_keeps_opposite_length_symmetric_mirrors():
    anchor = Anchor(10, 10, -2, 0, 4, 0, AnchorMode.SMOOTH)
    path = VectorPath((anchor, Anchor(20, 0), Anchor(20, 20)))
    moved = path.move_handle(0, "out", 10, 14).anchors[0]
    assert (moved.out_dx, moved.out_dy) == (0, 4)
    assert moved.in_dx == pytest.approx(0) and moved.in_dy == pytest.approx(-2)

    symmetric = VectorPath((replace_mode(anchor, AnchorMode.SYMMETRIC), *path.anchors[1:]))
    mirrored = symmetric.move_handle(0, "in", 7, 10).anchors[0]
    assert (mirrored.out_dx, mirrored.out_dy) == (3, 0)

    broken = path.move_handle(0, "out", 10, 14, break_handles=True).anchors[0]
    assert broken.mode is AnchorMode.CORNER and (broken.in_dx, broken.in_dy) == (-2, 0)


def replace_mode(anchor: Anchor, mode: AnchorMode) -> Anchor:
    return Anchor(anchor.x, anchor.y, anchor.in_dx, anchor.in_dy, anchor.out_dx, anchor.out_dy, mode)


def test_set_mode_converts_like_illustrator():
    path = _square(0, 0, 10)
    smooth = path.set_mode(0, AnchorMode.SMOOTH).anchors[0]
    assert smooth.mode is AnchorMode.SMOOTH and smooth.has_in_handle and smooth.has_out_handle
    # Generated handles are collinear (tangent to the neighbours).
    cross = smooth.in_dx * smooth.out_dy - smooth.in_dy * smooth.out_dx
    assert cross == pytest.approx(0)

    lopsided = VectorPath((Anchor(5, 5, -1, 0, 3, 1, AnchorMode.CORNER), *path.anchors[1:]))
    symmetric = lopsided.set_mode(0, AnchorMode.SYMMETRIC).anchors[0]
    assert math.hypot(symmetric.in_dx, symmetric.in_dy) == pytest.approx(
        math.hypot(symmetric.out_dx, symmetric.out_dy)
    )

    corner = lopsided.set_mode(0, AnchorMode.CORNER).anchors[0]
    assert not corner.has_in_handle and not corner.has_out_handle
    assert path.toggle_mode(1).anchors[1].mode is AnchorMode.SMOOTH


def test_insert_anchor_keeps_the_shape():
    yy, xx = np.mgrid[:60, :60]
    (path,) = trace_mask((xx - 30) ** 2 + (yy - 30) ** 2 < 20**2).paths
    split, index = path.insert_anchor(2, 0.4)
    assert len(split) == len(path) + 1 and index == 3
    before = VectorMask((path,)).rasterize((60, 60))
    np.testing.assert_array_equal(VectorMask((split,)).rasterize((60, 60)), before)
    straight, index = _square(0, 0, 10).insert_anchor(0, 0.5)
    assert straight.anchors[index].point == (5.0, 0.0)
    assert straight.anchors[index].mode is AnchorMode.CORNER


def test_remove_anchor_refuses_to_leave_fewer_than_three():
    path = _square(0, 0, 10)
    assert len(path.remove_anchor(0)) == 3
    assert path.remove_anchor(0).remove_anchor(0) is None
    with pytest.raises(ClassificationError):
        VectorPath((Anchor(0, 0), Anchor(1, 1)))


def test_closest_point_finds_the_segment():
    segment, t, distance = _square(0, 0, 10).closest_point(10.5, 5)
    assert segment == 1 and t == pytest.approx(0.5) and distance == pytest.approx(0.5)


def test_apply_vector_masks_claims_releases_and_restores():
    class_map = np.zeros((10, 10), dtype=np.int32)
    class_map[:, 5:] = 1
    result = _result(class_map)
    # Class 0 becomes the top-left 7x7 square, taking 14 pixels from class 1.
    grown = apply_vector_masks(result, {0: VectorMask((_square(0, 0, 7),))})
    np.testing.assert_array_equal(grown.class_map[:7, :7], 0)
    assert (grown.class_map[7:, :5] == -1).all()  # released by class 0
    assert (grown.class_map[:, 7:] == 1).all()
    assert grown.class_pixel_counts.tolist() == [49, 36]
    assert grown.one_hot_masks.sum() == 85
    # Shrinking class 0 again hands the claimed pixels back to class 1.
    shrunk = apply_vector_masks(result, {0: VectorMask((_square(0, 0, 3),))})
    assert (shrunk.class_map[:, 5:] == 1).all()
    assert apply_vector_masks(result, {}) is result


def test_layer_model_with_result_keeps_layer_state():
    class_map = np.zeros((4, 4), dtype=np.int32)
    class_map[2:] = 1
    result = _result(class_map)
    layers = ClassificationLayerModel(result)
    layers.rename_class(1, "Leaf")
    layers.set_class_visible(0, False)
    layers.set_class_opacity(1, 0.4)
    layers.set_global_opacity(0.5)
    edited = apply_vector_masks(result, {1: VectorMask((_square(0, 0, 2),))})
    carried = layers.with_result(edited)
    assert carried.result is edited
    assert [(l.name, l.visible, l.opacity) for l in carried.layers] == [
        ("Class 0", False, 1.0), ("Leaf", True, 0.4)
    ]
    assert carried.global_opacity == 0.5
    assert carried.layers[1].pixel_count == 4


def test_vector_mask_model_undo_coalesces_drags():
    model = VectorMaskModel()
    first = VectorMask((_square(0, 0, 4),))
    model.set_masks({1: first})
    for size in (5, 6, 7):
        model.set_mask(1, VectorMask((_square(0, 0, size),)), coalesce_key=("drag", 1))
    model.break_coalescing()
    model.set_mask(1, VectorMask((_square(0, 0, 9),)), coalesce_key=("drag", 1))
    assert model.undo() and model.get(1) == VectorMask((_square(0, 0, 7),))
    assert model.undo() and model.get(1) == first
    assert model.undo() and not model
    assert not model.undo()
    assert model.redo() and model.get(1) == first
    model.remove(1)
    assert 1 not in model and model.can_undo and not model.can_redo


def test_apply_vector_masks_adds_drawn_classes():
    class_map = np.zeros((10, 10), dtype=np.int32)
    class_map[:, 5:] = 1
    result = _result(class_map)
    # Class 3 is new; K-means IDs stay contiguous, so empty class 2 is added too.
    drawn = apply_vector_masks(result, {3: VectorMask((_square(0, 0, 4),))})
    assert drawn.n_classes == 4 and drawn.class_pixel_counts.tolist() == [34, 50, 0, 16]
    assert (drawn.class_map[:4, :4] == 3).all()
    assert drawn.cluster_centers.shape == (4, 2) and np.isnan(drawn.cluster_centers[3]).all()
    assert ClassificationLayerModel(drawn).class_ids == (0, 1, 2, 3)
    with pytest.raises(ClassificationError):
        apply_vector_masks(result, {-1: VectorMask()})
