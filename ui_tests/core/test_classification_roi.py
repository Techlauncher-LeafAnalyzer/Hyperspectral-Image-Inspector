"""Polygon crops exclude pixels from both classification algorithms."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from spectral.io import envi

from core import ClassificationError, ClassificationService, HSIReader
from core.classification_model import (
    SupervisedClassificationRequest,
    SupervisedClassifierType,
    UnsupervisedClassificationRequest,
)
from core.roi import polygon_mask


def _cube(tmp_path, name: str, values: np.ndarray):
    path = tmp_path / f"{name}.hdr"
    envi.save_image(
        str(path),
        values.astype(np.float32),
        ext=".bip",
        interleave="bip",
        metadata={"wavelength": [470.0, 550.0, 660.0]},
    )
    return HSIReader().open(path)


@pytest.mark.parametrize("classifier", tuple(SupervisedClassifierType))
def test_supervised_classifies_only_polygon_pixels(tmp_path, classifier):
    rng = np.random.default_rng(4)
    training_values = rng.normal(0, 0.03, size=(6, 6, 3)).astype(np.float32)
    training_values[:3] += 0.2
    training_values[3:] += 0.8
    training = _cube(tmp_path, "training", training_values)
    mask_path = tmp_path / "training_mask.png"
    training_labels = np.ones((6, 6), dtype=np.uint8)
    training_labels[3:] = 2
    Image.fromarray(training_labels).save(mask_path)

    roi = np.tri(6, 6, dtype=bool)
    target_values = training_values.copy()
    target_values[~roi] = np.nan
    target = _cube(tmp_path, "target", target_values)
    target.roi_mask = roi

    result = ClassificationService().classify_supervised(
        target,
        training,
        mask_path,
        SupervisedClassificationRequest(classifier, band_indices=(0, 1, 2)),
    )

    assert result.class_map.shape == roi.shape
    assert np.all(result.class_map[~roi] == -1)
    assert set(np.unique(result.class_map[roi])) <= {1, 2}
    assert np.all(result.one_hot_masks[:, ~roi] == 0)
    assert np.all(result.one_hot_masks[:, roi].sum(axis=0) == 1)
    assert result.class_pixel_counts.sum() == int(roi.sum())


def _scaled_pair(tmp_path, scale: float):
    rng = np.random.default_rng(7)
    values = rng.normal(0, 0.03, size=(8, 8, 3)).astype(np.float32)
    values[:4] += 0.2
    values[4:] += 0.8
    labels = np.ones((8, 8), dtype=np.uint8)
    labels[4:] = 2
    mask_path = tmp_path / "pair_mask.png"
    Image.fromarray(labels).save(mask_path)
    training = _cube(tmp_path, "pair_training", values)
    target = _cube(tmp_path, "pair_target", values * scale)
    return training, target, mask_path, labels


def test_supervised_rejects_mismatched_value_scales(tmp_path):
    training, target, mask_path, _ = _scaled_pair(tmp_path, 1000.0)

    with pytest.raises(ClassificationError, match="different value scales"):
        ClassificationService().classify_supervised(
            target,
            training,
            mask_path,
            SupervisedClassificationRequest(
                SupervisedClassifierType.MAHALANOBIS, band_indices=(0, 1, 2)
            ),
        )


@pytest.mark.parametrize("classifier", tuple(SupervisedClassifierType))
def test_supervised_standardization_bridges_value_scales(tmp_path, classifier):
    training, target, mask_path, labels = _scaled_pair(tmp_path, 1000.0)

    result = ClassificationService().classify_supervised(
        target,
        training,
        mask_path,
        SupervisedClassificationRequest(
            classifier, band_indices=(0, 1, 2), standardize_features=True
        ),
    )

    assert np.array_equal(result.class_map, labels)


def test_kmeans_ignores_nonfinite_pixels_outside_polygon(tmp_path):
    values = np.zeros((4, 4, 3), dtype=np.float32)
    values[:2, :, :] = 0.2
    values[2:, :, :] = 0.8
    roi = np.tri(4, 4, dtype=bool)
    values[~roi] = np.nan
    target = _cube(tmp_path, "target", values)
    target.roi_mask = roi

    result = ClassificationService().classify_unsupervised(
        target,
        UnsupervisedClassificationRequest(n_classes=2, max_iterations=5),
    )

    assert np.all(result.class_map[~roi] == -1)
    assert np.all(result.one_hot_masks[:, ~roi] == 0)
    assert result.class_pixel_counts.sum() == int(roi.sum())


def test_training_polygon_must_retain_both_classes(tmp_path):
    training = _cube(tmp_path, "training", np.ones((6, 6, 3)))
    training.roi_mask = np.zeros((6, 6), dtype=bool)
    training.roi_mask[:3] = True
    mask_path = tmp_path / "training_mask.png"
    labels = np.ones((6, 6), dtype=np.uint8)
    labels[3:] = 2
    Image.fromarray(labels).save(mask_path)

    with pytest.raises(ClassificationError, match="retain target and background"):
        ClassificationService().classify_supervised(
            training,
            training,
            mask_path,
            SupervisedClassificationRequest(
                SupervisedClassifierType.GAUSSIAN, band_indices=(0, 1, 2)
            ),
        )


@pytest.mark.parametrize("outside_value", [1e6, -1e6, np.nan])
def test_actual_polygon_crop_matches_fitting_only_selected_pixels(
    tmp_path, outside_value
):
    shape = (8, 10)
    vertices = [(2.0, 1.0), (9.0, 1.0), (7.0, 7.0), (2.0, 6.0)]
    selected = polygon_mask(vertices, shape)
    rows, columns = np.indices(shape)
    levels = ((rows + columns) % 5).astype(np.float32) / 4
    values = np.stack((levels, 1 - levels, np.full(shape, 0.5)), axis=-1)
    reference = _cube(tmp_path, "selected-only", values[selected][:, None, :])
    values[~selected] = outside_value
    target = _cube(tmp_path, "polygon", values)
    target.rgb_array = np.zeros((*shape, 3), dtype=np.uint8)
    target.mask_array = np.ones(shape, dtype=np.uint8)
    try:
        assert target.crop_polygon(vertices) == (7, 6)
        roi = target.roi_mask
        assert roi is not None
        assert roi.any() and not roi.all()
        service = ClassificationService()
        request = UnsupervisedClassificationRequest(n_classes=5)

        expected = service.classify_unsupervised(reference, request)
        actual = service.classify_unsupervised(target, request)

        np.testing.assert_array_equal(actual.class_map[roi], expected.class_map.ravel())
        np.testing.assert_array_equal(actual.cluster_centers, expected.cluster_centers)
        np.testing.assert_array_equal(actual.class_pixel_counts, expected.class_pixel_counts)
        assert actual.class_pixel_counts.sum() == int(selected.sum())
        assert np.all(actual.class_pixel_counts > 0)
        assert np.all(actual.class_map[~roi] == -1)
        assert not actual.one_hot_masks[:, ~roi].any()
        np.testing.assert_array_equal(actual.one_hot_masks[:, roi].sum(axis=0), 1)
    finally:
        target.close()
        reference.close()
