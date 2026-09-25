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
