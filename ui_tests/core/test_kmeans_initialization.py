"""K-means must seed real spectra, including when bands are anticorrelated."""

from __future__ import annotations

import numpy as np
import pytest
from spectral.io import envi

from core import (
    CancelledError,
    ClassificationService,
    HSIReader,
    UnsupervisedClassificationRequest,
)


def _open_cube(tmp_path, values):
    header = tmp_path / "spectra.hdr"
    envi.save_image(
        str(header),
        values.astype(np.float32),
        ext=".bip",
        interleave="bip",
        metadata={"wavelength": [500.0, 600.0, 700.0]},
    )
    return HSIReader().open(header)


@pytest.mark.parametrize("columns", [4, 1100])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("max_iterations", [1, 20])
def test_five_distinct_spectra_populate_all_five_classes(
    tmp_path, masked, max_iterations, columns
):
    # Opposing bands make SPy's default diagonal seeds invent all-low/all-high
    # spectra and collapse these five distinct groups to one class.
    levels = np.linspace(0, 1, 5, dtype=np.float32)
    spectra = np.column_stack((levels, 1 - levels, np.full(5, 0.5)))
    values = np.repeat(spectra[:, None, :], columns, axis=1)
    if masked:
        values = np.concatenate((values, np.full_like(values, np.nan)), axis=1)
    data = _open_cube(tmp_path, values)
    if masked:
        data.roi_mask = np.zeros(values.shape[:2], dtype=bool)
        data.roi_mask[:, :columns] = True
    try:
        service = ClassificationService()
        request = UnsupervisedClassificationRequest(
            n_classes=5, max_iterations=max_iterations, band_indices=(0, 1)
        )
        result = service.classify_unsupervised(data, request)
        repeated = service.classify_unsupervised(data, request)

        np.testing.assert_array_equal(result.class_pixel_counts, [columns] * 5)
        np.testing.assert_array_equal(result.class_map, repeated.class_map)
        np.testing.assert_array_equal(result.cluster_centers, repeated.cluster_centers)
        np.testing.assert_array_equal(result.one_hot_masks.sum(axis=(1, 2)), [columns] * 5)
        for class_id, center in enumerate(result.cluster_centers):
            assigned = values[result.class_map == class_id, :2]
            np.testing.assert_array_equal(assigned, np.tile(center, (columns, 1)))
        if masked:
            assert np.all(result.class_map[:, columns:] == -1)
            assert not result.one_hot_masks[:, :, columns:].any()
    finally:
        data.close()


@pytest.mark.parametrize("distinct_spectra", [1, 2])
def test_fewer_distinct_spectra_preserve_empty_masks(tmp_path, distinct_spectra):
    values = np.zeros((4, 4, 3), dtype=np.float32)
    if distinct_spectra == 2:
        values[2:] = 1
    data = _open_cube(tmp_path, values)
    try:
        result = ClassificationService().classify_unsupervised(
            data, UnsupervisedClassificationRequest(n_classes=5)
        )

        assert result.one_hot_masks.shape == (5, 4, 4)
        assert np.count_nonzero(result.class_pixel_counts) == distinct_spectra
        np.testing.assert_array_equal(result.one_hot_masks.sum(axis=0), 1)
        assert result.class_pixel_counts.sum() == 16
        assert np.isfinite(result.cluster_centers).all()
    finally:
        data.close()


def test_cancellation_during_center_initialization(tmp_path):
    data = _open_cube(tmp_path, np.zeros((4, 4, 3), dtype=np.float32))
    cancelled = False
    events = []

    def progress(value, message):
        nonlocal cancelled
        events.append(message)
        if value == 5:
            cancelled = True

    try:
        with pytest.raises(CancelledError):
            ClassificationService().classify_unsupervised(
                data,
                UnsupervisedClassificationRequest(n_classes=5),
                progress=progress,
                is_cancelled=lambda: cancelled,
            )
        assert "Running SPy K-means" not in events
    finally:
        data.close()
