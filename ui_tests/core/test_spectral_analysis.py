from __future__ import annotations

import csv

import numpy as np
import pytest

from core import (
    SpectralAnalysisService,
    SpectrumTransform,
    VisualizationError,
    polygon_mask,
)
import core.spectral_analysis as spectral_analysis


@pytest.fixture
def cube_data(camera_cube_factory):
    # 6 rows x 8 columns x 40 bands: more than one read chunk once shrunk below.
    return camera_cube_factory(np.linspace(400, 1000, 40))


def test_class_spectra_match_numpy_over_band_chunks(cube_data, monkeypatch):
    data, cube = cube_data
    monkeypatch.setattr(spectral_analysis, "_BAND_CHUNK", 7)
    class_map = np.zeros((6, 8), dtype=np.int32)
    class_map[:, 4:] = 1
    class_map[0, 0] = -1  # outside the region: belongs to no class
    spectra = SpectralAnalysisService().class_spectra(
        data, class_map, {0: "Leaf", 1: "Soil", 2: "Empty"}
    )
    assert [s.label for s in spectra] == ["Leaf", "Soil"]  # empty class omitted
    for spectrum, mask in zip(spectra, (class_map == 0, class_map == 1)):
        assert spectrum.pixel_count == int(mask.sum())
        np.testing.assert_allclose(spectrum.mean, cube[mask].mean(axis=0), rtol=1e-5)
        np.testing.assert_allclose(spectrum.std, cube[mask].std(axis=0), rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(spectrum.minimum, cube[mask].min(axis=0))
        np.testing.assert_allclose(spectrum.maximum, cube[mask].max(axis=0))


def test_image_spectrum_respects_polygon_roi(cube_data):
    data, cube = cube_data
    data.roi_mask = polygon_mask([(0, 0), (4, 0), (4, 4), (0, 4)], (6, 8))
    spectrum = SpectralAnalysisService().image_spectrum(data)
    assert spectrum.pixel_count == int(data.roi_mask.sum())
    np.testing.assert_allclose(spectrum.mean, cube[data.roi_mask].mean(axis=0), rtol=1e-5)


def test_class_map_shape_must_match_image(cube_data):
    data, _ = cube_data
    with pytest.raises(VisualizationError, match="shape"):
        SpectralAnalysisService().class_spectra(data, np.zeros((2, 2)), {0: "A"})


def test_spectral_angle_map_is_brightness_invariant(cube_data):
    data, cube = cube_data
    service = SpectralAnalysisService()
    reference = cube[2, 3] * 5.0  # same direction, different brightness
    result = service.spectral_angle_map(data, reference, "pixel")
    assert result.angles[2, 3] == pytest.approx(0.0, abs=1e-3)
    expected = service.spectral_angle(cube[4, 5], reference)
    assert result.angles[4, 5] == pytest.approx(expected, rel=1e-4)
    assert result.display_rgb.shape == (6, 8, 3)
    assert result.display_rgb.dtype == np.uint8


def test_spectral_angle_map_rejects_wrong_band_count(cube_data):
    data, _ = cube_data
    with pytest.raises(VisualizationError, match="bands"):
        SpectralAnalysisService().spectral_angle_map(data, np.ones(3))


def test_separability_is_symmetric_with_zero_diagonal(cube_data):
    data, _ = cube_data
    class_map = np.repeat(np.arange(4)[None, :], 6, axis=0).repeat(2, axis=1)
    service = SpectralAnalysisService()
    spectra = service.class_spectra(data, class_map, {i: str(i) for i in range(4)})
    matrix = service.separability(spectra)
    assert matrix.shape == (4, 4)
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(np.diag(matrix), 0)
    assert (matrix[~np.eye(4, dtype=bool)] > 0).all()


def test_continuum_removal_flattens_to_one_outside_absorption():
    x = np.linspace(400, 1000, 61)
    continuum = 0.2 + 0.0005 * (x - 400)
    dip = 1 - 0.4 * np.exp(-((x - 700) / 30) ** 2)
    removed = SpectralAnalysisService.transform(x, continuum * dip, SpectrumTransform.CONTINUUM_REMOVED)
    assert removed.max() <= 1 + 1e-9
    assert removed[0] == pytest.approx(1) and removed[-1] == pytest.approx(1)
    assert removed[np.argmin(np.abs(x - 700))] == pytest.approx(0.6, abs=0.02)


def test_derivative_and_normalisation():
    x = np.linspace(400, 1000, 61)
    y = 0.001 * x
    service = SpectralAnalysisService
    np.testing.assert_allclose(service.transform(x, y, SpectrumTransform.FIRST_DERIVATIVE), 0.001)
    normalized = service.transform(x, y, SpectrumTransform.NORMALIZED)
    assert np.trapezoid(normalized, x) == pytest.approx(1)
    np.testing.assert_array_equal(service.transform(x, y, SpectrumTransform.NONE), y)


def test_export_csv_writes_wavelength_and_series_columns(tmp_path):
    path = SpectralAnalysisService.export_csv(
        tmp_path / "spectra.csv", np.array([500.0, 600.0]),
        {"Leaf": np.array([0.1, 0.2]), "Soil": np.array([0.3, 0.4])},
    )
    with path.open() as handle:
        rows = list(csv.reader(handle))
    assert rows == [["wavelength_nm", "Leaf", "Soil"], ["500", "0.1", "0.3"], ["600", "0.2", "0.4"]]
