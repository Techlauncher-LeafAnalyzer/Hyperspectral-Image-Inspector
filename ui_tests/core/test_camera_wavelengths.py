from __future__ import annotations

import numpy as np
import pytest
from spectral import get_rgb

from core import (
    HSIHeaderError, HSIReader, SuperResolutionError, SuperResolutionRequest,
    SuperResolutionService, VisualizationMode, VisualizationRequest,
    VisualizationService, WavelengthError,
)


@pytest.mark.parametrize("wavelengths", [[550, 670, 800], [1000], [1000, 1200]])
def test_missing_rgb_uses_first_middle_last_without_changing_spectra(camera_cube_factory, wavelengths):
    data, original = camera_cube_factory(wavelengths)
    result = VisualizationService().render(data, VisualizationRequest("RGB"))
    selected = (0, data.bands // 2, data.bands - 1)
    assert tuple(result.band_indices.values()) == selected
    assert result.title.startswith("False-colour RGB")
    expected = np.round(get_rgb(data.image, selected, stretch=(.02, .98), stretch_all=True) * 255).astype(np.uint8)
    np.testing.assert_array_equal(result.display_rgb, expected)
    np.testing.assert_array_equal(data.read_bands(range(data.bands)), original)


def test_swir_camera_uses_fixed_false_colour_bands(camera_cube_factory):
    data, original = camera_cube_factory(np.linspace(935.613, 1720.233, 224))
    service = VisualizationService()
    result = service.render(data, VisualizationRequest("RGB"))
    assert result.title.startswith("False-colour RGB")
    for name, target in service.SWIR_FALSE_COLOUR_TARGETS.items():
        assert abs(result.band_wavelengths_nm[name] - target) <= service.RGB_TOLERANCE_NM
    np.testing.assert_array_equal(data.read_bands(range(data.bands)), original)


def test_visible_camera_retains_true_rgb_and_all_indices(camera_cube_factory):
    data, _ = camera_cube_factory(np.linspace(398.25, 1001.17, 224))
    service = VisualizationService()
    result = service.render(data, VisualizationRequest("RGB"))
    assert result.title.startswith("RGB (")
    for name, target in service.RGB_TARGETS.items():
        assert abs(result.band_wavelengths_nm[name] - target) <= service.RGB_TOLERANCE_NM
    water = {VisualizationMode.NDWI, VisualizationMode.NDMI}
    for mode in service.supported_modes:
        reason = service.unavailable_reason(data, mode)
        # The visible camera stops at 1001 nm, short of the 1070 nm water-index reference.
        assert (reason is not None) == (mode in water)


def test_index_availability_checks_nearest_bands_not_just_endpoints(camera_cube_factory, monkeypatch):
    # The range encloses every target, but the large internal gap contains
    # neither PRI target nor EVI's blue band within the allowed tolerance.
    data, _ = camera_cube_factory([400, 550, 670, 700, 800, 1000])
    monkeypatch.setattr(data.image, "read_bands", lambda *args: pytest.fail("Capability checks must not read pixels"))
    service = VisualizationService()
    assert service.unavailable_reason(data, "NDVI") is None
    assert service.unavailable_reason(data, "MCARI") is None
    assert "470" in service.unavailable_reason(data, "EVI")
    assert "531" in service.unavailable_reason(data, "PRI")
    with pytest.raises(WavelengthError, match="470"):
        service.render(data, VisualizationRequest("EVI"))


def test_swir_camera_keeps_rgb_band_and_hypercube_but_no_indices(camera_cube_factory):
    data, _ = camera_cube_factory(np.linspace(935.613, 1720.233, 224))
    service = VisualizationService()
    assert service.unavailable_reason(data, "RGB") is None
    assert service.unavailable_reason(data, "BAND") is None
    water = {VisualizationMode.NDWI, VisualizationMode.NDMI}
    for mode in service.DEFAULT_COLORMAPS:
        assert (service.unavailable_reason(data, mode) is None) == (mode in water)
    result = service.render(data, VisualizationRequest("NDMI"))
    assert result.band_wavelengths_nm["nir"] == pytest.approx(1070, abs=service.INDEX_TOLERANCE_NM)
    assert result.band_wavelengths_nm["swir"] == pytest.approx(1650, abs=service.INDEX_TOLERANCE_NM)
    cube = service.prepare_hypercube_view(data, max_spatial_side=8, max_spectral_bands=16)
    assert cube.top_rgb.shape == (6, 8, 3)


@pytest.mark.parametrize("wavelengths, error", [
    (np.linspace(398.25, 1001.17, 224), "exactly 480"),
    (np.linspace(935.613, 1720.233, 224), "exactly 480"),
    (np.linspace(935.613, 1720.233, 480), "coverage"),
    (np.linspace(400, 900, 480), "coverage"),
    (np.linspace(352.49, 850, 480), "coverage"),
])
def test_sr_rejects_wrong_camera_band_count_or_wavelength_range(camera_cube_factory, wavelengths, error):
    data, _ = camera_cube_factory(wavelengths)
    service = SuperResolutionService()
    assert error in service.compatibility_error(data)
    with pytest.raises(SuperResolutionError, match=error):
        service.validate(data, SuperResolutionRequest())


def test_sr_accepts_original_camera_range_and_minor_rounding(camera_cube_factory):
    data, _ = camera_cube_factory(np.linspace(353, 899, 480))
    service = SuperResolutionService()
    assert service.compatibility_error(data) is None
    service.validate(data, SuperResolutionRequest())


def test_micrometer_metadata_is_normalized_to_nanometers(camera_cube_factory):
    data, _ = camera_cube_factory(np.linspace(.35249, .89881, 480), units="Micrometers")
    np.testing.assert_allclose(data.wavelengths_nm[[0, -1]], [352.49, 898.81])
    assert SuperResolutionService().compatibility_error(data) is None
    assert VisualizationService().unavailable_reason(data, "NDVI") is None


@pytest.mark.parametrize("wavelengths", [[400, float("nan"), 800], [400, 600, float("inf")],
                                        [0, 600, 800], [400, 400, 800]])
def test_invalid_wavelength_metadata_is_rejected(wavelengths):
    with pytest.raises(HSIHeaderError):
        HSIReader._read_wavelengths({"wavelength": wavelengths}, 3)
