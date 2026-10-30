"""Custom index formulas and the Brightness (normalized band sum) mode."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from core import (
    CustomIndexError,
    DisplayStretch,
    VisualizationError,
    VisualizationMode,
    VisualizationRequest,
    VisualizationService,
    leaf_clusters,
    parse_custom_index,
)
from core.classification_model import (
    SupervisedClassificationResult,
    SupervisedClassifierType,
)


def evaluate(text: str, **bands: float) -> float:
    formula = parse_custom_index(text)
    values = {float(name[1:]): np.full((1, 1), value) for name, value in bands.items()}
    return float(formula.evaluate(values)[0, 0])


@pytest.mark.parametrize(
    "text, expected",
    [
        ("(R800 - R670) / (R800 + R670)", (8 - 6) / (8 + 6)),
        ("(1+0.25)/R800+8.0*R670", 1.25 / 8 + 8 * 6),
        ("2^3^2 * R800 / R800", 512),  # right-associative power
        ("-R800^2", -64),  # unary minus binds looser than ^
        ("2R800", 16),  # implicit multiplication
        ("2(R800)(R670)", 96),
        ("sqrt(R800 * 2)", 4),
        ("sq(R670) - abs(-R800)", 36 - 8),
        ("min(R800, R670) + max(R800, R670)", 14),
        ("ln(R800) + log10(R800) + exp(0 * R800)", np.log(8) + np.log10(8) + 1),
        ("R800 ** 2", 64),
        ("pi * R800 / e", np.pi * 8 / np.e),
        ("R800² − R670 × 2 ÷ √R800", 64 - 6 * 2 / np.sqrt(8)),
    ],
)
def test_formulas_follow_arithmetic_precedence(text, expected):
    assert evaluate(text, R800=8.0, R670=6.0) == pytest.approx(expected, rel=1e-6)


def test_band_terms_accept_decimals_and_lowercase():
    formula = parse_custom_index("r531.5 / R570 + R531.5")
    assert formula.band_tokens == ("R531.5", "R570")
    assert formula.requested_wavelengths_nm == (531.5, 570.0)


@pytest.mark.parametrize(
    "text, message, position",
    [
        ("", "Enter a formula", 0),
        ("R800 +", "ends unexpectedly", 6),
        ("(R800 - R670", "never closed", 0),
        ("R800 $ 2", "Unexpected character", 5),
        ("R800 )", "Unexpected ')'", 5),
        ("foo(R800)", "Unknown name 'foo'", 0),
        ("R800R670", "is not a band", 0),
        ("sqrt R800", "needs parentheses", 0),
        ("min(R800)", "takes 2 arguments", 0),
        ("3 + 4", "at least one band", None),
    ],
)
def test_invalid_formulas_explain_the_problem(text, message, position):
    with pytest.raises(CustomIndexError, match=re.escape(message)) as info:
        parse_custom_index(text)
    assert info.value.position == position
    assert isinstance(info.value, VisualizationError)


def test_requested_wavelengths_map_to_the_nearest_band():
    wavelengths = np.linspace(400, 1000, 101)  # 6 nm apart
    resolved = parse_custom_index("R801.4 / R397").resolve(wavelengths)
    assert [(band.token, band.band_index, band.wavelength_nm) for band in resolved] == [
        ("R801.4", 67, 802.0),
        ("R397", 0, 400.0),
    ]
    assert resolved[0].offset_nm == pytest.approx(0.6)


def test_wavelengths_outside_the_camera_range_are_rejected():
    with pytest.raises(CustomIndexError, match="R270 is outside this image's 400–1000 nm"):
        parse_custom_index("R800 + R270").resolve(np.linspace(400, 1000, 101))


def test_preview_latex_typesets_fractions_and_functions():
    latex = parse_custom_index("(R800 - R670) / sqrt(R800) + sq(R670)").latex()
    assert latex == (
        r"\frac{R_{800} - R_{670}}{\sqrt{R_{800}}} + {\left(R_{670}\right)}^{2}"
    )


def test_custom_mode_renders_with_nearest_bands(camera_cube_factory):
    data, cube = camera_cube_factory([500.0, 670.0, 720.0, 805.0])
    result = VisualizationService().render(
        data,
        VisualizationRequest(VisualizationMode.CUSTOM, formula="(R800 - R670) / (R800 + R670)"),
    )
    nir, red = cube[:, :, 3], cube[:, :, 1]
    np.testing.assert_allclose(result.values, (nir - red) / (nir + red), rtol=1e-5)
    assert dict(result.band_indices) == {"R800": 3, "R670": 1}
    assert dict(result.band_wavelengths_nm) == {"R800": 805.0, "R670": 670.0}
    assert result.title == "Custom index: (R800 - R670) / (R800 + R670)"
    assert result.display_rgb.shape == (*cube.shape[:2], 3)


def test_custom_mode_turns_invalid_arithmetic_into_zero(camera_cube_factory):
    data, _ = camera_cube_factory([500.0, 670.0])
    result = VisualizationService().render(
        data, VisualizationRequest("CUSTOM", formula="R670 / (R500 - R500)")
    )
    assert np.all(result.values == 0)


def test_custom_mode_requires_a_formula(camera_cube_factory):
    data, _ = camera_cube_factory([500.0, 670.0])
    with pytest.raises(VisualizationError, match="requires a formula"):
        VisualizationService().render(data, VisualizationRequest("CUSTOM"))


def test_band_sum_is_chunked_but_exact(camera_cube_factory, monkeypatch):
    data, cube = camera_cube_factory(np.linspace(400, 1000, 37))
    service = VisualizationService()
    monkeypatch.setattr(service, "BAND_SUM_CHUNK", 5)
    np.testing.assert_allclose(service.band_sum(data), cube.sum(axis=2), rtol=1e-6)


def test_brightness_scales_the_whole_image_to_unit_range(camera_cube_factory):
    data, cube = camera_cube_factory(np.linspace(400, 1000, 12))
    result = VisualizationService().render(data, VisualizationRequest("BRIGHTNESS"))
    total = cube.sum(axis=2)
    low, high = np.percentile(total, (2, 98))
    expected = np.clip((total - low) / (high - low), 0, 1)
    np.testing.assert_allclose(result.values, expected, rtol=1e-5, atol=1e-6)
    assert result.value_range == pytest.approx((0.0, 1.0))
    assert result.display_limits == (0.0, 1.0)
    assert "whole image" in result.title


def test_brightness_scales_each_segment_separately(camera_cube_factory):
    data, cube = camera_cube_factory(np.linspace(400, 1000, 12))
    segments = np.full(cube.shape[:2], -1)
    segments[:3, :4] = 0
    segments[3:, 4:] = 7
    result = VisualizationService().render(
        data, VisualizationRequest("BRIGHTNESS", segments=segments)
    )
    for label in (0, 7):
        region = result.values[segments == label]
        assert region.min() == pytest.approx(0.0)
        assert region.max() == pytest.approx(1.0)
    assert np.isnan(result.values[segments == -1]).all()
    assert "per cluster, 2 clusters" in result.title


def test_brightness_reuses_a_supplied_band_sum(camera_cube_factory, monkeypatch):
    data, cube = camera_cube_factory(np.linspace(400, 1000, 12))
    service = VisualizationService()
    total = service.band_sum(data)
    monkeypatch.setattr(data, "read_bands", lambda *_: pytest.fail("must not re-read"))
    result = service.render(data, VisualizationRequest("BRIGHTNESS", band_sum=total))
    assert np.nanmax(result.values) == pytest.approx(1.0)


def test_a_few_bright_pixels_do_not_darken_their_cluster():
    rng = np.random.default_rng(1)
    leaf = rng.uniform(10.0, 20.0, size=(20, 20))
    leaf[0, :3] = 1000.0  # Three glints: under 1% of the leaf.
    segments = np.zeros(leaf.shape, dtype=int)

    robust = VisualizationService.normalize_by_segment(leaf, segments)
    plain = VisualizationService.normalize_by_segment(
        leaf, segments, DisplayStretch(0, 100)
    )

    body = np.ones(leaf.shape, dtype=bool)
    body[0, :3] = False
    # Min-max squeezes the leaf into the bottom 1% of the scale...
    assert np.median(plain[body]) < 0.01
    # ...whereas percentile scaling keeps it spread out and clips the glints.
    assert 0.3 < np.median(robust[body]) < 0.7
    assert (robust[~body] == 1.0).all()


def test_constant_segments_map_to_zero():
    values = VisualizationService.normalize_by_segment(
        np.array([[5.0, 5.0], [1.0, 3.0]]), np.array([[0, 0], [1, 1]])
    )
    np.testing.assert_allclose(values, [[0.0, 0.0], [0.0, 1.0]])


def _supervised(class_map: np.ndarray) -> SupervisedClassificationResult:
    class_map = class_map.astype(np.int32)
    class_ids = (1, 2)
    masks = (class_map[None] == np.asarray(class_ids)[:, None, None]).astype(np.uint8)
    return SupervisedClassificationResult(
        class_map=class_map,
        one_hot_masks=masks,
        class_ids=class_ids,
        class_pixel_counts=masks.reshape(2, -1).sum(axis=1),
        training_sample_counts=np.asarray([10, 20]),
        classifier=SupervisedClassifierType.GAUSSIAN,
        training_band_indices=(0,),
        target_band_indices=(0,),
        band_wavelengths_nm=np.asarray([550.0]),
        training_mask_path=Path("mask.png"),
        training_cube_path=Path("cube.img"),
    )


def test_leaf_clusters_split_the_plant_and_drop_background():
    class_map = np.full((40, 60), 2)
    class_map[5:25, 5:25] = 1
    class_map[5:20, 35:50] = 1
    clusters = leaf_clusters(_supervised(class_map))
    assert clusters.count == 2
    assert (clusters.labels[class_map == 2] == -1).all()
    assert clusters.labels[10, 10] != clusters.labels[10, 40]


def test_leaf_clusters_keep_an_unsplittable_plant_as_one_cluster():
    class_map = np.full((10, 10), 2)
    class_map[4, 4] = 1  # Too small to count as a leaf.
    clusters = leaf_clusters(_supervised(class_map))
    assert clusters.count == 1
    assert clusters.labels[4, 4] == 1
