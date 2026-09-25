from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from spectral.io import envi

from core import (
    CalibrationError,
    CalibrationFrameResolver,
    CalibrationRequest,
    CalibrationService,
    CancelledError,
    HSIReader,
    order_calibration_frame_paths,
)


def _touch_cube_pair(directory: Path, stem: str) -> Path:
    header = directory / f"{stem}.hdr"
    header.write_text("ENVI\n", encoding="utf-8")
    (directory / f"{stem}.bil").write_bytes(b"cube")
    return header


def _cube(
    directory: Path,
    name: str,
    values: np.ndarray,
    wavelengths: tuple[float, ...] = (470.0, 550.0, 660.0),
):
    path = directory / f"{name}.hdr"
    envi.save_image(
        str(path),
        np.asarray(values, dtype=np.float32),
        ext=".bip",
        interleave="bip",
        metadata={"wavelength": wavelengths, "band names": ["B", "G", "R"]},
    )
    return HSIReader().open(path)


def test_calibrate_averages_reference_lines_and_streams_reflectance(tmp_path):
    dark_mean = np.array([[2, 3, 4], [3, 4, 5]], dtype=np.float32)
    white_mean = np.array([[11, 13, 15], [13, 15, 17]], dtype=np.float32)
    dark_values = np.broadcast_to(dark_mean, (23, 2, 3)).copy()
    white_values = np.broadcast_to(white_mean, (25, 2, 3)).copy()
    # A 23-row frame uses 1:22 and a 25-row frame uses 2:23. Extreme edge
    # values prove calibration ignores every row outside the central 21.
    dark_values[[0, 22]] = 500
    white_values[[0, 1, 23, 24]] = 900
    reflectance = np.array(
        [
            [[0.0, 0.25, 0.5], [0.2, 0.4, 0.6]],
            [[0.5, 0.75, 1.0], [0.1, 0.3, 0.5]],
            [[1.1, -0.1, 0.45], [0.9, 0.7, 0.2]],
        ],
        dtype=np.float32,
    )
    raw_values = dark_mean[None, :, :] + reflectance * (
        white_mean - dark_mean
    )[None, :, :]
    source = _cube(tmp_path, "source", raw_values)
    dark = _cube(tmp_path, "dark", dark_values)
    white = _cube(tmp_path, "white", white_values)
    progress = []

    result = CalibrationService().calibrate(
        source,
        dark,
        white,
        CalibrationRequest(line_chunk_size=2),
        progress=lambda value, message: progress.append((value, message)),
    )

    assert result.data.shape == source.shape
    assert result.input_shape == source.shape
    assert result.reference_line_counts == (23, 25)
    assert result.reference_row_ranges == ((1, 22), (2, 23))
    assert result.original_input_shape == source.shape
    assert result.source_spatial_bounds == ((0, 3), (0, 2))
    assert result.dark_reference_path == dark.source_path
    assert result.white_reference_path == white.source_path
    np.testing.assert_allclose(
        result.data.read_bands(range(result.data.bands)), reflectance, atol=1e-6
    )
    np.testing.assert_allclose(result.data.wavelengths_nm, source.wavelengths_nm)
    assert result.data.metadata["band names"] == ["B", "G", "R"]
    assert result.reflectance_range == pytest.approx((-0.1, 1.1))
    assert progress[0][0] == 0 and progress[-1][0] == 100
    storage_path = Path(result._storage.name)
    result.cleanup()
    assert not storage_path.exists()


def test_calibrate_preserves_polygon_roi_for_preview(tmp_path):
    source = _cube(tmp_path, "source", np.full((3, 2, 3), 5))
    source.roi_mask = np.array([[False, True], [True, True], [False, True]])
    dark = _cube(tmp_path, "dark", np.ones((21, 2, 3)))
    white = _cube(tmp_path, "white", np.full((21, 2, 3), 10))

    result = CalibrationService().calibrate(source, dark, white)
    try:
        np.testing.assert_array_equal(result.data.roi_mask, source.roi_mask)
        assert result.data.roi_mask is not source.roi_mask
    finally:
        result.cleanup()


def test_calibrate_super_resolved_cube_interpolates_cropped_reference_profiles(tmp_path):
    low = _cube(tmp_path, "low", np.full((3, 5, 3), 10, dtype=np.float32))
    low.rgb_array = np.zeros((3, 5, 3), dtype=np.uint8)
    low.mask_array = np.zeros((3, 5), dtype=np.uint8)
    assert low.crop(1, 0, 4, 3) == (3, 3)

    dark_row = np.arange(5, dtype=np.float32)[:, None] * np.ones((1, 3))
    bright_row = dark_row + 10
    cropped_dark = dark_row[1:4]
    interpolated_dark = CalibrationService._interpolate_reference(cropped_dark, 6)
    raw = np.broadcast_to(interpolated_dark[None] + 5, (6, 6, 3)).copy()
    high = _cube(tmp_path, "high", raw)
    high.roi_mask = np.ones((6, 6), dtype=bool)
    high.roi_mask[0, 0] = False
    dark = _cube(tmp_path, "dark", np.broadcast_to(dark_row, (21, 5, 3)))
    bright = _cube(tmp_path, "bright", np.broadcast_to(bright_row, (21, 5, 3)))

    result = CalibrationService().calibrate(
        high, dark, bright, reference_source=low
    )
    try:
        np.testing.assert_allclose(result.data.read_bands(range(3)), 0.5, atol=1e-6)
        np.testing.assert_array_equal(result.data.roi_mask, high.roi_mask)
        assert result.input_shape == high.shape
    finally:
        result.cleanup()


@pytest.mark.parametrize("reference_name", ["dark", "white"])
def test_calibrate_rejects_reference_column_or_band_mismatch(
    tmp_path, reference_name
):
    source = _cube(tmp_path, "source", np.full((2, 3, 3), 5))
    dark_shape = (21, 4, 3) if reference_name == "dark" else (21, 3, 3)
    white_shape = (21, 4, 3) if reference_name == "white" else (21, 3, 3)
    dark = _cube(tmp_path, "dark", np.ones(dark_shape))
    white = _cube(tmp_path, "white", np.full(white_shape, 10))

    with pytest.raises(CalibrationError, match="columns and 3 bands"):
        CalibrationService().calibrate(source, dark, white)


def test_calibrate_rejects_mismatched_wavelengths(tmp_path):
    source = _cube(tmp_path, "source", np.full((2, 2, 3), 5))
    dark = _cube(tmp_path, "dark", np.ones((21, 2, 3)))
    white = _cube(
        tmp_path,
        "white",
        np.full((21, 2, 3), 10),
        wavelengths=(470.0, 551.0, 660.0),
    )

    with pytest.raises(CalibrationError, match="differs from the source"):
        CalibrationService().calibrate(source, dark, white)


def test_calibrate_rejects_nonpositive_white_response(tmp_path):
    source = _cube(tmp_path, "source", np.full((2, 2, 3), 5))
    dark = _cube(tmp_path, "dark", np.full((21, 2, 3), 2))
    white_values = np.full((21, 2, 3), 10, dtype=np.float32)
    white_values[:, 1, 2] = 2
    white = _cube(tmp_path, "white", white_values)

    with pytest.raises(CalibrationError) as error:
        CalibrationService().calibrate(source, dark, white)
    assert "1 column/band position(s) (first: column 1, band 2)" in str(error.value)
    assert str(error.value).endswith("\nTry cropping before calibration.")


def test_calibrate_honours_cancellation_before_reading(tmp_path):
    source = _cube(tmp_path, "source", np.full((2, 2, 3), 5))
    dark = _cube(tmp_path, "dark", np.ones((21, 2, 3)))
    white = _cube(tmp_path, "white", np.full((21, 2, 3), 10))

    with pytest.raises(CancelledError, match="cancelled"):
        CalibrationService().calibrate(
            source, dark, white, is_cancelled=lambda: True
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"line_chunk_size": 1.5}, "integer"),
        ({"line_chunk_size": 0}, "positive"),
        ({"wavelength_tolerance_nm": np.inf}, "finite"),
        ({"minimum_response": 0}, "positive"),
    ],
)
def test_calibration_request_rejects_invalid_settings(kwargs, message):
    with pytest.raises(CalibrationError, match=message):
        CalibrationRequest(**kwargs)


def test_calibrate_does_not_modify_source(tmp_path):
    raw = np.arange(18, dtype=np.float32).reshape(2, 3, 3) + 4
    source = _cube(tmp_path, "source", raw)
    dark = _cube(tmp_path, "dark", np.ones((21, 3, 3)))
    white = _cube(tmp_path, "white", np.full((21, 3, 3), 20))

    CalibrationService().calibrate(source, dark, white)

    np.testing.assert_array_equal(source.read_bands(range(source.bands)), raw)


def test_calibrate_expands_to_original_rows_then_applies_nested_source_crop(
    tmp_path,
):
    original_shape = (6, 5, 3)
    dark_row = np.arange(15, dtype=np.float32).reshape(5, 3) + 2
    bright_row = dark_row + 20
    reflectance = (
        np.arange(np.prod(original_shape), dtype=np.float32).reshape(original_shape)
        / 100
    )
    raw = dark_row[None, :, :] + reflectance * 20
    source = _cube(tmp_path, "source", raw)
    source.rgb_array = np.zeros((6, 5, 3), dtype=np.uint8)
    source.mask_array = np.zeros((6, 5), dtype=np.uint8)
    assert source.crop(1, 1, 5, 6) == (4, 5)
    assert source.crop(1, 1, 4, 4) == (3, 3)

    dark = _cube(
        tmp_path,
        "dark",
        np.broadcast_to(dark_row, (21, 5, 3)),
    )
    bright = _cube(
        tmp_path,
        "bright",
        np.broadcast_to(bright_row, (21, 5, 3)),
    )

    result = CalibrationService().calibrate(source, dark, bright)

    assert source.original_shape == original_shape
    assert source.spatial_bounds == ((2, 5), (2, 5))
    assert result.input_shape == (3, 3, 3)
    assert result.original_input_shape == original_shape
    assert result.source_spatial_bounds == ((2, 5), (2, 5))
    np.testing.assert_allclose(
        result.data.read_bands(range(3)),
        reflectance[2:5, 2:5, :],
        atol=1e-6,
    )


def test_calibrate_rejects_reference_with_fewer_than_21_rows(tmp_path):
    source = _cube(tmp_path, "source", np.full((2, 2, 3), 5))
    dark = _cube(tmp_path, "dark", np.ones((20, 2, 3)))
    bright = _cube(tmp_path, "bright", np.full((21, 2, 3), 10))

    with pytest.raises(CalibrationError, match="has 20 rows; at least 21"):
        CalibrationService().calibrate(source, dark, bright)


def test_timestamped_calibration_frames_are_ordered_earlier_dark_later_bright():
    earlier = Path("2025-04-17--12-49-24_round-0_cam-1_calibFrame.hdr")
    later = Path("2025-04-17--12-49-29_round-0_cam-1_calibFrame.bil")

    assert order_calibration_frame_paths(later, earlier) == (earlier, later)


def test_legacy_calibration_names_keep_explicit_selection_order():
    first = Path("manually_selected_dark.hdr")
    second = Path("manually_selected_bright.hdr")

    assert order_calibration_frame_paths(first, second) == (first, second)


def test_resolver_selects_closest_earlier_pair_within_one_minute(tmp_path):
    source = _touch_cube_pair(
        tmp_path, "2025-04-17--12-51-23_round-0_cam-1_tray-1"
    )
    _touch_cube_pair(
        tmp_path, "2025-04-17--12-49-24_round-0_cam-1_calibFrame"
    )
    _touch_cube_pair(
        tmp_path, "2025-04-17--12-49-29_round-0_cam-1_calibFrame"
    )
    nearest_dark = _touch_cube_pair(
        tmp_path, "2025-04-17--12-50-40_round-0_cam-1_calibFrame"
    )
    nearest_bright = _touch_cube_pair(
        tmp_path, "2025-04-17--12-50-55_round-0_cam-1_calibFrame"
    )
    # A calibration frame after the source is ineligible.
    _touch_cube_pair(
        tmp_path, "2025-04-17--12-51-24_round-0_cam-1_calibFrame"
    )

    pair = CalibrationFrameResolver().resolve(source)

    assert pair is not None
    assert pair.dark_path == nearest_dark.resolve()
    assert pair.bright_path == nearest_bright.resolve()
    assert pair.frame_interval_seconds == 15
    assert pair.seconds_before_source == 28


def test_resolver_ignores_incomplete_and_over_one_minute_pairs(tmp_path):
    source = _touch_cube_pair(
        tmp_path, "2025-04-17--13-00-00_round-0_cam-1_tray-1"
    )
    _touch_cube_pair(
        tmp_path, "2025-04-17--12-57-00_round-0_cam-1_calibFrame"
    )
    _touch_cube_pair(
        tmp_path, "2025-04-17--12-58-01_round-0_cam-1_calibFrame"
    )
    # A header without its binary partner must not be auto-selected.
    (tmp_path / "2025-04-17--12-59-50_round-0_cam-1_calibFrame.hdr").write_text(
        "ENVI\n", encoding="utf-8"
    )

    assert CalibrationFrameResolver().resolve(source) is None


def test_resolver_returns_none_when_source_name_has_no_timestamp(tmp_path):
    source = _touch_cube_pair(tmp_path, "leaf_capture")
    _touch_cube_pair(
        tmp_path, "2025-04-17--12-49-24_round-0_cam-1_calibFrame"
    )
    _touch_cube_pair(
        tmp_path, "2025-04-17--12-49-29_round-0_cam-1_calibFrame"
    )

    assert CalibrationFrameResolver().resolve(source) is None
