from __future__ import annotations

import json

import numpy as np
import pytest
from spectral.io import envi

from core import (
    CalibrationFrameResolver, HSIFileError, HSIHeaderError, HSIReader,
    TrainingPairResolver,
)
from core.hsi_reader import DATA_EXTENSIONS, HSI_FILE_FILTER, METADATA_EXTENSIONS


def write_json_capture(directory, *, name="capture", extension=".bil", nested=True,
                       string_wavelengths=True, offset=0):
    values = np.arange(4 * 5 * 3, dtype=np.uint16).reshape(4, 5, 3)
    header = directory / f"{name}.hdr"
    envi.save_image(str(header), values, ext=extension, interleave="bil",
                    metadata={"wavelength": [1000, 1200, 1400]})
    metadata = envi.read_envi_header(str(header))
    if string_wavelengths:
        metadata["wavelength"] = "{ 1000, 1200, 1400 }"
    if offset:
        data = header.with_suffix(extension)
        data.write_bytes(b"\0" * offset + data.read_bytes())
        metadata["header offset"] = offset
    document = {"bil_hdr": metadata, "frames": [{"frame_id": 1}],
                "info": {"frame_count": 4}} if nested else metadata
    json_path = header.with_suffix(".json")
    json_path.write_text(json.dumps(document), encoding="utf-8")
    header.unlink()
    return json_path, values


@pytest.mark.parametrize("extension", DATA_EXTENSIONS)
@pytest.mark.parametrize("nested,string_wavelengths", [(True, True), (False, False)])
def test_reader_opens_json_or_cube_and_preserves_metadata_and_pixels(
    tmp_path, extension, nested, string_wavelengths
):
    metadata, values = write_json_capture(tmp_path, extension=extension,
                                           nested=nested, string_wavelengths=string_wavelengths)
    original_metadata = metadata.read_bytes()
    for selected in (metadata, metadata.with_suffix(extension)):
        data = HSIReader().open(selected)
        try:
            assert data.header_path == metadata
            assert data.header_format == "JSON"
            assert data.source_path == selected
            assert data.shape == (4, 5, 3)
            np.testing.assert_array_equal(data.wavelengths_nm, [1000, 1200, 1400])
            np.testing.assert_array_equal(data.read_bands(range(3)), values)
        finally:
            data.close()
    assert metadata.read_bytes() == original_metadata
    assert not metadata.with_suffix(".hdr").exists()


def test_pairing_is_case_insensitive_and_prefers_hdr_for_binary_cube(tmp_path):
    metadata, _ = write_json_capture(tmp_path, extension=".raw")
    upper_json = metadata.with_name("CAPTURE.JSON")
    metadata.rename(upper_json)
    raw = metadata.with_suffix(".raw")
    upper_raw = raw.with_name("Capture.RAW")
    raw.rename(upper_raw)
    assert HSIReader.resolve_pair(upper_raw) == (upper_json, upper_raw)
    assert HSIReader.resolve_pair(upper_json) == (upper_json, upper_raw)
    header = tmp_path / "CAPTURE.HDR"
    header.write_text("ENVI\n", encoding="utf-8")
    assert HSIReader.resolve_pair(upper_raw) == (header, upper_raw)
    assert HSIReader.resolve_pair(upper_json) == (upper_json, upper_raw)


def test_json_metadata_without_pixel_file_has_actionable_error(tmp_path):
    metadata, _ = write_json_capture(tmp_path)
    metadata.with_suffix(".bil").unlink()
    with pytest.raises(HSIFileError, match="Metadata does not contain spectral pixels"):
        HSIReader().open(metadata)


@pytest.mark.parametrize("field,value,message", [
    ("data type", 4, "data type"),
    ("lines", 2, "dimensions"),
    ("interleave", "bip", "interleave"),
    ("byte order", 1, "byte order"),
    ("wavelength", [935, 1200, 1720], "wavelengths"),
])
def test_json_conflicting_with_companion_hdr_is_rejected(tmp_path, field, value, message):
    metadata, values = write_json_capture(tmp_path)
    document = json.loads(metadata.read_text())
    envi.write_envi_header(str(metadata.with_suffix(".hdr")), document["bil_hdr"])
    document["bil_hdr"][field] = value
    metadata.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(HSIHeaderError, match=f"conflicts.*{message}"):
        HSIReader().open(metadata)
    # Opening the authoritative ENVI header remains supported, regardless of
    # acquisition metadata in an adjacent JSON file.
    data = HSIReader().open(metadata.with_suffix(".hdr"))
    np.testing.assert_array_equal(data.read_bands(range(3)), values)
    data.close()


def test_json_matching_companion_hdr_remains_loadable(tmp_path):
    metadata, values = write_json_capture(tmp_path)
    document = json.loads(metadata.read_text())
    envi.write_envi_header(str(metadata.with_suffix(".hdr")), document["bil_hdr"])
    data = HSIReader().open(metadata)
    np.testing.assert_array_equal(data.read_bands(range(3)), values)
    data.close()


def test_json_with_wrong_storage_type_is_rejected_without_companion_hdr(tmp_path):
    metadata, _ = write_json_capture(tmp_path)
    document = json.loads(metadata.read_text())
    # uint16 pixels incorrectly described as uint8 previously passed the
    # minimum-size check and silently decoded only half the binary cube.
    document["bil_hdr"]["data type"] = 1
    metadata.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(HSIFileError, match="JSON metadata describes.*bytes"):
        HSIReader().open(metadata)


def test_x10_style_float32_envi_header_loads_without_using_uint16_json(tmp_path):
    header = tmp_path / "005-specim-fx10.hdr"
    wavelengths = np.linspace(398.25, 1001.17, 224)
    values = np.arange(4 * 5 * 224, dtype=np.float32).reshape(4, 5, 224)
    envi.save_image(str(header), values, ext=".bil", interleave="bil",
                    metadata={"wavelength": wavelengths.tolist(), "wavelength units": "nm",
                              "description": "Experiment: anutest; Batch: 20230419; Run: 005."})
    metadata = envi.read_envi_header(str(header))
    metadata["data type"] = 12
    json_path = header.with_suffix(".json")
    json_path.write_text(json.dumps({"bil_hdr": metadata}), encoding="utf-8")
    for source in (header, header.with_suffix(".bil")):
        data = HSIReader().open(source)
        assert np.dtype(data.image.dtype) == np.dtype(np.float32)
        np.testing.assert_array_equal(data.read_bands(range(224)), values)
        data.close()
    with pytest.raises(HSIHeaderError, match="conflicts.*data type"):
        HSIReader().open(json_path)


@pytest.mark.parametrize("document,message", [
    ("{ invalid", "malformed"),
    ("[]", "must be an object"),
    ('{"bil_hdr": []}', "must be a metadata object"),
    ('{"frames": [{"frame_id": 1}]}', "missing samples"),
])
def test_invalid_or_unrelated_json_is_not_treated_as_a_cube(tmp_path, document, message):
    metadata = tmp_path / "invalid.json"
    metadata.write_text(document, encoding="utf-8")
    metadata.with_suffix(".bil").write_bytes(b"\0" * 64)
    with pytest.raises(HSIHeaderError, match=message):
        HSIReader().open(metadata)


def test_json_header_offset_is_used_and_truncation_is_detected(tmp_path):
    metadata, values = write_json_capture(tmp_path, offset=16)
    data = HSIReader().open(metadata)
    np.testing.assert_array_equal(data.read_bands(range(3)), values)
    data.close()
    binary = metadata.with_suffix(".bil")
    binary.write_bytes(binary.read_bytes()[:-1])
    with pytest.raises(HSIFileError, match="truncated"):
        HSIReader().open(metadata)


def test_json_metadata_updates_are_not_hidden_by_adapter_cache(tmp_path):
    metadata, _ = write_json_capture(tmp_path)
    first = HSIReader().open(metadata)
    first.close()
    document = json.loads(metadata.read_text())
    document["bil_hdr"]["wavelength"] = [1100, 1300, 1500]
    metadata.write_text(json.dumps(document), encoding="utf-8")
    second = HSIReader().open(metadata)
    np.testing.assert_array_equal(second.wavelengths_nm, [1100, 1300, 1500])
    second.close()


def test_training_pair_accepts_json_automatically_and_manually(tmp_path):
    metadata, _ = write_json_capture(tmp_path)
    mask = tmp_path / "capture_mask.png"
    mask.write_bytes(b"mask")
    resolver = TrainingPairResolver()
    pair = resolver.resolve(mask)
    assert pair.cube_path == metadata.with_suffix(".bil")
    assert resolver.resolve_manual(mask, metadata).cube_path == metadata
    data = HSIReader().open(pair.cube_path)
    assert data.header_format == "JSON"
    data.close()


def test_calibration_pair_discovers_json_and_does_not_double_count_sidecars(tmp_path):
    dark, _ = write_json_capture(tmp_path, name="2026-01-01--12-00-00_calibFrame")
    bright, _ = write_json_capture(tmp_path, name="2026-01-01--12-00-20_calibFrame")
    source = tmp_path / "2026-01-01--12-01-00_leaf.raw"
    pair = CalibrationFrameResolver().resolve(source)
    assert (pair.dark_path, pair.bright_path) == (dark, bright)
    for metadata in (dark, bright):
        metadata.with_suffix(".hdr").write_text("ENVI\n", encoding="utf-8")
    pair = CalibrationFrameResolver().resolve(source)
    assert pair.dark_path == dark.with_suffix(".hdr")
    assert pair.bright_path == bright.with_suffix(".hdr")
    bright.with_suffix(".hdr").unlink()
    bright.unlink()
    assert CalibrationFrameResolver().resolve(source) is None


def test_file_filter_includes_all_supported_metadata_and_data_suffixes():
    assert all(f"*{extension}" in HSI_FILE_FILTER
               for extension in (*METADATA_EXTENSIONS, *DATA_EXTENSIONS))
