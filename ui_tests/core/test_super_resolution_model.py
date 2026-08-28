from pathlib import Path

import numpy as np
import pytest
from spectral.io import envi

from core import (
    CancelledError, HSIData, HSIReader, SuperResolutionError,
    SuperResolutionRequest, SuperResolutionService, VisualizationRequest,
    VisualizationService,
)
from core.super_resolution_model import DEFAULT_CHECKPOINT


@pytest.fixture
def sr_source(tmp_path):
    # Band-dependent, signed values detect lost/reordered bands and unintended
    # per-band normalization. Odd, rectangular dimensions exercise reconstruction.
    cube = (np.arange(7 * 9 * 480).reshape(7, 9, 480) / 1000 - 2).astype(np.float32)
    path = tmp_path / "source.hdr"
    envi.save_image(str(path), cube, ext=".bip", interleave="bip",
                    metadata={"wavelength": np.linspace(352.49, 898.81, 480).tolist()})
    return HSIReader().open(path), cube


@pytest.fixture
def identity_service(monkeypatch):
    torch = pytest.importorskip("torch")
    scipy = pytest.importorskip("scipy.ndimage")
    service = SuperResolutionService()

    def model(low, baseline):
        assert low.ndim == 4 and low.shape[:2] == (1, 480)
        assert baseline.shape == (1, 480, low.shape[2] * 2, low.shape[3] * 2)
        return low.repeat_interleave(2, 2).repeat_interleave(2, 3)

    monkeypatch.setattr(service, "_load_model", lambda request:
                        (torch, scipy.zoom, model, "cpu", Path("test.pth")))
    return service


def test_rejects_unloaded_or_wrong_band_count(synthetic_cube_path):
    service = SuperResolutionService()
    with pytest.raises(SuperResolutionError, match="Load a hyperspectral"):
        service.run(HSIData())
    with pytest.raises(SuperResolutionError, match="exactly 480.*has 8"):
        service.run(HSIReader().open(synthetic_cube_path))


def test_missing_checkpoint_is_actionable(sr_source, tmp_path):
    with pytest.raises(SuperResolutionError, match="SR model not found"):
        SuperResolutionService().run(sr_source[0], SuperResolutionRequest(tmp_path / "missing.pth"))


@pytest.mark.parametrize("size,context", [(64, 8), (4, 2), (1, 0)])
def test_reconstruction_preserves_all_bands_and_values(sr_source, identity_service, size, context):
    source, cube = sr_source
    progress = []
    result = identity_service.run(source, SuperResolutionRequest(tile_size=size, context=context),
                                  progress=lambda value, message: progress.append(value))
    assert result.data.shape == (14, 18, 480)
    expected = cube.repeat(2, 0).repeat(2, 1)
    np.testing.assert_array_equal(result.data.read_bands(range(480)), expected)
    np.testing.assert_array_equal(result.data.read_pixel(9, 11), cube[4, 5])
    np.testing.assert_array_equal(result.data.wavelengths_nm, source.wavelengths_nm)
    np.testing.assert_array_equal(source.read_bands(range(480)), cube)
    assert result.data.metadata["lines"] == "14"
    assert result.data.metadata["samples"] == "18"
    assert progress == sorted(progress) and progress[-1] == 100
    assert result.tiled == (size < 9)
    assert VisualizationService().render(result.data, VisualizationRequest("RGB")).display_rgb.shape == (14, 18, 3)


def test_cropped_input_uses_current_cube_not_original_file(sr_source, identity_service):
    data, cube = sr_source
    data.rgb_array = np.zeros((7, 9, 3), dtype=np.uint8)
    data.mask_array = np.zeros((7, 9), dtype=np.uint8)
    data.crop(2, 1, 8, 6)
    result = identity_service.run(data)
    np.testing.assert_array_equal(result.data.read_bands(range(480)), cube[1:6, 2:8].repeat(2, 0).repeat(2, 1))


def test_cancellation_cleans_partial_output(sr_source, identity_service, monkeypatch, tmp_path):
    import core.super_resolution_model as sr
    monkeypatch.setattr(sr.tempfile, "tempdir", str(tmp_path))
    cancelled = False

    def progress(value, message):
        nonlocal cancelled
        cancelled = value > 5

    with pytest.raises(CancelledError):
        identity_service.run(sr_source[0], SuperResolutionRequest(tile_size=4),
                             progress=progress, is_cancelled=lambda: cancelled)
    assert not list(tmp_path.glob("hsi-sr-*"))


def test_rejects_nonfinite_input(sr_source, identity_service, monkeypatch):
    source, cube = sr_source
    cube[0, 0, 0] = np.nan
    monkeypatch.setattr(source.image, "read_subregion", lambda *args: cube)
    with pytest.raises(SuperResolutionError, match="NaN/infinite pixels"):
        identity_service.run(source)


@pytest.mark.parametrize("bad_output", ["shape", "nan"])
def test_rejects_invalid_model_output(sr_source, identity_service, monkeypatch, bad_output):
    torch, zoom, _, device, path = identity_service._load_model(None)

    def model(low, baseline):
        return baseline[:, :1] if bad_output == "shape" else baseline * float("nan")

    monkeypatch.setattr(identity_service, "_load_model", lambda request: (torch, zoom, model, device, path))
    with pytest.raises(SuperResolutionError, match="Invalid model output"):
        identity_service.run(sr_source[0])


def test_rejects_corrupt_checkpoint(sr_source, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("scipy")
    path = tmp_path / "bad.pth"
    path.write_bytes(b"invalid checkpoint")
    with pytest.raises(SuperResolutionError, match="Cannot load bad.pth"):
        SuperResolutionService().run(sr_source[0], SuperResolutionRequest(path))


@pytest.mark.sr_model
def test_actual_checkpoint_pipeline(sr_source):
    torch = pytest.importorskip("torch")
    scipy = pytest.importorskip("scipy.ndimage")
    if not DEFAULT_CHECKPOINT.is_file():
        pytest.skip("Supply model/fin_msdformer.pth to test actual inference")
    source, cube = sr_source
    service = SuperResolutionService()
    result = service.run(source)
    actual = result.data.read_bands(range(480))
    assert actual.shape == (14, 18, 480)
    assert actual.dtype == np.float32 and np.isfinite(actual).all()
    # Compare the service with direct network inference, not a mock or RGB-only path.
    cubic = np.stack([scipy.zoom(cube[:, :, i], 2, order=3) for i in range(480)])
    with torch.inference_mode():
        expected = service._model(torch.from_numpy(cube.transpose(2, 0, 1).copy())[None],
                                  torch.from_numpy(cubic)[None])[0].numpy().transpose(1, 2, 0)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(result.data.wavelengths_nm, source.wavelengths_nm)
