from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PyQt6.QtCore import QRectF

from core import VisualizationMode
from core.hsi_reader import DATA_EXTENSIONS

RESOURCES_DIR = Path(__file__).resolve().parents[1] / "resources"

MODE_BUTTONS = (
    ("modeRGB", VisualizationMode.RGB),
    ("modeNDVI", VisualizationMode.NDVI),
    ("modeEVI", VisualizationMode.EVI),
    ("modeMCARI", VisualizationMode.MCARI),
    ("modeMTVI", VisualizationMode.MTVI),
    ("modeOSAVI", VisualizationMode.OSAVI),
    ("modePRI", VisualizationMode.PRI),
)


def _discover_sample_headers(base_dir: Path) -> list[Path]:
    """Return every ``.hdr`` file under ``base_dir`` with a matching data file.

    Testers drop their own captures into ``ui_tests/resources/`` locally
    (too large to commit — see that folder's README) rather than the suite
    shipping any. A header without a same-stem data file (e.g. a PSI-style
    reference header kept beside its already-converted ENVI sibling) isn't
    directly loadable by ``HSIReader``, so it's skipped here too.
    """
    if not base_dir.is_dir():
        return []
    return sorted(
        hdr_path
        for hdr_path in base_dir.rglob("*.hdr")
        if any(hdr_path.with_suffix(ext).is_file() for ext in DATA_EXTENSIONS)
    )


SAMPLE_HEADERS = _discover_sample_headers(RESOURCES_DIR)
_NO_SAMPLES_REASON = (
    f"No real sample images found under {RESOURCES_DIR} — see "
    "ui_tests/resources/README.md to add your own."
)


@pytest.fixture(params=SAMPLE_HEADERS or [None], ids=lambda p: p.stem if p else "no-sample-images")
def sample_header_path(request) -> Path:
    if request.param is None:
        pytest.skip(_NO_SAMPLES_REASON)
    return request.param


def test_real_image_loads_and_populates_state(window, sample_header_path):
    window.load_image_from_path(sample_header_path)

    assert window._hsi_data.is_loaded()
    assert window.viewer.has_photo()
    assert window._hsi_data.rows > 0
    assert window._hsi_data.columns > 0
    assert sample_header_path.stem in window.imageFilePath.text()


@pytest.mark.parametrize("button_name, mode", MODE_BUTTONS)
def test_real_image_renders_every_supported_mode(
    window, sample_header_path, button_name, mode
):
    window.load_image_from_path(sample_header_path)

    getattr(window, button_name).click()

    result = window._visualization_results.get(mode)
    if result is None:
        pytest.skip(f"{mode.value} needs wavelength coverage this capture doesn't have")
    assert result.display_rgb.dtype == np.uint8
    assert result.display_rgb.shape[:2] == (window._hsi_data.rows, window._hsi_data.columns)


def test_real_image_crop_then_save_round_trips(window, sample_header_path, tmp_path, file_dialog):
    window.load_image_from_path(sample_header_path)
    rows, columns = window._hsi_data.rows, window._hsi_data.columns
    crop_rows, crop_columns = max(1, rows // 4), max(1, columns // 4)

    window.viewer.cropRequested.emit(QRectF(0, 0, crop_columns, crop_rows))

    assert window._hsi_data.rgb_array.shape[:2] == (crop_rows, crop_columns)

    target = tmp_path / "real-image-crop.png"
    file_dialog.save_return = (str(target), "")
    window.actionSaveImage.trigger()

    assert Image.open(target).size == (crop_columns, crop_rows)
