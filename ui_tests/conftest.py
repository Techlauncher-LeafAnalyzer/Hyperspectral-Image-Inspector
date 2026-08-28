from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from spectral.io import envi

from core import HSIReader
from ui.main_window import MainWindowController

# Wavelengths (nm) covering every band VisualizationService needs: RGB
# (470/550/660), NDVI/EVI/MTVI/OSAVI (670/800), EVI blue (470), MCARI
# (550/670/700), PRI (531/570).
SYNTHETIC_WAVELENGTHS_NM = [470.0, 531.0, 550.0, 570.0, 660.0, 670.0, 700.0, 800.0]
SYNTHETIC_ROWS = 8
SYNTHETIC_COLUMNS = 8


def _write_synthetic_envi_pair(directory: Path, *, name: str = "synthetic") -> Path:
    """Write a small synthetic ENVI cube covering every visualization mode."""
    rng = np.random.default_rng(0)
    cube = rng.uniform(
        0.05, 0.95, size=(SYNTHETIC_ROWS, SYNTHETIC_COLUMNS, len(SYNTHETIC_WAVELENGTHS_NM))
    ).astype(np.float32)
    hdr_path = directory / f"{name}.hdr"
    envi.save_image(
        str(hdr_path),
        cube,
        dtype=np.float32,
        interleave="bsq",
        ext=".img",
        metadata={"wavelength": SYNTHETIC_WAVELENGTHS_NM},
    )
    return hdr_path


@pytest.fixture
def synthetic_cube_path(tmp_path) -> Path:
    """Header path for a small, fully-loadable synthetic hyperspectral cube."""
    return _write_synthetic_envi_pair(tmp_path)


@pytest.fixture
def synthetic_shape() -> tuple[int, int, int]:
    """``(rows, columns, bands)`` matching ``synthetic_cube_path``'s cube."""
    return (SYNTHETIC_ROWS, SYNTHETIC_COLUMNS, len(SYNTHETIC_WAVELENGTHS_NM))


@pytest.fixture
def dialogs(monkeypatch):
    """Spy on QMessageBox popups so headless tests never block on them."""
    calls = SimpleNamespace(critical=[], information=[])
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        staticmethod(lambda *args, **kwargs: calls.critical.append(args)),
    )
    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(lambda *args, **kwargs: calls.information.append(args)),
    )
    return calls


@pytest.fixture
def file_dialog(monkeypatch):
    """Stub QFileDialog's native pickers with test-controlled return values."""
    stub = SimpleNamespace(open_return=("", ""), save_return=("", ""))
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *args, **kwargs: stub.open_return),
    )
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: stub.save_return),
    )
    return stub


@pytest.fixture
def window(qtbot, dialogs):
    win = MainWindowController()
    qtbot.addWidget(win)
    return win


@pytest.fixture
def loaded_window(window, synthetic_cube_path):
    window.load_image_from_path(synthetic_cube_path)
    return window


@pytest.fixture
def sr_source(tmp_path):
    # Band-dependent, signed values detect lost/reordered bands and unintended
    # per-band normalization. Odd, rectangular dimensions exercise reconstruction.
    cube = (np.arange(7 * 9 * 480).reshape(7, 9, 480) / 1000 - 2).astype(np.float32)
    path = tmp_path / "source.hdr"
    envi.save_image(str(path), cube, ext=".bip", interleave="bip",
                    metadata={"wavelength": np.linspace(352.49, 898.81, 480).tolist()})
    return HSIReader().open(path), cube
