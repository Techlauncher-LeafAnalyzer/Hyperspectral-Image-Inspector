from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PyQt6.QtWidgets import QFileDialog
from spectral.io import envi

from conftest import SYNTHETIC_WAVELENGTHS_NM


def _write_cube(directory: Path, name: str, seed: int, rows: int = 6, columns: int = 6) -> Path:
    rng = np.random.default_rng(seed)
    cube = rng.uniform(
        0.05, 0.95, size=(rows, columns, len(SYNTHETIC_WAVELENGTHS_NM))
    ).astype(np.float32)
    path = directory / f"{name}.hdr"
    envi.save_image(
        str(path),
        cube,
        dtype=np.float32,
        interleave="bsq",
        ext=".img",
        metadata={"wavelength": SYNTHETIC_WAVELENGTHS_NM},
    )
    return path


def _file_names_stub(monkeypatch):
    stub = SimpleNamespace(paths=[])
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileNames",
        staticmethod(lambda *args, **kwargs: (stub.paths, "")),
    )
    return stub


def _existing_directory_stub(monkeypatch):
    stub = SimpleNamespace(directory="")
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        staticmethod(lambda *args, **kwargs: stub.directory),
    )
    return stub


def _wait_for_batch(qtbot, window):
    qtbot.waitUntil(lambda: not window._batch_controller.is_running(), timeout=30000)


def test_add_files_and_run_unsupervised_batch(window, tmp_path, monkeypatch, qtbot):
    file_names = _file_names_stub(monkeypatch)
    directory = _existing_directory_stub(monkeypatch)

    sample_a = _write_cube(tmp_path, "sample_a", seed=1)
    sample_b = _write_cube(tmp_path, "sample_b", seed=2)
    file_names.paths = [str(sample_a), str(sample_b)]
    window.batchAddFilesButton.click()
    assert window.batchFileList.count() == 2
    assert not window.batchRunButton.isEnabled()

    output_dir = tmp_path / "out"
    directory.directory = str(output_dir)
    window.batchOutputDirButton.click()
    assert window.batchRunButton.isEnabled()

    window.batchClassifyCheck.setChecked(True)
    window.batchNumClassesEdit.setText("3")
    window.batchMaxIterationsEdit.setText("5")

    window.batchRunButton.click()
    _wait_for_batch(qtbot, window)

    assert window.batchResultsList.count() == 2
    assert (output_dir / "sample_a.png").exists()
    assert (output_dir / "sample_b.png").exists()
    assert "complete" in window.batchStatusLabel.text().lower()


def test_remove_and_clear_files(window, tmp_path, monkeypatch):
    file_names = _file_names_stub(monkeypatch)
    sample_a = _write_cube(tmp_path, "sample_a", seed=3)
    sample_b = _write_cube(tmp_path, "sample_b", seed=4)
    file_names.paths = [str(sample_a), str(sample_b)]
    window.batchAddFilesButton.click()
    assert window.batchFileList.count() == 2

    window.batchFileList.setCurrentRow(0)
    window.batchFileList.item(0).setSelected(True)
    window.batchRemoveButton.click()
    assert window.batchFileList.count() == 1

    window.batchClearButton.click()
    assert window.batchFileList.count() == 0
    assert not window.batchRunButton.isEnabled()


def test_missing_calibration_pair_records_failure_without_crashing(
    window, tmp_path, monkeypatch, qtbot
):
    file_names = _file_names_stub(monkeypatch)
    directory = _existing_directory_stub(monkeypatch)

    lonely = _write_cube(tmp_path, "lonely", seed=5)
    file_names.paths = [str(lonely)]
    window.batchAddFilesButton.click()

    output_dir = tmp_path / "out"
    directory.directory = str(output_dir)
    window.batchOutputDirButton.click()

    window.batchCalibrateCheck.setChecked(True)
    window.batchRunButton.click()
    _wait_for_batch(qtbot, window)

    assert window.batchResultsList.count() == 1
    assert "✗" in window.batchResultsList.item(0).text()


def test_batch_saves_raw_preview_for_every_file(window, tmp_path, monkeypatch, qtbot):
    file_names = _file_names_stub(monkeypatch)
    directory = _existing_directory_stub(monkeypatch)

    sample = _write_cube(tmp_path, "sample", seed=6)
    file_names.paths = [str(sample)]
    window.batchAddFilesButton.click()
    output_dir = tmp_path / "out"
    directory.directory = str(output_dir)
    window.batchOutputDirButton.click()

    window.batchRunButton.click()
    _wait_for_batch(qtbot, window)

    assert (output_dir / "sample.png").exists()
    assert (output_dir / "sample_raw.png").exists()


def _write_marker_cube(directory: Path, name: str) -> Path:
    size, tag = 120, 14
    cube = np.full((size, size, len(SYNTHETIC_WAVELENGTHS_NM)), 0.9, dtype=np.float32)
    for top, left in ((20, 20), (20, 86), (86, 20), (86, 86)):
        cube[top : top + tag, left : left + tag, :] = 0.02
    path = directory / f"{name}.hdr"
    envi.save_image(
        str(path), cube, dtype=np.float32, interleave="bsq", ext=".img",
        metadata={"wavelength": SYNTHETIC_WAVELENGTHS_NM},
    )
    return path


def test_batch_service_auto_crops_marker_pattern(tmp_path):
    from PIL import Image

    from core import BatchRequest, BatchService

    sample = _write_marker_cube(tmp_path, "leaf")
    output_dir = tmp_path / "out"
    result = BatchService().run(BatchRequest((sample,), output_dir))

    file_result = result.file_results[0]
    assert file_result.succeeded, file_result.error
    assert file_result.auto_cropped
    for path in (file_result.output_path, file_result.raw_output_path):
        with Image.open(path) as image:
            assert image.mode == "RGBA"
            assert image.size[0] < 120 and image.size[1] < 120
            alpha = np.asarray(image)[:, :, 3]
            assert alpha[0, 0] == 0
            assert alpha[alpha.shape[0] // 2, alpha.shape[1] // 2] == 255


def test_batch_csv_checkbox_exports_class_summary(window, tmp_path, monkeypatch, qtbot):
    import csv

    file_names = _file_names_stub(monkeypatch)
    directory = _existing_directory_stub(monkeypatch)

    sample = _write_cube(tmp_path, "sample", seed=7)
    file_names.paths = [str(sample)]
    window.batchAddFilesButton.click()
    output_dir = tmp_path / "out"
    directory.directory = str(output_dir)
    window.batchOutputDirButton.click()

    window.batchClassifyCheck.setChecked(True)
    window.batchNumClassesEdit.setText("3")
    window.batchMaxIterationsEdit.setText("5")
    window.batchExportCsvCheck.setChecked(True)
    window.batchRunButton.click()
    _wait_for_batch(qtbot, window)

    with (output_dir / "sample.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["class_id"] for row in rows] == ["0", "1", "2"]
    assert sum(int(row["pixel_count"]) for row in rows) == 36
    assert "NDVI_mean" in rows[0] and "RGB_R_mean" in rows[0]


def test_batch_csv_without_classification_writes_whole_image_row(tmp_path):
    import csv

    from core import BatchRequest, BatchService

    sample = _write_cube(tmp_path, "sample", seed=8)
    output_dir = tmp_path / "out"
    result = BatchService().run(BatchRequest((sample,), output_dir, export_csv=True))

    file_result = result.file_results[0]
    assert file_result.succeeded, file_result.error
    with file_result.csv_output_path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["class_name"] == "Whole image"
    assert rows[0]["pixel_count"] == "36"
    assert rows[0]["NDVI_mean"]


def test_batch_without_csv_option_writes_no_csv(tmp_path):
    from core import BatchRequest, BatchService

    sample = _write_cube(tmp_path, "sample", seed=9)
    output_dir = tmp_path / "out"
    BatchService().run(BatchRequest((sample,), output_dir))
    assert not (output_dir / "sample.csv").exists()
