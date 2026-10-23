"""Visualization tab's Custom Index and Brightness modes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6 import QtWidgets

from core import VisualizationMode
from core.classification_model import (
    SupervisedClassificationResult,
    SupervisedClassifierType,
)
from ui.custom_index_dialog import CustomIndexDialog, CustomIndexStore
from ui.main_window import MainWindowController

NDVI_FORMULA = "(R800 - R670) / (R800 + R670)"


def fake_dialog_factory(formula: str | None, calls: list | None = None):
    """Stand in for the modal editor: accept `formula`, or cancel if None."""

    def create(wavelengths, current, parent=None):
        if calls is not None:
            calls.append(SimpleNamespace(wavelengths=wavelengths, current=current))
        accepted = QtWidgets.QDialog.DialogCode.Accepted
        rejected = QtWidgets.QDialog.DialogCode.Rejected
        return SimpleNamespace(
            exec=lambda: accepted if formula else rejected, formula=formula
        )

    return create


def test_choosing_custom_without_a_formula_opens_the_editor(loaded_window):
    calls = []
    loaded_window._custom_index_dialog_factory = fake_dialog_factory(NDVI_FORMULA, calls)

    loaded_window.modeCustom.click()

    assert len(calls) == 1
    np.testing.assert_array_equal(
        calls[0].wavelengths, loaded_window._hsi_data.wavelengths_nm
    )
    assert loaded_window._active_visualization_mode is VisualizationMode.CUSTOM
    result = loaded_window._visualization_results[VisualizationMode.CUSTOM]
    assert result.title == f"Custom index: {NDVI_FORMULA}"
    caption = loaded_window.visualizationModeCaption
    assert caption.isVisibleTo(loaded_window)
    assert "R800 → 800.0 nm" in caption.text()


def test_cancelling_the_editor_returns_to_the_previous_mode(loaded_window):
    loaded_window.modeNDVI.click()
    loaded_window._custom_index_dialog_factory = fake_dialog_factory(None)

    loaded_window.modeCustom.click()

    assert loaded_window.modeNDVI.isChecked()
    assert not loaded_window.modeCustom.isChecked()
    assert loaded_window._active_visualization_mode is VisualizationMode.NDVI
    assert VisualizationMode.CUSTOM not in loaded_window._visualization_results


def test_applied_formula_is_saved_and_restored_next_session(loaded_window, qtbot, synthetic_cube_path):
    loaded_window._custom_index_dialog_factory = fake_dialog_factory("R800 / R670")
    loaded_window.editCustomIndexButton.click()
    assert loaded_window.modeCustom.isChecked()
    loaded_window.close()

    assert CustomIndexStore().load() == "R800 / R670"
    reopened = MainWindowController()
    qtbot.addWidget(reopened)
    assert reopened._custom_formula == "R800 / R670"

    reopened.load_image_from_path(synthetic_cube_path)
    # Restored formulas render with the other indices, so hover shows them.
    assert "CUSTOM" in reopened._pixel_values_at(0, 0)
    assert reopened.viewer.available_on_demand_indices == {"CUSTOM", "BRIGHTNESS"}


def test_custom_formula_outside_the_camera_range_is_explained(loaded_window):
    loaded_window._custom_index_dialog_factory = fake_dialog_factory("R800 / R270")

    loaded_window.modeCustom.click()

    caption = loaded_window.visualizationModeCaption
    assert loaded_window.modeCustom.isChecked()
    assert VisualizationMode.CUSTOM not in loaded_window._visualization_results
    assert "R270 is outside" in caption.text()
    assert caption.property("state") == "warning"


def test_brightness_defaults_to_whole_image(loaded_window):
    loaded_window.modeBrightness.click()

    result = loaded_window._visualization_results[VisualizationMode.BRIGHTNESS]
    assert np.nanmin(result.values) == pytest.approx(0.0)
    assert np.nanmax(result.values) == pytest.approx(1.0)
    caption = loaded_window.visualizationModeCaption.text()
    assert "Whole image" in caption
    assert "sum of all 8 bands" in caption
    assert "whole image" in loaded_window.modeBrightness.toolTip()


def _two_leaf_result(shape: tuple[int, int]) -> SupervisedClassificationResult:
    """A result as Separate Leaves leaves it: leaves 3 and 4, background 2."""
    class_map = np.full(shape, 2, dtype=np.int32)
    class_map[:, :3] = 3
    class_map[:, 5:] = 4
    class_ids = (3, 4, 2)
    masks = (class_map[None] == np.asarray(class_ids)[:, None, None]).astype(np.uint8)
    return SupervisedClassificationResult(
        class_map=class_map,
        one_hot_masks=masks,
        class_ids=class_ids,
        class_pixel_counts=masks.reshape(3, -1).sum(axis=1),
        training_sample_counts=np.asarray([0, 0, 20]),
        classifier=SupervisedClassifierType.GAUSSIAN,
        training_band_indices=(0,),
        target_band_indices=(0,),
        band_wavelengths_nm=np.asarray([550.0]),
        training_mask_path=Path("mask.png"),
        training_cube_path=Path("cube.img"),
    )


def test_supervised_result_switches_brightness_to_per_leaf(loaded_window, qtbot, monkeypatch):
    window = loaded_window
    window.modeBrightness.click()
    band_reads = []
    original_band_sum = window._visualization_service.band_sum
    monkeypatch.setattr(
        window._visualization_service,
        "band_sum",
        lambda *args, **kwargs: band_reads.append(1) or original_band_sum(*args, **kwargs),
    )

    controller = window._classification_controller
    controller._on_result(_two_leaf_result((8, 8)))
    controller._current_slot.leaves_separated = True
    qtbot.waitUntil(
        lambda: "Per leaf cluster" in window.visualizationModeCaption.text()
    )

    brightness = window._visualization_results[VisualizationMode.BRIGHTNESS]
    assert band_reads == []  # The cached band sum is reused.
    for columns in (slice(0, 3), slice(5, 8)):
        leaf = brightness.values[:, columns]
        assert leaf.min() == pytest.approx(0.0)
        assert leaf.max() == pytest.approx(1.0)
    assert np.isnan(brightness.values[:, 3:5]).all()
    assert "2 clusters" in window.visualizationModeCaption.text()
    assert "per leaf cluster" in window.modeBrightness.toolTip()

    controller.clear_result()
    qtbot.waitUntil(lambda: "Whole image" in window.visualizationModeCaption.text())


def test_caption_hides_for_built_in_modes_and_hypercube(loaded_window):
    loaded_window.modeBrightness.click()
    assert loaded_window.visualizationModeCaption.isVisibleTo(loaded_window)

    loaded_window.modeHyperCube.click()
    assert not loaded_window.visualizationModeCaption.isVisibleTo(loaded_window)

    loaded_window.modeNDVI.click()
    assert not loaded_window.visualizationModeCaption.isVisibleTo(loaded_window)


def test_brightness_mean_uses_friendly_label(loaded_window, monkeypatch):
    shown = []
    monkeypatch.setattr(
        "ui.main_window.IndexMeanDialog",
        lambda name, mean, *args, **kwargs: shown.append((name, mean))
        or SimpleNamespace(show=lambda: None, raise_=lambda: None, activateWindow=lambda: None),
    )

    loaded_window._on_mean_index("BRIGHTNESS")

    assert shown and shown[0][0] == "Brightness"
    assert 0.0 <= shown[0][1] <= 1.0


def test_dialog_validates_and_maps_bands(qtbot):
    dialog = CustomIndexDialog(np.array([470.0, 670.0, 802.5]), "R800/R670")
    qtbot.addWidget(dialog)

    assert dialog.applyButton.isEnabled()
    assert "3 bands from 470.0 to 802.5 nm" in dialog.rangeLabel.text()
    assert "R800</b> → band 2 at 802.5 nm (+2.5 nm)" in dialog.statusLabel.text()
    assert dialog.previewLabel.pixmap() is not None and not dialog.previewLabel.pixmap().isNull()

    dialog.formulaEdit.setText("R800 / R300")
    dialog._validate()
    assert not dialog.applyButton.isEnabled()
    assert "R300 is outside" in dialog.statusLabel.text()

    dialog.formulaEdit.setText("R800 +")
    dialog._validate()
    assert not dialog.applyButton.isEnabled()
    assert "ends unexpectedly" in dialog.statusLabel.text()


def test_dialog_palette_inserts_at_the_cursor(qtbot):
    dialog = CustomIndexDialog(np.array([470.0, 670.0, 800.0]))
    qtbot.addWidget(dialog)
    dialog._insert("sqrt()", 1)
    dialog.wavelengthSpin.setValue(671.0)
    assert "nearest band 1 at 670.0 nm" in dialog.nearestLabel.text()
    dialog._insert(f"R{dialog.wavelengthSpin.value():g}", 0)
    assert dialog.formulaEdit.text() == "sqrt(R671)"

    dialog._apply()
    assert dialog.formula == "sqrt(R671)"
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
