from __future__ import annotations

from PyQt6.QtGui import QShortcut
from PyQt6.QtCore import QRectF

from core import AnnotationKind


def _open_classification_tab(window) -> None:
    window.tabWidget.setCurrentWidget(window.Classification)
    assert window._active_viewer is window.classificationViewer


def _draw_rect(window) -> None:
    window.classificationViewer.annotationShapeDrawn.emit(
        AnnotationKind.RECTANGLE, QRectF(1, 1, 4, 4)
    )


def _count(window) -> int:
    return len(window._classification_controller._annotation_slot.layers)


def test_undo_redo_target_annotations_on_the_classification_tab(loaded_window):
    _open_classification_tab(loaded_window)
    _draw_rect(loaded_window)
    _draw_rect(loaded_window)
    assert _count(loaded_window) == 2

    loaded_window._undo_crop()
    assert _count(loaded_window) == 1
    loaded_window._redo_crop()
    assert _count(loaded_window) == 2
    assert len(loaded_window._classification_controller._annotation_slot.layers.layers) == 2


def test_undo_falls_back_to_crop_when_no_annotation_history(loaded_window):
    loaded_window.viewer.cropRequested.emit(QRectF(1, 1, 4, 4))
    _open_classification_tab(loaded_window)
    assert loaded_window._hsi_data.rgb_array.shape[:2] == (4, 4)
    loaded_window._undo_crop()
    assert loaded_window._hsi_data.rgb_array.shape[:2] == (8, 8)


def test_annotation_undo_does_not_touch_crop_on_other_tabs(loaded_window):
    _open_classification_tab(loaded_window)
    _draw_rect(loaded_window)
    loaded_window.tabWidget.setCurrentWidget(loaded_window.tabWidget.widget(0))
    loaded_window.viewer.cropRequested.emit(QRectF(1, 1, 4, 4))
    # Cropping clears annotations with the rest of the classification state;
    # undoing from the main tab must undo the crop, not annotations.
    assert _count(loaded_window) == 0
    loaded_window._undo_crop()
    assert loaded_window._hsi_data.rgb_array.shape[:2] == (8, 8)


def _shortcut_for(window, key: str) -> QShortcut:
    # Offscreen windows never become the active window, so real key events are
    # not delivered to shortcuts; trigger the registered shortcut directly.
    for shortcut in window.findChildren(QShortcut):
        if key in [k.toString() for k in shortcut.keys()]:
            return shortcut
    raise AssertionError(f"No shortcut registered for {key}")


def test_ctrl_z_and_ctrl_shift_z_shortcuts_drive_annotation_history(loaded_window):
    _open_classification_tab(loaded_window)
    _draw_rect(loaded_window)
    _draw_rect(loaded_window)

    _shortcut_for(loaded_window, "Ctrl+Z").activated.emit()
    assert _count(loaded_window) == 1
    _shortcut_for(loaded_window, "Ctrl+Shift+Z").activated.emit()
    assert _count(loaded_window) == 2


def test_moving_an_annotation_is_a_single_undo_step(loaded_window):
    _open_classification_tab(loaded_window)
    _draw_rect(loaded_window)
    controller = loaded_window._classification_controller
    layer_id = controller._annotation_slot.selected_id
    viewer = loaded_window.classificationViewer
    for x in (2, 3, 4):  # one drag reporting several positions
        viewer.annotationGeometryChanged.emit(layer_id, x, 1, 4, 4)
    viewer.annotationInteractionFinished.emit()
    assert controller._annotation_slot.layers.get(layer_id).x == 4

    loaded_window._undo_crop()
    assert controller._annotation_slot.layers.get(layer_id).x == 1
    loaded_window._redo_crop()
    assert controller._annotation_slot.layers.get(layer_id).x == 4


def test_deselecting_clears_selection_but_keeps_the_layer_and_history(loaded_window):
    _open_classification_tab(loaded_window)
    _draw_rect(loaded_window)
    controller = loaded_window._classification_controller
    assert controller._annotation_slot.selected_id is not None
    loaded_window.classificationViewer.annotationDeselected.emit()
    assert controller._annotation_slot.selected_id is None
    assert _count(loaded_window) == 1
    assert loaded_window.classificationLayerPanel.annotation_page.selected_layer_id() is None
    loaded_window.classificationViewer.annotationDeselected.emit()  # idempotent
    loaded_window._undo_crop()
    assert _count(loaded_window) == 0


def test_rotating_is_a_single_undo_step_and_survives_deselect(loaded_window):
    _open_classification_tab(loaded_window)
    _draw_rect(loaded_window)
    controller = loaded_window._classification_controller
    layer_id = controller._annotation_slot.selected_id
    viewer = loaded_window.classificationViewer
    for degrees in (10, 25, 40):
        viewer.annotationRotationChanged.emit(layer_id, degrees)
    viewer.annotationInteractionFinished.emit()
    assert controller._annotation_slot.layers.get(layer_id).rotation == 40
    viewer.annotationDeselected.emit()
    assert controller._annotation_slot.layers.get(layer_id).rotation == 40
    loaded_window._undo_crop()
    assert controller._annotation_slot.layers.get(layer_id).rotation == 0
