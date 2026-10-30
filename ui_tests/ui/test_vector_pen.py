from __future__ import annotations

import numpy as np
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

from core import Anchor, AnchorMode, VectorMask, VectorPath, VectorTraceOptions
from ui.annotation_panel import PEN_TOOL
from ui.vector_pen import VectorOutline
from ui.viewer import HSIViewer

from test_classification_layer_panel import (
    _make_controller,
    _make_unsupervised_result,
)

LEFT = Qt.MouseButton.LeftButton


def _viewer(qtbot) -> HSIViewer:
    viewer = HSIViewer()
    qtbot.addWidget(viewer)
    viewer.resize(300, 300)
    viewer.show()
    pixmap = QPixmap(100, 100)
    pixmap.fill()
    viewer.set_photo(pixmap)
    viewer.set_pen_tool(True)
    return viewer


def _square(x: float, y: float, size: float) -> VectorPath:
    return VectorPath((
        Anchor(x, y), Anchor(x + size, y), Anchor(x + size, y + size), Anchor(x, y + size)
    ))


def _click(qtbot, viewer, x, y, modifier=Qt.KeyboardModifier.NoModifier):
    pos = viewer.mapFromScene(x, y)
    qtbot.mousePress(viewer.viewport(), LEFT, modifier, pos)
    qtbot.mouseRelease(viewer.viewport(), LEFT, modifier, pos)


def _drag(qtbot, viewer, start, end, modifier=Qt.KeyboardModifier.NoModifier):
    a, b = viewer.mapFromScene(*start), viewer.mapFromScene(*end)
    qtbot.mousePress(viewer.viewport(), LEFT, modifier, a)
    middle = (a + b) / 2
    for pos in (middle, b):
        viewer.mouseMoveEvent(_move_event(viewer, pos, modifier))
    qtbot.mouseRelease(viewer.viewport(), LEFT, modifier, b)


def _move_event(viewer, pos, modifier):
    from PyQt6.QtGui import QMouseEvent

    return QMouseEvent(
        QtCore.QEvent.Type.MouseMove,
        QtCore.QPointF(pos),
        QtCore.QPointF(viewer.viewport().mapToGlobal(pos)),
        LEFT,
        LEFT,
        modifier,
    )


def _record(viewer):
    edits = []
    viewer.vectorMaskEdited.connect(lambda cid, mask, final: edits.append((cid, mask, final)))
    return edits


def test_clicking_points_and_the_first_point_closes_a_new_path(qtbot):
    viewer = _viewer(qtbot)
    viewer.set_vector_masks((VectorOutline(3, (255, 0, 0), VectorMask()),), draw_target=3)
    edits = _record(viewer)
    for x, y in ((10, 10), (60, 10), (60, 50)):
        _click(qtbot, viewer, x, y)
    assert viewer._pen.is_drafting and not edits
    _click(qtbot, viewer, 10, 10)  # back on the first point closes it
    class_id, mask, final = edits[-1]
    assert class_id == 3 and final and len(mask.paths) == 1
    assert [a.point for a in mask.paths[0].anchors] == [(10, 10), (60, 10), (60, 50)]
    assert viewer.vector_selection == (3, 0, -1)


def test_dragging_while_placing_pulls_symmetric_handles(qtbot):
    viewer = _viewer(qtbot)
    viewer.set_vector_masks((VectorOutline(0, (0, 0, 255), VectorMask()),), draw_target=0)
    edits = _record(viewer)
    _click(qtbot, viewer, 10, 10)
    _drag(qtbot, viewer, (50, 10), (60, 20))
    _click(qtbot, viewer, 30, 60)
    viewer.keyPressEvent(_key(Qt.Key.Key_Return))
    anchor = edits[-1][1].paths[0].anchors[1]
    assert anchor.mode is AnchorMode.SYMMETRIC
    assert (round(anchor.out_dx), round(anchor.out_dy)) == (10, 10)
    assert (round(anchor.in_dx), round(anchor.in_dy)) == (-10, -10)


def _key(key):
    from PyQt6.QtGui import QKeyEvent

    return QKeyEvent(QtCore.QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)


def test_select_outline_then_drag_a_point_and_toggle_its_mode(qtbot):
    viewer = _viewer(qtbot)
    mask = VectorMask((_square(20, 20, 40),))
    viewer.set_vector_masks((VectorOutline(1, (0, 255, 0), mask),), draw_target=1)
    edits = _record(viewer)
    _click(qtbot, viewer, 40, 20)  # on the top edge: selects the path
    assert viewer.vector_selection == (1, 0, -1) and not edits
    _drag(qtbot, viewer, (60, 20), (70, 10))  # drag the top-right point
    final = [e for e in edits if e[2]][-1][1]
    assert final.paths[0].anchors[1].point == (70, 10)
    assert viewer.vector_selection == (1, 0, 1)
    viewer.set_vector_masks((VectorOutline(1, (0, 255, 0), final),), viewer.vector_selection, 1)
    _click(qtbot, viewer, 70, 10, Qt.KeyboardModifier.AltModifier)  # Alt-click toggles
    assert edits[-1][1].paths[0].anchors[1].mode is AnchorMode.SMOOTH


def test_click_on_selected_outline_adds_point_and_delete_removes_it(qtbot):
    viewer = _viewer(qtbot)
    mask = VectorMask((_square(20, 20, 40),))
    viewer.set_vector_masks((VectorOutline(1, (0, 255, 0), mask),), (1, 0, -1), 1)
    edits = _record(viewer)
    _click(qtbot, viewer, 40, 20)
    added = edits[-1][1].paths[0]
    assert len(added) == 5 and added.anchors[1].point == (40, 20)
    assert viewer.vector_selection == (1, 0, 1)
    viewer.keyPressEvent(_key(Qt.Key.Key_Delete))
    assert len(edits[-1][1].paths[0]) == 4


def test_vectorize_and_edit_updates_the_class_region(qtbot, monkeypatch):
    controller, viewer, panel, _source = _make_controller(qtbot)
    class_map = np.zeros((20, 20), dtype=np.int32)
    class_map[5:15, 5:15] = 1
    controller._on_result(_make_unsupervised_result(class_map))

    class _Dialog:
        def __init__(self, classes, parent):
            pass

        def exec(self):
            return QtWidgets.QDialog.DialogCode.Accepted

        def selected_class_ids(self):
            return (1,)

        def options(self):
            return VectorTraceOptions()

    monkeypatch.setattr("ui.classification_controller.VectorMaskDialog", _Dialog)
    panel.vectorizeRequested.emit()
    slot = controller._current_slot
    assert 1 in slot.vectors and viewer.pen_active
    assert panel.mode == "annotations"
    page = panel.annotation_page
    assert page._tool_buttons[PEN_TOOL].isChecked()
    np.testing.assert_array_equal(slot.result.class_map == 1, class_map == 1)

    # Enlarge the class through an edit, as the pen would report it.
    viewer.vectorMaskEdited.emit(1, VectorMask((_square(2, 2, 16),)), True)
    # analysis_regions() (spectra) and the means both read slot.result.
    region = slot.result.class_map == 1
    assert region.sum() == 256 and slot.layers.layers[1].pixel_count == 256
    assert region[3, 3] and slot.layers.mask_for_class(1).sum() == 256

    # Point controls: select a point and convert it to smooth.
    viewer.vectorSelectionChanged.emit(1, 0, 0)
    assert page._anchor_mode_buttons[AnchorMode.CORNER].isChecked()
    page._anchor_mode_buttons[AnchorMode.SMOOTH].click()
    assert slot.vectors.get(1).paths[0].anchors[0].mode is AnchorMode.SMOOTH

    # With the pen armed, undo steps the vector history.
    assert controller.undo_annotation()
    assert slot.vectors.get(1).paths[0].anchors[0].mode is AnchorMode.CORNER
    page._revert_vector_button.click()
    assert 1 not in slot.vectors
    np.testing.assert_array_equal(slot.result.class_map, class_map)


def test_closing_a_path_with_no_target_draws_a_new_class(qtbot):
    viewer = _viewer(qtbot)
    edits = _record(viewer)
    drawn = []
    viewer.vectorClassDrawn.connect(drawn.append)
    for x, y in ((10, 10), (60, 10), (60, 50), (10, 10)):
        _click(qtbot, viewer, x, y)
    assert not edits and len(drawn) == 1
    assert [a.point for a in drawn[0].paths[0].anchors] == [(10, 10), (60, 10), (60, 50)]


def test_pen_draws_classes_from_scratch_without_a_classification(qtbot):
    from types import SimpleNamespace

    controller, viewer, panel, source = _make_controller(qtbot)
    source.data = SimpleNamespace(
        rows=20, columns=30, is_loaded=lambda: True, rgb_array=None, roi_mask=None
    )
    changes = []
    controller.regionsChanged.connect(lambda: changes.append(True))
    page = panel.annotation_page
    page.set_tool(PEN_TOOL)
    controller._on_annotation_tool_changed(PEN_TOOL)
    assert page.selected_vector_class() is None  # "New class"

    viewer.vectorClassDrawn.emit(VectorMask((_square(2, 2, 5),)))
    slot = controller._current_slot
    assert slot.layers.class_ids == (0,) and slot.layers.layers[0].name == "Drawn class 1"
    assert (slot.result.class_map == 0).sum() == 25 and changes
    assert page.selected_vector_class() == 0
    regions = controller.analysis_regions()[1]
    assert regions["class:0"].mask.sum() == 25

    # Esc (deselect) aims the pen at a new class again.
    viewer.vectorSelectionChanged.emit(-1, -1, -1)
    assert page.selected_vector_class() is None
    color = controller.analysis_regions()[1]["class:0"].color
    viewer.vectorClassDrawn.emit(VectorMask((_square(10, 10, 4),)))
    assert slot.layers.class_ids == (0, 1)
    assert slot.layers.layers[1].name == "Drawn class 2"
    # Adding a class leaves the existing class's colour alone.
    assert controller.analysis_regions()[1]["class:0"].color == color

    # Undo removes the drawn class; redo brings it back with its name.
    assert controller.undo_annotation()
    assert slot.layers.class_ids == (0,)
    assert controller.redo_annotation()
    assert slot.layers.layers[1].name == "Drawn class 2"
    page._vector_list.setCurrentRow(2)  # row 0 is "New class"
    page._revert_vector_button.click()
    assert slot.layers.class_ids == (0,)

    # Removing a class below another leaves an empty slot, reused next time.
    page._vector_list.setCurrentRow(1)
    page._revert_vector_button.click()
    assert slot.layers.class_ids == () or slot.layers.layers[0].pixel_count == 0
    viewer.vectorClassDrawn.emit(VectorMask((_square(10, 10, 4),)))
    assert slot.layers.class_ids == (0,) and slot.layers.layers[0].pixel_count == 16


def test_drawn_class_joins_an_existing_classification(qtbot):
    controller, viewer, panel, _source = _make_controller(qtbot)
    class_map = np.zeros((20, 20), dtype=np.int32)
    class_map[5:15, 5:15] = 1
    controller._on_result(_make_unsupervised_result(class_map))
    viewer.vectorClassDrawn.emit(VectorMask((_square(0, 0, 8),)))
    slot = controller._current_slot
    assert slot.layers.class_ids == (0, 1, 2)
    assert (slot.result.class_map[:8, :8] == 2).all()
    # The drawn class claims its pixels from the classes beneath it.
    assert slot.layers.layers[1].pixel_count == 100 - 9
