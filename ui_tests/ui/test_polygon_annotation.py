from __future__ import annotations

import numpy as np
import pytest
from PyQt6 import QtCore
from PyQt6.QtGui import QPixmap

from core import (
    AnnotationError,
    AnnotationKind,
    AnnotationLayerModel,
    AnnotationStyle,
    region_mask,
)
from ui.annotation_panel import AnnotationToolsPage
from ui.annotation_render import render_annotations
from ui.viewer import HSIViewer

STYLE = AnnotationStyle()
TRIANGLE = [(10.0, 10.0), (50.0, 10.0), (10.0, 50.0)]
LEFT = QtCore.Qt.MouseButton.LeftButton


def _viewer(qtbot) -> HSIViewer:
    viewer = HSIViewer()
    qtbot.addWidget(viewer)
    viewer.resize(300, 300)
    viewer.show()
    pixmap = QPixmap(100, 100)
    pixmap.fill()
    viewer.set_photo(pixmap)
    return viewer


def _click(qtbot, viewer, x, y, button=LEFT):
    qtbot.mouseClick(viewer.viewport(), button, pos=viewer.mapFromScene(x, y))


# --- Model ------------------------------------------------------------------ #


def test_polygon_keeps_vertices_and_bounding_box_and_measures_by_default():
    model = AnnotationLayerModel()
    layer = model.add_polygon(TRIANGLE, STYLE)
    assert layer.kind is AnnotationKind.POLYGON and layer.name == "Polygon 1"
    assert layer.points == tuple(TRIANGLE)
    assert (layer.x, layer.y, layer.width, layer.height) == (10, 10, 40, 40)
    assert layer.measure_index
    assert model.get(layer.layer_id).points == tuple(TRIANGLE)


def test_polygon_drops_repeated_corners_and_rejects_degenerate_ones():
    model = AnnotationLayerModel()
    layer = model.add_polygon(TRIANGLE + [TRIANGLE[-1], TRIANGLE[0]], STYLE)
    assert layer.points == tuple(TRIANGLE)
    with pytest.raises(AnnotationError):
        model.add_polygon([(0, 0), (5, 5), (5, 5)], STYLE)
    with pytest.raises(AnnotationError):
        model.add_polygon([(0, 0), (5, 0), (9, 0)], STYLE)  # zero height
    with pytest.raises(AnnotationError):
        model.add_shape(AnnotationKind.POLYGON, 0, 0, 5, 5, STYLE)


def test_moving_and_resizing_a_polygon_maps_its_vertices_and_undoes():
    model = AnnotationLayerModel()
    layer = model.add_polygon(TRIANGLE, STYLE)
    model.set_geometry(layer.layer_id, 20, 30, 80, 20)
    moved = model.get(layer.layer_id)
    assert moved.points == ((20, 30), (100, 30), (20, 50))
    assert model.undo() and model.get(layer.layer_id).points == tuple(TRIANGLE)
    assert model.undo() and len(model) == 0


# --- Region and rendering ----------------------------------------------------- #


def test_polygon_mask_matches_its_outline_and_clips_to_the_image():
    model = AnnotationLayerModel()
    # Off the pixel grid, so no pixel centre lies on the hypotenuse.
    layer = model.add_polygon([(0, 0), (8.25, 0), (0, 8.25)], STYLE)
    mask = region_mask(layer, (6, 6))
    assert mask[0, 0] and mask[0, 5] and mask[5, 0]
    assert not mask[5, 5] and not mask[4, 4]
    rows, columns = np.mgrid[0:6, 0:6]
    assert np.array_equal(mask, rows + columns + 1 < 8.25)


def test_filled_polygon_is_burnt_into_exports(qtbot):
    rgb = np.full((60, 60, 3), 100, dtype=np.uint8)
    model = AnnotationLayerModel()
    red = (255, 0, 0)
    model.add_polygon(
        TRIANGLE, AnnotationStyle(fill=red, fill_enabled=True, stroke_width=0)
    )
    out = render_annotations(rgb, model.layers)
    assert tuple(out[15, 15]) == red
    assert tuple(out[45, 45]) == (100, 100, 100)  # past the hypotenuse


# --- Viewer ------------------------------------------------------------------- #


def test_polygon_tool_places_corners_and_double_click_finishes(qtbot):
    viewer = _viewer(qtbot)
    viewer.set_annotation_tool(AnnotationKind.POLYGON)
    drawn = []
    viewer.annotationPolygonDrawn.connect(drawn.append)
    for x, y in TRIANGLE:
        _click(qtbot, viewer, x, y)
    assert viewer._annotation_preview is not None and not drawn
    # A real double-click is a click then a double-click event; QTest sends
    # only the latter, so the first click above stands in for it.
    qtbot.mouseDClick(viewer.viewport(), LEFT, pos=viewer.mapFromScene(*TRIANGLE[2]))
    assert viewer._annotation_preview is None
    (points,) = drawn
    assert len(points) == 3
    for (x, y), (ex, ey) in zip(points, TRIANGLE):
        assert abs(x - ex) <= 2 and abs(y - ey) <= 2


def test_polygon_keys_undo_a_corner_cancel_and_finish(qtbot):
    viewer = _viewer(qtbot)
    viewer.set_annotation_tool(AnnotationKind.POLYGON)
    drawn = []
    viewer.annotationPolygonDrawn.connect(drawn.append)
    for x, y in TRIANGLE + [(80, 80)]:
        _click(qtbot, viewer, x, y)
    qtbot.keyClick(viewer, QtCore.Qt.Key.Key_Backspace)
    assert len(viewer._annotation_polygon) == 3
    qtbot.keyClick(viewer, QtCore.Qt.Key.Key_Return)
    assert len(drawn) == 1 and len(drawn[0]) == 3

    for x, y in TRIANGLE:
        _click(qtbot, viewer, x, y)
    qtbot.keyClick(viewer, QtCore.Qt.Key.Key_Escape)
    assert viewer._annotation_preview is None and not viewer._annotation_polygon
    assert len(drawn) == 1


def test_right_click_finishes_without_opening_the_menu_and_too_few_corners_draw_nothing(
    qtbot,
):
    viewer = _viewer(qtbot)
    viewer.set_annotation_tool(AnnotationKind.POLYGON)
    drawn = []
    viewer.annotationPolygonDrawn.connect(drawn.append)
    for x, y in TRIANGLE:
        _click(qtbot, viewer, x, y)
    qtbot.mousePress(
        viewer.viewport(), QtCore.Qt.MouseButton.RightButton,
        pos=viewer.mapFromScene(60, 60),
    )
    assert len(drawn) == 1 and viewer._suppress_context_menu

    _click(qtbot, viewer, 10, 10)
    _click(qtbot, viewer, 20, 20)
    qtbot.keyClick(viewer, QtCore.Qt.Key.Key_Return)
    assert len(drawn) == 1


def test_switching_tools_discards_an_unfinished_polygon(qtbot):
    viewer = _viewer(qtbot)
    viewer.set_annotation_tool(AnnotationKind.POLYGON)
    _click(qtbot, viewer, 10, 10)
    viewer.set_annotation_tool(None)
    assert not viewer._annotation_polygon and viewer._annotation_preview is None


def test_polygon_is_hit_on_its_edges_and_inside_only_when_filled(qtbot):
    viewer = _viewer(qtbot)
    model = AnnotationLayerModel()
    hollow = model.add_polygon(TRIANGLE, STYLE)
    viewer.set_annotations(model.layers)
    assert viewer._annotation_at(QtCore.QPointF(30, 10)) == hollow.layer_id
    assert viewer._annotation_at(QtCore.QPointF(20, 20)) is None
    assert viewer._annotation_at(QtCore.QPointF(45, 45)) is None

    model.set_style(hollow.layer_id, AnnotationStyle(fill_enabled=True))
    viewer.set_annotations(model.layers)
    assert viewer._annotation_at(QtCore.QPointF(20, 20)) == hollow.layer_id
    assert viewer._annotation_at(QtCore.QPointF(45, 45)) is None


def test_panel_offers_the_polygon_tool_and_lets_it_measure(qtbot):
    page = AnnotationToolsPage()
    qtbot.addWidget(page)
    button = page._tool_buttons[AnnotationKind.POLYGON]
    assert not button.icon().isNull()
    tools = []
    page.toolChanged.connect(tools.append)
    button.click()
    assert tools == [AnnotationKind.POLYGON]

    model = AnnotationLayerModel()
    layer = model.add_polygon(TRIANGLE, STYLE)
    page.set_layers(model.layers, layer.layer_id)
    assert page._measure_check.isEnabled() and page._measure_check.isChecked()


# --- Window ------------------------------------------------------------------- #


def test_drawn_polygon_becomes_a_selected_measured_annotation(loaded_window):
    loaded_window.tabWidget.setCurrentWidget(loaded_window.Classification)
    viewer = loaded_window.classificationViewer
    controller = loaded_window._classification_controller
    viewer.annotationPolygonDrawn.emit([(0.0, 0.0), (8.25, 0.0), (0.0, 8.25)])

    slot = controller._annotation_slot
    (layer,) = slot.layers.layers
    assert layer.kind is AnnotationKind.POLYGON and slot.selected_id == layer.layer_id
    measured = controller._measured_regions()
    assert [m.pixel_count for _, m in measured] == [36]  # 8x8 fixture, row + col <= 7

    loaded_window._undo_crop()
    assert len(slot.layers) == 0
