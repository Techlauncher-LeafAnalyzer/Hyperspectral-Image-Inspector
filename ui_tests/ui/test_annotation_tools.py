from __future__ import annotations

import numpy as np
from PyQt6 import QtCore
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QPixmap

from core import AnnotationKind, AnnotationLayerModel, AnnotationStyle
from ui.annotation_panel import AnnotationToolsPage
from ui.classification_layer_panel import ClassificationLayerPanel
from ui.viewer import HSIViewer

STYLE = AnnotationStyle()


def _viewer(qtbot) -> HSIViewer:
    viewer = HSIViewer()
    qtbot.addWidget(viewer)
    viewer.resize(300, 300)
    viewer.show()
    pixmap = QPixmap(100, 100)
    pixmap.fill()
    viewer.set_photo(pixmap)
    return viewer


def test_panel_toggles_between_classes_and_annotations(qtbot):
    panel = ClassificationLayerPanel()
    qtbot.addWidget(panel)
    panel.show()
    modes = []
    panel.modeChanged.connect(modes.append)
    assert panel.mode == "classes" and panel._body.isVisible()

    panel._annotations_mode_button.click()
    assert panel.mode == "annotations" and panel.annotation_page.isVisible()
    assert not panel._body.isVisible()
    panel._classes_mode_button.click()
    assert panel.mode == "classes" and panel._body.isVisible()
    assert modes == ["annotations", "classes"]


def test_layer_list_is_top_first_and_reports_actions(qtbot):
    page = AnnotationToolsPage()
    qtbot.addWidget(page)
    model = AnnotationLayerModel()
    bottom = model.add_shape(AnnotationKind.RECTANGLE, 0, 0, 5, 5, STYLE)
    top = model.add_shape(AnnotationKind.ELLIPSE, 0, 0, 5, 5, STYLE)
    page.set_layers(model.layers, bottom.layer_id)
    assert page._list.item(0).text() == top.name
    assert page.selected_layer_id() == bottom.layer_id
    assert not page._down_button.isEnabled() and page._up_button.isEnabled()

    with qtbot.waitSignal(page.moveRequested) as moved:
        page._up_button.click()
    assert moved.args == [bottom.layer_id, 1]
    with qtbot.waitSignal(page.deleteRequested):
        page._delete_button.click()
    with qtbot.waitSignal(page.visibilityChanged) as vis:
        page._list.item(0).setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert vis.args == [top.layer_id, False]


def test_style_controls_emit_current_style(qtbot):
    page = AnnotationToolsPage()
    qtbot.addWidget(page)
    with qtbot.waitSignal(page.styleChanged) as changed:
        page._stroke_width.setValue(7)
    assert changed.args[0].stroke_width == 7
    page.set_style(AnnotationStyle(fill_enabled=True, font_size=40))
    assert page.current_style().fill_enabled and page.current_style().font_size == 40


def test_viewer_stacks_items_in_layer_order_and_skips_hidden(qtbot):
    viewer = _viewer(qtbot)
    model = AnnotationLayerModel()
    a = model.add_shape(AnnotationKind.RECTANGLE, 0, 0, 20, 20, STYLE)
    b = model.add_shape(AnnotationKind.ELLIPSE, 5, 5, 20, 20, STYLE)
    model.add_text(1, 1, "hi", STYLE)
    c = model.layers[-1]
    model.set_visible(b.layer_id, False)
    viewer.set_annotations(model.layers)
    z = [item.zValue() for item in viewer._annotation_items]
    assert len(z) == 2 and z == sorted(z) and z[0] < z[1] < 10
    model.move(a.layer_id, 1)  # rect now above hidden ellipse, below text
    viewer.set_annotations(model.layers, c.layer_id)
    assert len(viewer._annotation_items) == 3  # 2 visible + selection outline


def test_drag_with_shape_tool_emits_clamped_rect(qtbot):
    viewer = _viewer(qtbot)
    viewer.set_annotation_tool(AnnotationKind.RECTANGLE)
    start = viewer.mapFromScene(10, 10)
    end = viewer.mapFromScene(60, 40)
    drawn = []
    viewer.annotationShapeDrawn.connect(lambda k, r: drawn.append((k, r)))
    qtbot.mousePress(viewer.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=start)
    qtbot.mouseMove(viewer.viewport(), end)
    assert viewer._annotation_preview is not None
    qtbot.mouseRelease(viewer.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=end)
    assert viewer._annotation_preview is None
    kind, rect = drawn[0]
    assert kind is AnnotationKind.RECTANGLE
    assert abs(rect.width() - 50) <= 2 and abs(rect.height() - 30) <= 2


def test_text_tool_click_requests_text_and_pan_tool_does_not_draw(qtbot):
    viewer = _viewer(qtbot)
    requested = []
    viewer.annotationTextRequested.connect(requested.append)
    viewer.set_annotation_tool(AnnotationKind.TEXT)
    qtbot.mouseClick(viewer.viewport(), QtCore.Qt.MouseButton.LeftButton,
                     pos=viewer.mapFromScene(30, 30))
    assert len(requested) == 1

    viewer.set_annotation_tool(None)
    qtbot.mouseClick(viewer.viewport(), QtCore.Qt.MouseButton.LeftButton,
                     pos=viewer.mapFromScene(30, 30))
    assert len(requested) == 1


def test_arrow_tool_drag_emits_tail_and_head_and_previews(qtbot):
    viewer = _viewer(qtbot)
    viewer.set_annotation_tool(AnnotationKind.ARROW)
    got = []
    viewer.annotationArrowDrawn.connect(lambda a, b: got.append((a, b)))
    start, end = viewer.mapFromScene(80, 70), viewer.mapFromScene(20, 20)
    qtbot.mousePress(viewer.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=start)
    qtbot.mouseMove(viewer.viewport(), end)
    assert viewer._annotation_preview is not None
    qtbot.mouseRelease(viewer.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=end)
    tail, head = got[0]
    assert abs(tail.x() - 80) <= 2 and abs(head.y() - 20) <= 2  # direction kept


def test_arrow_and_opacity_render(qtbot):
    viewer = _viewer(qtbot)
    model = AnnotationLayerModel()
    model.add_arrow(10, 10, 60, 40, AnnotationStyle(opacity=0.4))
    model.add_shape(AnnotationKind.RECTANGLE, 5, 5, 20, 20, AnnotationStyle(opacity=0.5))
    model.add_text(1, 1, "hi", AnnotationStyle(opacity=0.3))
    viewer.set_annotations(model.layers)
    assert [round(i.opacity(), 2) for i in viewer._annotation_items] == [0.4, 0.5, 0.3]


def test_opacity_slider_round_trips_through_style(qtbot):
    page = AnnotationToolsPage()
    qtbot.addWidget(page)
    with qtbot.waitSignal(page.styleChanged) as changed:
        page._opacity.setValue(40)
    assert changed.args[0].opacity == 0.4 and page._opacity_value.text() == "40%"
    page.set_style(AnnotationStyle(opacity=0.8))
    assert page._opacity.value() == 80 and page._opacity_value.text() == "80%"
    assert AnnotationKind.ARROW in page._tool_buttons


def _press_move_release(qtbot, viewer, start, end):
    vp = viewer.viewport()
    left = QtCore.Qt.MouseButton.LeftButton
    qtbot.mousePress(vp, left, pos=viewer.mapFromScene(*start))
    qtbot.mouseMove(vp, viewer.mapFromScene(*end))
    qtbot.mouseRelease(vp, left, pos=viewer.mapFromScene(*end))


def _record(viewer):
    log = {"geom": [], "drawn": [], "finished": 0, "picked": []}
    viewer.annotationGeometryChanged.connect(lambda *a: log["geom"].append(a))
    viewer.annotationShapeDrawn.connect(lambda k, r: log["drawn"].append((k, r)))
    viewer.annotationInteractionFinished.connect(lambda: log.__setitem__("finished", log["finished"] + 1))
    viewer.annotationPicked.connect(log["picked"].append)
    return log


FILLED = AnnotationStyle(fill_enabled=True)


def _selected_rect_viewer(qtbot, tool=None, style=FILLED):
    viewer = _viewer(qtbot)
    model = AnnotationLayerModel()
    rect = model.add_shape(AnnotationKind.RECTANGLE, 20, 20, 40, 30, style)
    viewer.set_annotations(model.layers, rect.layer_id)
    viewer.set_annotation_tool(tool)
    return viewer, rect, _record(viewer)


def test_selected_annotation_moves_even_with_a_drawing_tool_active(qtbot):
    viewer, rect, log = _selected_rect_viewer(qtbot, AnnotationKind.ELLIPSE)
    _press_move_release(qtbot, viewer, (40, 35), (50, 45))
    assert not log["drawn"]  # did not start drawing a circle
    layer_id, x, y, w, h = log["geom"][-1]
    assert layer_id == rect.layer_id
    assert abs(x - 30) <= 2 and abs(y - 30) <= 2 and (round(w), round(h)) == (40, 30)
    assert log["finished"] == 1


def test_unselected_annotation_does_not_block_drawing_with_a_tool(qtbot):
    viewer, rect, log = _selected_rect_viewer(qtbot, AnnotationKind.RECTANGLE)
    viewer.set_annotations(
        (rect,), None  # present but not selected
    )
    _press_move_release(qtbot, viewer, (40, 35), (80, 80))
    assert not log["geom"] and len(log["drawn"]) == 1


def test_selected_annotation_moves_with_no_tool_instead_of_panning(qtbot):
    viewer, rect, log = _selected_rect_viewer(qtbot, None)
    _press_move_release(qtbot, viewer, (40, 35), (45, 40))
    assert log["geom"] and not log["picked"]


def test_resize_handles_change_size_and_corner_anchors_opposite(qtbot):
    viewer, rect, log = _selected_rect_viewer(qtbot, AnnotationKind.RECTANGLE)
    _press_move_release(qtbot, viewer, (60, 50), (80, 70))  # bottom-right handle
    _, x, y, w, h = log["geom"][-1]
    assert (round(x), round(y)) == (20, 20)
    assert abs(w - 60) <= 2 and abs(h - 50) <= 2
    assert not log["drawn"]


def test_dragging_past_the_opposite_edge_flips_instead_of_collapsing(qtbot):
    viewer, rect, log = _selected_rect_viewer(qtbot)
    _press_move_release(qtbot, viewer, (60, 50), (10, 10))
    _, x, y, w, h = log["geom"][-1]
    assert w > 0 and h > 0 and x < 20 and y < 20


def test_moves_are_clamped_to_the_image(qtbot):
    viewer, rect, log = _selected_rect_viewer(qtbot)
    _press_move_release(qtbot, viewer, (40, 35), (200, 200))
    _, x, y, w, h = log["geom"][-1]
    assert x + w <= 100.5 and y + h <= 100.5


def test_clicking_an_unselected_annotation_picks_it_without_a_tool(qtbot):
    viewer, rect, log = _selected_rect_viewer(qtbot)
    viewer.set_annotations((rect,), None)
    qtbot.mouseClick(viewer.viewport(), QtCore.Qt.MouseButton.LeftButton,
                     pos=viewer.mapFromScene(40, 35))
    assert log["picked"] == [rect.layer_id]
    qtbot.mouseClick(viewer.viewport(), QtCore.Qt.MouseButton.LeftButton,
                     pos=viewer.mapFromScene(95, 95))
    assert log["picked"] == [rect.layer_id]  # empty space picks nothing


def test_arrow_endpoints_move_independently(qtbot):
    viewer = _viewer(qtbot)
    model = AnnotationLayerModel()
    arrow = model.add_arrow(10, 10, 60, 60, STYLE)
    viewer.set_annotations(model.layers, arrow.layer_id)
    log = _record(viewer)
    _press_move_release(qtbot, viewer, (60, 60), (80, 40))  # drag the head
    _, x, y, w, h = log["geom"][-1]
    assert (round(x), round(y)) == (10, 10) and abs(w - 70) <= 2 and abs(h - 30) <= 2


def test_tool_buttons_toggle_off_and_are_only_as_wide_as_their_text(qtbot):
    page = AnnotationToolsPage()
    qtbot.addWidget(page)
    page.resize(236, 400)
    page.show()
    tools = []
    page.toolChanged.connect(tools.append)
    rect = page._tool_buttons[AnnotationKind.RECTANGLE]
    circle = page._tool_buttons[AnnotationKind.ELLIPSE]
    rect.click()
    circle.click()
    assert not rect.isChecked() and circle.isChecked()  # still exclusive
    circle.click()
    assert tools == [AnnotationKind.RECTANGLE, AnnotationKind.ELLIPSE, None]
    assert not any(b.isChecked() for b in page._tool_buttons.values())
    for button in page._tool_buttons.values():
        assert button.width() == button.sizeHint().width()
    assert (
        page._tool_buttons[AnnotationKind.RECTANGLE].width()
        != page._tool_buttons[AnnotationKind.ELLIPSE].width()
    )


def _log_deselect(viewer, log):
    log["deselected"] = 0
    viewer.annotationDeselected.connect(lambda: log.__setitem__("deselected", log["deselected"] + 1))


def test_clicking_empty_image_deselects_with_and_without_a_tool(qtbot):
    for tool in (None, AnnotationKind.RECTANGLE, AnnotationKind.ELLIPSE, AnnotationKind.ARROW):
        viewer, rect, log = _selected_rect_viewer(qtbot, tool)
        _log_deselect(viewer, log)
        qtbot.mouseClick(viewer.viewport(), QtCore.Qt.MouseButton.LeftButton,
                         pos=viewer.mapFromScene(90, 90))
        assert log["deselected"] == 1, tool
        assert not log["drawn"] and not log["geom"]


def test_dragging_on_empty_image_does_not_deselect(qtbot):
    viewer, rect, log = _selected_rect_viewer(qtbot, None)
    _log_deselect(viewer, log)
    _press_move_release(qtbot, viewer, (80, 80), (90, 95))  # a pan, not a click
    assert log["deselected"] == 0


def test_clicking_the_selected_annotation_does_not_deselect_it(qtbot):
    viewer, rect, log = _selected_rect_viewer(qtbot, None)
    _log_deselect(viewer, log)
    qtbot.mouseClick(viewer.viewport(), QtCore.Qt.MouseButton.LeftButton,
                     pos=viewer.mapFromScene(40, 35))
    assert log["deselected"] == 0


def test_unfilled_shape_is_hit_on_and_just_outside_its_stroke_not_inside(qtbot):
    for kind in (AnnotationKind.RECTANGLE, AnnotationKind.ELLIPSE):
        viewer = _viewer(qtbot)
        model = AnnotationLayerModel()
        layer = model.add_shape(kind, 20, 20, 60, 60, AnnotationStyle(stroke_width=2.0))
        viewer.set_annotations(model.layers, None)
        scale = viewer._view_scale()
        slack = 4 / scale
        left_mid = (20, 50)  # on the outline (leftmost point of both shapes)
        assert viewer._annotation_at(QPointF(*left_mid)) == layer.layer_id
        assert viewer._annotation_at(QPointF(20 - 1 - slack * 0.8, 50)) == layer.layer_id
        assert viewer._annotation_at(QPointF(20 - 1 - slack * 1.5, 50)) is None
        assert viewer._annotation_at(QPointF(50, 50)) is None  # hollow centre


def test_filled_shape_is_hit_everywhere_inside(qtbot):
    viewer = _viewer(qtbot)
    model = AnnotationLayerModel()
    layer = model.add_shape(AnnotationKind.ELLIPSE, 20, 20, 60, 60, FILLED)
    viewer.set_annotations(model.layers, None)
    assert viewer._annotation_at(QPointF(50, 50)) == layer.layer_id
    assert viewer._annotation_at(QPointF(2, 2)) is None  # outside the ellipse


def test_clicking_through_an_unfilled_shapes_centre_selects_what_is_beneath(qtbot):
    viewer = _viewer(qtbot)
    model = AnnotationLayerModel()
    below = model.add_shape(AnnotationKind.RECTANGLE, 30, 30, 20, 20, FILLED)
    model.add_shape(AnnotationKind.RECTANGLE, 10, 10, 80, 80, AnnotationStyle())
    viewer.set_annotations(model.layers, None)
    assert viewer._annotation_at(QPointF(40, 40)) == below.layer_id


def test_text_tool_click_on_empty_image_places_text_instead_of_deselecting(qtbot):
    viewer, rect, log = _selected_rect_viewer(qtbot, AnnotationKind.TEXT)
    _log_deselect(viewer, log)
    requested = []
    viewer.annotationTextRequested.connect(requested.append)
    qtbot.mouseClick(viewer.viewport(), QtCore.Qt.MouseButton.LeftButton,
                     pos=viewer.mapFromScene(90, 90))
    assert len(requested) == 1 and log["deselected"] == 0
