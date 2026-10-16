from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QFontDatabase, QPixmap

from core import AnnotationKind, AnnotationLayerModel, AnnotationStyle
from ui.annotation_panel import AnnotationToolsPage
from ui.classification_layer_panel import ClassificationLayerPanel
from ui.theme import apply_theme
from ui.viewer import HSIViewer

STYLE = AnnotationStyle()


@pytest.fixture(params=["system", "DejaVu Sans"])
def themed_app(qapp, request):
    previous_style = qapp.style().objectName()
    previous_font = qapp.font()
    previous_stylesheet = qapp.styleSheet()
    apply_theme(qapp)
    font_ids = []
    if request.param == "DejaVu Sans":
        # Use Matplotlib's bundled fonts to reproduce Ubuntu's font metrics
        # on macOS and Windows without adding a dependency or downloading fonts.
        font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        for filename in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
            font_id = QFontDatabase.addApplicationFont(str(font_dir / filename))
            assert font_id >= 0
            font_ids.append(font_id)
        font = qapp.font()
        font.setFamily("DejaVu Sans")
        qapp.setFont(font)
    yield qapp
    qapp.setStyleSheet(previous_stylesheet)
    qapp.setStyle(previous_style)
    qapp.setFont(previous_font)
    for font_id in font_ids:
        QFontDatabase.removeApplicationFont(font_id)


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
    assert len(viewer._annotation_items) == 4  # 2 visible + outline + text rotate handle


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
    _drag_handle(qtbot, viewer, "br", 20, 20)
    _, x, y, w, h = log["geom"][-1]
    assert (round(x), round(y)) == (20, 20)
    assert abs(w - 60) <= 2 and abs(h - 50) <= 2
    assert not log["drawn"]


def test_dragging_past_the_opposite_edge_flips_instead_of_collapsing(qtbot):
    viewer, rect, log = _selected_rect_viewer(qtbot)
    _drag_handle(qtbot, viewer, "br", -50, -40)
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


@pytest.mark.parametrize("width,height", [(236, 450), (236, 750), (520, 850)])
def test_tool_buttons_toggle_off_and_fit_the_inspector(qtbot, themed_app, width, height):
    panel = ClassificationLayerPanel()
    qtbot.addWidget(panel)
    panel.set_expanded_width(width)
    panel.resize(width, height)
    panel.set_mode("annotations")
    page = panel.annotation_page
    page.set_layers(())
    panel.show()
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
    scroll = page.findChild(QtWidgets.QScrollArea)
    qtbot.waitUntil(lambda: scroll.widget().width() <= scroll.viewport().width())
    for button in page._tool_buttons.values():
        assert not button.icon().isNull()
        assert button.width() >= button.minimumSizeHint().width()
    arrow = page._tool_buttons[AnnotationKind.ARROW]
    text = page._tool_buttons[AnnotationKind.TEXT]
    # Qt respects each label's minimum width before sharing remaining space;
    # wider fonts may need unequal columns to keep both labels readable.
    assert rect.x() == arrow.x()
    assert circle.x() == text.x()
    assert rect.y() == circle.y()
    assert arrow.y() == text.y()
    assert rect.geometry().right() < circle.x()
    assert rect.geometry().bottom() < arrow.y()

    hint = next(label for label in page.findChildren(QtWidgets.QLabel)
                if label.text().startswith("Click the active tool"))
    appearance = page.findChild(QtWidgets.QFrame, "annotationAppearance")
    assert hint.height() >= hint.heightForWidth(hint.width())
    assert hint.geometry().bottom() < appearance.y()
    assert appearance.height() >= appearance.minimumSizeHint().height()
    assert page._fill_button.geometry().bottom() < page._fill_check.y()
    assert page._fill_check.height() >= page._fill_check.minimumSizeHint().height()
    assert page._fill_check.geometry().bottom() < page._stroke_width.y()


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


def _rotated_viewer(qtbot, rotation, kind=AnnotationKind.RECTANGLE):
    viewer = _viewer(qtbot)
    model = AnnotationLayerModel()
    layer = model.add_shape(kind, 30, 40, 40, 10, FILLED)  # wide, short box
    if rotation:
        model.set_rotation(layer.layer_id, rotation)
    viewer.set_annotations(model.layers, layer.layer_id)
    return viewer, model, layer, _record(viewer)


def test_hit_testing_follows_the_rotated_shape(qtbot):
    viewer, model, layer, _ = _rotated_viewer(qtbot, 90)
    # Centre (50, 45); at 90 degrees the 40x10 box becomes 10 wide, 40 tall.
    assert viewer._annotation_at(QPointF(50, 28)) == layer.layer_id   # now inside (was outside)
    assert viewer._annotation_at(QPointF(50, 62)) == layer.layer_id
    assert viewer._annotation_at(QPointF(33, 45)) is None             # was inside before rotating


def test_rotate_handle_exists_for_rectangle_and_text_but_not_ellipse_or_arrow(qtbot):
    viewer = _viewer(qtbot)
    model = AnnotationLayerModel()
    ids = {
        "rect": model.add_shape(AnnotationKind.RECTANGLE, 10, 40, 30, 20, STYLE).layer_id,
        "ellipse": model.add_shape(AnnotationKind.ELLIPSE, 10, 40, 30, 20, STYLE).layer_id,
        "arrow": model.add_arrow(10, 10, 50, 10, STYLE).layer_id,
        "text": model.add_text(10, 10, "hi", STYLE).layer_id,
    }
    present = {}
    for name, layer_id in ids.items():
        viewer.set_annotations(model.layers, layer_id)
        present[name] = "rotate" in viewer._annotation_handles(viewer._selected_layer)
    assert present == {"rect": True, "ellipse": False, "arrow": False, "text": True}


def test_dragging_the_rotate_handle_emits_the_angle_swept(qtbot):
    viewer, model, layer, log = _rotated_viewer(qtbot, 0)
    rotations = []
    viewer.annotationRotationChanged.connect(lambda i, d: rotations.append((i, d)))
    handle = viewer._annotation_handles(viewer._selected_layer)["rotate"]
    assert handle.y() < 40  # above the top edge
    _press_move_release(qtbot, viewer, (handle.x(), handle.y()), (80, 45))  # to the right of centre
    layer_id, degrees = rotations[-1]
    assert layer_id == layer.layer_id and abs(degrees - 90) <= 3
    assert not log["geom"] and log["finished"] == 1  # a rotation, not a move


def test_shift_snaps_rotation_to_fifteen_degrees(qtbot):
    viewer, model, layer, log = _rotated_viewer(qtbot, 0)
    rotations = []
    viewer.annotationRotationChanged.connect(lambda i, d: rotations.append(d))
    handle = viewer._annotation_handles(viewer._selected_layer)["rotate"]
    vp, left = viewer.viewport(), QtCore.Qt.MouseButton.LeftButton
    shift = QtCore.Qt.KeyboardModifier.ShiftModifier
    qtbot.mousePress(vp, left, pos=viewer.mapFromScene(handle.x(), handle.y()))
    qtbot.mouseMove(vp, viewer.mapFromScene(75, 38))  # an awkward angle
    # QTest's mouseMove carries no modifiers, so drive the move directly.
    viewer._drag_annotation(QPointF(75, 38), QPointF(75, 38), snap=True)
    qtbot.mouseRelease(vp, left, pos=viewer.mapFromScene(75, 38))
    assert rotations[-1] % 15 == 0 and rotations[-1] != rotations[0]


def test_resizing_a_rotated_rectangle_follows_its_own_edges(qtbot):
    viewer, model, layer, log = _rotated_viewer(qtbot, 90)
    # Centre (50, 45). Rotated 90 degrees clockwise, the box's right edge ("r",
    # originally at x=70) now sits below the selection border, at (50, 68).
    handles = viewer._annotation_handles(viewer._selected_layer)
    r = handles["r"]
    assert abs(r.x() - 50) < 0.5 and abs(r.y() - 68) < 0.5
    _drag_handle(qtbot, viewer, "r", 0, 10)  # drag it 10px further out
    _, x, y, w, h = log["geom"][-1]
    assert abs(w - 50) <= 2 and abs(h - 10) <= 0.5  # grew along its own width only
    # the opposite (left) edge stayed put on screen: it is at the top, (50, 25)
    new_center_y = y + h / 2
    assert abs((new_center_y - w / 2) - 25) <= 2


def test_rotated_text_is_hit_and_has_a_rotate_handle(qtbot):
    viewer = _viewer(qtbot)
    model = AnnotationLayerModel()
    text = model.add_text(10, 40, "Hello", AnnotationStyle(font_size=14))
    model.set_rotation(text.layer_id, 90)
    viewer.set_annotations(model.layers, text.layer_id)
    rect = viewer._unrotated_rect(viewer._selected_layer)
    assert rect.width() > rect.height()
    c = rect.center()
    assert viewer._annotation_at(QPointF(c.x(), c.y() - rect.width() / 2 + 2)) == text.layer_id
    # Stay outside the rotated text plus its hit margin for every font.
    outside_x = c.x() - rect.height() / 2 - 4 / viewer._view_scale() - 2
    assert viewer._annotation_at(QPointF(outside_x, c.y())) is None



def _drag_handle(qtbot, viewer, name, dx, dy):
    handle = viewer._annotation_handles(viewer._selected_layer)[name]
    _press_move_release(qtbot, viewer, (handle.x(), handle.y()),
                        (handle.x() + dx, handle.y() + dy))


def _hover(qtbot, viewer, x, y):
    qtbot.mouseMove(viewer.viewport(), viewer.mapFromScene(x, y))


@pytest.mark.parametrize("kind,rotation", [
    (AnnotationKind.RECTANGLE, 0), (AnnotationKind.RECTANGLE, 35),
    (AnnotationKind.ELLIPSE, 0),
])
@pytest.mark.parametrize("scale", [1, 4])
def test_resize_handles_are_on_the_drawn_selection_border(qtbot, kind, rotation, scale):
    viewer, model, layer, _ = _rotated_viewer(qtbot, rotation, kind)
    viewer.resetTransform()
    viewer.scale(scale, scale)
    handles = viewer._annotation_handles(viewer._selected_layer)
    outline = next(item for item in viewer._annotation_items
                   if isinstance(item, QtWidgets.QGraphicsPolygonItem)).polygon()
    corners = [outline[i] for i in range(4)]
    for name, expected in zip(("tl", "tr", "br", "bl"), corners):
        assert QtCore.QLineF(handles[name], expected).length() < 1e-6
    for name, a, b in (("t", 0, 1), ("r", 1, 2), ("b", 2, 3), ("l", 3, 0)):
        assert QtCore.QLineF(handles[name], (corners[a] + corners[b]) / 2).length() < 1e-6
    drawn_handles = [item for item in viewer._annotation_items
                     if isinstance(item, QtWidgets.QGraphicsRectItem)
                     and item.flags() & item.GraphicsItemFlag.ItemIgnoresTransformations
                     and item.rect().width() == 8]
    assert len(drawn_handles) == 8
    for item in drawn_handles:
        assert item.pos() in handles.values()
        assert viewer._selected_annotation_hit(item.pos()) in handles
        assert item.rect().size() == QtCore.QSizeF(8, 8)


def test_pressing_a_border_handle_does_not_change_geometry_until_dragged(qtbot):
    viewer, rect, log = _selected_rect_viewer(qtbot)
    handle = viewer._annotation_handles(rect)["br"]
    # An off-centre press still preserves the grab offset.
    press = handle + QPointF(1 / viewer._view_scale(), 0)
    bounds = viewer._annotation_bounds(rect)
    viewer._annotation_interaction = (rect.layer_id, "br", press, rect, bounds)
    viewer._drag_annotation(press, press)
    assert log["geom"][-1][1:] == (rect.x, rect.y, rect.width, rect.height)
    viewer._drag_annotation(press + QPointF(10, 5), press + QPointF(10, 5))
    assert log["geom"][-1][1:] == (rect.x, rect.y, rect.width + 10, rect.height + 5)


def test_border_handle_resizing_stays_within_the_image(qtbot):
    viewer, rect, log = _selected_rect_viewer(qtbot)
    _drag_handle(qtbot, viewer, "br", 150, 150)
    _, x, y, w, h = log["geom"][-1]
    assert x + w == 100 and y + h == 100


def test_cursor_updates_immediately_when_selecting_deselecting_and_reselecting(qtbot):
    viewer, rect, log = _selected_rect_viewer(qtbot)
    viewer.set_annotations((rect,), None)
    viewer.annotationPicked.connect(lambda selected: viewer.set_annotations((rect,), selected))
    viewer.annotationDeselected.connect(lambda: viewer.set_annotations((rect,), None))
    cursor = QtCore.Qt.CursorShape
    _hover(qtbot, viewer, 40, 35)
    assert viewer.viewport().cursor().shape() == cursor.PointingHandCursor
    qtbot.mouseClick(viewer.viewport(), QtCore.Qt.MouseButton.LeftButton,
                     pos=viewer.mapFromScene(40, 35))
    assert viewer.viewport().cursor().shape() == cursor.SizeAllCursor
    # Panel deselection/reselection also refreshes a stationary pointer.
    viewer.set_annotations((rect,), None)
    assert viewer.viewport().cursor().shape() == cursor.PointingHandCursor
    viewer.set_annotations((rect,), rect.layer_id)
    assert viewer.viewport().cursor().shape() == cursor.SizeAllCursor
    qtbot.mouseClick(viewer.viewport(), QtCore.Qt.MouseButton.LeftButton,
                     pos=viewer.mapFromScene(90, 90))
    assert viewer.viewport().cursor().shape() == cursor.OpenHandCursor


@pytest.mark.parametrize("tool", [None, AnnotationKind.ELLIPSE])
def test_move_cursor_is_locked_during_drag_and_restored_on_release(qtbot, tool):
    viewer, rect, log = _selected_rect_viewer(qtbot, tool)
    model = AnnotationLayerModel()
    layer = model.add_shape(AnnotationKind.RECTANGLE, 20, 20, 40, 30, FILLED)
    def update_geometry(layer_id, *geometry):
        model.set_geometry(layer_id, *geometry)
        viewer.set_annotations(model.layers, layer_id)

    viewer.annotationGeometryChanged.connect(update_geometry)
    cursor = QtCore.Qt.CursorShape
    _hover(qtbot, viewer, 40, 35)
    assert viewer.viewport().cursor().shape() == cursor.SizeAllCursor
    qtbot.mousePress(viewer.viewport(), QtCore.Qt.MouseButton.LeftButton,
                     pos=viewer.mapFromScene(40, 35))
    assert viewer.viewport().cursor().shape() == cursor.ClosedHandCursor
    _hover(qtbot, viewer, 45, 40)
    assert viewer.viewport().cursor().shape() == cursor.ClosedHandCursor
    qtbot.mouseRelease(viewer.viewport(), QtCore.Qt.MouseButton.LeftButton,
                       pos=viewer.mapFromScene(45, 40))
    assert viewer.viewport().cursor().shape() == cursor.SizeAllCursor


@pytest.mark.parametrize("rotation,handle,expected", [
    (0, "r", QtCore.Qt.CursorShape.SizeHorCursor),
    (0, "t", QtCore.Qt.CursorShape.SizeVerCursor),
    (0, "br", QtCore.Qt.CursorShape.SizeFDiagCursor),
    (0, "tr", QtCore.Qt.CursorShape.SizeBDiagCursor),
    (90, "r", QtCore.Qt.CursorShape.SizeVerCursor),
    (45, "r", QtCore.Qt.CursorShape.SizeFDiagCursor),
    (45, "br", QtCore.Qt.CursorShape.SizeVerCursor),
])
def test_resize_cursor_matches_rotated_handle_and_persists_during_drag(
    qtbot, rotation, handle, expected
):
    viewer, model, layer, log = _rotated_viewer(qtbot, rotation)
    p = viewer._annotation_handles(viewer._selected_layer)[handle]
    _hover(qtbot, viewer, p.x(), p.y())
    assert viewer.viewport().cursor().shape() == expected
    qtbot.mousePress(viewer.viewport(), QtCore.Qt.MouseButton.LeftButton,
                     pos=viewer.mapFromScene(p))
    assert viewer.viewport().cursor().shape() == expected
    _hover(qtbot, viewer, 90, 90)
    assert viewer.viewport().cursor().shape() == expected
    qtbot.mouseRelease(viewer.viewport(), QtCore.Qt.MouseButton.LeftButton,
                       pos=viewer.mapFromScene(90, 90))
    assert viewer.viewport().cursor().shape() == QtCore.Qt.CursorShape.OpenHandCursor


def test_tool_and_ctrl_cursor_restore_the_hover_action(qtbot):
    viewer, rect, log = _selected_rect_viewer(qtbot)
    cursor = QtCore.Qt.CursorShape
    _hover(qtbot, viewer, 40, 35)
    qtbot.keyPress(viewer, QtCore.Qt.Key.Key_Control)
    assert viewer.viewport().cursor().shape() == cursor.CrossCursor
    qtbot.keyRelease(viewer, QtCore.Qt.Key.Key_Control)
    assert viewer.viewport().cursor().shape() == cursor.SizeAllCursor
    viewer.set_annotation_tool(AnnotationKind.ELLIPSE)
    assert viewer.viewport().cursor().shape() == cursor.SizeAllCursor
    _hover(qtbot, viewer, 90, 90)
    assert viewer.viewport().cursor().shape() == cursor.CrossCursor
    viewer.set_annotation_tool(None)
    assert viewer.viewport().cursor().shape() == cursor.OpenHandCursor
    viewer._begin_crop_mode()
    assert viewer.viewport().cursor().shape() == cursor.CrossCursor
    viewer._end_crop_mode()
    assert viewer.viewport().cursor().shape() == cursor.OpenHandCursor


def test_clicking_another_annotation_reselects_it_with_a_drawing_tool(qtbot):
    viewer, rect, log = _selected_rect_viewer(qtbot, AnnotationKind.ELLIPSE)
    model = AnnotationLayerModel()
    first = model.add_shape(AnnotationKind.RECTANGLE, 10, 10, 20, 20, FILLED)
    second = model.add_shape(AnnotationKind.RECTANGLE, 60, 60, 20, 20, FILLED)
    viewer.set_annotations(model.layers, first.layer_id)
    viewer.annotationPicked.connect(lambda selected: viewer.set_annotations(model.layers, selected))
    qtbot.mouseClick(viewer.viewport(), QtCore.Qt.MouseButton.LeftButton,
                     pos=viewer.mapFromScene(70, 70))
    assert log["picked"] == [second.layer_id] and not log["drawn"]
    assert viewer.viewport().cursor().shape() == QtCore.Qt.CursorShape.SizeAllCursor



def test_annotation_spin_box_arrows_are_visible_and_change_values(qtbot, themed_app):
    panel = ClassificationLayerPanel()
    qtbot.addWidget(panel)
    panel.set_expanded_width(236)
    panel.resize(236, 620)
    panel.set_mode("annotations")
    panel.show()
    qtbot.wait(10)
    page = panel.annotation_page
    for spin in (page._stroke_width, page._font_size):
        option = QtWidgets.QStyleOptionSpinBox()
        spin.initStyleOption(option)
        for control, change in (
            (QtWidgets.QStyle.SubControl.SC_SpinBoxUp, 1),
            (QtWidgets.QStyle.SubControl.SC_SpinBoxDown, -1),
        ):
            rect = spin.style().subControlRect(
                QtWidgets.QStyle.ComplexControl.CC_SpinBox, option, control, spin
            )
            assert not rect.isEmpty() and spin.rect().contains(rect)
            # Check the rendered glyph, excluding button borders/background.
            image = spin.grab().toImage()
            ratio = image.devicePixelRatio()
            center = rect.center()
            pixels = [image.pixelColor(round(x * ratio), round(y * ratio))
                      for x in range(center.x() - 4, center.x() + 5)
                      for y in range(center.y() - 2, center.y() + 3)]
            assert sum(c.red() < 140 and c.green() < 160 and c.blue() < 160
                       for c in pixels) >= 3
            previous = spin.value()
            with qtbot.waitSignal(page.styleChanged) as changed:
                qtbot.mouseClick(spin, QtCore.Qt.MouseButton.LeftButton, pos=center)
            assert spin.value() == previous + change
            style = changed.args[0]
            assert style.stroke_width == page._stroke_width.value()
            assert style.font_size == page._font_size.value()
        spin.setValue(spin.maximum())
        editor = spin.lineEdit()
        assert editor.contentsRect().width() >= editor.fontMetrics().horizontalAdvance(spin.text())



def test_measure_default_checkmark_is_visible_while_disabled(qtbot, themed_app):
    page = AnnotationToolsPage()
    qtbot.addWidget(page)
    page.show()
    check = page._measure_check
    assert check.isChecked() and not check.isEnabled()
    checked = check.grab().toImage()
    check.setChecked(False)
    unchecked = check.grab().toImage()
    check.setChecked(True)
    assert checked != unchecked
    page.set_layers(())
    assert check.isChecked()
