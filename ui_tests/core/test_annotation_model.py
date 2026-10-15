from __future__ import annotations

import pytest

from core import AnnotationError, AnnotationKind, AnnotationLayerModel, AnnotationStyle

STYLE = AnnotationStyle()


def _model_with_three() -> tuple[AnnotationLayerModel, list[int]]:
    model = AnnotationLayerModel()
    ids = [
        model.add_shape(AnnotationKind.RECTANGLE, 0, 0, 5, 5, STYLE).layer_id,
        model.add_shape(AnnotationKind.ELLIPSE, 1, 1, 5, 5, STYLE).layer_id,
        model.add_text(2, 2, "hello", STYLE).layer_id,
    ]
    return model, ids


def test_layers_are_stacked_in_creation_order_with_unique_names():
    model = AnnotationLayerModel()
    first = model.add_shape(AnnotationKind.RECTANGLE, 0, 0, 5, 5, STYLE)
    second = model.add_shape(AnnotationKind.RECTANGLE, 0, 0, 5, 5, STYLE)
    assert [layer.name for layer in model.layers] == ["Rectangle 1", "Rectangle 2"]
    assert model.layers[-1].layer_id == second.layer_id != first.layer_id


def test_move_reorders_and_clamps_at_the_ends():
    model, (rect, ellipse, text) = _model_with_three()
    assert model.move(rect, 1)
    assert [l.layer_id for l in model.layers] == [ellipse, rect, text]
    assert model.move(rect, 5)
    assert [l.layer_id for l in model.layers] == [ellipse, text, rect]
    assert not model.move(rect, 1)
    assert model.move(rect, -9)
    assert model.layers[0].layer_id == rect


def test_ids_are_not_reused_after_removal():
    model, (rect, *_rest) = _model_with_three()
    model.remove(rect)
    assert model.add_shape(AnnotationKind.ELLIPSE, 0, 0, 2, 2, STYLE).layer_id not in _rest + [rect]


def test_style_and_visibility_updates_are_per_layer():
    model, (rect, ellipse, _text) = _model_with_three()
    model.set_style(rect, AnnotationStyle(stroke_width=9.0))
    model.set_visible(ellipse, False)
    assert model.get(rect).style.stroke_width == 9.0
    assert model.get(ellipse).style.stroke_width == STYLE.stroke_width
    assert not model.get(ellipse).visible and model.get(rect).visible


def test_text_editing_is_text_only_and_non_empty():
    model, (rect, _e, text) = _model_with_three()
    model.set_text(text, "changed")
    assert model.get(text).text == "changed"
    with pytest.raises(AnnotationError):
        model.set_text(rect, "x")
    with pytest.raises(AnnotationError):
        model.set_text(text, "   ")


@pytest.mark.parametrize("w,h", [(0, 5), (5, 0), (-1, 5)])
def test_degenerate_shapes_are_rejected(w, h):
    with pytest.raises(AnnotationError):
        AnnotationLayerModel().add_shape(AnnotationKind.RECTANGLE, 0, 0, w, h, STYLE)


def test_unknown_layer_and_bad_style_raise():
    model = AnnotationLayerModel()
    with pytest.raises(AnnotationError):
        model.remove(99)
    assert model.selected_or_none(99) is None
    with pytest.raises(AnnotationError):
        AnnotationStyle(stroke_width=-1)


def test_arrow_stores_signed_extent_and_rejects_zero_length():
    model = AnnotationLayerModel()
    arrow = model.add_arrow(10, 10, 4, 20, STYLE)
    assert (arrow.width, arrow.height) == (-6, 10) and arrow.name == "Arrow 1"
    with pytest.raises(AnnotationError):
        model.add_arrow(3, 3, 3, 3, STYLE)
    with pytest.raises(AnnotationError):
        model.add_shape(AnnotationKind.ARROW, 0, 0, 5, 5, STYLE)


def test_opacity_is_validated_and_per_layer():
    with pytest.raises(AnnotationError):
        AnnotationStyle(opacity=1.5)
    model, (rect, ellipse, _t) = _model_with_three()
    model.set_style(rect, AnnotationStyle(opacity=0.25))
    assert model.get(rect).style.opacity == 0.25 and model.get(ellipse).style.opacity == 1.0


def test_undo_redo_restores_each_kind_of_change():
    model, (rect, ellipse, text) = _model_with_three()
    assert model.undo() and len(model) == 2  # undo add_text
    assert model.redo() and len(model) == 3
    model.remove(rect)
    assert model.undo() and model.get(rect)
    model.move(rect, 2)
    assert model.undo() and [l.layer_id for l in model.layers] == [rect, ellipse, text]
    model.set_visible(rect, False)
    assert model.undo() and model.get(rect).visible


def test_new_change_clears_redo_and_empty_history_is_noop():
    model = AnnotationLayerModel()
    assert not model.undo() and not model.redo()
    model.add_shape(AnnotationKind.RECTANGLE, 0, 0, 5, 5, STYLE)
    model.undo()
    assert model.can_redo
    model.add_shape(AnnotationKind.ELLIPSE, 0, 0, 5, 5, STYLE)
    assert not model.can_redo


def test_slider_drag_collapses_to_one_undo_step_but_other_fields_do_not():
    model = AnnotationLayerModel()
    rect = model.add_shape(AnnotationKind.RECTANGLE, 0, 0, 5, 5, STYLE).layer_id
    for value in (0.9, 0.7, 0.5):
        model.set_style(rect, AnnotationStyle(opacity=value))
    model.set_style(rect, AnnotationStyle(opacity=0.5, stroke_width=8.0))
    assert model.undo() and model.get(rect).style.stroke_width == STYLE.stroke_width
    assert model.get(rect).style.opacity == 0.5
    assert model.undo() and model.get(rect).style.opacity == 1.0
    model.set_style(rect, AnnotationStyle(opacity=1.0))  # unchanged: no history
    assert model.undo() and len(model) == 0


def test_set_geometry_moves_resizes_and_validates():
    model, (rect, _e, text) = _model_with_three()
    model.set_geometry(rect, 3, 4, 10, 12)
    layer = model.get(rect)
    assert (layer.x, layer.y, layer.width, layer.height) == (3, 4, 10, 12)
    model.set_geometry(text, 7, 8, 99, 99)  # text keeps no extent
    assert (model.get(text).x, model.get(text).width) == (7, 0.0)
    with pytest.raises(AnnotationError):
        model.set_geometry(rect, 0, 0, 0, 5)
    arrow = model.add_arrow(0, 0, 5, 5, STYLE).layer_id
    model.set_geometry(arrow, 5, 5, -5, -5)
    with pytest.raises(AnnotationError):
        model.set_geometry(arrow, 5, 5, 0, 0)


def test_a_drag_of_many_geometry_updates_is_one_undo_step():
    model = AnnotationLayerModel()
    rect = model.add_shape(AnnotationKind.RECTANGLE, 0, 0, 5, 5, STYLE).layer_id
    for x in (1, 2, 3, 4):
        model.set_geometry(rect, x, 0, 5, 5)
    model.break_coalescing()
    model.set_geometry(rect, 9, 0, 5, 5)  # a second drag
    assert model.undo() and model.get(rect).x == 4
    assert model.undo() and model.get(rect).x == 0
    model.set_geometry(rect, 0, 0, 5, 5)  # unchanged: no history
    assert model.undo() and len(model) == 0
