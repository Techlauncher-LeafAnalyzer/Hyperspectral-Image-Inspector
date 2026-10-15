from __future__ import annotations

import numpy as np
from PIL import Image
from PyQt6.QtCore import QRectF

from core import AnnotationKind, AnnotationLayerModel, AnnotationStyle
from ui.annotation_render import render_annotations

RED = (255, 0, 0)
BLUE = (0, 0, 255)


def _canvas(size: int = 40) -> np.ndarray:
    return np.full((size, size, 3), 100, dtype=np.uint8)


def _solid(color, **kw) -> AnnotationStyle:
    return AnnotationStyle(fill=color, fill_enabled=True, stroke=color, stroke_width=0, **kw)


def test_nothing_visible_returns_the_input_untouched(qtbot):
    rgb = _canvas()
    model = AnnotationLayerModel()
    layer = model.add_shape(AnnotationKind.RECTANGLE, 5, 5, 10, 10, _solid(RED))
    model.set_visible(layer.layer_id, False)
    assert render_annotations(rgb, model.layers) is rgb
    assert render_annotations(rgb, ()) is rgb


def test_filled_rectangle_is_drawn_at_image_pixels_and_input_is_not_modified(qtbot):
    rgb = _canvas()
    before = rgb.copy()
    model = AnnotationLayerModel()
    model.add_shape(AnnotationKind.RECTANGLE, 10, 10, 10, 10, _solid(RED))
    out = render_annotations(rgb, model.layers)
    assert out.shape == rgb.shape and out.dtype == np.uint8
    assert tuple(out[15, 15]) == RED
    assert tuple(out[5, 5]) == (100, 100, 100) and tuple(out[25, 25]) == (100, 100, 100)
    assert np.array_equal(rgb, before)


def test_unfilled_shape_leaves_its_centre_and_draws_the_stroke(qtbot):
    model = AnnotationLayerModel()
    model.add_shape(
        AnnotationKind.RECTANGLE, 5, 5, 30, 30,
        AnnotationStyle(stroke=RED, stroke_width=4.0, fill_enabled=False),
    )
    out = render_annotations(_canvas(), model.layers)
    assert tuple(out[20, 20]) == (100, 100, 100)  # hollow centre
    assert tuple(out[20, 5]) == RED  # on the left edge


def test_stacking_order_matches_layer_order(qtbot):
    model = AnnotationLayerModel()
    a = model.add_shape(AnnotationKind.RECTANGLE, 5, 5, 20, 20, _solid(RED)).layer_id
    model.add_shape(AnnotationKind.RECTANGLE, 10, 10, 20, 20, _solid(BLUE))
    assert tuple(render_annotations(_canvas(), model.layers)[15, 15]) == BLUE
    model.move(a, 1)  # red now on top
    assert tuple(render_annotations(_canvas(), model.layers)[15, 15]) == RED


def test_opacity_blends_with_the_image(qtbot):
    model = AnnotationLayerModel()
    model.add_shape(AnnotationKind.RECTANGLE, 5, 5, 20, 20, _solid(RED, opacity=0.5))
    px = render_annotations(_canvas(), model.layers)[15, 15].astype(int)
    assert abs(px[0] - (255 + 100) / 2) <= 2 and abs(px[1] - 50) <= 2


def test_arrow_and_text_leave_marks(qtbot):
    model = AnnotationLayerModel()
    model.add_arrow(4, 20, 36, 20, AnnotationStyle(stroke=RED, stroke_width=3.0))
    model.add_text(2, 2, "Hi", AnnotationStyle(fill=BLUE, stroke_width=0, font_size=16))
    out = render_annotations(_canvas(), model.layers)
    assert (out[18:23, 8:30] == RED).all(axis=-1).any()  # shaft
    assert (out[0:20, 0:30] == BLUE).all(axis=-1).any()  # glyph pixels
    assert not np.array_equal(out, _canvas())


def test_selection_outline_and_handles_are_never_exported(qtbot):
    model = AnnotationLayerModel()
    model.add_shape(AnnotationKind.RECTANGLE, 10, 10, 10, 10, _solid(RED))
    out = render_annotations(_canvas(), model.layers)
    changed = np.argwhere((out != _canvas()).any(axis=-1))
    assert changed.min() >= 9 and changed.max() <= 20  # nothing outside the shape


def _annotate_and_save(window, tmp_path, file_dialog, style):
    window.tabWidget.setCurrentWidget(window.Classification)
    window.classificationViewer.annotationShapeDrawn.emit(
        AnnotationKind.RECTANGLE, QRectF(1, 1, 4, 4)
    )
    layer = window._classification_controller._annotation_slot.layers.layers[-1]
    window._classification_controller._annotation_slot.layers.set_style(layer.layer_id, style)
    target = tmp_path / "annotated.png"
    file_dialog.save_return = (str(target), "")
    window.actionSaveImage.trigger()
    return np.array(Image.open(target).convert("RGB"))


def test_save_image_includes_annotations_on_the_classification_tab(
    loaded_window, tmp_path, file_dialog
):
    saved = _annotate_and_save(loaded_window, tmp_path, file_dialog, _solid(RED))
    plain = loaded_window._display_data().rgb_array
    assert saved.shape == plain.shape
    assert tuple(saved[3, 3]) == RED
    assert not np.array_equal(saved, plain)


def test_save_image_respects_hidden_annotations_and_other_tabs(
    loaded_window, tmp_path, file_dialog
):
    window = loaded_window
    window.tabWidget.setCurrentWidget(window.Classification)
    window.classificationViewer.annotationShapeDrawn.emit(
        AnnotationKind.RECTANGLE, QRectF(1, 1, 4, 4)
    )
    controller = window._classification_controller
    layers = controller._annotation_slot.layers
    layers.set_style(layers.layers[-1].layer_id, _solid(RED))

    # Only the Classification tab's save carries its annotations.
    window.tabWidget.setCurrentWidget(window.tabWidget.widget(0))
    other = tmp_path / "other.png"
    file_dialog.save_return = (str(other), "")
    window.actionSaveImage.trigger()
    expected = window._visualization_results[window._active_visualization_mode].display_rgb
    assert np.array_equal(np.array(Image.open(other).convert("RGB")), expected)

    window.tabWidget.setCurrentWidget(window.Classification)
    layers.set_visible(layers.layers[-1].layer_id, False)
    hidden = tmp_path / "hidden.png"
    file_dialog.save_return = (str(hidden), "")
    window.actionSaveImage.trigger()
    assert np.array_equal(
        np.array(Image.open(hidden).convert("RGB")), window._display_data().rgb_array
    )


def test_save_image_after_classification_draws_annotations_over_the_class_colours(
    loaded_window, tmp_path, file_dialog, qtbot
):
    window = loaded_window
    window.numOfClassesEdit.setText("2")
    window.maxIterationsEdit.setText("3")
    window.unsupervisedClassifyButton.click()
    qtbot.waitUntil(lambda: not window._classification_controller.is_running(), timeout=5000)
    window.tabWidget.setCurrentWidget(window.Classification)
    classified = window._classification_controller.composited_rgb()
    assert classified is not None

    saved = _annotate_and_save(window, tmp_path, file_dialog, _solid(RED))
    assert tuple(saved[3, 3]) == RED
    outside = np.ones(saved.shape[:2], dtype=bool)
    outside[1:5, 1:5] = False
    assert np.array_equal(saved[outside], classified[outside])  # class colours intact elsewhere
    # And the composite itself stays annotation-free, so the viewer isn't doubled up.
    assert tuple(window._classification_controller.composited_rgb()[3, 3]) != RED


def test_rotated_rectangle_is_exported_rotated(qtbot):
    model = AnnotationLayerModel()
    layer = model.add_shape(AnnotationKind.RECTANGLE, 5, 17, 30, 6, _solid(RED))
    flat = render_annotations(_canvas(), model.layers)
    model.set_rotation(layer.layer_id, 90)
    turned = render_annotations(_canvas(), model.layers)
    red = lambda img: (img == RED).all(axis=-1)
    flat_rows, flat_cols = np.where(red(flat))
    rows, cols = np.where(red(turned))
    assert np.ptp(flat_cols) > np.ptp(flat_rows)          # wide before
    assert np.ptp(rows) > np.ptp(cols)                     # tall after
    assert abs((rows.min() + rows.max()) - (flat_rows.min() + flat_rows.max())) <= 1  # same centre
    assert abs((cols.min() + cols.max()) - (flat_cols.min() + flat_cols.max())) <= 1


def test_rotated_text_is_exported_rotated(qtbot):
    model = AnnotationLayerModel()
    style = AnnotationStyle(fill=RED, stroke_width=0, font_size=16)
    layer = model.add_text(2, 14, "Hello", style)
    flat = render_annotations(_canvas(60), model.layers)
    model.set_rotation(layer.layer_id, 90)
    turned = render_annotations(_canvas(60), model.layers)
    red = lambda img: np.argwhere((img == RED).all(axis=-1))
    f, t = red(flat), red(turned)
    assert np.ptp(f[:, 1]) > np.ptp(f[:, 0]) and np.ptp(t[:, 0]) > np.ptp(t[:, 1])
