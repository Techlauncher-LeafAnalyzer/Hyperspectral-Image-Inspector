from __future__ import annotations

import numpy as np
import pytest
from PyQt6 import QtCore

from core import SpectrumTransform


@pytest.fixture
def inspector(loaded_window):
    controller = loaded_window._spectrum_controller
    controller.show()
    # Follow cursor starts off; most tests exercise the live trace.
    controller.inspector._follow_check.setChecked(True)
    return controller.inspector


def _classify(window, qtbot):
    window.numOfClassesEdit.setText("3")
    window.maxIterationsEdit.setText("5")
    window.unsupervisedClassifyButton.click()
    qtbot.waitUntil(lambda: not window._classification_controller.is_running(), timeout=10000)


def test_hover_updates_live_trace_from_any_viewer(loaded_window, inspector):
    data = loaded_window._display_data()
    for viewer in (loaded_window.viewer, loaded_window.classificationViewer):
        viewer.pixelHovered.emit(3, 5)
        np.testing.assert_array_equal(inspector._live.values, data.read_pixel(3, 5))
        viewer.pixelHovered.emit(1, 2)
        np.testing.assert_array_equal(inspector._live.values, data.read_pixel(1, 2))


def test_follow_cursor_starts_off(loaded_window):
    controller = loaded_window._spectrum_controller
    controller.show()
    assert not controller.inspector.is_following()
    loaded_window.viewer.pixelHovered.emit(3, 5)
    assert controller.inspector._live is None


def _send_ctrl(viewer, kind):
    from PyQt6 import QtGui, QtWidgets

    QtWidgets.QApplication.sendEvent(viewer, QtGui.QKeyEvent(
        kind, QtCore.Qt.Key.Key_Control, QtCore.Qt.KeyboardModifier.ControlModifier
    ))


def test_holding_ctrl_follows_the_cursor_while_the_panel_is_open(loaded_window):
    controller = loaded_window._spectrum_controller
    inspector = controller.inspector
    viewer = loaded_window.viewer
    controller.show()
    _send_ctrl(viewer, QtCore.QEvent.Type.KeyPress)
    assert inspector.is_following() and not inspector._follow_check.isChecked()
    viewer.pixelHovered.emit(3, 5)
    assert (inspector._live.row, inspector._live.column) == (3, 5)
    assert inspector._live_line is not None

    _send_ctrl(viewer, QtCore.QEvent.Type.KeyRelease)
    assert not inspector.is_following()
    assert inspector._live_line is None
    viewer.pixelHovered.emit(1, 2)
    assert (inspector._live.row, inspector._live.column) == (3, 5)

    # Leaving the image counts as a release (the key-up may go elsewhere).
    _send_ctrl(viewer, QtCore.QEvent.Type.KeyPress)
    assert inspector.is_following()
    viewer.leaveEvent(QtCore.QEvent(QtCore.QEvent.Type.Leave))
    assert not inspector.is_following()

    # A closed panel ignores Ctrl.
    loaded_window.spectrumTool.hide()
    _send_ctrl(viewer, QtCore.QEvent.Type.KeyPress)
    assert not inspector.is_following()
    _send_ctrl(viewer, QtCore.QEvent.Type.KeyRelease)


def test_viewer_shows_a_pointer_while_following(loaded_window, inspector):
    viewer = loaded_window.viewer
    viewer.resize(400, 400)
    viewer.fit_in_view()
    viewer._pointer_view_pos = viewer.mapFromScene(QtCore.QPointF(5.5, 3.5))
    viewer._apply_idle_cursor()
    assert viewer.viewport().cursor().shape() == QtCore.Qt.CursorShape.PointingHandCursor

    inspector._follow_check.setChecked(False)
    viewer._apply_idle_cursor()
    assert viewer.viewport().cursor().shape() == QtCore.Qt.CursorShape.OpenHandCursor

    inspector._follow_check.setChecked(True)
    loaded_window.spectrumTool.hide()
    viewer._apply_idle_cursor()
    assert viewer.viewport().cursor().shape() == QtCore.Qt.CursorShape.OpenHandCursor


def test_hover_is_ignored_when_inspector_hidden_or_frozen(loaded_window, inspector):
    inspector._follow_check.setChecked(False)
    loaded_window.viewer.pixelHovered.emit(3, 5)
    assert inspector._live is None
    inspector._follow_check.setChecked(True)
    loaded_window.spectrumTool.hide()
    loaded_window.viewer.pixelHovered.emit(3, 5)
    assert inspector._live is None


def test_hover_skips_reads_while_hypercube_builds(loaded_window, inspector, monkeypatch):
    hypercube = loaded_window._hypercube_controller
    stops = []
    monkeypatch.setattr(hypercube, "is_building", lambda: True)
    monkeypatch.setattr(hypercube, "stop_and_wait", lambda: stops.append(True))
    loaded_window.viewer.pixelHovered.emit(3, 5)
    assert inspector._live is None
    assert stops == []  # hover must never block on the hypercube worker


def test_viewer_emits_pixel_hovered_once_per_pixel(loaded_window, qtbot):
    viewer = loaded_window.viewer
    viewer.resize(400, 400)
    viewer.fit_in_view()
    received = []
    viewer.pixelHovered.connect(lambda row, column: received.append((row, column)))
    target = viewer.mapFromScene(QtCore.QPointF(5.5, 3.5))
    for _ in range(2):
        viewer._emit_hovered_pixel(target)
    assert received == [(3, 5)]


def test_spectrum_plot_action_pins_and_shows(loaded_window):
    loaded_window.viewer.spectrumPlotRequested.emit(QtCore.QPointF(5, 3))
    inspector = loaded_window._spectrum_controller.inspector
    assert loaded_window.spectrumTool.is_open()
    pins = inspector.pinned_spectra()
    assert len(pins) == 1 and pins[0].label.startswith("#1 · Pixel (3, 5)")


def test_band_markers_follow_visualization_mode(loaded_window, inspector):
    loaded_window.modeNDVI.setChecked(True)
    assert set(inspector._markers) == {"nir", "red"}
    loaded_window.modeRGB.setChecked(True)
    assert set(inspector._markers) == {"red", "green", "blue"}


def test_class_spectra_button_plots_each_class_in_its_colour(loaded_window, qtbot, inspector):
    _classify(loaded_window, qtbot)
    panel = loaded_window.classificationLayerPanel
    assert panel._spectra_button.isEnabled()
    panel._spectra_button.click()
    spectra = inspector.region_spectra()
    result = loaded_window._classification_controller._current_slot.result
    data = loaded_window._display_data()
    cube = data.read_bands(range(data.bands))
    populated = [c for c in range(3) if (result.class_map == c).any()]
    assert [s.key for s in spectra] == [f"class:{c}" for c in populated]
    for spectrum, class_id in zip(spectra, populated):
        np.testing.assert_allclose(
            spectrum.mean, cube[result.class_map == class_id].mean(axis=0), rtol=1e-5
        )
    assert inspector._separability_table.rowCount() == len(spectra)
    # Re-plotting replaces the class series rather than duplicating them.
    panel._spectra_button.click()
    assert len(inspector.region_spectra()) == len(spectra)


def test_similarity_mode_shows_angle_map_and_saves_it(
    loaded_window, inspector, file_dialog, tmp_path
):
    from PIL import Image

    loaded_window.viewer.spectrumPlotRequested.emit(QtCore.QPointF(5, 3))
    inspector._similarity_button.click()
    assert loaded_window.modeSimilarity.isChecked()
    similarity = loaded_window._spectrum_controller.similarity_for(loaded_window._display_data())
    assert similarity is not None
    assert similarity.angles[3, 5] == pytest.approx(0, abs=1e-3)
    np.testing.assert_array_equal(loaded_window.viewer.rgb, loaded_window._display_data().rgb_array)

    output = tmp_path / "similarity.png"
    file_dialog.save_return = (str(output), "")
    loaded_window.tabWidget.setCurrentWidget(loaded_window.Visualization)
    loaded_window.actionSaveImage.trigger()
    with Image.open(output) as image:
        np.testing.assert_array_equal(np.asarray(image)[..., :3], similarity.display_rgb)

    # Hover reports the angle at the cursor.
    loaded_window.viewer.pixelHovered.emit(3, 5)
    assert "0.00°" in inspector._status_label.text()


def test_crop_and_reload_discard_similarity(loaded_window, inspector, synthetic_cube_path):
    loaded_window.viewer.spectrumPlotRequested.emit(QtCore.QPointF(5, 3))
    inspector._similarity_button.click()
    loaded_window.viewer.cropRequested.emit(QtCore.QRectF(0, 0, 4, 4))
    assert loaded_window._spectrum_controller.similarity_for(loaded_window._display_data()) is None
    loaded_window.load_image_from_path(synthetic_cube_path)
    assert inspector.pinned_spectra() == ()


def test_transform_and_export(loaded_window, inspector, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    loaded_window.viewer.spectrumPlotRequested.emit(QtCore.QPointF(5, 3))
    inspector.set_transform(SpectrumTransform.FIRST_DERIVATIVE)
    output = tmp_path / "spectra"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(output), ""))
    )
    inspector._export_button.click()
    lines = (tmp_path / "spectra.csv").read_text().splitlines()
    assert lines[0] == 'wavelength_nm,"#1 · Pixel (3, 5) Visualization",Cursor'
    assert len(lines) == loaded_window._display_data().bands + 1


def test_ctrl_click_pins_pixel_without_leaving_the_image(loaded_window, qtbot):
    from PyQt6.QtCore import Qt

    viewer = loaded_window.viewer
    loaded_window.show()
    viewer.resize(400, 400)
    viewer.fit_in_view()
    target = viewer.mapFromScene(QtCore.QPointF(5.5, 3.5))
    qtbot.mouseClick(
        viewer.viewport(), Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ControlModifier, target,
    )
    inspector = loaded_window._spectrum_controller.inspector
    pins = inspector.pinned_spectra()
    assert len(pins) == 1 and pins[0].label.startswith("#1 · Pixel (3, 5)")
    assert inspector.isVisible()
    assert len(viewer.input_points) == 0  # no segmentation prompt marker
    assert "Ctrl+Click" in loaded_window.statusbar.currentMessage()


def test_series_can_be_hidden_selected_and_removed(loaded_window, inspector):
    from PyQt6.QtCore import Qt

    for column in (1, 2, 3):
        loaded_window.viewer.spectrumPlotRequested.emit(QtCore.QPointF(column, 0))
    items = inspector._series_list
    assert items.count() == 3

    # Untick hides from the plot and export, but keeps the list entry.
    items.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert "#1 · Pixel (0, 1) Visualization" not in inspector._series()
    assert items.count() == 3

    # Selecting highlights; Remove deletes only the selection.
    items.clearSelection()
    items.item(1).setSelected(True)
    assert inspector.selected_series() == items.item(1).data(Qt.ItemDataRole.UserRole)
    assert inspector._remove_button.isEnabled()
    inspector._remove_button.click()
    labels = [pin.label for pin in inspector.pinned_spectra()]
    assert labels == ["#1 · Pixel (0, 1) Visualization", "#3 · Pixel (0, 3) Visualization"]
    assert inspector.selected_series() is None

    # Clicking a plotted line selects it.
    line, key = next(iter(inspector._lines.items()))
    inspector._on_line_picked(type("Pick", (), {"artist": line})())
    assert inspector.selected_series() == key


def test_separability_shows_help_until_two_regions(loaded_window, qtbot, inspector):
    stack = inspector._separability_stack
    assert stack.currentWidget() is inspector._separability_help
    assert "How to fill it" in inspector._separability_help.text()
    inspector._image_mean_button.click()
    assert stack.currentWidget() is inspector._separability_help
    _classify(loaded_window, qtbot)
    loaded_window.classificationLayerPanel._spectra_button.click()
    assert stack.currentWidget() is inspector._separability_data
    inspector.clear()
    assert stack.currentWidget() is inspector._separability_help


def test_class_spectra_only_adds_selected_classes(loaded_window, qtbot, inspector):
    _classify(loaded_window, qtbot)
    controller = loaded_window._classification_controller
    layers = controller._current_slot.layers
    populated = [l.class_id for l in layers.layers if l.pixel_count]
    hidden = populated[0]
    controller._on_layer_visibility_changed(hidden, False)
    loaded_window.classificationLayerPanel._spectra_button.click()
    keys = {s.key for s in inspector.region_spectra()}
    assert f"class:{hidden}" not in keys
    assert keys == {f"class:{c}" for c in populated[1:]}

    controller._on_set_all_visible_requested(False)
    loaded_window.classificationLayerPanel._spectra_button.click()
    assert "Nothing is selected" in loaded_window.statusbar.currentMessage()


def test_classification_context_menu_offers_class_spectra(loaded_window):
    def spectrum_actions(viewer):
        menu = viewer._build_context_menu(QtCore.QPointF(1, 1))
        submenu = next(a.menu() for a in menu.actions() if a.text() == "Spectrum Analyzer")
        return {a.text().split("\t")[0] for a in submenu.actions() if a.text()}

    assert "Add Class Spectra" in spectrum_actions(loaded_window.classificationViewer)
    assert "Add Class Spectra" not in spectrum_actions(loaded_window.viewer)


def _marker_numbers(viewer):
    return [
        child.text()
        for item in viewer._pin_marker_items
        for child in item.childItems()
        if hasattr(child, "text")
    ]


def test_pins_draw_numbered_markers_matching_their_spectra(loaded_window, inspector):
    from PyQt6.QtCore import Qt

    loaded_window.viewer.spectrumPlotRequested.emit(QtCore.QPointF(5, 3))
    loaded_window.viewer.spectrumPlotRequested.emit(QtCore.QPointF(1, 6))
    # Every viewer showing the same cube marks both pins, numbered as listed.
    for viewer in (loaded_window.viewer, loaded_window.classificationViewer):
        assert _marker_numbers(viewer) == ["1", "2"]
    first, second = loaded_window.viewer._pin_marker_items
    assert first.pos() == QtCore.QPointF(5.5, 3.5)  # centre of pixel (3, 5)
    pins = inspector.pinned_spectra()
    circle = next(c for c in first.childItems() if hasattr(c, "rect"))
    assert circle.brush().color().getRgb()[:3] == tuple(
        round(channel * 255) for channel in pins[0].color
    )
    # The newest pin is selected and drawn larger.
    assert second.childItems()[0].rect().width() > circle.rect().width()

    # Hiding a spectrum hides its marker; removing renumbers nothing.
    inspector._series_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert _marker_numbers(loaded_window.viewer) == ["2"]
    inspector.clear()
    assert loaded_window.viewer._pin_marker_items == []


def test_markers_only_on_viewers_showing_the_pinned_cube(loaded_window, inspector, monkeypatch):
    loaded_window.viewer.spectrumPlotRequested.emit(QtCore.QPointF(5, 3))
    other = object()
    controller = loaded_window._spectrum_controller
    original = controller._data_for_viewer
    monkeypatch.setattr(
        controller, "_data_for_viewer",
        lambda viewer: other if viewer is loaded_window.calibrationViewer else original(viewer),
    )
    controller.refresh_markers()
    assert _marker_numbers(loaded_window.viewer) == ["1"]
    assert loaded_window.calibrationViewer._pin_marker_items == []


def test_markers_survive_mode_switch_and_clear_on_crop(loaded_window, inspector):
    loaded_window.viewer.spectrumPlotRequested.emit(QtCore.QPointF(5, 3))
    loaded_window.modeNDVI.setChecked(True)
    assert _marker_numbers(loaded_window.viewer) == ["1"]
    loaded_window.viewer.cropRequested.emit(QtCore.QRectF(0, 0, 4, 4))
    assert inspector.pinned_spectra() == ()
    assert loaded_window.viewer._pin_marker_items == []


def test_unfollowing_hides_the_cursor_line_and_following_restores_it(loaded_window, inspector):
    loaded_window.viewer.pixelHovered.emit(3, 5)
    assert inspector._live_line is not None
    assert "Cursor" in inspector._series()

    inspector._follow_check.setChecked(False)
    assert inspector._live_line is None
    assert "Cursor" not in inspector._series()  # nor in the CSV export
    assert "Cursor" not in inspector._axes.get_legend_handles_labels()[1]

    # Pinning still works without bringing the cursor line back.
    loaded_window.viewer.spectrumPlotRequested.emit(QtCore.QPointF(1, 6))
    assert len(inspector.pinned_spectra()) == 1
    assert inspector._live_line is None

    inspector._follow_check.setChecked(True)
    assert inspector._live_line is not None
    assert "row 6, column 1" in inspector._status_label.text()


def test_tiny_or_hidden_canvas_never_warns_about_layout(loaded_window, qtbot):
    import warnings

    loaded_window.show()
    inspector = loaded_window._spectrum_controller.inspector
    loaded_window.viewer.spectrumPlotRequested.emit(QtCore.QPointF(5, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any matplotlib layout warning fails
        for size in ((1, 1), (40, 30), (200, 150), (900, 600)):
            inspector._canvas.resize(*size)
            inspector._redraw()
            qtbot.wait(5)
        loaded_window.spectrumTool.hide()
        inspector._redraw()
        for _ in range(2):
            loaded_window.spectrumTool.set_floating(True)
            loaded_window.spectrumTool.set_floating(False)
        qtbot.wait(20)


def _draw_rectangle(window, x, y, width, height):
    from core import AnnotationKind

    window.classificationViewer.annotationShapeDrawn.emit(
        AnnotationKind.RECTANGLE, QtCore.QRectF(x, y, width, height)
    )
    layers = window._classification_controller._annotation_slot.layers.layers
    return layers[-1]


def test_class_spectra_include_selected_annotation_regions(loaded_window, qtbot, inspector):
    window = loaded_window
    _classify(window, qtbot)
    rect = _draw_rectangle(window, 2, 1, 3, 2)  # columns 2-4, rows 1-2
    panel = window.classificationLayerPanel
    assert rect.layer_id in panel._region_rows
    region_row = panel._region_rows[rect.layer_id]
    assert region_row.is_selected()
    assert panel._spectra_button.isEnabled()

    panel._spectra_button.click()
    keys = [s.key for s in inspector.region_spectra()]
    assert f"annotation:{rect.layer_id}" in keys
    assert any(key.startswith("class:") for key in keys)
    region = next(s for s in inspector.region_spectra() if s.key == f"annotation:{rect.layer_id}")
    data = window._display_data()
    cube = data.read_bands(range(data.bands))
    assert region.pixel_count == 6
    np.testing.assert_allclose(region.mean, cube[1:3, 2:5].reshape(-1, data.bands).mean(axis=0), rtol=1e-5)

    # Unticking the region leaves it out of the next plot (and keeps it drawn).
    region_row._toggle.setChecked(False)
    panel._spectra_button.click()
    assert f"annotation:{rect.layer_id}" not in [s.key for s in inspector.region_spectra()]
    assert rect.visible


def test_region_selection_survives_moving_the_shape(loaded_window, qtbot):
    window = loaded_window
    rect = _draw_rectangle(window, 1, 1, 3, 3)
    panel = window.classificationLayerPanel
    panel._region_rows[rect.layer_id]._toggle.setChecked(False)
    window.classificationViewer.annotationGeometryChanged.emit(rect.layer_id, 2, 2, 3, 3)
    window._classification_controller._refresh_regions()
    assert not panel._region_rows[rect.layer_id].is_selected()
    assert "0 of 1 classes selected" in panel._summary_label.text()


def test_regions_work_without_a_classification(loaded_window, inspector):
    window = loaded_window
    rect = _draw_rectangle(window, 0, 0, 2, 2)
    panel = window.classificationLayerPanel
    assert panel._spectra_button.isEnabled() and panel._toggle_all_button.isEnabled()
    panel._spectra_button.click()
    assert [s.key for s in inspector.region_spectra()] == [f"annotation:{rect.layer_id}"]


def test_inspect_buttons_add_one_class_or_region(loaded_window, qtbot, inspector):
    window = loaded_window
    _classify(window, qtbot)
    rect = _draw_rectangle(window, 2, 2, 3, 3)
    panel = window.classificationLayerPanel
    class_row = next(row for row in panel._rows.values() if row._inspect_button.isEnabled())
    class_row._inspect_button.click()
    assert [s.key for s in inspector.region_spectra()] == [f"class:{class_row.class_id}"]
    assert inspector.selected_series() == f"class:{class_row.class_id}"
    assert window.spectrumTool.is_open()

    # Inspect adds rather than replacing, and works on hidden classes too.
    window._classification_controller._on_layer_visibility_changed(class_row.class_id, False)
    panel._region_rows[rect.layer_id]._inspect_button.click()
    assert [s.key for s in inspector.region_spectra()] == [
        f"class:{class_row.class_id}", f"annotation:{rect.layer_id}",
    ]
    # Region colour follows the annotation's stroke.
    series = inspector._series_by_key[f"annotation:{rect.layer_id}"]
    assert tuple(round(c * 255) for c in series.color) == tuple(rect.style.stroke[:3])


def test_select_all_covers_classes_and_regions(loaded_window, qtbot):
    window = loaded_window
    _classify(window, qtbot)
    rect = _draw_rectangle(window, 1, 1, 2, 2)
    panel = window.classificationLayerPanel
    panel._toggle_all_button.click()  # everything was selected → deselect all
    assert not panel._region_rows[rect.layer_id].is_selected()
    assert not any(row._toggle.isChecked() for row in panel._rows.values())
    panel._toggle_all_button.click()
    assert panel._region_rows[rect.layer_id].is_selected()
    assert all(row._toggle.isChecked() for row in panel._rows.values())


def _pin_view_pos(viewer, row, column):
    return viewer.mapFromScene(QtCore.QPointF(column + 0.5, row + 0.5))


@pytest.fixture
def pinned_viewer(loaded_window, qtbot):
    window = loaded_window
    window.resize(1100, 800)
    window.show()
    qtbot.waitExposed(window)
    viewer = window.viewer
    viewer.fit_in_view()
    window.viewer.spectrumPlotRequested.emit(QtCore.QPointF(2, 3))  # pin #1 at (3, 2)
    qtbot.waitUntil(lambda: not window._hypercube_controller.is_building(), timeout=10000)
    return window, viewer


def test_dragging_a_pin_moves_it_and_updates_its_spectrum(pinned_viewer, qtbot):
    from PyQt6.QtCore import Qt

    window, viewer = pinned_viewer
    inspector = window._spectrum_controller.inspector
    start = _pin_view_pos(viewer, 3, 2)
    qtbot.mousePress(viewer.viewport(), Qt.MouseButton.LeftButton, pos=start)
    assert viewer._pin_drag is not None
    # Live update while dragging: the plot follows, number and colour stay.
    qtbot.mouseMove(viewer.viewport(), pos=_pin_view_pos(viewer, 5, 6))
    pin = inspector.pinned_spectra()[0]
    data = window._display_data()
    np.testing.assert_array_equal(pin.values, data.read_pixel(5, 6))
    color = pin.color
    qtbot.mouseRelease(viewer.viewport(), Qt.MouseButton.LeftButton, pos=_pin_view_pos(viewer, 5, 6))

    (pin,) = inspector.pinned_spectra()
    assert (pin.row, pin.column, pin.number, pin.color) == (5, 6, 1, color)
    assert pin.label.startswith("#1 · Pixel (5, 6)")
    assert window.statusbar.currentMessage() == "Moved pin to pixel (5, 6)"
    # The marker is redrawn at the new pixel on every viewer of this cube.
    for v in (viewer, window.classificationViewer):
        (item,) = v._pin_marker_items
        assert item.pos() == QtCore.QPointF(6.5, 5.5)
    # The drag did not pan the view or add a second pin.
    assert len(inspector.pinned_spectra()) == 1


def test_ctrl_click_on_a_pin_picks_it_instead_of_adding_another(pinned_viewer, qtbot):
    from PyQt6.QtCore import Qt

    window, viewer = pinned_viewer
    inspector = window._spectrum_controller.inspector
    window.viewer.spectrumPlotRequested.emit(QtCore.QPointF(6, 6))  # pin #2, now selected
    first_key = inspector.pin_markers()[0].key
    qtbot.mouseClick(
        viewer.viewport(), Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ControlModifier, _pin_view_pos(viewer, 3, 2),
    )
    assert len(inspector.pinned_spectra()) == 2
    assert inspector.selected_series() == first_key


def test_pin_drag_is_clamped_to_the_image(pinned_viewer, qtbot):
    from PyQt6.QtCore import Qt

    window, viewer = pinned_viewer
    qtbot.mousePress(viewer.viewport(), Qt.MouseButton.LeftButton, pos=_pin_view_pos(viewer, 3, 2))
    outside = viewer.mapFromScene(QtCore.QPointF(-50, 400))
    qtbot.mouseMove(viewer.viewport(), pos=outside)
    qtbot.mouseRelease(viewer.viewport(), Qt.MouseButton.LeftButton, pos=outside)
    (pin,) = window._spectrum_controller.inspector.pinned_spectra()
    assert (pin.row, pin.column) == (7, 0)


def test_pin_drag_is_refused_while_the_pipeline_is_busy(pinned_viewer, qtbot, monkeypatch):
    from PyQt6.QtCore import Qt

    window, viewer = pinned_viewer
    monkeypatch.setattr(window._classification_controller, "is_running", lambda: True)
    qtbot.mousePress(viewer.viewport(), Qt.MouseButton.LeftButton, pos=_pin_view_pos(viewer, 3, 2))
    qtbot.mouseMove(viewer.viewport(), pos=_pin_view_pos(viewer, 6, 6))
    qtbot.mouseRelease(viewer.viewport(), Qt.MouseButton.LeftButton, pos=_pin_view_pos(viewer, 6, 6))
    (pin,) = window._spectrum_controller.inspector.pinned_spectra()
    assert (pin.row, pin.column) == (3, 2)
    (item,) = viewer._pin_marker_items
    assert item.pos() == QtCore.QPointF(2.5, 3.5)  # snapped back
    monkeypatch.undo()  # let the window close normally at teardown


def test_plotted_spectra_follow_region_edits(loaded_window, qtbot, inspector):
    window = loaded_window
    _classify(window, qtbot)
    rect = _draw_rectangle(window, 0, 0, 2, 2)
    panel = window.classificationLayerPanel
    panel._region_rows[rect.layer_id]._inspect_button.click()
    key = f"annotation:{rect.layer_id}"
    assert inspector._series_by_key[key].region.pixel_count == 4

    viewer = window.classificationViewer
    viewer.annotationGeometryChanged.emit(rect.layer_id, 0, 0, 3, 2)
    viewer.annotationInteractionFinished.emit()
    qtbot.waitUntil(lambda: inspector._series_by_key[key].region.pixel_count == 6)

    window._classification_controller._on_annotation_rename_requested(rect.layer_id, "Spot")
    qtbot.waitUntil(lambda: inspector._series_by_key[key].label == "Spot")

    window._classification_controller._on_annotation_delete_requested(rect.layer_id)
    qtbot.waitUntil(lambda: key not in inspector._series_by_key)


def test_plotted_class_spectrum_follows_vector_edits(loaded_window, qtbot, inspector):
    from core import Anchor, VectorMask, VectorPath

    window = loaded_window
    _classify(window, qtbot)
    panel = window.classificationLayerPanel
    row = next(row for row in panel._rows.values() if row._inspect_button.isEnabled())
    row._inspect_button.click()
    key = f"class:{row.class_id}"
    square = VectorPath((Anchor(0, 0), Anchor(3, 0), Anchor(3, 3), Anchor(0, 3)))
    window.classificationViewer.vectorMaskEdited.emit(row.class_id, VectorMask((square,)), True)
    qtbot.waitUntil(lambda: inspector._series_by_key[key].region.pixel_count == 9)
    data = window._display_data()
    cube = data.read_bands(range(data.bands))
    np.testing.assert_allclose(
        inspector._series_by_key[key].region.mean,
        cube[:3, :3].reshape(-1, data.bands).mean(axis=0), rtol=1e-5,
    )
