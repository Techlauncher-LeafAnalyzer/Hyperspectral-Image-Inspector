from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import (
    QComboBox,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QWidget,
)
from PyQt6.QtGui import QAction

from core import (
    ClassificationLayer,
    ClassificationLayerModel,
    ClassificationService,
    HSIData,
    HSIReader,
    TrainingPairResolver,
    VisualizationMode,
    VisualizationResult,
)
from core.classification_model import UnsupervisedClassificationResult
from ui.classification_colors import classification_palette
from ui.classification_controller import _LAYER_COMPOSITE_BACKGROUND, ClassificationController
from ui.classification_layer_panel import ClassificationLayerPanel
from ui.viewer import HSIViewer, PixelValueEntry


@dataclass
class _FakeDisplaySource:
    """Stands in for MainWindowController's ``_display_data``/SR-toggle state."""

    data: HSIData = field(default_factory=HSIData)
    is_super_resolution_active: bool = False


def _make_layers(*, count: int = 3) -> tuple[ClassificationLayer, ...]:
    return tuple(
        ClassificationLayer(
            class_id=class_id,
            name=f"Class {class_id}",
            pixel_count=10 * (class_id + 1),
            visible=True,
            opacity=1.0,
        )
        for class_id in range(count)
    )


def _make_unsupervised_result(class_map: np.ndarray) -> UnsupervisedClassificationResult:
    n_classes = int(class_map.max()) + 1
    masks = np.stack(
        [(class_map == class_id).astype(np.uint8) for class_id in range(n_classes)]
    )
    return UnsupervisedClassificationResult(
        class_map=class_map,
        one_hot_masks=masks,
        cluster_centers=np.zeros((n_classes, 2), dtype=np.float32),
        class_pixel_counts=np.array(
            [int(np.sum(class_map == class_id)) for class_id in range(n_classes)]
        ),
        band_indices=(0, 1),
        band_wavelengths_nm=np.array([500.0, 600.0]),
        iterations_completed=1,
    )


def _make_visualization(
    mode: VisualizationMode,
    display_rgb: np.ndarray,
    values: np.ndarray | None,
) -> VisualizationResult:
    is_index = values is not None
    return VisualizationResult(
        mode=mode,
        display_rgb=display_rgb,
        values=values,
        title=mode.value,
        value_range=(0.0, 1.0),
        band_indices={},
        band_wavelengths_nm={},
        colormap="gray" if is_index else None,
        display_limits=(0.0, 1.0) if is_index else None,
    )


def _make_controller(qtbot):
    """Build a ClassificationController against real, minimal Qt widgets.

    No MainWindow involved -- mirrors ``test_hypercube_controller.py``'s
    ``_make_controller`` in exercising the controller's own public surface.
    """
    viewer = HSIViewer()
    qtbot.addWidget(viewer)
    layer_panel = ClassificationLayerPanel()
    qtbot.addWidget(layer_panel)
    statusbar = QStatusBar()
    qtbot.addWidget(statusbar)
    unsupervised_button = QPushButton()
    supervised_button = QPushButton()
    groundtruth_button = QPushButton()
    hyperspectral_button = QPushButton()
    classifier_combo = QComboBox()
    num_classes_edit = QLineEdit()
    max_iterations_edit = QLineEdit()
    groundtruth_path_edit = QLineEdit()
    hyperspectral_path_edit = QLineEdit()
    parent = QWidget()
    qtbot.addWidget(parent)
    for widget in (
        unsupervised_button,
        supervised_button,
        groundtruth_button,
        hyperspectral_button,
        classifier_combo,
        num_classes_edit,
        max_iterations_edit,
        groundtruth_path_edit,
        hyperspectral_path_edit,
    ):
        qtbot.addWidget(widget)
    load_image_action = QAction(parent)
    source = _FakeDisplaySource()

    controller = ClassificationController(
        lambda: source.data,
        lambda: source.is_super_resolution_active,
        ClassificationService(),
        TrainingPairResolver(),
        viewer,
        layer_panel,
        statusbar,
        unsupervised_button,
        supervised_button,
        groundtruth_button,
        hyperspectral_button,
        classifier_combo,
        num_classes_edit,
        max_iterations_edit,
        groundtruth_path_edit,
        hyperspectral_path_edit,
        load_image_action,
        lambda: None,
        parent,
    )
    return controller, viewer, layer_panel, source


# ---------------------------------------------------------------------- #
# ClassificationLayerPanel widget behaviour                                #
# ---------------------------------------------------------------------- #


def test_set_layers_builds_one_row_per_layer(qtbot):
    panel = ClassificationLayerPanel()
    qtbot.addWidget(panel)

    panel.set_layers(_make_layers(count=3))

    assert len(panel._rows) == 3
    assert panel._empty_label.isHidden()
    for class_id, row in panel._rows.items():
        assert row._toggle.isChecked()
        assert row._opacity_slider.value() == 100
        assert row._color_swatch.color == classification_palette(range(3))[class_id]


def test_layer_summary_tracks_visible_classes(qtbot):
    panel = ClassificationLayerPanel()
    qtbot.addWidget(panel)
    panel.set_layers(_make_layers(count=3))

    assert panel._summary_label.text() == "3 of 3 classes selected"
    panel._rows[1]._toggle.setChecked(False)
    assert panel._summary_label.text() == "2 of 3 classes selected"


def test_panel_has_no_collapse_control(qtbot):
    # Showing and hiding the whole inspector is the Segment pill's job.
    panel = ClassificationLayerPanel()
    qtbot.addWidget(panel)
    assert panel.findChild(QtWidgets.QToolButton, "classificationLayerCollapseButton") is None
    assert not hasattr(panel, "set_collapsed")


def test_empty_state_shown_before_any_layers(qtbot):
    panel = ClassificationLayerPanel()
    qtbot.addWidget(panel)
    panel.show()

    assert panel._empty_label.isVisible()
    assert not panel._rows_container.isVisible()


def test_clear_removes_rows_and_shows_empty_state(qtbot):
    panel = ClassificationLayerPanel()
    qtbot.addWidget(panel)
    panel.show()
    panel.set_layers(_make_layers(count=2))

    panel.clear()

    assert len(panel._rows) == 0
    assert panel._empty_label.isVisible()


def test_toggling_a_row_checkbox_emits_visibility_changed(qtbot):
    panel = ClassificationLayerPanel()
    qtbot.addWidget(panel)
    panel.set_layers(_make_layers(count=2))
    row = panel._rows[1]

    with qtbot.waitSignal(panel.visibilityChanged, timeout=1000) as blocker:
        row._toggle.setChecked(False)

    assert blocker.args == [1, False]


def test_moving_a_row_slider_emits_opacity_changed(qtbot):
    panel = ClassificationLayerPanel()
    qtbot.addWidget(panel)
    panel.set_layers(_make_layers(count=2))
    row = panel._rows[0]

    with qtbot.waitSignal(panel.opacityChanged, timeout=1000) as blocker:
        row._opacity_slider.setValue(40)

    assert blocker.args == [0, 0.4]


def test_toggle_all_button_disables_then_enables_every_row(qtbot):
    panel = ClassificationLayerPanel()
    qtbot.addWidget(panel)
    panel.set_layers(_make_layers(count=3))
    assert panel._toggle_all_button.text() == "Deselect all"

    with qtbot.waitSignal(panel.setAllVisibleRequested, timeout=1000) as blocker:
        panel._toggle_all_button.click()
    assert blocker.args == [False]

    # Simulate the controller applying the request and rebuilding rows.
    hidden_layers = tuple(
        ClassificationLayer(
            class_id=layer.class_id,
            name=layer.name,
            pixel_count=layer.pixel_count,
            visible=False,
            opacity=layer.opacity,
        )
        for layer in _make_layers(count=3)
    )
    panel.set_layers(hidden_layers)
    assert panel._toggle_all_button.text() == "Select all"

    with qtbot.waitSignal(panel.setAllVisibleRequested, timeout=1000) as blocker:
        panel._toggle_all_button.click()
    assert blocker.args == [True]


def test_toggle_all_button_text_follows_a_single_row_toggle(qtbot):
    panel = ClassificationLayerPanel()
    qtbot.addWidget(panel)
    panel.set_layers(_make_layers(count=2))

    panel._rows[0]._toggle.setChecked(False)
    assert panel._toggle_all_button.text() == "Deselect all"

    panel._rows[1]._toggle.setChecked(False)
    assert panel._toggle_all_button.text() == "Select all"


def test_outline_toggle_button_emits_outline_mode_changed(qtbot):
    panel = ClassificationLayerPanel()
    qtbot.addWidget(panel)
    panel.set_layers(_make_layers(count=2))

    with qtbot.waitSignal(panel.outlineModeChanged, timeout=1000) as blocker:
        panel._outline_toggle_button.setChecked(True)
    assert blocker.args == [True]
    assert panel._outline_toggle_button.text() == "Show fill"

    with qtbot.waitSignal(panel.outlineModeChanged, timeout=1000) as blocker:
        panel._outline_toggle_button.setChecked(False)
    assert blocker.args == [False]
    assert panel._outline_toggle_button.text() == "Borders only"


def test_moving_the_global_opacity_slider_emits_global_opacity_changed(qtbot):
    panel = ClassificationLayerPanel()
    qtbot.addWidget(panel)
    panel.set_layers(_make_layers(count=2))

    with qtbot.waitSignal(panel.globalOpacityChanged, timeout=1000) as blocker:
        panel._global_opacity_slider.setValue(30)

    assert blocker.args == [0.3]
    assert panel._global_opacity_value_label.text() == "30%"


def test_set_layers_resets_global_controls(qtbot):
    panel = ClassificationLayerPanel()
    qtbot.addWidget(panel)
    panel.set_layers(_make_layers(count=2))
    panel._global_opacity_slider.setValue(20)
    panel._outline_toggle_button.setChecked(True)

    panel.set_layers(_make_layers(count=3))

    assert panel._global_opacity_slider.value() == 100
    assert not panel._outline_toggle_button.isChecked()
    assert panel._outline_toggle_button.text() == "Borders only"


# ---------------------------------------------------------------------- #
# End-to-end wiring through ClassificationController                       #
# ---------------------------------------------------------------------- #


def test_result_populates_layer_panel(qtbot):
    controller, viewer, layer_panel, source = _make_controller(qtbot)
    class_map = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int32)
    result = _make_unsupervised_result(class_map)

    controller._on_result(result)

    assert len(layer_panel._rows) == 3
    assert controller._current_slot.layers is not None
    assert controller._current_slot.layers.visible_class_ids == (0, 1, 2)
    for class_id, row in layer_panel._rows.items():
        rendered_color = tuple(controller.rgb[class_map == class_id][0])
        assert row._color_swatch.color == rendered_color


def test_hiding_a_layer_updates_the_rendered_pixmap(qtbot):
    controller, viewer, layer_panel, source = _make_controller(qtbot)
    class_map = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int32)
    result = _make_unsupervised_result(class_map)
    controller._on_result(result)
    before = viewer._photo.pixmap().toImage().copy()

    controller._on_layer_visibility_changed(1, False)

    after = viewer._photo.pixmap().toImage()
    assert controller._current_slot.layers.visible_class_ids == (0, 2)
    assert before != after


def test_hiding_a_layer_reveals_the_true_colour_base_image(qtbot):
    controller, viewer, layer_panel, source = _make_controller(qtbot)
    class_map = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int32)
    result = _make_unsupervised_result(class_map)
    base_rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    base_rgb[..., 2] = 40  # a distinct, otherwise-unused blue base image
    source.data.rgb_array = base_rgb
    controller._current_slot.active_data = source.data  # set by the real click handler pre-launch
    controller._on_result(result)

    controller._on_layer_visibility_changed(1, False)

    image = viewer._photo.pixmap().toImage()
    # Class 1 occupies (0, 2) and (1, 0) in (row, col) order.
    for row, col in ((0, 2), (1, 0)):
        color = image.pixelColor(col, row)
        assert (color.red(), color.green(), color.blue()) == (0, 0, 40)


def test_changing_opacity_debounces_then_updates_the_pixmap(qtbot):
    controller, viewer, layer_panel, source = _make_controller(qtbot)
    class_map = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int32)
    result = _make_unsupervised_result(class_map)
    controller._on_result(result)
    before = viewer._photo.pixmap().toImage().copy()

    controller._on_layer_opacity_changed(2, 0.5)
    assert controller._current_slot.layers.layers[2].opacity == 0.5

    qtbot.wait(150)

    after = viewer._photo.pixmap().toImage()
    assert before != after


def test_clear_result_clears_the_layer_panel(qtbot):
    controller, viewer, layer_panel, source = _make_controller(qtbot)
    class_map = np.array([[0, 1]], dtype=np.int32)
    controller._on_result(_make_unsupervised_result(class_map))

    controller.clear_result()

    assert len(layer_panel._rows) == 0
    assert controller._current_slot.layers is None


# ---------------------------------------------------------------------- #
# Enable/disable all, global opacity, and border-only mode                 #
# ---------------------------------------------------------------------- #


def test_disabling_all_hides_every_class_and_rebuilds_unchecked_rows(qtbot):
    controller, viewer, layer_panel, source = _make_controller(qtbot)
    class_map = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int32)
    controller._on_result(_make_unsupervised_result(class_map))
    before = viewer._photo.pixmap().toImage().copy()

    controller._on_set_all_visible_requested(False)

    assert controller._current_slot.layers.visible_class_ids == ()
    assert all(not row._toggle.isChecked() for row in layer_panel._rows.values())
    after = viewer._photo.pixmap().toImage()
    assert before != after


def test_enabling_all_restores_every_class(qtbot):
    controller, viewer, layer_panel, source = _make_controller(qtbot)
    class_map = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int32)
    controller._on_result(_make_unsupervised_result(class_map))
    controller._on_layer_visibility_changed(1, False)

    controller._on_set_all_visible_requested(True)

    assert controller._current_slot.layers.visible_class_ids == (0, 1, 2)
    assert all(row._toggle.isChecked() for row in layer_panel._rows.values())


def test_toggle_all_button_click_flows_through_to_the_model(qtbot):
    controller, viewer, layer_panel, source = _make_controller(qtbot)
    class_map = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int32)
    controller._on_result(_make_unsupervised_result(class_map))

    layer_panel._toggle_all_button.click()

    assert controller._current_slot.layers.visible_class_ids == ()


def test_global_opacity_scales_the_composite_after_debounce(qtbot):
    controller, viewer, layer_panel, source = _make_controller(qtbot)
    class_map = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int32)
    controller._on_result(_make_unsupervised_result(class_map))

    controller._on_global_opacity_changed(0.5)
    assert controller._current_slot.layers.global_opacity == 0.5

    qtbot.wait(150)

    composite = controller._composite()
    assert composite is not None
    # Every visible-class pixel should now be a 50/50 blend toward background,
    # never the fully-opaque class colour.
    assert not np.array_equal(composite.display_rgb, controller._current_slot.rgb)


def test_outline_mode_reveals_background_in_class_interiors(qtbot):
    controller, viewer, layer_panel, source = _make_controller(qtbot)
    class_map = np.full((6, 6), 0, dtype=np.int32)
    class_map[1:5, 1:5] = 1  # a solid interior block guaranteed to have non-border pixels
    controller._on_result(_make_unsupervised_result(class_map))

    controller._on_outline_mode_changed(True)

    assert controller._current_slot.layers.outline_mode
    composite = controller._composite()
    # (2, 2) is inside class 1's block, away from every edge -> pure interior.
    interior_pixel = composite.display_rgb[2, 2]
    background_pixel = np.array(_LAYER_COMPOSITE_BACKGROUND, dtype=np.uint8)
    assert np.array_equal(interior_pixel, background_pixel)


# ---------------------------------------------------------------------- #
# Super-Resolution-aware classification                                    #
# ---------------------------------------------------------------------- #


def test_classifying_with_super_resolution_active_targets_its_data_without_notice(
    qtbot, synthetic_cube_path, monkeypatch
):
    controller, viewer, layer_panel, source = _make_controller(qtbot)
    source.data = HSIReader().open(synthetic_cube_path)
    source.is_super_resolution_active = True
    controller._num_classes_edit.setText("2")
    controller._max_iterations_edit.setText("3")

    infos: list[tuple] = []
    monkeypatch.setattr(
        QMessageBox, "information", staticmethod(lambda *args: infos.append(args))
    )

    controller._on_unsupervised_classify_clicked()
    qtbot.waitUntil(lambda: not controller.is_running(), timeout=5000)

    assert controller._current_slot.active_data is source.data
    assert not infos


def test_classifying_without_super_resolution_active_shows_no_notice(
    qtbot, synthetic_cube_path, monkeypatch
):
    controller, viewer, layer_panel, source = _make_controller(qtbot)
    source.data = HSIReader().open(synthetic_cube_path)
    source.is_super_resolution_active = False
    controller._num_classes_edit.setText("2")
    controller._max_iterations_edit.setText("3")

    infos: list[tuple] = []
    monkeypatch.setattr(
        QMessageBox, "information", staticmethod(lambda *args: infos.append(args))
    )

    controller._on_unsupervised_classify_clicked()
    qtbot.waitUntil(lambda: not controller.is_running(), timeout=5000)

    assert controller._current_slot.active_data is source.data
    assert len(infos) == 0


def test_repeated_super_resolution_classification_shows_no_notice(
    qtbot, synthetic_cube_path, monkeypatch
):
    controller, viewer, layer_panel, source = _make_controller(qtbot)
    source.data = HSIReader().open(synthetic_cube_path)
    source.is_super_resolution_active = True
    controller._num_classes_edit.setText("2")
    controller._max_iterations_edit.setText("3")

    infos: list[tuple] = []
    monkeypatch.setattr(
        QMessageBox, "information", staticmethod(lambda *args: infos.append(args))
    )

    controller._on_unsupervised_classify_clicked()
    qtbot.waitUntil(lambda: not controller.is_running(), timeout=5000)
    controller._on_unsupervised_classify_clicked()
    qtbot.waitUntil(lambda: not controller.is_running(), timeout=5000)

    assert not infos


# ---------------------------------------------------------------------- #
# Per-resolution classification storage and low/high-res swapping          #
# ---------------------------------------------------------------------- #


def _classify_into_slot(controller, source, slot_key: bool, class_map: np.ndarray) -> None:
    """Simulate a completed classification job launched at ``slot_key``.

    Mirrors what the real click handler records (``_pending_slot_key`` and
    the slot's ``active_data``) before a worker's ``resultReady`` signal
    reaches ``_on_result``, without spinning up an actual QThread.
    """

    source.is_super_resolution_active = slot_key
    controller._pending_slot_key = slot_key
    controller._slots[slot_key].active_data = source.data
    controller._on_result(_make_unsupervised_result(class_map))


def test_low_and_high_res_results_are_stored_in_separate_slots(qtbot):
    controller, viewer, layer_panel, source = _make_controller(qtbot)
    low_res_data = HSIData()
    low_res_data.rgb_array = np.zeros((2, 3, 3), dtype=np.uint8)
    high_res_data = HSIData()
    high_res_data.rgb_array = np.zeros((4, 6, 3), dtype=np.uint8)

    source.data = low_res_data
    _classify_into_slot(
        controller, source, False, np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int32)
    )
    low_slot_layers = controller._slots[False].layers

    source.data = high_res_data
    _classify_into_slot(
        controller,
        source,
        True,
        np.array([[0, 0, 1, 1], [1, 2, 2, 0], [0, 1, 1, 2], [2, 2, 0, 0]], dtype=np.int32),
    )
    high_slot_layers = controller._slots[True].layers

    assert low_slot_layers is not None and high_slot_layers is not None
    assert low_slot_layers is not high_slot_layers
    assert controller._slots[False].active_data is low_res_data
    assert controller._slots[True].active_data is high_res_data
    assert controller._slots[False].result.class_map.shape == (2, 3)
    assert controller._slots[True].result.class_map.shape == (4, 4)


def test_editing_one_resolutions_layers_does_not_affect_the_other(qtbot):
    controller, viewer, layer_panel, source = _make_controller(qtbot)
    class_map = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int32)
    _classify_into_slot(controller, source, False, class_map)
    _classify_into_slot(controller, source, True, class_map)

    # Editing while viewing high-res must not touch the low-res slot.
    source.is_super_resolution_active = True
    controller._on_layer_visibility_changed(1, False)
    controller._on_global_opacity_changed(0.4)

    assert controller._slots[True].layers.visible_class_ids == (0, 2)
    assert controller._slots[True].layers.global_opacity == 0.4
    assert controller._slots[False].layers.visible_class_ids == (0, 1, 2)
    assert controller._slots[False].layers.global_opacity == 1.0


def test_refresh_display_swaps_the_layer_panel_to_the_active_resolution(qtbot):
    controller, viewer, layer_panel, source = _make_controller(qtbot)
    class_map = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int32)
    _classify_into_slot(controller, source, False, class_map)
    controller._on_layer_visibility_changed(1, False)  # fade a class in low-res only

    # Switch to high-res, which has never been classified.
    source.is_super_resolution_active = True
    controller.refresh_display()
    assert len(layer_panel._rows) == 0

    # Switch back to low-res: its rows and visibility choice must be intact.
    source.is_super_resolution_active = False
    controller.refresh_display()
    assert len(layer_panel._rows) == 3
    assert not layer_panel._rows[1]._toggle.isChecked()


def test_refresh_display_restores_global_opacity_and_outline_mode(qtbot):
    controller, viewer, layer_panel, source = _make_controller(qtbot)
    class_map = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int32)
    _classify_into_slot(controller, source, False, class_map)
    controller._on_global_opacity_changed(0.6)
    controller._on_outline_mode_changed(True)

    source.is_super_resolution_active = True
    controller.refresh_display()
    assert layer_panel._global_opacity_slider.value() == 100
    assert not layer_panel._outline_toggle_button.isChecked()

    source.is_super_resolution_active = False
    controller.refresh_display()
    assert layer_panel._global_opacity_slider.value() == 60
    assert layer_panel._outline_toggle_button.isChecked()


def test_clear_super_resolution_result_only_discards_the_high_res_slot(qtbot):
    controller, viewer, layer_panel, source = _make_controller(qtbot)
    class_map = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int32)
    _classify_into_slot(controller, source, False, class_map)
    _classify_into_slot(controller, source, True, class_map)

    controller.clear_super_resolution_result()

    assert controller._slots[True].result is None
    assert controller._slots[True].layers is None
    assert controller._slots[False].result is not None
    assert controller._slots[False].layers is not None


def test_clear_result_discards_both_resolution_slots(qtbot):
    controller, viewer, layer_panel, source = _make_controller(qtbot)
    class_map = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int32)
    _classify_into_slot(controller, source, False, class_map)
    _classify_into_slot(controller, source, True, class_map)

    controller.clear_result()

    assert controller._slots[False].result is None
    assert controller._slots[True].result is None
    assert len(layer_panel._rows) == 0


def test_a_completed_worker_does_not_overwrite_the_slot_toggled_away_from(qtbot):
    """A result must land in the slot active at launch, not at completion."""

    controller, viewer, layer_panel, source = _make_controller(qtbot)
    class_map = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int32)

    # Launch while low-res is active...
    source.is_super_resolution_active = False
    controller._pending_slot_key = False
    controller._slots[False].active_data = source.data
    # ...but the user flips to high-res before the worker's result arrives.
    source.is_super_resolution_active = True

    controller._on_result(_make_unsupervised_result(class_map))

    assert controller._slots[False].result is not None
    assert controller._slots[True].result is None
    # The currently-active (high-res) panel/viewer must not have been
    # disturbed by a result that belongs to the other resolution.
    assert len(layer_panel._rows) == 0


# ---------------------------------------------------------------------- #
# Per-class visualization means                                           #
# ---------------------------------------------------------------------- #


def test_visualization_means_average_each_class():
    class_map = np.array([[0, 0, 1], [1, 1, 1]], dtype=np.int32)
    model = ClassificationLayerModel(_make_unsupervised_result(class_map))
    display = np.zeros((2, 3, 3), dtype=np.uint8)
    display[0, 0] = (10, 20, 30)
    display[0, 1] = (30, 40, 50)
    display[class_map == 1] = (200, 0, 0)
    display[1, 2] = (0, 0, 200)
    values = np.array([[0.1, 0.3, 0.0], [0.5, 0.4, np.nan]], dtype=np.float32)

    rgb_means = model.visualization_means(
        _make_visualization(VisualizationMode.RGB, display, None)
    )
    ndvi_means = model.visualization_means(
        _make_visualization(VisualizationMode.NDVI, display, values)
    )

    assert rgb_means[0].value == (20.0, 30.0, 40.0)
    assert rgb_means[0].color == (20, 30, 40)
    assert abs(ndvi_means[0].value - 0.2) < 1e-6
    # Class 1 ignores the NaN pixel, and its swatch is the mean run through
    # the visualization's own mapping (gray over 0..1), not a pixel colour.
    assert abs(ndvi_means[1].value - 0.3) < 1e-6
    assert ndvi_means[1].color == (76, 76, 76)


def test_rows_show_index_means_with_swatches(qtbot):
    panel = ClassificationLayerPanel()
    qtbot.addWidget(panel)

    panel.set_layers(
        _make_layers(count=2),
        means={
            0: {
                "RGB": PixelValueEntry(value=(1.0, 2.0, 3.0), color=(1, 2, 3)),
                "NDVI": PixelValueEntry(value=0.4567, color=(0, 128, 0)),
            }
        },
    )

    html = panel._rows[0]._means_label.text()
    assert "RGB: (1, 2, 3)" in html
    assert "NDVI: 0.457" in html
    assert 'bgcolor="#008000"' in html
    assert panel._rows[1]._means_label is None


def test_controller_populates_panel_with_class_means(qtbot):
    class_map = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int32)
    display = np.full((2, 3, 3), 50, dtype=np.uint8)
    values = np.array([[0.2, 0.4, 0.6], [0.6, 0.8, 1.0]], dtype=np.float32)
    results = {
        VisualizationMode.RGB: _make_visualization(VisualizationMode.RGB, display, None),
        VisualizationMode.NDVI: _make_visualization(VisualizationMode.NDVI, display, values),
    }
    controller, viewer, layer_panel, source = _make_controller(qtbot)
    controller.set_visualization_results(results)

    _classify_into_slot(controller, source, False, class_map)

    html = layer_panel._rows[2]._means_label.text()
    assert "NDVI: 0.900" in html
    assert "RGB: (50, 50, 50)" in html
    assert "EVI: —" in html


def test_new_visualization_results_rebuild_cached_means(qtbot):
    class_map = np.array([[0, 0, 1], [1, 1, 1]], dtype=np.int32)
    display = np.zeros((2, 3, 3), dtype=np.uint8)
    controller, viewer, layer_panel, source = _make_controller(qtbot)
    controller.set_visualization_results(
        {VisualizationMode.NDVI: _make_visualization(
            VisualizationMode.NDVI, display, np.full((2, 3), 0.25, dtype=np.float32)
        )}
    )
    _classify_into_slot(controller, source, False, class_map)

    controller.set_visualization_results(
        {VisualizationMode.NDVI: _make_visualization(
            VisualizationMode.NDVI, display, np.full((2, 3), 0.75, dtype=np.float32)
        )}
    )
    controller.refresh_display()

    assert "NDVI: 0.750" in layer_panel._rows[0]._means_label.text()


def test_panel_width_is_user_resizable_within_limits(qtbot):
    panel = ClassificationLayerPanel()
    qtbot.addWidget(panel)
    panel.show()
    assert panel.panel_width == 300 and panel.maximumWidth() == 300
    panel.set_panel_width(340)
    assert panel.panel_width == 340 and panel.maximumWidth() == 340
    panel.set_panel_width(10)
    assert panel.panel_width == 270  # never below the minimum
    panel.set_panel_width(5000)
    assert panel.panel_width == 520  # nor above the maximum


def test_dragging_the_grip_widens_the_panel(qtbot):
    from PyQt6 import QtCore

    panel = ClassificationLayerPanel()
    qtbot.addWidget(panel)
    panel.resize(300, 500)
    panel.show()
    grip = panel._resize_grip
    assert grip.isVisible() and grip.geometry().left() == 0
    start = grip.rect().center()
    qtbot.mousePress(grip, QtCore.Qt.MouseButton.LeftButton, pos=start)
    qtbot.mouseMove(grip, start + QtCore.QPoint(-80, 0))
    qtbot.mouseRelease(grip, QtCore.Qt.MouseButton.LeftButton, pos=start + QtCore.QPoint(-80, 0))
    assert panel.panel_width > 300
    assert panel.maximumWidth() == panel.panel_width


# ---------------------------------------------------------------------- #
# Measured annotation regions                                              #
# ---------------------------------------------------------------------- #


def _region_setup(qtbot):
    controller, viewer, layer_panel, source = _make_controller(qtbot)
    values = np.zeros((20, 20), dtype=np.float32)
    values[:, 10:] = 1.0
    ndvi = _make_visualization(
        VisualizationMode.NDVI, np.zeros((20, 20, 3), dtype=np.uint8), values
    )
    controller.set_visualization_results({VisualizationMode.NDVI: ndvi})
    return controller, layer_panel


def _draw(controller, kind, x, y, w, h):
    from PyQt6.QtCore import QRectF

    controller._on_annotation_shape_drawn(kind, QRectF(x, y, w, h))
    return controller._annotation_slot.selected_id


def test_measured_shape_appears_as_a_region_row_and_recalculates_on_move(qtbot):
    from core import AnnotationKind

    controller, panel = _region_setup(qtbot)
    layer_id = _draw(controller, AnnotationKind.RECTANGLE, 0, 0, 5, 5)
    assert layer_id in panel._region_rows
    row = panel._region_rows[layer_id]
    assert "NDVI: 0.000" in row._means_label.text()

    controller._on_annotation_geometry_changed(layer_id, 12, 0, 5, 5)
    qtbot.waitUntil(
        lambda: "NDVI: 1.000" in panel._region_rows[layer_id]._means_label.text(),
        timeout=1000,
    )


def test_region_rows_follow_measure_toggle_visibility_and_delete(qtbot):
    from core import AnnotationKind

    controller, panel = _region_setup(qtbot)
    layer_id = _draw(controller, AnnotationKind.ELLIPSE, 2, 2, 6, 6)
    controller._on_annotation_measure_changed(layer_id, True)
    assert layer_id in panel._region_rows

    controller._on_annotation_visibility_changed(layer_id, False)
    assert not panel._region_rows
    controller._on_annotation_visibility_changed(layer_id, True)
    assert layer_id in panel._region_rows

    controller._on_annotation_measure_changed(layer_id, False)
    assert not panel._region_rows
    controller._on_annotation_measure_changed(layer_id, True)
    controller._on_annotation_delete_requested(layer_id)
    assert not panel._region_rows


def test_measure_checkbox_tracks_selected_shape_kind(qtbot):
    from core import AnnotationKind

    controller, panel = _region_setup(qtbot)
    page = panel.annotation_page
    rect_id = _draw(controller, AnnotationKind.RECTANGLE, 0, 0, 5, 5)
    assert page._measure_check.isEnabled() and page._measure_check.isChecked()

    with qtbot.waitSignal(page.measureChanged, timeout=1000) as blocker:
        page._measure_check.setChecked(False)
    assert blocker.args == [rect_id, False]
    assert not controller._annotation_slot.layers.get(rect_id).measure_index
    assert not panel._region_rows

    from PyQt6.QtCore import QPointF

    controller._on_annotation_arrow_drawn(QPointF(0, 0), QPointF(8, 8))
    assert not page._measure_check.isEnabled()


def test_region_rows_do_not_change_class_rows(qtbot):
    from core import AnnotationKind

    controller, panel = _region_setup(qtbot)
    panel.set_layers(_make_layers(count=2))
    layer_id = _draw(controller, AnnotationKind.RECTANGLE, 0, 0, 5, 5)
    controller._on_annotation_measure_changed(layer_id, True)
    assert len(panel._rows) == 2
    assert len(panel._region_rows) == 1
    panel.set_layers(_make_layers(count=2))  # class refresh keeps region rows
    assert len(panel._region_rows) == 1


def test_region_row_is_renamed_by_double_click_editor(qtbot):
    from core import AnnotationKind

    controller, panel = _region_setup(qtbot)
    panel.show()  # the inline editor only commits while it is visible
    layer_id = _draw(controller, AnnotationKind.RECTANGLE, 0, 0, 5, 5)
    row = panel._region_rows[layer_id]

    row.begin_rename()
    row._name_edit.setText("Leaf patch")
    row._name_edit.editingFinished.emit()

    assert controller._annotation_slot.layers.get(layer_id).name == "Leaf patch"
    assert panel._region_rows[layer_id]._name_label.text() == "Leaf patch"
    names = [panel.annotation_page._list.item(i).text() for i in range(panel.annotation_page._list.count())]
    assert "Leaf patch" in names
    # Renaming is one undoable step.
    controller.undo_annotation()
    assert controller._annotation_slot.layers.get(layer_id).name == "Rectangle 1"


def test_blank_annotation_name_is_rejected(qtbot):
    import pytest as _pytest

    from core import AnnotationError, AnnotationKind, AnnotationLayerModel, AnnotationStyle

    model = AnnotationLayerModel()
    layer = model.add_shape(AnnotationKind.RECTANGLE, 0, 0, 5, 5, AnnotationStyle())
    with _pytest.raises(AnnotationError):
        model.rename(layer.layer_id, "   ")


def test_measure_checkbox_is_disabled_for_text_and_arrow(qtbot):
    from PyQt6.QtCore import QPointF

    from core import AnnotationKind

    controller, panel = _region_setup(qtbot)
    page = panel.annotation_page
    _draw(controller, AnnotationKind.RECTANGLE, 0, 0, 5, 5)
    assert page._measure_check.isEnabled()

    controller._on_annotation_arrow_drawn(QPointF(0, 0), QPointF(8, 8))
    assert not page._measure_check.isEnabled()
    assert not page._measure_check.isChecked()

    _draw(controller, AnnotationKind.ELLIPSE, 1, 1, 4, 4)
    assert page._measure_check.isEnabled()


def test_export_button_enabled_by_measured_annotation_without_classification(qtbot):
    from core import AnnotationKind

    controller, panel = _region_setup(qtbot)
    assert not panel._export_button.isEnabled()

    layer_id = _draw(controller, AnnotationKind.RECTANGLE, 0, 0, 5, 5)
    assert panel._export_button.isEnabled()  # new shapes are measured immediately

    controller._on_annotation_measure_changed(layer_id, False)
    assert not panel._export_button.isEnabled()


def test_export_writes_measured_annotations_with_and_without_classes(qtbot, tmp_path, monkeypatch):
    import csv

    from PyQt6.QtWidgets import QFileDialog

    from core import AnnotationKind

    controller, panel = _region_setup(qtbot)
    layer_id = _draw(controller, AnnotationKind.RECTANGLE, 12, 0, 4, 4)
    controller._on_annotation_rename_requested(layer_id, "Leaf patch")
    controller._on_annotation_measure_changed(layer_id, True)

    path = tmp_path / "out.csv"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(path), ""))
    )
    controller._on_export_requested()

    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["class_name"] for r in rows] == ["Leaf patch"]
    assert rows[0]["class_id"] == ""
    assert rows[0]["pixel_count"] == "16"
    assert float(rows[0]["NDVI_mean"]) == 1.0



def test_measure_defaults_on_at_startup_and_preserves_individual_opt_out(qtbot):
    from core import AnnotationKind

    controller, panel = _region_setup(qtbot)
    page = panel.annotation_page
    assert page._measure_check.isChecked()
    assert not page._measure_check.isEnabled()
    page.set_layers(())
    assert page._measure_check.isChecked()

    rect = _draw(controller, AnnotationKind.RECTANGLE, 0, 0, 5, 5)
    assert page._measure_check.isChecked() and rect in panel._region_rows
    page._measure_check.setChecked(False)
    assert rect not in panel._region_rows

    ellipse = _draw(controller, AnnotationKind.ELLIPSE, 2, 2, 6, 6)
    assert page._measure_check.isChecked() and ellipse in panel._region_rows
    controller._on_annotation_picked(rect)
    assert not page._measure_check.isChecked()
    controller._on_annotation_deselected()
    assert page._measure_check.isChecked() and not page._measure_check.isEnabled()


def test_whole_inspector_is_the_only_scrolling_element(qtbot):
    panel = ClassificationLayerPanel()
    qtbot.addWidget(panel)
    panel.resize(300, 400)
    panel.set_layers(_make_layers(count=8))
    panel.show()
    qtbot.wait(20)
    # One scroll area wraps everything: header, mode switch and page.
    scroll_areas = panel.findChildren(QtWidgets.QScrollArea)
    assert scroll_areas == [panel._scroll_area]
    assert panel._scroll_area.widget().isAncestorOf(panel._rows_container)
    assert panel._scroll_area.widget().isAncestorOf(panel._mode_row)
    # The class rows take their full height, so the inspector scrolls.
    assert panel._rows_container.height() >= panel._rows_container.sizeHint().height()
    assert panel._scroll_area.verticalScrollBar().maximum() > 0
    # The annotation layer list never scrolls on its own.
    layer_list = panel.annotation_page.findChild(QtWidgets.QListWidget, "annotationLayerList")
    assert layer_list.verticalScrollBarPolicy() == QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
