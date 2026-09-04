from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
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
    ClassificationService,
    HSIData,
    HSIReader,
    TrainingPairResolver,
)
from core.classification_model import UnsupervisedClassificationResult
from ui.classification_controller import ClassificationController
from ui.classification_layer_panel import ClassificationLayerPanel
from ui.viewer import HSIViewer


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
    classifier_combo = QComboBox()
    num_classes_edit = QLineEdit()
    max_iterations_edit = QLineEdit()
    groundtruth_path_edit = QLineEdit()
    parent = QWidget()
    qtbot.addWidget(parent)
    for widget in (
        unsupervised_button,
        supervised_button,
        groundtruth_button,
        classifier_combo,
        num_classes_edit,
        max_iterations_edit,
        groundtruth_path_edit,
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
        classifier_combo,
        num_classes_edit,
        max_iterations_edit,
        groundtruth_path_edit,
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


def test_empty_state_shown_before_any_layers(qtbot):
    panel = ClassificationLayerPanel()
    qtbot.addWidget(panel)
    panel.show()

    assert panel._empty_label.isVisible()
    assert not panel._scroll_area.isVisible()


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


# ---------------------------------------------------------------------- #
# End-to-end wiring through ClassificationController                       #
# ---------------------------------------------------------------------- #


def test_result_populates_layer_panel(qtbot):
    controller, viewer, layer_panel, source = _make_controller(qtbot)
    class_map = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int32)
    result = _make_unsupervised_result(class_map)

    controller._on_result(result)

    assert len(layer_panel._rows) == 3
    assert controller._layers is not None
    assert controller._layers.visible_class_ids == (0, 1, 2)


def test_hiding_a_layer_updates_the_rendered_pixmap(qtbot):
    controller, viewer, layer_panel, source = _make_controller(qtbot)
    class_map = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int32)
    result = _make_unsupervised_result(class_map)
    controller._on_result(result)
    before = viewer._photo.pixmap().toImage().copy()

    controller._on_layer_visibility_changed(1, False)

    after = viewer._photo.pixmap().toImage()
    assert controller._layers.visible_class_ids == (0, 2)
    assert before != after


def test_hiding_a_layer_reveals_the_true_colour_base_image(qtbot):
    controller, viewer, layer_panel, source = _make_controller(qtbot)
    class_map = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int32)
    result = _make_unsupervised_result(class_map)
    base_rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    base_rgb[..., 2] = 40  # a distinct, otherwise-unused blue base image
    source.data.rgb_array = base_rgb
    controller._active_data = source.data  # set by the real click handler pre-launch
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
    assert controller._layers.layers[2].opacity == 0.5

    qtbot.wait(150)

    after = viewer._photo.pixmap().toImage()
    assert before != after


def test_clear_result_clears_the_layer_panel(qtbot):
    controller, viewer, layer_panel, source = _make_controller(qtbot)
    class_map = np.array([[0, 1]], dtype=np.int32)
    controller._on_result(_make_unsupervised_result(class_map))

    controller.clear_result()

    assert len(layer_panel._rows) == 0
    assert controller._layers is None


# ---------------------------------------------------------------------- #
# Super-Resolution-aware classification                                    #
# ---------------------------------------------------------------------- #


def test_classifying_with_super_resolution_active_targets_its_data_and_notifies(
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

    assert controller._active_data is source.data
    assert len(infos) == 1
    assert "Super-Resolution" in infos[0][1]


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

    assert controller._active_data is source.data
    assert len(infos) == 0


def test_super_resolution_notice_is_shown_only_once(
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

    assert len(infos) == 1
