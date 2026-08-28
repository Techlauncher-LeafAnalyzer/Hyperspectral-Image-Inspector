from __future__ import annotations

import numpy as np
import pytest

from core import HSIReader, VisualizationService
from ui.hypercube_widget import HypercubeWidget, _BRIGHT_SPECTRAL_COLORS, _colorize_slices


@pytest.fixture
def hypercube_view_data(synthetic_cube_path):
    data = HSIReader().open(synthetic_cube_path)
    return VisualizationService().prepare_hypercube_view(data)


def test_colorize_slices_returns_uint8_rgb_per_slice():
    slices = [
        np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32),
        np.array([[1.0, 0.0], [0.5, 0.75]], dtype=np.float32),
    ]

    colored = _colorize_slices(slices)

    assert len(colored) == 2
    for original, image in zip(slices, colored):
        assert image.shape == (*original.shape, 3)
        assert image.dtype == np.uint8
    assert tuple(colored[0][0, 0]) == (35, 85, 215)


def test_colorize_slices_handles_uniform_input():
    slices = [np.full((3, 3), 0.4, dtype=np.float32)]

    colored = _colorize_slices(slices)

    assert colored[0].shape == (3, 3, 3)
    assert colored[0].dtype == np.uint8


def test_colorize_slices_uses_the_full_shared_data_range():
    values = np.concatenate((np.arange(100, dtype=np.float32), [10_000.0]))

    colored = _colorize_slices([values])[0]

    assert tuple(colored[0]) == tuple(_BRIGHT_SPECTRAL_COLORS[0])
    assert tuple(colored[-1]) == tuple(_BRIGHT_SPECTRAL_COLORS[-1])
    # A percentile stretch would clip 99 to the maximum colour here. The
    # reference instead preserves the full range shared by all side faces.
    assert tuple(colored[99]) != tuple(_BRIGHT_SPECTRAL_COLORS[-1])


def test_side_faces_map_spatial_width_and_spectral_height():
    view_data = type("ViewData", (), {
        "top_rgb": np.zeros((3, 5, 3), dtype=np.uint8),
        "surface_cube": np.arange(3 * 5 * 7, dtype=np.float32).reshape(3, 5, 7),
    })()

    _, front, right, back, left, _ = HypercubeWidget._build_face_images(view_data)

    # GL texture width is each face's spatial axis and height is depth/bands.
    assert front.shape == back.shape == (7, 5, 3)
    assert right.shape == left.shape == (7, 3, 3)


def test_cube_faces_preserve_model_surface_order(hypercube_view_data):
    faces = dict(HypercubeWidget._cube_faces(hypercube_view_data))
    top = faces["top"]
    front = faces["front"]
    right = faces["right"]
    back = faces["back"]
    left = faces["left"]

    # The model defines row 0 as back / last row as front and column 0 as
    # left / last column as right.  The top texture shares these axes.
    assert all(vertex[1] == top[2][1] for vertex in front)
    assert all(vertex[1] == top[0][1] for vertex in back)
    assert all(vertex[0] == top[1][0] for vertex in right)
    assert all(vertex[0] == top[0][0] for vertex in left)

    top_width = top[1][0] - top[0][0]
    top_height = top[2][1] - top[1][1]
    assert top_width == top_height  # the fixture has equal row/column spans


def test_cube_faces_preserve_model_spatial_aspect_ratio():
    view_data = type("ViewData", (), {
        "row_indices": np.array([0, 3]),
        "column_indices": np.array([0, 7]),
    })()

    top, *_ = HypercubeWidget._cube_faces(view_data)
    _, vertices = top
    width = vertices[1][0] - vertices[0][0]
    height = vertices[2][1] - vertices[1][1]

    assert width == 2 * height


def test_reference_view_camera_defaults_and_bounded_zoom(qtbot):
    widget = HypercubeWidget()
    qtbot.addWidget(widget)

    assert (
        widget._camera_distance,
        widget._camera_theta,
        widget._camera_phi,
    ) == HypercubeWidget.DEFAULT_CAMERA
    assert HypercubeWidget.DEFAULT_CAMERA == (7.0, 55.0, 135.0)
    assert HypercubeWidget.zoomed_camera_distance(2.0, 120) == 2.0
    assert HypercubeWidget.zoomed_camera_distance(9.0, -120) == 9.0

    widget._camera_distance = 3.0
    widget._target = [1.0, 1.0, 1.0]
    widget._reset_button.click()

    assert widget._camera_distance == HypercubeWidget.DEFAULT_CAMERA[0]
    assert widget._target == [0.0, 0.0, 0.0]
    assert widget._export_button.text() == "Export current view"


def test_reference_view_axis_geometry_uses_model_axes():
    view_data = type("ViewData", (), {
        "row_indices": np.array([0, 3]),
        "column_indices": np.array([0, 7]),
    })()

    origin, endpoints = HypercubeWidget.axis_geometry(view_data)

    assert endpoints["Rows"][1] > origin[1]
    assert endpoints["Columns"][0] > origin[0]
    assert endpoints["Wavelength"][2] > origin[2]


def test_reference_view_uses_a_white_canvas_with_colored_axis_labels():
    assert HypercubeWidget.BACKGROUND_COLOR == (1.0, 1.0, 1.0, 1.0)
    assert HypercubeWidget.AXIS_COLORS == {
        "Rows": (0.78, 0.16, 0.16),
        "Columns": (0.10, 0.50, 0.20),
        "Wavelength": (0.08, 0.34, 0.74),
    }


def test_axis_label_projection_handles_opengl_column_major_matrices():
    model = np.eye(4)
    model[3, 0] = 0.5  # OpenGL's column-major translation component.
    projection = np.eye(4)
    viewport = np.array([0.0, 0.0, 200.0, 100.0])

    projected = HypercubeWidget._project_point((0.0, 0.0, 0.0), model, projection, viewport)

    assert projected == (150.0, 50.0)


def test_axis_label_projection_retains_depth_for_occlusion_checks():
    model = np.eye(4)
    projection = np.eye(4)
    viewport = np.array([0.0, 0.0, 200.0, 100.0])

    projected = HypercubeWidget._project_point_with_depth(
        (0.0, 0.0, 0.0), model, projection, viewport
    )

    assert projected == (100.0, 50.0, 0.5)


def test_axis_label_occlusion_matches_reference_depth_tolerance():
    assert HypercubeWidget._is_depth_visible(0.5, 0.498)
    assert not HypercubeWidget._is_depth_visible(0.5, 0.497)


def test_reference_view_projects_arrowheads_in_the_axis_direction():
    left, right = HypercubeWidget.screen_arrowhead_points((20.0, 10.0), (0.0, 10.0))

    assert left[0] < 20.0
    assert right[0] < 20.0


def test_reference_view_has_a_3d_arrowhead_for_each_axis():
    endpoint = (1.0, 2.0, 3.0)

    for label in HypercubeWidget.AXIS_COLORS:
        edges = HypercubeWidget.axis_arrowhead_vertices(label, endpoint)

        assert len(edges) == 2
        assert all(tip == endpoint for tip, _wing in edges)


def test_set_data_accepts_real_hypercube_view_data(qtbot, hypercube_view_data):
    widget = HypercubeWidget()
    qtbot.addWidget(widget)

    widget.set_data(hypercube_view_data)  # must not raise

    assert widget._view_data is hypercube_view_data


def test_set_data_none_clears_state(qtbot, hypercube_view_data):
    widget = HypercubeWidget()
    qtbot.addWidget(widget)
    widget.set_data(hypercube_view_data)

    widget.set_data(None)

    assert widget._view_data is None


def test_set_status_message_toggles_label(qtbot):
    # Deliberately does not call widget.show(): QOpenGLWidget.show() under
    # this project's offscreen test platform aborts the process (a genuine
    # Qt-side fatal assertion, not a catchable exception) when it tries to
    # realize a real GL context. isHidden() reflects the label's own
    # explicit visibility flag without requiring the ancestor chain to be
    # shown, so it is used here instead of isVisible().
    widget = HypercubeWidget()
    qtbot.addWidget(widget)

    widget.set_status_message("Computing hypercube…")
    assert not widget._status_label.isHidden()
    assert widget._status_label.text() == "Computing hypercube…"

    widget.set_status_message(None)
    assert widget._status_label.isHidden()


def test_widget_shows_and_repaints_without_raising(qtbot, hypercube_view_data):
    widget = HypercubeWidget()
    qtbot.addWidget(widget)
    widget.resize(200, 200)
    widget.set_data(hypercube_view_data)

    # show() is what actually triggers resizeGL/paintGL under Qt (a bare
    # repaint() on a never-shown widget does not) -- this is the regression
    # check for OpenGL.GLU: gluPerspective/gluLookAt raised
    # OpenGL.error.NullFunctionError and aborted the process on a machine
    # without libGLU installed, and only surfaced via an actual show().
    widget.show()
    widget.repaint()
