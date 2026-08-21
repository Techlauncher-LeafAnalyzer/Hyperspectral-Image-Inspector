from __future__ import annotations

import numpy as np
import pytest

from core import HSIReader, VisualizationService
from ui.hypercube_widget import HypercubeWidget, _colorize_slices


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


def test_colorize_slices_handles_uniform_input():
    slices = [np.full((3, 3), 0.4, dtype=np.float32)]

    colored = _colorize_slices(slices)

    assert colored[0].shape == (3, 3, 3)
    assert colored[0].dtype == np.uint8


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
