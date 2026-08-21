from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtTest import QSignalSpy

from core import hsi_utils
from ui.viewer import HSIViewer


@pytest.fixture
def viewer_with_photo(qtbot):
    view = HSIViewer()
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    view.set_photo(hsi_utils.numpy_to_qpixmap(rgb))
    qtbot.wait(50)  # let the post-set_photo fit-in-view settle
    return view


def test_crop_drag_emits_crop_requested_with_the_dragged_rect(viewer_with_photo, qtbot):
    # Anchor both drag points near the viewport's center rather than fixed
    # pixel coordinates: the platform is free to size the top-level widget
    # however it likes, and fit-in-view letterboxes the photo within it, so
    # only points near the center are guaranteed to land on the photo. The
    # offset must also scale with the current zoom so the drag covers at
    # least one scene unit in each direction (fit-in-view may zoom in a lot
    # on a small widget).
    scale = viewer_with_photo.transform().m11()
    offset = max(5.0, scale * 1.5)
    center = viewer_with_photo.viewport().rect().center()
    start_view_pos = QPointF(center) - QPointF(offset, offset)
    end_view_pos = QPointF(center) + QPointF(offset, offset)
    expected_start = viewer_with_photo.mapToScene(start_view_pos.toPoint())
    expected_end = viewer_with_photo.mapToScene(end_view_pos.toPoint())
    image_rect = QRectF(viewer_with_photo._photo.pixmap().rect())
    expected_rect = QRectF(expected_start, expected_end).normalized().intersected(image_rect)
    assert expected_rect.width() >= 1 and expected_rect.height() >= 1

    with qtbot.waitSignal(viewer_with_photo.cropRequested, timeout=1000) as blocker:
        viewer_with_photo._begin_crop_mode()
        viewer_with_photo._crop_start = expected_start
        viewer_with_photo.mouseReleaseEvent(_fake_release_event(end_view_pos))

    assert blocker.args[0] == expected_rect


def test_crop_mode_ends_after_a_release(viewer_with_photo):
    viewer_with_photo._begin_crop_mode()
    viewer_with_photo._crop_start = QPointF(1, 1)

    viewer_with_photo.mouseReleaseEvent(_fake_release_event(QPointF(50, 50)))

    assert viewer_with_photo._cropping is False


def test_view_state_round_trips_through_get_and_set(viewer_with_photo):
    original_state = (2.0, QPointF(4.0, 4.0))

    viewer_with_photo.set_view_state(original_state)
    scale, center = viewer_with_photo.get_view_state()

    assert scale == pytest.approx(2.0)
    assert center.x() == pytest.approx(4.0, abs=1.0)
    assert center.y() == pytest.approx(4.0, abs=1.0)


def test_get_view_state_is_none_without_a_photo(qtbot):
    view = HSIViewer()
    qtbot.addWidget(view)

    assert view.get_view_state() is None


def test_visualization_context_menu_keeps_actions_and_polished_identity(
    viewer_with_photo,
):
    viewer_with_photo.setObjectName("viewer")
    viewer_with_photo._pixel_overlay_enabled = True

    menu = viewer_with_photo._build_context_menu(QPointF(2, 3))
    actions = {action.text(): action for action in menu.actions() if action.text()}

    assert menu.objectName() == "visualizationContextMenu"
    assert menu.minimumWidth() == 252
    assert {
        "Spectrum Plot",
        "Clear Selection",
        "Index Mean",
        "Show Pixel Values",
        "Crop",
    } <= set(actions)
    assert actions["Show Pixel Values"].isCheckable()
    assert actions["Show Pixel Values"].isChecked()
    assert actions["Index Mean"].menu().objectName() == "visualizationIndexMenu"
    assert [action.text() for action in actions["Index Mean"].menu().actions()] == [
        "NDVI",
        "EVI",
        "MCARI",
        "MTVI",
        "OSAVI",
        "PRI",
    ]
    assert all(not action.icon().isNull() for action in actions.values())


def test_context_menu_polish_is_scoped_to_visualization_viewer(viewer_with_photo):
    viewer_with_photo.setObjectName("superResViewer")

    menu = viewer_with_photo._build_context_menu(QPointF())

    assert menu.objectName() == ""
    assert menu.actions()[2].menu().objectName() == ""


def test_context_menu_actions_keep_existing_signal_wiring(viewer_with_photo):
    scene_pos = QPointF(2, 3)
    spectrum_spy = QSignalSpy(viewer_with_photo.spectrumPlotRequested)
    mean_spy = QSignalSpy(viewer_with_photo.meanIndexRequested)
    menu = viewer_with_photo._build_context_menu(scene_pos)
    actions = {action.text(): action for action in menu.actions() if action.text()}

    actions["Spectrum Plot"].trigger()
    actions["Index Mean"].menu().actions()[0].trigger()
    actions["Show Pixel Values"].trigger()
    actions["Crop"].trigger()

    assert list(spectrum_spy[0]) == [scene_pos]
    assert list(mean_spy[0]) == ["NDVI"]
    assert viewer_with_photo._pixel_overlay_enabled is True
    assert viewer_with_photo._cropping is True


def _fake_release_event(scene_pos: QPointF):
    """A minimal stand-in for a left-button mouse release at ``scene_pos``."""
    from PyQt6.QtCore import QPointF as _QPointF
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QMouseEvent

    return QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        _QPointF(scene_pos),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
