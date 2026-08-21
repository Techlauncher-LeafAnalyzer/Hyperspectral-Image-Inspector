from __future__ import annotations

import math
from typing import Optional

import numpy as np
from matplotlib import colormaps
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QMouseEvent, QResizeEvent, QSurfaceFormat, QWheelEvent
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import QLabel, QWidget

from core import HypercubeViewData


def _colorize_slices(slices: list[np.ndarray]) -> list[np.ndarray]:
    """Map same-scale float slices to ``uint8`` RGB via one shared stretch.

    All slices are stretched together (not independently) so a viewer can
    compare values across the cube's four side faces on one visual scale.
    """
    stacked = np.concatenate([values.ravel() for values in slices])
    finite = stacked[np.isfinite(stacked)]
    cmap = colormaps["gray"]
    if finite.size == 0:
        return [
            np.zeros((*values.shape, 3), dtype=np.uint8) for values in slices
        ]

    low, high = np.percentile(finite, (2.0, 98.0))
    if high <= low:
        high = low + np.finfo(np.float32).eps

    colored = []
    for values in slices:
        normalized = np.clip((values - low) / (high - low), 0, 1).astype(np.float32)
        rgb = np.round(cmap(normalized)[..., :3] * 255).astype(np.uint8)
        colored.append(np.ascontiguousarray(rgb))
    return colored


class HypercubeWidget(QOpenGLWidget):
    """Interactive OpenGL cube rendering a :class:`core.HypercubeViewData`.

    Left-drag rotates, Ctrl+left-drag zooms, Shift+left-drag pans, and the
    scroll wheel also zooms. Pure PyQt6 + PyOpenGL: no ``spectral.graphics``
    or PySide6 import, since this app cannot load a second Qt binding.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        self.setFormat(fmt)

        self._view_data: Optional[HypercubeViewData] = None
        self._textures: Optional[list[int]] = None
        self._textures_dirty = False

        self._camera_distance = 5.0
        self._camera_theta = 55.0
        self._camera_phi = 35.0
        self._target = [0.0, 0.0, 0.0]

        self._drag_start: Optional[QPoint] = None
        self._drag_modifiers = Qt.KeyboardModifier.NoModifier

        self._status_label = QLabel(self)
        self._status_label.setObjectName("hypercubeStatusLabel")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet(
            "QLabel#hypercubeStatusLabel {"
            "  background-color: rgba(20, 20, 20, 200);"
            "  color: #f5f5f5;"
            "  border: 1px solid #555555;"
            "  border-radius: 4px;"
            "  padding: 8px 12px;"
            "  font-size: 12px;"
            "}"
        )
        self._status_label.hide()
        self.set_status_message("Load an image to view its hypercube.")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def set_data(self, view_data: Optional[HypercubeViewData]) -> None:
        self._view_data = view_data
        self._textures_dirty = True
        self.update()

    def set_status_message(self, message: Optional[str]) -> None:
        if message:
            self._status_label.setText(message)
            self._status_label.adjustSize()
            self._status_label.show()
            self._position_status_label()
        else:
            self._status_label.hide()
        self._status_label.raise_()

    # ------------------------------------------------------------------ #
    # QOpenGLWidget overrides                                              #
    # ------------------------------------------------------------------ #

    def initializeGL(self) -> None:
        import OpenGL.GL as gl

        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glClearColor(0.12, 0.12, 0.12, 1.0)
        gl.glShadeModel(gl.GL_FLAT)

    def resizeGL(self, width: int, height: int) -> None:
        import OpenGL.GL as gl
        import OpenGL.GLU as glu

        gl.glViewport(0, 0, max(width, 1), max(height, 1))
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(45.0, width / max(height, 1), 0.1, 100.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def paintGL(self) -> None:
        import OpenGL.GL as gl
        import OpenGL.GLU as glu

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        if self._textures_dirty:
            self._rebuild_textures()
        if self._textures is None:
            return

        gl.glLoadIdentity()
        eye = self._camera_position()
        glu.gluLookAt(
            eye[0], eye[1], eye[2],
            self._target[0], self._target[1], self._target[2],
            0.0, 0.0, 1.0,
        )
        self._draw_cube()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._position_status_label()

    # ------------------------------------------------------------------ #
    # Cube construction                                                    #
    # ------------------------------------------------------------------ #

    def _camera_position(self) -> tuple[float, float, float]:
        theta = math.radians(self._camera_theta)
        phi = math.radians(self._camera_phi)
        r = self._camera_distance
        x = r * math.sin(theta) * math.cos(phi)
        y = r * math.sin(theta) * math.sin(phi)
        z = r * math.cos(theta)
        return (x, y, z)

    def _rebuild_textures(self) -> None:
        import OpenGL.GL as gl

        self._textures_dirty = False
        if self._textures is not None:
            gl.glDeleteTextures(self._textures)
            self._textures = None
        if self._view_data is None:
            return

        images = self._build_face_images(self._view_data)
        texture_ids = [int(value) for value in gl.glGenTextures(len(images))]
        for texture_id, image in zip(texture_ids, images):
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            height, width = image.shape[:2]
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0,
                gl.GL_RGB, gl.GL_UNSIGNED_BYTE, np.ascontiguousarray(image),
            )
        self._textures = texture_ids

    @staticmethod
    def _build_face_images(view_data: HypercubeViewData) -> list[np.ndarray]:
        """Return ``[top, front, right, back, left, bottom]`` uint8 RGB images.

        Face naming matches ``VisualizationService.prepare_hypercube_view``'s
        own boundary layout: front = last row, right = last column,
        back = first row, left = first column.
        """
        surface = view_data.surface_cube
        front = surface[-1, :, :]
        right = surface[:, -1, :]
        back = surface[0, :, :]
        left = surface[:, 0, :]
        colored_front, colored_right, colored_back, colored_left = _colorize_slices(
            [front, right, back, left]
        )
        top = np.ascontiguousarray(view_data.top_rgb)
        return [top, colored_front, colored_right, colored_back, colored_left, top]

    def _draw_cube(self) -> None:
        import OpenGL.GL as gl

        assert self._textures is not None
        top, front, right, back, left, bottom = self._textures
        hw = hh = 1.0
        hz = 0.6

        faces = (
            (top, ((-hw, -hh, hz), (hw, -hh, hz), (hw, hh, hz), (-hw, hh, hz))),
            (front, ((-hw, -hh, -hz), (hw, -hh, -hz), (hw, -hh, hz), (-hw, -hh, hz))),
            (right, ((hw, -hh, -hz), (hw, hh, -hz), (hw, hh, hz), (hw, -hh, hz))),
            (back, ((hw, hh, -hz), (-hw, hh, -hz), (-hw, hh, hz), (hw, hh, hz))),
            (left, ((-hw, hh, -hz), (-hw, -hh, -hz), (-hw, -hh, hz), (-hw, hh, hz))),
            (bottom, ((-hw, hh, -hz), (hw, hh, -hz), (hw, -hh, -hz), (-hw, -hh, -hz))),
        )
        tex_coords = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        for texture_id, vertices in faces:
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
            gl.glBegin(gl.GL_QUADS)
            for (tx, ty), (vx, vy, vz) in zip(tex_coords, vertices):
                gl.glTexCoord2f(tx, ty)
                gl.glVertex3f(vx, vy, vz)
            gl.glEnd()

    # ------------------------------------------------------------------ #
    # Mouse / wheel interaction                                            #
    # ------------------------------------------------------------------ #

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position().toPoint()
            self._drag_modifiers = event.modifiers()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = None

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_start is None:
            return
        pos = event.position().toPoint()
        dx = pos.x() - self._drag_start.x()
        dy = pos.y() - self._drag_start.y()
        self._drag_start = pos
        modifiers = self._drag_modifiers

        if modifiers & Qt.KeyboardModifier.ControlModifier:
            self._camera_distance = max(
                1.5, min(20.0, self._camera_distance * (1.0 - dy / 200.0))
            )
        elif modifiers & Qt.KeyboardModifier.ShiftModifier:
            self._target[0] += dx / 100.0
            self._target[2] -= dy / 100.0
        else:
            self._camera_phi = (self._camera_phi - dx * 0.5) % 360.0
            self._camera_theta = max(5.0, min(175.0, self._camera_theta - dy * 0.5))
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.1 if event.angleDelta().y() > 0 else 1 / 1.1
        self._camera_distance = max(1.5, min(20.0, self._camera_distance / factor))
        self.update()

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _position_status_label(self) -> None:
        if not self._status_label.isVisible():
            return
        x = (self.width() - self._status_label.width()) // 2
        y = (self.height() - self._status_label.height()) // 2
        self._status_label.move(max(0, x), max(0, y))
