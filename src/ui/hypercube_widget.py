from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import (
    QColor,
    QFont,
    QMouseEvent,
    QPainter,
    QPen,
    QResizeEvent,
    QSurfaceFormat,
    QWheelEvent,
)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import QFileDialog, QLabel, QMessageBox, QPushButton, QWidget

from core import HypercubeViewData


_BRIGHT_SPECTRAL_COLORS = np.array(
    (
        (35, 85, 215),
        (0, 190, 255),
        (75, 230, 135),
        (255, 220, 55),
        (255, 130, 45),
        (255, 248, 245),
    ),
    dtype=int,
)
_SPECTRAL_COLORMAP = LinearSegmentedColormap.from_list(
    "hypercube_spectral_faces", _BRIGHT_SPECTRAL_COLORS / 255.0
)


def _colorize_slices(slices: list[np.ndarray]) -> list[np.ndarray]:
    """Map float side slices to a bright, shared ``uint8`` RGB scale.

    All slices are stretched together over their full finite range (not
    independently) so a viewer can compare values across the cube's four side
    faces on the same scale used by the PySide6 reference implementation.
    """
    stacked = np.concatenate([values.ravel() for values in slices])
    finite = stacked[np.isfinite(stacked)]
    if finite.size == 0:
        return [
            np.zeros((*values.shape, 3), dtype=np.uint8) for values in slices
        ]

    low = float(finite.min())
    high = float(finite.max())
    if high <= low:
        high = low + np.finfo(np.float32).eps

    colored = []
    for values in slices:
        normalized = np.clip((values - low) / (high - low), 0, 1).astype(np.float32)
        rgb = np.round(_SPECTRAL_COLORMAP(normalized)[..., :3] * 255).astype(np.uint8)
        colored.append(np.ascontiguousarray(rgb))
    return colored


class HypercubeWidget(QOpenGLWidget):
    """Interactive OpenGL cube rendering a :class:`core.HypercubeViewData`.

    Left-drag rotates, Ctrl+left-drag zooms, Shift+left-drag pans, and the
    scroll wheel also zooms. The view ports the presentation behaviour of the
    SPy reference implementation without importing its PySide6 widget, since
    this application must use one Qt binding (PyQt6) per process.
    """

    AXIS_COLORS = {
        "Rows": (0.78, 0.16, 0.16),
        "Columns": (0.10, 0.50, 0.20),
        "Wavelength": (0.08, 0.34, 0.74),
    }
    BACKGROUND_COLOR = (1.0, 1.0, 1.0, 1.0)
    MIN_CAMERA_DISTANCE = 2.0
    MAX_CAMERA_DISTANCE = 9.0
    # Rotate the reference view one quarter-turn around the wavelength axis
    # so its default presentation matches the application's axis convention.
    DEFAULT_CAMERA = (7.0, 55.0, 135.0)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        self.setFormat(fmt)

        self._view_data: Optional[HypercubeViewData] = None
        self._textures: Optional[list[int]] = None
        self._textures_dirty = False

        self._camera_distance, self._camera_theta, self._camera_phi = self.DEFAULT_CAMERA
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

        self._reset_button = QPushButton("Reset view", self)
        self._reset_button.setObjectName("hypercubeResetButton")
        self._reset_button.clicked.connect(self.reset_view)
        self._export_button = QPushButton("Export current view", self)
        self._export_button.setObjectName("hypercubeExportButton")
        self._export_button.clicked.connect(self._choose_export)
        self._position_controls()

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

    def reset_view(self) -> None:
        """Restore the reference view's initial camera position."""
        self._camera_distance, self._camera_theta, self._camera_phi = self.DEFAULT_CAMERA
        self._target = [0.0, 0.0, 0.0]
        self.update()

    def export_current_view(self, path: str | Path) -> bool:
        """Save the current framebuffer as PNG (by default) or another Qt format."""
        output_path = Path(path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".png")
        image = self.grabFramebuffer()
        return not image.isNull() and image.save(str(output_path))

    # ------------------------------------------------------------------ #
    # QOpenGLWidget overrides                                              #
    # ------------------------------------------------------------------ #

    def initializeGL(self) -> None:
        import OpenGL.GL as gl

        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glClearColor(*self.BACKGROUND_COLOR)
        gl.glShadeModel(gl.GL_FLAT)

    def resizeGL(self, width: int, height: int) -> None:
        import OpenGL.GL as gl

        gl.glViewport(0, 0, max(width, 1), max(height, 1))
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        self._apply_perspective(45.0, width / max(height, 1), 0.1, 100.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def paintGL(self) -> None:
        import OpenGL.GL as gl

        # Qt requires raw OpenGL commands to be enclosed by
        # beginNativePainting/endNativePainting when they are mixed with a
        # QPainter overlay on a QOpenGLWidget.  The painter then renders the
        # axis text reliably after the 3D scene has completed.
        painter = QPainter(self)
        painter.beginNativePainting()
        axis_overlay = None
        try:
            gl.glEnable(gl.GL_TEXTURE_2D)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthFunc(gl.GL_LEQUAL)
            gl.glDisable(gl.GL_BLEND)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            # QPainter's own paint-engine setup (triggered by constructing it
            # above) reprograms the fixed-function projection matrix for its
            # own 2D drawing; beginNativePainting() makes raw GL calls safe
            # to issue but does not restore this widget's 3D projection, so
            # it must be reapplied every paintGL call rather than relying on
            # the matrix resizeGL last configured.
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            self._apply_perspective(
                45.0, self.width() / max(self.height(), 1), 0.1, 100.0
            )
            gl.glMatrixMode(gl.GL_MODELVIEW)
            if self._textures_dirty:
                self._rebuild_textures()
            if self._textures is not None:
                gl.glLoadIdentity()
                gl.glTranslatef(0.0, 0.0, -self._camera_distance)
                gl.glRotatef(self._camera_theta - 90.0, 1.0, 0.0, 0.0)
                gl.glRotatef(-self._camera_phi, 0.0, 0.0, 1.0)
                gl.glTranslatef(-self._target[0], -self._target[1], -self._target[2])
                self._draw_cube()
                axis_overlay = self._draw_axes()
        finally:
            painter.endNativePainting()

        if axis_overlay is not None:
            self._draw_axis_labels(painter, *axis_overlay)
        painter.end()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._position_status_label()
        self._position_controls()

    # ------------------------------------------------------------------ #
    # Cube construction                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _apply_perspective(fovy_deg: float, aspect: float, z_near: float, z_far: float) -> None:
        """Apply a perspective projection without OpenGL.GLU.

        gluPerspective lives in libGLU, a separate, optional legacy library
        not guaranteed to be installed alongside OpenGL itself (confirmed
        absent on this machine) -- importing OpenGL.GLU succeeds either way,
        but calling an unresolved GLU function raises
        OpenGL.error.NullFunctionError, which is fatal when it happens inside
        a Qt virtual method override (PyQt aborts the process; it cannot be
        caught as an ordinary exception). glFrustum is core (non-GLU) OpenGL
        and produces the identical projection matrix from the same
        field-of-view/aspect/near/far inputs.
        """
        import OpenGL.GL as gl

        top = z_near * math.tan(math.radians(fovy_deg) / 2.0)
        right = top * max(aspect, 1e-6)
        gl.glFrustum(-right, right, -top, top, z_near, z_far)

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
        # NumPy RGB rows are tightly packed (three bytes per pixel), whereas
        # OpenGL defaults to four-byte row alignment.  Without this, images
        # whose width is not a multiple of four have corrupted texture rows.
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
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
        # Restore the default alignment immediately: it is global context
        # state, and Qt's own text/glyph texture uploads (used to paint the
        # axis labels below) assume the default of 4. Leaving it at 1 here
        # corrupted every subsequent QPainter-drawn text glyph with a
        # scanline-shifted, ghosted appearance.
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
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
        # A side slice is shaped ``(spatial_samples, spectral_bands)``.  Its
        # spatial axis must map to the face's horizontal texture axis and its
        # bands to the cube depth, so transpose it before OpenGL uploads it
        # as a conventional ``(height, width, channels)`` image.
        side_images = [
            np.ascontiguousarray(image.transpose(1, 0, 2))
            for image in (colored_front, colored_right, colored_back, colored_left)
        ]
        return [top, *side_images, top]

    def _draw_cube(self) -> None:
        import OpenGL.GL as gl

        assert self._textures is not None
        assert self._view_data is not None
        faces = self._cube_faces(self._view_data)
        tex_coords = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        for texture_id, (_, vertices) in zip(self._textures, faces):
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
            gl.glBegin(gl.GL_QUADS)
            for (tx, ty), (vx, vy, vz) in zip(tex_coords, vertices):
                gl.glTexCoord2f(tx, ty)
                gl.glVertex3f(vx, vy, vz)
            gl.glEnd()

    def _draw_axes(
        self,
    ) -> tuple[
        tuple[float, float, float],
        dict[str, tuple[float, float, float]],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        set[str],
    ]:
        """Draw the model's row, column, and wavelength axes over the cube."""
        import OpenGL.GL as gl

        assert self._view_data is not None
        origin, endpoints = self.axis_geometry(self._view_data)
        gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glLineWidth(3.0)
        gl.glBegin(gl.GL_LINES)
        for label, endpoint in endpoints.items():
            gl.glColor3f(*self.AXIS_COLORS[label])
            gl.glVertex3f(*origin)
            gl.glVertex3f(*endpoint)
            for tip, wing in self.axis_arrowhead_vertices(label, endpoint):
                gl.glVertex3f(*tip)
                gl.glVertex3f(*wing)
        gl.glEnd()

        model = np.asarray(gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX), dtype=float)
        projection = np.asarray(gl.glGetDoublev(gl.GL_PROJECTION_MATRIX), dtype=float)
        viewport = np.asarray(gl.glGetIntegerv(gl.GL_VIEWPORT), dtype=float)
        visible_labels: set[str] = set()
        for label, endpoint in endpoints.items():
            projected = self._project_point_with_depth(
                endpoint, model, projection, viewport
            )
            if projected is None:
                continue
            x, y, depth = projected
            if not (
                viewport[0] <= x < viewport[0] + viewport[2]
                and viewport[1] <= y < viewport[1] + viewport[3]
            ):
                continue
            buffer_depth = float(
                np.asarray(
                    gl.glReadPixels(
                        int(x),
                        int(y),
                        1,
                        1,
                        gl.GL_DEPTH_COMPONENT,
                        gl.GL_FLOAT,
                    )
                ).reshape(-1)[0]
            )
            if self._is_depth_visible(depth, buffer_depth):
                visible_labels.add(label)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glColor4f(1.0, 1.0, 1.0, 1.0)
        return origin, endpoints, model, projection, viewport, visible_labels

    @staticmethod
    def _cube_half_extents(view_data: HypercubeViewData) -> tuple[float, float, float]:
        row_span = max(
            int(view_data.row_indices[-1] - view_data.row_indices[0]) + 1,
            1,
        )
        column_span = max(
            int(view_data.column_indices[-1] - view_data.column_indices[0]) + 1, 1
        )
        longest_spatial_span = max(row_span, column_span)
        return (
            column_span / longest_spatial_span,
            row_span / longest_spatial_span,
            0.6,
        )

    @classmethod
    def axis_geometry(
        cls, view_data: HypercubeViewData
    ) -> tuple[tuple[float, float, float], dict[str, tuple[float, float, float]]]:
        """Return the axis origin and positive row/column/wavelength tips."""
        half_columns, half_rows, half_spectral = cls._cube_half_extents(view_data)
        extension = 0.5
        origin = (-half_columns, -half_rows, -half_spectral)
        return origin, {
            "Rows": (origin[0], half_rows + extension, origin[2]),
            "Columns": (half_columns + extension, origin[1], origin[2]),
            "Wavelength": (origin[0], origin[1], half_spectral + extension),
        }

    @staticmethod
    def _project_point(
        point: tuple[float, float, float],
        model: np.ndarray,
        projection: np.ndarray,
        viewport: np.ndarray,
    ) -> tuple[float, float] | None:
        """Project one OpenGL point without depending on optional GLU."""
        projected = HypercubeWidget._project_point_with_depth(
            point, model, projection, viewport
        )
        return projected[:2] if projected is not None else None

    @staticmethod
    def _project_point_with_depth(
        point: tuple[float, float, float],
        model: np.ndarray,
        projection: np.ndarray,
        viewport: np.ndarray,
    ) -> tuple[float, float, float] | None:
        """Project a point to window coordinates, retaining its depth value."""
        # OpenGL returns column-major matrices; transpose after NumPy's
        # row-major reshape before multiplying a conventional column vector.
        model_matrix = model.reshape(4, 4).T
        projection_matrix = projection.reshape(4, 4).T
        clip = projection_matrix @ model_matrix @ np.array((*point, 1.0))
        if abs(clip[3]) < np.finfo(float).eps:
            return None
        normalized = clip[:3] / clip[3]
        if not (-1.0 <= normalized[0] <= 1.0 and -1.0 <= normalized[1] <= 1.0):
            return None
        return (
            viewport[0] + (normalized[0] + 1.0) * viewport[2] / 2.0,
            viewport[1] + (normalized[1] + 1.0) * viewport[3] / 2.0,
            (normalized[2] + 1.0) / 2.0,
        )

    @staticmethod
    def _is_depth_visible(projected_depth: float, buffer_depth: float) -> bool:
        """Match the reference's small tolerance for an endpoint's depth."""
        return projected_depth <= buffer_depth + 0.002

    @staticmethod
    def screen_arrowhead_points(
        tip: tuple[float, float], anchor: tuple[float, float]
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return camera-facing arrowhead wings for a projected axis line."""
        dx = tip[0] - anchor[0]
        dy = tip[1] - anchor[1]
        magnitude = max(math.hypot(dx, dy), 1e-6)
        unit_x, unit_y = dx / magnitude, dy / magnitude
        perpendicular_x, perpendicular_y = -unit_y, unit_x
        base_x = tip[0] - unit_x * 14.0
        base_y = tip[1] - unit_y * 14.0
        return (
            (base_x + perpendicular_x * 6.5, base_y + perpendicular_y * 6.5),
            (base_x - perpendicular_x * 6.5, base_y - perpendicular_y * 6.5),
        )

    @staticmethod
    def axis_arrowhead_vertices(
        label: str, endpoint: tuple[float, float, float]
    ) -> tuple[
        tuple[tuple[float, float, float], tuple[float, float, float]],
        tuple[tuple[float, float, float], tuple[float, float, float]],
    ]:
        """Return two 3D arrowhead edges pointing along one positive axis."""
        x, y, z = endpoint
        length = 0.18
        half_width = 0.08
        if label == "Rows":
            wings = ((x - half_width, y - length, z), (x + half_width, y - length, z))
        elif label == "Columns":
            wings = ((x - length, y - half_width, z), (x - length, y + half_width, z))
        elif label == "Wavelength":
            wings = ((x - half_width, y, z - length), (x + half_width, y, z - length))
        else:
            raise ValueError(f"Unknown axis label: {label}")
        return ((endpoint, wings[0]), (endpoint, wings[1]))

    def _draw_axis_labels(
        self,
        painter: QPainter,
        origin: tuple[float, float, float],
        endpoints: dict[str, tuple[float, float, float]],
        model: np.ndarray,
        projection: np.ndarray,
        viewport: np.ndarray,
        visible_labels: set[str],
    ) -> None:
        projected_origin = self._project_point(origin, model, projection, viewport)
        projected = {
            label: self._project_point(endpoint, model, projection, viewport)
            for label, endpoint in endpoints.items()
        }
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont(painter.font())
        font.setBold(True)
        font.setPointSize(10)
        painter.setFont(font)
        for label, position in projected.items():
            if label not in visible_labels or position is None:
                continue
            x, y = position
            color = QColor.fromRgbF(*self.AXIS_COLORS[label])
            pen = QPen(color)
            pen.setWidthF(3.0)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            tip = QPointF(x, self.height() - y)
            origin_tip = (
                (projected_origin[0], self.height() - projected_origin[1])
                if projected_origin is not None
                else (tip.x() - 24.0, tip.y())
            )
            left, right = self.screen_arrowhead_points((tip.x(), tip.y()), origin_tip)
            painter.drawLine(tip, QPointF(*left))
            painter.drawLine(tip, QPointF(*right))
            label_text = f"{label}{' (nm)' if label == 'Wavelength' else ''}"
            text_width = painter.fontMetrics().horizontalAdvance(label_text)
            text_x = x - text_width - 26.0 if label == "Rows" else x + 7.0
            painter.drawText(round(text_x), self.height() - round(y) - 5, label_text)
            painter.drawEllipse(tip, 2.5, 2.5)

    @staticmethod
    def _cube_faces(
        view_data: HypercubeViewData,
    ) -> tuple[tuple[str, tuple[tuple[float, float, float], ...]], ...]:
        """Build textured face geometry from the model payload's axes.

        ``prepare_hypercube_view`` stores ordinary image rows in ascending
        order: row 0 is the back surface and the final row is the front.
        Columns similarly run from the left surface to the right.  Keeping
        that ordering in both the top texture and side-face vertices prevents
        a rotated cube from showing a mirrored dataset.
        """
        hw, hh, hz = HypercubeWidget._cube_half_extents(view_data)

        # Texture coordinates are shared by every face: u advances along the
        # model's first spatial axis and v from the first to final band.  The
        # vertex order below maps u/v directly onto those same axes.
        return (
            ("top", ((-hw, -hh, hz), (hw, -hh, hz), (hw, hh, hz), (-hw, hh, hz))),
            ("front", ((-hw, hh, -hz), (hw, hh, -hz), (hw, hh, hz), (-hw, hh, hz))),
            ("right", ((hw, -hh, -hz), (hw, hh, -hz), (hw, hh, hz), (hw, -hh, hz))),
            ("back", ((-hw, -hh, -hz), (hw, -hh, -hz), (hw, -hh, hz), (-hw, -hh, hz))),
            ("left", ((-hw, -hh, -hz), (-hw, hh, -hz), (-hw, hh, hz), (-hw, -hh, hz))),
            ("bottom", ((-hw, -hh, -hz), (hw, -hh, -hz), (hw, hh, -hz), (-hw, hh, -hz))),
        )

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
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if delta:
            self._camera_distance = self.zoomed_camera_distance(
                self._camera_distance, delta
            )
            self.update()
            event.accept()
            return
        super().wheelEvent(event)

    @classmethod
    def zoomed_camera_distance(cls, distance: float, wheel_delta: int) -> float:
        """Return the reference view's bounded zoom radius for a wheel delta."""
        steps = wheel_delta / 120.0
        zoom_factor = 0.85**steps
        return min(
            cls.MAX_CAMERA_DISTANCE,
            max(cls.MIN_CAMERA_DISTANCE, float(distance) * zoom_factor),
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _position_status_label(self) -> None:
        if not self._status_label.isVisible():
            return
        x = (self.width() - self._status_label.width()) // 2
        y = (self.height() - self._status_label.height()) // 2
        self._status_label.move(max(0, x), max(0, y))

    def _position_controls(self) -> None:
        """Keep the ported reset/export controls in the lower-right corner."""
        margin = 12
        self._export_button.adjustSize()
        self._reset_button.adjustSize()
        export_x = max(margin, self.width() - self._export_button.width() - margin)
        export_y = max(margin, self.height() - self._export_button.height() - margin)
        self._export_button.move(export_x, export_y)
        self._reset_button.move(
            max(margin, export_x - self._reset_button.width() - 8), export_y
        )

    def _choose_export(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export current hypercube view",
            "hypercube-view.png",
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg)",
        )
        if not filename:
            return
        if not self.export_current_view(filename):
            QMessageBox.critical(self, "Hypercube", "Could not capture the cube view.")
