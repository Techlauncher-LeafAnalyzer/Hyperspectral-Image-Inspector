from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from spectral.graphics.hypercube import HypercubeWindow
from spectral.graphics.colorscale import ColorScale


class AxisHypercubeWindow(HypercubeWindow):
    """SPy hypercube with camera-aware row, column, and wavelength axes."""

    AXIS_COLORS = {
        "Rows": (0.78, 0.16, 0.16),
        "Columns": (0.10, 0.50, 0.20),
        "Wavelength": (0.08, 0.34, 0.74),
    }
    MIN_CAMERA_DISTANCE = 2.0
    MAX_CAMERA_DISTANCE = 9.0
    # Positive row/column/wavelength octant: diagonally opposite the shared
    # negative-corner axis origin used by ``axis_geometry``.
    DEFAULT_CAMERA = (7.0, 55.0, 45.0)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.camera_pos_rtp = list(self.DEFAULT_CAMERA)

    @staticmethod
    def bright_color_scale() -> ColorScale:
        """Return an opaque, high-luminance SPy scale for spectral faces."""
        colors = np.array(
            [
                [35, 85, 215],
                [0, 190, 255],
                [75, 230, 135],
                [255, 220, 55],
                [255, 130, 45],
                [255, 248, 245],
            ],
            dtype=int,
        )
        scale = ColorScale(np.arange(6, dtype=float), colors, num_tics=256)
        scale.set_background_color(colors[0])
        return scale

    @classmethod
    def zoomed_camera_distance(cls, distance: float, wheel_delta: int) -> float:
        """Return a bounded camera radius for a standard Qt wheel delta."""
        steps = wheel_delta / 120.0
        zoom_factor = 0.85**steps
        return min(
            cls.MAX_CAMERA_DISTANCE,
            max(cls.MIN_CAMERA_DISTANCE, float(distance) * zoom_factor),
        )

    @staticmethod
    def axis_geometry(
        shape: tuple[int, int, int], cube_height: float
    ) -> tuple[tuple[float, float, float], dict[str, tuple[float, float, float]]]:
        divisor = max(shape[:2])
        half_rows, half_columns = (float(value) / divisor for value in shape[:2])
        half_spectral = float(cube_height)
        extension = 0.50
        origin = (-half_rows, -half_columns, -half_spectral)
        endpoints = {
            "Rows": (half_rows + extension, origin[1], origin[2]),
            "Columns": (origin[0], half_columns + extension, origin[2]),
            "Wavelength": (origin[0], origin[1], half_spectral + extension),
        }
        return origin, endpoints

    @staticmethod
    def arrowhead_anchor(
        label: str, endpoint: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Return a point behind the tip for projected arrow direction."""
        x, y, z = endpoint
        length = 0.18
        if label == "Rows":
            return (x - length, y, z)
        if label == "Columns":
            return (x, y - length, z)
        if label == "Wavelength":
            return (x, y, z - length)
        raise ValueError(f"Unknown axis label: {label}")

    @staticmethod
    def screen_arrowhead_points(
        tip: tuple[float, float], anchor: tuple[float, float]
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return two camera-facing line-arrow wing endpoints in screen pixels."""
        dx = tip[0] - anchor[0]
        dy = tip[1] - anchor[1]
        magnitude = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        unit_x, unit_y = dx / magnitude, dy / magnitude
        perpendicular_x, perpendicular_y = -unit_y, unit_x
        length = 14.0
        half_width = 6.5
        base_x = tip[0] - unit_x * length
        base_y = tip[1] - unit_y * length
        return (
            (
                base_x + perpendicular_x * half_width,
                base_y + perpendicular_y * half_width,
            ),
            (
                base_x - perpendicular_x * half_width,
                base_y - perpendicular_y * half_width,
            ),
        )

    @staticmethod
    def axis_label_x(label: str, tip_x: float, text_width: int) -> float:
        """Place Rows left of its arrow; keep other labels on the right."""
        if label == "Rows":
            return tip_x - text_width - 26.0
        return tip_x + 7.0

    def paintGL(self) -> None:  # noqa: N802 - Qt/OpenGL API name
        import OpenGL.GL as gl
        import OpenGL.GLU as glu

        # QPainter is used for axis labels after each OpenGL pass and may
        # alter compatibility-profile state. Restore depth/opacity explicitly
        # before SPy draws the next frame so rear faces cannot bleed through.
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_TEXTURE_2D)
        super().paintGL()

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()
        glu.gluLookAt(
            *(list(self._camera_xyz()) + list(self.target_pos) + list(self.up))
        )
        origin, endpoints = self.axis_geometry(self.hsi.shape, self.cubeHeight)

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
        gl.glEnd()

        model = gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX)
        projection = gl.glGetDoublev(gl.GL_PROJECTION_MATRIX)
        viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
        projected_positions = {
            label: glu.gluProject(*endpoint, model, projection, viewport)
            for label, endpoint in endpoints.items()
        }
        projected_anchors = {
            label: glu.gluProject(
                *self.arrowhead_anchor(label, endpoint), model, projection, viewport
            )
            for label, endpoint in endpoints.items()
        }
        screen_positions = {}
        for label, projected in projected_positions.items():
            x, y, depth = projected
            if not (0 <= x < viewport[2] and 0 <= y < viewport[3]):
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
            if depth <= buffer_depth + 0.002:
                screen_positions[label] = (projected, projected_anchors[label])

        gl.glDepthFunc(gl.GL_LESS)
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glPopMatrix()
        gl.glFlush()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont(painter.font())
        font.setBold(True)
        font.setPointSize(10)
        painter.setFont(font)
        for label, (projected, projected_anchor) in screen_positions.items():
            x, y, _depth = projected
            anchor_x, anchor_y, _anchor_depth = projected_anchor
            color = self.AXIS_COLORS[label]
            pen = QPen(QColor.fromRgbF(*color))
            pen.setWidthF(3.0)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            tip = (float(x), float(self.height() - y))
            anchor = (float(anchor_x), float(self.height() - anchor_y))
            left, right = self.screen_arrowhead_points(tip, anchor)
            painter.drawLine(QPointF(*tip), QPointF(*left))
            painter.drawLine(QPointF(*tip), QPointF(*right))
            suffix = " (nm)" if label == "Wavelength" else ""
            text = label + suffix
            text_x = self.axis_label_x(
                label, float(x), painter.fontMetrics().horizontalAdvance(text)
            )
            painter.drawText(round(text_x), self.height() - round(y) - 5, text)
        painter.end()

    def draw_cube(self, *args, **kwargs) -> None:
        """Draw opaque faces with RGB only on SPy's upper cube face."""
        import OpenGL.GL as gl

        gl.glDisable(gl.GL_BLEND)
        gl.glColor4f(1.0, 1.0, 1.0, 1.0)
        gl.glTexEnvi(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_REPLACE)
        super().draw_cube(*args, **kwargs)

        # SPy applies texture 0 to both top and bottom. Cover only the lower
        # face with a solid opaque neutral color, leaving RGB exclusively on
        # the positive spectral (upper) face.
        divisor = max(self.hsi.shape[:2])
        half_rows, half_columns = (
            float(value) / divisor for value in self.hsi.shape[:2]
        )
        half_spectral = self.cubeHeight
        gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glColor4f(0.88, 0.91, 0.95, 1.0)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex3f(half_rows, -half_columns, -half_spectral)
        gl.glVertex3f(half_rows, half_columns, -half_spectral)
        gl.glVertex3f(-half_rows, half_columns, -half_spectral)
        gl.glVertex3f(-half_rows, -half_columns, -half_spectral)
        gl.glEnd()
        gl.glDepthFunc(gl.GL_LESS)
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glColor4f(1.0, 1.0, 1.0, 1.0)

    def _camera_xyz(self) -> list[float]:
        from spectral.graphics.hypercube import rtp_to_xyz

        return rtp_to_xyz(*self.camera_pos_rtp)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt API name
        delta = event.angleDelta().y()
        if delta == 0:
            delta = event.pixelDelta().y()
        if delta:
            self.camera_pos_rtp[0] = self.zoomed_camera_distance(
                self.camera_pos_rtp[0], delta
            )
            self.update()
            event.accept()
            return
        super().wheelEvent(event)


class HypercubeViewer(QMainWindow):
    """Hosts SPy's interactive OpenGL cube and adds presentation export."""

    def __init__(
        self,
        surface_cube: np.ndarray,
        top_rgb: np.ndarray,
        title: str,
    ) -> None:
        super().__init__()
        self.setWindowTitle(f"LeafSpectra — {title}")
        self.resize(800, 720)

        self.cube = AxisHypercubeWindow(
            surface_cube,
            self,
            -1,
            top=top_rgb,
            scale=AxisHypercubeWindow.bright_color_scale(),
            background=(1.0, 1.0, 1.0),
            title=title,
        )
        export_button = QPushButton("Export current view…")
        export_button.clicked.connect(self._choose_export)
        reset_button = QPushButton("Reset view")
        reset_button.clicked.connect(self.reset_view)
        help_label = QLabel(
            "Red: rows   |   Green: columns   |   Blue: wavelength   |   "
            "Drag: rotate   |   Wheel: zoom"
        )

        controls = QHBoxLayout()
        controls.addWidget(help_label, 1)
        controls.addWidget(reset_button)
        controls.addWidget(export_button)

        layout = QVBoxLayout()
        layout.addWidget(self.cube, 1)
        layout.addLayout(controls)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    def reset_view(self) -> None:
        self.cube.camera_pos_rtp = list(AxisHypercubeWindow.DEFAULT_CAMERA)
        self.cube.target_pos = [0.0, 0.0, 0.0]
        self.cube.cubeHeight = 1.0
        self.cube.light = False
        self.cube.update()

    def export_current_view(self, path: str | Path) -> bool:
        output = Path(path)
        if not output.suffix:
            output = output.with_suffix(".png")
        image = self.cube.grabFramebuffer()
        if image.isNull():
            return False
        return image.save(str(output))

    def _choose_export(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export current hypercube view",
            "hypercube-view.png",
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg)",
        )
        if not filename:
            return
        if self.export_current_view(filename):
            self.statusBar().showMessage(f"Exported {filename}", 5000)
        else:
            QMessageBox.critical(self, "LeafSpectra", "Could not capture the cube view.")


class MainWindow(QMainWindow):
    """Small test view; all hyperspectral work is delegated to the Controller."""

    open_requested = Signal(str)
    render_requested = Signal(str, object)
    closing = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._cube_window: HypercubeViewer | None = None
        self.setWindowTitle("LeafSpectra — Visualization Tester")
        self.resize(920, 760)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select a hyperspectral .hdr or data file")
        self.path_edit.returnPressed.connect(self._emit_open)
        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self._browse)
        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self._emit_open)

        file_row = QHBoxLayout()
        file_row.addWidget(self.path_edit, 1)
        file_row.addWidget(browse_button)
        file_row.addWidget(self.load_button)

        self.mode_combo = QComboBox()
        self.mode_combo.currentTextChanged.connect(self._mode_changed)
        self.band_spin = QSpinBox()
        self.band_spin.setRange(0, 0)
        self.band_spin.setPrefix("Band ")
        self.band_spin.setVisible(False)
        self.render_button = QPushButton("Render")
        self.render_button.setEnabled(False)
        self.render_button.clicked.connect(self._emit_render)

        control_row = QHBoxLayout()
        control_row.addWidget(QLabel("View:"))
        control_row.addWidget(self.mode_combo)
        control_row.addWidget(self.band_spin)
        control_row.addStretch(1)
        control_row.addWidget(self.render_button)

        self.dataset_label = QLabel("No cube loaded")
        self.dataset_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.image_label = QLabel("Load a cube and click Render")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(480, 360)
        self.image_label.setStyleSheet("QLabel { background: #1d1f21; color: #c9c9c9; }")
        scroll = QScrollArea()
        scroll.setWidget(self.image_label)
        scroll.setWidgetResizable(False)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.status_label = QLabel("Ready")

        status_row = QHBoxLayout()
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.progress)

        layout = QVBoxLayout()
        layout.addLayout(file_row)
        layout.addLayout(control_row)
        layout.addWidget(self.dataset_label)
        layout.addWidget(scroll, 1)
        layout.addLayout(status_row)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    def set_modes(self, modes: list[str]) -> None:
        self.mode_combo.clear()
        self.mode_combo.addItems(modes)

    def set_loaded_cube(self, path: Path, shape: tuple[int, int, int]) -> None:
        self.path_edit.setText(str(path))
        rows, columns, bands = shape
        self.dataset_label.setText(
            f"{path.name}  |  {rows} rows × {columns} columns × {bands} bands"
        )
        self.band_spin.setRange(0, max(0, bands - 1))
        self.render_button.setEnabled(True)
        self.status_label.setText("Cube loaded; choose a view and render")

    def set_busy(self, busy: bool) -> None:
        self.load_button.setEnabled(not busy)
        self.render_button.setEnabled(not busy and bool(self.path_edit.text().strip()))
        self.mode_combo.setEnabled(not busy)
        self.band_spin.setEnabled(not busy)
        if busy:
            self.progress.setValue(0)

    def set_progress(self, value: int, message: str) -> None:
        self.progress.setValue(value)
        self.status_label.setText(message)

    def show_visualization(self, rgb: np.ndarray, title: str) -> None:
        image = np.ascontiguousarray(rgb, dtype=np.uint8)
        height, width, channels = image.shape
        if channels != 3:
            raise ValueError("Visualization display data must have three RGB channels.")
        qimage = QImage(
            image.data,
            width,
            height,
            int(image.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap)
        self.image_label.resize(pixmap.size())
        self.status_label.setText(title)
        self.progress.setValue(100)

    def show_hypercube(
        self,
        surface_cube: np.ndarray,
        top_rgb: np.ndarray,
        title: str,
    ) -> None:
        if self._cube_window is not None:
            self._cube_window.close()
        self._cube_window = HypercubeViewer(surface_cube, top_rgb, title)
        self._cube_window.show()
        self._cube_window.raise_()
        self._cube_window.activateWindow()
        self.status_label.setText("Interactive hypercube opened")
        self.progress.setValue(100)

    def show_error(self, message: str) -> None:
        self.status_label.setText(message)
        QMessageBox.critical(self, "LeafSpectra", message)

    def _browse(self) -> None:
        start = self.path_edit.text().strip() or str(Path.cwd())
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open hyperspectral image",
            start,
            "Hyperspectral files (*.hdr *.bil *.bip *.bsq);;All files (*)",
        )
        if filename:
            self.path_edit.setText(filename)
            self.open_requested.emit(filename)

    def _emit_open(self) -> None:
        path = self.path_edit.text().strip()
        if path:
            self.open_requested.emit(path)

    def _emit_render(self) -> None:
        band = self.band_spin.value() if self.mode_combo.currentText() == "BAND" else None
        self.render_requested.emit(self.mode_combo.currentText(), band)

    def _mode_changed(self, mode: str) -> None:
        self.band_spin.setVisible(mode == "BAND")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API name
        if self._cube_window is not None:
            self._cube_window.close()
        self.closing.emit()
        super().closeEvent(event)
