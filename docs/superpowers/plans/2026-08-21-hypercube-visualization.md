# Hypercube Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `VisualizationService.prepare_hypercube_view()` into the UI as an interactive, rotatable 3D cube reachable from the existing (currently disabled) `modeHyperCube` button.

**Architecture:** A `HypercubeWorker` (QObject + QThread pair) computes `HypercubeViewData` off the GUI thread whenever an image loads or is cropped. A `HypercubeWidget` (`QOpenGLWidget` + PyOpenGL) renders that data as a 6-textured-quad cube with mouse-driven rotate/zoom/pan, ported from SPy's own OpenGL technique (not its PySide6 code, which this PyQt6 app cannot load). `MainWindowController` swaps a new `QStackedWidget` between the existing 2D `HSIViewer` and the new `HypercubeWidget` based on which mode radio button is active.

**Tech Stack:** PyQt6 (`QOpenGLWidget`, `QThread`), PyOpenGL (new dependency), NumPy, Matplotlib colormaps (already a dependency).

**Spec:** `docs/superpowers/specs/2026-08-21-hypercube-visualization-design.md`

## Global Constraints

- No import of `spectral.graphics` or PySide6 anywhere in this feature — this app runs PyQt6 only, and the two bindings cannot coexist in one process.
- New dependency is `PyOpenGL` only (added to `requirements.txt`). Do not add `pyqtgraph` or any other 3D/graphics package.
- Hypercube computation runs on a background `QThread`, triggered from the same point the other 7 modes already recompute (`_push_image_to_viewers`, after every load and every crop).
- The worker must receive `dataclasses.replace(self._hsi_data)` (a field-value snapshot), never the live mutable `self._hsi_data` reference — see spec §4 for why.
- A stale worker result (superseded by a newer load/crop before it finished) must never overwrite newer state — enforced via a monotonically increasing generation counter, not by blocking/joining threads.
- Headless tests (`QT_QPA_PLATFORM=offscreen`, set in `ui_tests/conftest.py`) can exercise `HypercubeWidget` code paths (confirmed: `QOpenGLWidget` + real `OpenGL.GL` calls do not raise under offscreen) but cannot assert on rendered pixel output — don't write tests that try.
- Never import or call `OpenGL.GLU` (`gluPerspective`, `gluLookAt`, etc.). `libGLU` is confirmed **not installed** on the reference dev machine — calling an unresolved GLU function raises `OpenGL.error.NullFunctionError`, which is fatal (uncatchable process abort) when raised from inside a Qt virtual method override such as `resizeGL`/`paintGL`, and this reproduces under a real `.show()`, not just the offscreen test platform. Use core (non-GLU) `OpenGL.GL` equivalents instead: `glFrustum` (with fovy/aspect converted to frustum bounds) for `gluPerspective`, and `glTranslatef`/`glRotatef` sequences for camera orbiting instead of `gluLookAt`.
- Regenerate `src/ui/generated/MainWindow.py` with `python -m PyQt6.uic.pyuic src/qt/MainWindow.ui -o src/ui/generated/MainWindow.py` after any `.ui` edit. Never hand-edit that generated file.
- Follow existing test conventions: tests live under `ui_tests/ui/`, use the `qtbot`/`loaded_window`/`synthetic_cube_path` fixtures already in `ui_tests/conftest.py`, and reach into controller/widget internals directly (e.g. `window._hypercube_view_data`) the same way `ui_tests/ui/test_visualization_modes.py` already does — this codebase does not use mocking layers for its own objects.
- **User ruling (2026-08-21, mid-implementation):** automated tests that require `qtbot.waitSignal`/`waitUntil` on a real cross-thread `QThread` signal proved unreliable in this environment. Do not write or chase such tests. The `HypercubeWorker`/`HypercubeWidget` implementations must still be written correctly per their task text — only the requirement to prove the background thread's *end-to-end* behavior via automated pytest is waived; that gets verified by the user running the app manually. Tests that don't depend on a real thread completing (pure functions, synchronous UI state changes, direct calls to callback methods with hand-built arguments) should still be written.

---

### Task 1: `HypercubeWorker` — background computation with cancellation and generation-safety

**Files:**
- Create: `src/ui/hypercube_worker.py`
- Test: `ui_tests/ui/test_hypercube_worker.py`

**Interfaces:**
- Consumes: `core.VisualizationService.prepare_hypercube_view(data, *, progress=None, is_cancelled=None) -> HypercubeViewData` (existing, unchanged). `core.HSIData` (existing). `core.errors.CancelledError`, `core.VisualizationError`, `core.WavelengthError` (existing).
- Produces: `HypercubeWorker(service, data)` — construct on the GUI thread. Takes **no** `parent` argument by design: `QObject.moveToThread()` refuses (prints an error and does not move) an object that has a Qt parent, and this worker calls `moveToThread` in `__init__`, so it must never be parented — lifetime is managed by `self._thread.finished` driving `deleteLater()` instead. `.start() -> None` — begins the background computation. `.cancel() -> None` — best-effort early-exit signal (checked by the model between surface reads). Signals: `progress = pyqtSignal(int, str)`, `finished = pyqtSignal(object)` (payload: `HypercubeViewData`), `failed = pyqtSignal(str)`. Exactly one of `finished`/`failed` fires per completed run; a successful `cancel()` before completion fires neither. `._thread` (a `QThread`) is accessible for tests to wait on Qt's own `finished` signal regardless of outcome.

- [ ] **Step 1: Write the failing tests**

Create `ui_tests/ui/test_hypercube_worker.py`:

```python
from __future__ import annotations

import pytest

from core import HSIReader, VisualizationError, VisualizationService
from core.errors import CancelledError
from ui.hypercube_worker import HypercubeWorker


class _StubService:
    """A fake VisualizationService that fails or cancels on demand."""

    def __init__(self, outcome: str) -> None:
        self._outcome = outcome

    def prepare_hypercube_view(self, data, *, progress=None, is_cancelled=None):
        if progress is not None:
            progress(50, "halfway")
        if self._outcome == "cancel":
            assert is_cancelled is not None and is_cancelled(), (
                "worker must set is_cancelled() True before this stub runs"
            )
            raise CancelledError("cancelled")
        raise VisualizationError("boom")


def test_worker_emits_finished_with_hypercube_view_data(qtbot, synthetic_cube_path):
    data = HSIReader().open(synthetic_cube_path)
    worker = HypercubeWorker(VisualizationService(), data)

    with qtbot.waitSignal(worker.finished, timeout=5000) as blocker:
        worker.start()

    result = blocker.args[0]
    assert result.top_rgb.shape == (8, 8, 3)
    assert result.surface_cube.shape[2] == len(result.wavelengths_nm)


def test_worker_reports_progress(qtbot, synthetic_cube_path):
    data = HSIReader().open(synthetic_cube_path)
    worker = HypercubeWorker(VisualizationService(), data)

    with qtbot.waitSignal(worker.progress, timeout=5000):
        worker.start()


def test_worker_emits_failed_on_visualization_error(qtbot, synthetic_cube_path):
    data = HSIReader().open(synthetic_cube_path)
    worker = HypercubeWorker(_StubService("fail"), data)

    with qtbot.waitSignal(worker.failed, timeout=5000) as blocker:
        worker.start()

    assert blocker.args[0] == "boom"


def test_worker_cancel_suppresses_finished_and_failed(qtbot, synthetic_cube_path):
    data = HSIReader().open(synthetic_cube_path)
    worker = HypercubeWorker(_StubService("cancel"), data)
    seen_finished = []
    seen_failed = []
    worker.finished.connect(seen_finished.append)
    worker.failed.connect(seen_failed.append)

    worker.cancel()
    with qtbot.waitSignal(worker._thread.finished, timeout=5000):
        worker.start()

    assert seen_finished == []
    assert seen_failed == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest ui_tests/ui/test_hypercube_worker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ui.hypercube_worker'`

- [ ] **Step 3: Write the implementation**

Create `src/ui/hypercube_worker.py`:

```python
from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from core import HSIData, VisualizationError, WavelengthError
from core.errors import CancelledError


class HypercubeWorker(QObject):
    """Computes ``HypercubeViewData`` on a background ``QThread``.

    Construct on the GUI thread with the visualization service and the
    ``HSIData`` to read, then call :meth:`start`. Exactly one of
    ``finished``/``failed`` fires per run; a ``cancel()`` that lands before
    completion fires neither, matching ``CancelledError``'s documented
    "neutral outcome" contract in ``core.errors``.
    """

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)  # HypercubeViewData
    failed = pyqtSignal(str)

    def __init__(self, service: Any, data: HSIData) -> None:
        super().__init__()
        self._service = service
        self._data = data
        self._cancelled = False

        self._thread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._run)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self.deleteLater)

    def start(self) -> None:
        """Begin computing on the background thread."""
        self._thread.start()

    def cancel(self) -> None:
        """Ask the in-flight computation to stop at its next checkpoint."""
        self._cancelled = True

    def _run(self) -> None:
        try:
            result = self._service.prepare_hypercube_view(
                self._data,
                progress=lambda value, message: self.progress.emit(value, message),
                is_cancelled=lambda: self._cancelled,
            )
        except CancelledError:
            pass
        except (VisualizationError, WavelengthError) as exc:
            self.failed.emit(str(exc))
        else:
            self.finished.emit(result)
        self._thread.quit()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest ui_tests/ui/test_hypercube_worker.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ui/hypercube_worker.py ui_tests/ui/test_hypercube_worker.py
git commit -m "LEAF-150: add HypercubeWorker for background hypercube computation"
```

---

### Task 2: `HypercubeWidget` — OpenGL cube rendering and interaction

**Files:**
- Modify: `requirements.txt` (add `PyOpenGL`)
- Create: `src/ui/hypercube_widget.py`
- Test: `ui_tests/ui/test_hypercube_widget.py`

**Interfaces:**
- Consumes: `core.HypercubeViewData` (existing dataclass with `top_rgb: np.ndarray`, `surface_cube: np.ndarray`, `wavelengths_nm`, `row_indices`, `column_indices`, `band_indices`).
- Produces: `HypercubeWidget(parent=None)` — a `QOpenGLWidget` subclass. `.set_data(view_data: HypercubeViewData | None) -> None`. `.set_status_message(message: str | None) -> None` — shows/hides centered overlay text; pass `None` to hide it. Module-level `_colorize_slices(slices: list[np.ndarray]) -> list[np.ndarray]` — pure-numpy helper (no GL/Qt dependency), returns one `uint8` RGB image per input slice, jointly percentile-stretched.

- [ ] **Step 1: Add the new dependency**

Add `PyOpenGL` to `requirements.txt` (append a line, matching the existing `name~=version` style used by the other entries, e.g. `PyOpenGL~=3.1.10`), then install it:

```bash
.venv/bin/python -m pip install "PyOpenGL~=3.1.10"
```

- [ ] **Step 2: Write the failing tests**

Create `ui_tests/ui/test_hypercube_widget.py`:

```python
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
    # Deliberately checks isHidden(), not isVisible(): isVisible() reflects
    # composed on-screen visibility (false while any ancestor, including
    # this never-shown widget itself, hasn't been shown), regardless of the
    # label's own show()/hide() calls. isHidden() reflects the label's own
    # explicit flag, so it doesn't require showing the widget.
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest ui_tests/ui/test_hypercube_widget.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ui.hypercube_widget'`

- [ ] **Step 4: Write the implementation**

Create `src/ui/hypercube_widget.py`:

```python
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

        gl.glViewport(0, 0, max(width, 1), max(height, 1))
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        self._apply_perspective(45.0, width / max(height, 1), 0.1, 100.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def paintGL(self) -> None:
        import OpenGL.GL as gl

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        if self._textures_dirty:
            self._rebuild_textures()
        if self._textures is None:
            return

        gl.glLoadIdentity()
        gl.glTranslatef(0.0, 0.0, -self._camera_distance)
        gl.glRotatef(self._camera_theta - 90.0, 1.0, 0.0, 0.0)
        gl.glRotatef(-self._camera_phi, 0.0, 0.0, 1.0)
        gl.glTranslatef(-self._target[0], -self._target[1], -self._target[2])
        self._draw_cube()

    @staticmethod
    def _apply_perspective(fovy_deg: float, aspect: float, z_near: float, z_far: float) -> None:
        """Apply a perspective projection without `OpenGL.GLU`.

        `gluPerspective` lives in `libGLU`, a separate, optional legacy
        library that is not guaranteed to be installed alongside OpenGL
        itself (confirmed absent on the reference dev machine) — importing
        `OpenGL.GLU` succeeds either way, but calling an unresolved GLU
        function raises `OpenGL.error.NullFunctionError`, which is fatal
        when it happens inside a Qt virtual method override like
        `resizeGL`/`paintGL` (PyQt aborts the process; it cannot be caught
        as an ordinary exception from the call site). `glFrustum` is core
        (non-GLU) OpenGL and produces the identical projection matrix from
        the same field-of-view/aspect/near/far inputs.
        """
        import OpenGL.GL as gl

        top = z_near * math.tan(math.radians(fovy_deg) / 2.0)
        right = top * max(aspect, 1e-6)
        gl.glFrustum(-right, right, -top, top, z_near, z_far)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._position_status_label()

    # ------------------------------------------------------------------ #
    # Cube construction                                                    #
    # ------------------------------------------------------------------ #

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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest ui_tests/ui/test_hypercube_widget.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/ui/hypercube_widget.py ui_tests/ui/test_hypercube_widget.py
git commit -m "LEAF-150: add HypercubeWidget OpenGL cube renderer"
```

---

### Task 3: Wire HyperCube mode into `MainWindowController` and `MainWindow.ui`

**Files:**
- Modify: `src/qt/MainWindow.ui`
- Modify (regenerated, not hand-edited): `src/ui/generated/MainWindow.py`
- Modify: `src/ui/main_window.py`
- Test: `ui_tests/ui/test_hypercube.py`

**Interfaces:**
- Consumes: `ui.hypercube_worker.HypercubeWorker` (Task 1), `ui.hypercube_widget.HypercubeWidget` (Task 2), both already implemented.
- Produces: `MainWindowController.visualizationStack` (new `QStackedWidget`, page 0 = `viewer`, page 1 = `hypercubeWidget`), `MainWindowController.hypercubeWidget` (new promoted widget), `MainWindowController._hypercube_view_data: HypercubeViewData | None`, `MainWindowController._hypercube_error: str | None`, `MainWindowController._hypercube_worker: HypercubeWorker | None`, `MainWindowController._hypercube_generation: int`.

- [ ] **Step 1: Edit `MainWindow.ui` to add the stacked widget and promoted class**

In `src/qt/MainWindow.ui`, replace the Visualization tab's `viewer` item (the block starting `<item>\n         <widget class="HSIViewer" name="viewer">` and ending at its matching `</item>`, currently lines 34-40) with:

```xml
        <item>
         <widget class="QStackedWidget" name="visualizationStack">
          <widget class="HSIViewer" name="viewer">
           <property name="layoutDirection">
            <enum>Qt::LayoutDirection::LeftToRight</enum>
           </property>
          </widget>
          <widget class="HypercubeWidget" name="hypercubeWidget"/>
         </widget>
        </item>
```

Then add a second `<customwidget>` entry to the existing `<customwidgets>` block (near the end of the file, alongside the existing `HSIViewer` entry):

```xml
  <customwidget>
   <class>HypercubeWidget</class>
   <extends>QOpenGLWidget</extends>
   <header>ui.hypercube_widget</header>
  </customwidget>
```

- [ ] **Step 2: Regenerate the UI module**

Run: `.venv/bin/python -m PyQt6.uic.pyuic src/qt/MainWindow.ui -o src/ui/generated/MainWindow.py`

Confirm the diff adds `self.visualizationStack = QtWidgets.QStackedWidget(...)` and `self.hypercubeWidget = HypercubeWidget(parent=self.visualizationStack)`, and an `from ui.hypercube_widget import HypercubeWidget` (or equivalent) import line near the top alongside the existing `from ui.viewer import HSIViewer`.

- [ ] **Step 3: Write the failing tests**

Per the user ruling in Global Constraints, none of these tests wait on a
real background `QThread` (no `qtbot.waitSignal`/`waitUntil` on threaded
completion). The mode-switching and generation-counter effects of
`_start_hypercube_worker`/`_on_hypercube_mode_toggled` are synchronous, so
they're asserted directly. The `finished`/`failed` callback logic is
exercised by calling `_on_hypercube_finished`/`_on_hypercube_failed`
directly with a real, synchronously-computed `HypercubeViewData` (via
`VisualizationService().prepare_hypercube_view(...)`, called straight from
the test, not through a worker thread) — this verifies the wiring
deterministically without ever waiting on a thread. End-to-end proof that
the real worker thread reaches these callbacks is left to manual testing.

Create `ui_tests/ui/test_hypercube.py`:

```python
from __future__ import annotations

from PyQt6.QtCore import QRectF

from core import VisualizationService


def test_hypercube_button_is_enabled_before_and_after_load(window, synthetic_cube_path):
    assert window.modeHyperCube.isEnabled()
    window.load_image_from_path(synthetic_cube_path)
    assert window.modeHyperCube.isEnabled()


def test_loading_image_starts_a_hypercube_worker(loaded_window):
    # Synchronous effects only: starting the worker is fire-and-forget.
    # Whether the background thread has finished by now is not asserted --
    # end-to-end completion is verified by running the app manually.
    assert loaded_window._hypercube_worker is not None
    assert loaded_window._hypercube_generation >= 1


def test_selecting_hypercube_mode_switches_stack_page(loaded_window):
    loaded_window.modeHyperCube.click()

    assert loaded_window.visualizationStack.currentWidget() is loaded_window.hypercubeWidget


def test_leaving_hypercube_mode_restores_viewer_page(loaded_window):
    loaded_window.modeHyperCube.click()

    loaded_window.modeRGB.click()

    assert loaded_window.visualizationStack.currentWidget() is loaded_window.viewer


def test_cropping_starts_a_new_hypercube_generation(loaded_window):
    first_generation = loaded_window._hypercube_generation

    # left=1, top=1, right=5, bottom=5 -> a 4x4 crop out of the 8x8 fixture,
    # emitted the same way ui_tests/ui/test_crop.py already exercises crop.
    loaded_window.viewer.cropRequested.emit(QRectF(1, 1, 4, 4))

    assert loaded_window._hypercube_generation > first_generation


def test_hypercube_finished_callback_updates_state_and_widget(loaded_window):
    loaded_window.modeHyperCube.click()
    result = VisualizationService().prepare_hypercube_view(loaded_window._hsi_data)

    loaded_window._on_hypercube_finished(loaded_window._hypercube_generation, result)

    assert loaded_window._hypercube_view_data is result
    assert loaded_window._hypercube_error is None
    assert loaded_window.hypercubeWidget._view_data is result


def test_stale_hypercube_generation_is_ignored(loaded_window):
    result = VisualizationService().prepare_hypercube_view(loaded_window._hsi_data)
    stale_generation = loaded_window._hypercube_generation - 1

    loaded_window._on_hypercube_finished(stale_generation, result)

    assert loaded_window._hypercube_view_data is not result


def test_hypercube_failure_is_recorded_and_shown(loaded_window):
    loaded_window.modeHyperCube.click()

    loaded_window._on_hypercube_failed(loaded_window._hypercube_generation, "no wavelengths")

    assert loaded_window._hypercube_error == "no wavelengths"
    assert loaded_window.hypercubeWidget._view_data is None
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest ui_tests/ui/test_hypercube.py -v`
Expected: FAIL — `modeHyperCube` is currently force-disabled and no `_hypercube_*` attributes exist yet.

- [ ] **Step 5: Wire `main_window.py`**

In `src/ui/main_window.py`:

Add to the imports:
```python
from core import HypercubeViewData
from ui.hypercube_widget import HypercubeWidget
from ui.hypercube_worker import HypercubeWorker
```
(`HypercubeWidget` is imported here only so type checkers/readers can see it; the generated UI module already constructs the instance.)

Add `import dataclasses` to the top-level imports.

In `__init__`, alongside the other visualization state (near `self._visualization_results: dict[...] = {}`), add:
```python
self._hypercube_view_data: HypercubeViewData | None = None
self._hypercube_error: str | None = None
self._hypercube_worker: HypercubeWorker | None = None
self._hypercube_generation: int = 0
```

In `_connect_signals`, delete these two lines:
```python
self.modeHyperCube.setEnabled(False)
self.modeHyperCube.setToolTip("Hypercube view is not implemented yet")
```
and in their place add:
```python
self.modeHyperCube.toggled.connect(self._on_hypercube_mode_toggled)
```

Add these new methods in the "Private: visualization mode / pixel values" section, after `_refresh_viewers_display`:

```python
def _on_hypercube_mode_toggled(self, checked: bool) -> None:
    if not checked:
        return
    self.visualizationStack.setCurrentWidget(self.hypercubeWidget)
    self._refresh_hypercube_display()

def _start_hypercube_worker(self) -> None:
    if self._hypercube_worker is not None:
        self._hypercube_worker.cancel()
    self._hypercube_generation += 1
    generation = self._hypercube_generation
    self._hypercube_view_data = None
    self._hypercube_error = None
    if self.modeHyperCube.isChecked():
        self._refresh_hypercube_display()

    snapshot = dataclasses.replace(self._hsi_data)
    worker = HypercubeWorker(self._visualization_service, snapshot)
    worker.progress.connect(self._on_hypercube_progress)
    worker.finished.connect(
        lambda result, gen=generation: self._on_hypercube_finished(gen, result)
    )
    worker.failed.connect(
        lambda message, gen=generation: self._on_hypercube_failed(gen, message)
    )
    self._hypercube_worker = worker
    worker.start()

def _on_hypercube_progress(self, _value: int, message: str) -> None:
    self.statusbar.showMessage(f"Hypercube: {message}")

def _on_hypercube_finished(self, generation: int, result: HypercubeViewData) -> None:
    if generation != self._hypercube_generation:
        return
    self._hypercube_view_data = result
    self._hypercube_error = None
    if self.modeHyperCube.isChecked():
        self._refresh_hypercube_display()

def _on_hypercube_failed(self, generation: int, message: str) -> None:
    if generation != self._hypercube_generation:
        return
    self._hypercube_view_data = None
    self._hypercube_error = message
    LOGGER.info("Hypercube unavailable: %s", message)
    if self.modeHyperCube.isChecked():
        self._refresh_hypercube_display()

def _refresh_hypercube_display(self) -> None:
    if self._hypercube_error is not None:
        self.hypercubeWidget.set_data(None)
        self.hypercubeWidget.set_status_message(self._hypercube_error)
    elif self._hypercube_view_data is not None:
        self.hypercubeWidget.set_data(self._hypercube_view_data)
        self.hypercubeWidget.set_status_message(None)
    elif not self._hsi_data.is_loaded():
        self.hypercubeWidget.set_data(None)
        self.hypercubeWidget.set_status_message("Load an image to view its hypercube.")
    else:
        self.hypercubeWidget.set_data(None)
        self.hypercubeWidget.set_status_message("Computing hypercube…")
```

Update `_on_visualization_mode_toggled` to restore the 2D page when a non-HyperCube mode is selected — add one line at the top of the method body:

```python
def _on_visualization_mode_toggled(self, mode: VisualizationMode, checked: bool) -> None:
    if not checked:
        return
    self.visualizationStack.setCurrentWidget(self.viewer)
    self._active_visualization_mode = mode
    if self._hsi_data.is_loaded():
        self._refresh_viewers_display()
```

Update `_push_image_to_viewers` to also (re)start the hypercube worker:

```python
def _push_image_to_viewers(self) -> None:
    self._recompute_visualizations()
    self._refresh_viewers_display()
    self._start_hypercube_worker()
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest ui_tests/ui/test_hypercube.py -v`
Expected: PASS (8 tests)

- [ ] **Step 7: Run the full test suite to check for regressions**

Run: `.venv/bin/python -m pytest ui_tests/ -v`
Expected: PASS (all tests, including the pre-existing suites)

- [ ] **Step 8: Commit**

```bash
git add src/qt/MainWindow.ui src/ui/generated/MainWindow.py src/ui/main_window.py ui_tests/ui/test_hypercube.py
git commit -m "LEAF-150: wire HyperCube mode into MainWindowController"
```

---

## Post-plan notes

- Manual verification of the actual rotate/zoom/pan feel (not covered by headless tests — see Global Constraints) should happen by running the app normally: `.venv/bin/python src/main.py` with `QT_QPA_PLATFORM` unset, loading a real image, and exercising the HyperCube button and mouse drag.
- `HypercubeData` (the older `row_side_values`/`column_side_values` type) is untouched and remains unused — out of scope per the spec.
