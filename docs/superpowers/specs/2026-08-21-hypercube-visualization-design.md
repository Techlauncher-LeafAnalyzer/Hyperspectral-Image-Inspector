# Hypercube visualization — design (LEAF-150)

## Context

`core/visualization_model.py` already exposes `VisualizationService.prepare_hypercube_view()`,
returning a `HypercubeViewData` payload (a downsampled RGB top face plus real
values on the cube's four boundary surfaces). The UI has a `modeHyperCube`
radio button that is force-disabled with a "not implemented yet" tooltip
(`main_window.py`). This spec wires the two together with an interactive,
rotatable cube view.

**Constraint:** SPy's own interactive cube widget
(`spectral.graphics.hypercube.HypercubeWindow`) hard-imports **PySide6**.
This app runs entirely on **PyQt6**. Two Qt bindings cannot safely share one
process (separate QApplication/event-loop implementations), so SPy's widget
cannot be embedded directly — a PyQt6-native cube view must be built instead.

## Approach

Port SPy's cube-rendering *technique* (not its code) to PyQt6 + PyOpenGL:
SPy's rendering is plain `OpenGL.GL`/`OpenGL.GLU` calls (binding-agnostic);
only the widget shell (`QOpenGLWidget`, `QSurfaceFormat`, `Qt` enums) is
PySide6-specific and has direct PyQt6 equivalents. This gives a proven
rotate/zoom/pan interaction model and 6-textured-quad cube geometry for the
cost of one new dependency, `PyOpenGL`.

Rejected: `pyqtgraph.GLViewWidget` (orbit camera for free, but texturing 6
independent faces isn't a first-class feature of its high-level items —
would still need a custom low-level GL item, i.e. most of the same code,
plus a heavier dependency). Hand-rolled camera math (no proven reference,
more risk, no upside over porting SPy's approach).

**Headless-render risk, checked:** `QOpenGLWidget` plus real `OpenGL.GL`
calls (`glGenTextures`, `glTexImage2D`, `glClear`, etc.) were spiked under
this project's test platform (`QT_QPA_PLATFORM=offscreen`, as set in
`ui_tests/conftest.py`). Qt logs `"QOpenGLWidget is not supported on this
platform"` / `"No fbo, cannot render"` but nothing raises — `initializeGL`
and `paintGL` execute safely as no-ops. So headless tests can exercise the
widget's code paths but **cannot** assert on rendered pixels.

## Changes

### 1. Dependency

Add `PyOpenGL` to `requirements.txt`.

### 2. `core/visualization_model.py`

No changes. `prepare_hypercube_view()` already returns the right shape of
data; this spec only adds a consumer.

### 3. `src/ui/hypercube_worker.py` (new)

A `QObject`-based worker plus `QThread` pair:

- `HypercubeWorker.run(data: HSIData)` calls
  `VisualizationService.prepare_hypercube_view(data, progress=self._emit_progress,
  is_cancelled=self._is_cancelled)` off the GUI thread.
- Signals: `progress(int, str)`, `finished(HypercubeViewData)`,
  `failed(str)`.
- `cancel()` sets an internal flag read by `_is_cancelled`; the model already
  checks this between surface reads and raises `CancelledError`, which the
  worker swallows (no `failed` signal — cancellation is a neutral outcome,
  matching `CancelledError`'s documented contract in `core/errors.py`).

### 4. `main_window.py`

- `_push_image_to_viewers()` (called after every load and every crop) starts
  a fresh `HypercubeWorker` after cancelling/discarding any worker still
  in flight, so a rapid second crop can't let a stale result win the race.
  The worker is given a `dataclasses.replace(self._hsi_data)` snapshot, not
  the live `self._hsi_data` reference — `HSIData` is a mutable dataclass the
  GUI thread can still reassign (e.g. `spectral_obj` on the next crop) while
  the worker reads it from another thread, so each worker gets its own
  decoupled copy of the field values as they stood when it was started.
- New state: `_hypercube_view_data: HypercubeViewData | None`,
  `_hypercube_error: str | None`, `_hypercube_worker`.
- On `finished`: cache the data; if HyperCube is the active mode, push it
  into `hypercubeWidget.set_data(...)` immediately. Clear `_hypercube_error`.
- On `failed`: cache the message in `_hypercube_error`; if HyperCube is
  active, show it (see §6). Logged via `LOGGER.info`, matching how
  `_recompute_visualizations` already handles `VisualizationError`/
  `WavelengthError` for the other 7 modes.
- `progress` forwards to `self.statusbar.showMessage(...)`.
- Remove the `modeHyperCube.setEnabled(False)` / tooltip lines. None of the
  other mode buttons are gated on `_hsi_data.is_loaded()` either — clicking
  one before an image loads just updates internal state with no visible
  effect until an image exists. HyperCube follows the same rule; the
  pre-load state is handled entirely by the pane's own placeholder text
  (§7), not by disabling the button.
- `_on_visualization_mode_toggled`: switching **to** HyperCube flips
  `visualizationStack` to page 1 (see §5) and shows cached data / a
  "Computing hypercube…" placeholder / the cached error, whichever applies.
  Switching **away** flips back to page 0 and calls the existing
  `_refresh_viewers_display()` — the 2D modes are untouched by this feature.

### 5. `qt/MainWindow.ui` + regenerated `ui/generated/MainWindow.py`

Wrap the Visualization tab's `viewer` in a `QStackedWidget` named
`visualizationStack`:
- page 0: existing `HSIViewer` (`viewer`), unchanged.
- page 1: new promoted widget `hypercubeWidget` (class `HypercubeWidget`,
  header `ui.hypercube_widget`), the same promotion mechanism already used
  for `HSIViewer` in this file.

Regenerate with the documented command:
```
python -m PyQt6.uic.pyuic src/qt/MainWindow.ui -o src/ui/generated/MainWindow.py
```

### 6. `src/ui/hypercube_widget.py` (new)

`HypercubeWidget(QOpenGLWidget)`:

- `set_data(view_data: HypercubeViewData | None) -> None` — the only data
  entry point. `None` clears to an empty/placeholder state.
- `set_status_message(message: str | None) -> None` — used for "Computing
  hypercube…" and error text, painted as centered text when there is no
  cube to show yet.
- Texture build (on `set_data`, realized in `initializeGL`/next `paintGL`
  if the context isn't ready yet): `top_rgb` supplies the top and bottom
  face textures directly (already `uint8` RGB). The four boundary slices of
  `surface_cube` (front = row `rows-1`, right = column `columns-1`,
  back = row `0`, left = column `0`, matching `prepare_hypercube_view`'s
  own comments) are colorized into `uint8` RGB via a shared percentile
  stretch across all four slices (reusing the min/max-then-normalize shape
  of `VisualizationService._percentile_stretch`) mapped through the `"gray"`
  colormap — consistent with how BAND mode already renders single-band
  data.
- Geometry/camera: a `glBegin`/`glEnd` six-quad cube (adequate for a fixed
  shape; no shader pipeline needed) with a spherical camera driven by
  mouse events: left-drag rotates, Ctrl+left-drag zooms, Shift+left-drag
  pans — reimplemented against `PyQt6.QtGui.QMouseEvent`/`Qt.KeyboardModifier`,
  no import from `spectral.graphics` or PySide6 anywhere.
- No public API depends on `HSIData` or the visualization service directly
  — it only ever receives `HypercubeViewData`, keeping it swappable/testable
  like `HSIViewer`.

### 7. Error / empty states

- No image loaded: page 1 shows "Load an image to view its hypercube."
- Worker running: "Computing hypercube…" (plus the forwarded progress
  message where useful).
- Worker failed (`VisualizationError`/`WavelengthError`, e.g. RGB
  wavelengths unavailable): the caught message, styled the same as the
  existing `QMessageBox.critical` failure text elsewhere in this file, but
  inline in the pane rather than a modal (HyperCube is a whole-pane view,
  not an overlay on existing pixels).

## Testing

Headless (`ui_tests/`, `QT_QPA_PLATFORM=offscreen`) can and should cover:
- `modeHyperCube` is enabled and clickable regardless of load state, same
  as every other mode button.
- Selecting/deselecting HyperCube flips `visualizationStack`'s current
  index.
- A load or crop starts a `HypercubeWorker`; a second load/crop before the
  first finishes cancels/discards the first (no stale data wins).
- `finished`/`failed` signals update `_hypercube_view_data` /
  `_hypercube_error` and, when HyperCube is active, reach
  `hypercubeWidget.set_data(...)` / `set_status_message(...)`.
- `HypercubeWidget.set_data(...)` does not raise when given real
  `HypercubeViewData` (from the 8×8×8 synthetic cube already used by
  `ui_tests/conftest.py`) or `None`.

Out of scope for automated tests, called out explicitly rather than faked:
actual rendered pixel output and real mouse-drag rotation feel — the
offscreen Qt platform cannot produce a real framebuffer (see the spike
above). Manual verification of the rotate/zoom/pan feel happens by running
the app normally (`QT_QPA_PLATFORM` unset).

## Out of scope

- The Calibration/Super-Resolution/Classification tabs keep their existing
  single 2D `HSIViewer` — this feature only touches the Visualization tab.
- `HypercubeData` (the older, simpler "custom future cube renderer" type
  with `row_side_values`/`column_side_values`) is not used by this design
  and is left as-is.
- Lighting, texture filtering quality, and cube proportions beyond a
  reasonable default are not tunable from the UI in this pass.
