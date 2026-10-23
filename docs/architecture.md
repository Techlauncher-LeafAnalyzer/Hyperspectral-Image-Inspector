# Hyperspectral Image Inspector — Architecture Reference

> **2026-08 update:** the UI pivoted from a sidebar + swappable-panel design to a
> static `QTabWidget` layout (PRs #5–#11). The now-unused `src/ui/panels/*.py`
> classes and their per-panel `.ui` files have since been deleted — see
> "Sidebar + swappable panels → static tabs" under Design Decisions.

> **2026-08 update (`LEAF-156`):** `HSIViewer`'s box-prompt mode (`PromptMode`,
> `draw_rectangle`, `rect_item`, `input_box`, `start_point`, `newInputBoxList`/
> `allInputBoxList`), its point-annotation undo/redo/history mechanism (`history`,
> `redo_stack`, `undo()`, `redo()`, the `historyChanged` and `photoClicked`
> signals), the `avatar` overlay (`set_avatar`), and the unused `set_mask()`
> setter were removed as dead code — none had a caller or any UI control wiring
> them up. `spectrumPlotRequested` and `meanIndexRequested` are no longer
> unconnected either (see "Known gaps" below) — only `historyChanged`/
> `photoClicked` were still dangling, and both are gone now. All of `HSIViewer`'s
> styling (the inline pixel-overlay stylesheet, scene-drawn colours) also moved
> into `ui/theme.py`. Re-add any of this only once a real feature needs it.

## Package Structure

```
src/
├── main.py                        Entry point — builds QApplication, applies theme, shows window
├── core/                          Pure Python — zero Qt imports
│   ├── calibration_model.py       Cropped, chunked dark/bright correction
│   ├── hsi_data.py                HSIData dataclass + enums
│   └── hsi_utils.py               Stateless utility functions
└── ui/                            All Qt-touching code
    ├── theme.py                   App-wide QSS stylesheet + font loading
    ├── viewer.py                  HSIViewer custom QGraphicsView
    ├── calibration_controller.py  Calibration controls and result lifecycle
    ├── calibration_worker.py      Calibration/reference I/O + RGB worker
    ├── main_window.py             MainWindowController (tab-based)
    └── generated/
        ├── MainWindow.py          Build artifact — never hand-edited
        └── WorkflowPanels.py      Build artifact — never hand-edited
```

UI files live in `src/qt/`: `MainWindow.ui` (the canvas pages and status bar) and `WorkflowPanels.ui` (the five right-sidebar panels). Regenerate after editing either:

```bash
python -m PyQt6.uic.pyuic src/qt/MainWindow.ui -o src/ui/generated/MainWindow.py
python -m PyQt6.uic.pyuic src/qt/WorkflowPanels.ui -o src/ui/generated/WorkflowPanels.py
```

---

## Dependency Graph

```
main.py
  ├─ ui.theme.apply_theme                     (global QSS + fonts)
  └─ ui.main_window.MainWindowController
       ├─ ui.generated.MainWindow.Ui_MainWindow   (tab layout, all controls)
       │    ├─ ui.viewer.HSIViewer  × 4            (one per tab: viewer,
       │    │                                       calibrationViewer, superResViewer,
       │    │                                       classificationViewer)
       │    └─ _TabTransitionController            (tab-switch animation, private to main_window.py)
       ├─ core.hsi_data.HSIData                    (shared state)
       └─ core.hsi_utils                           (I/O helpers)
```

`HSIViewer` has **no import from `ui.main_window`**. Communication flows upward through Qt signals only; `MainWindowController._connect_signals` connects all three (`spectrumPlotRequested`, `meanIndexRequested`, `cropRequested`).

---

## `core/hsi_data.py`

### `ImageFormat`

```python
class ImageFormat(Enum):
    ENVI = auto()
    PSI  = auto()
```

Discriminates between native ENVI headers and PSI-proprietary headers that require conversion before opening.

---

### `Functionality`

```python
class Functionality(Enum):
    VISUALIZATION    = auto()
    SUPER_RESOLUTION = auto()
    CALIBRATION      = auto()
    CLASSIFICATION   = auto()
```

Defined for the old swappable-panel dispatch. `MainWindowController` no longer references it — tab selection is handled entirely by `QTabWidget`. Now that `ui/panels/` has been deleted, nothing in the codebase references `Functionality` either; it's safe to remove.

---

### `HSIData`

```python
@dataclass
class HSIData:
    image_path:   Optional[Path]
    header_path:  Optional[Path]
    image_format: Optional[ImageFormat]
    spectral_obj: Optional[object]          # spectral.io.envi image object
    wavelengths:  list[float]
    rgb_array:    Optional[NDArray[uint8]]  # shape (H, W, 3)
    mask_array:   Optional[NDArray[uint8]]  # shape (H, W)

    def is_loaded(self) -> bool: ...
    def clear(self) -> None: ...
```

Single source of truth for all loaded image state. Created once in `MainWindowController.__init__` and held on `self._hsi_data`. Only `_load_image` writes to it. Nothing currently reads `image_format`, `header_path`, or `is_loaded()`/`clear()` — no panel or handler consumes `HSIData` at all right now, since the four viewers are populated directly from `_load_image`'s local variables rather than by reading back through `self._hsi_data`.

---

## `core/hsi_utils.py`

Unchanged. Stateless, pure functions with no Qt dependency. Safe to call from tests without a display.

```python
def find_rgb_bands(wavelengths: list[float]) -> Optional[tuple[int, int, int]]:
    """Return (r_idx, g_idx, b_idx) closest to 682.5 / 532.5 / 472.5 nm.
    Returns evenly-spaced fallback indices when the image lacks full RGB coverage.
    Returns None when fewer than 3 bands are present.
    Assumes wavelengths are in ascending order.
    """

def find_red_nir_bands(wavelengths: list[float]) -> Optional[tuple[int, int]]:
    """Return (r_idx, nir_idx) closest to 682.5 nm and 850.0 nm.
    Returns None when the image lacks the required spectral range.
    Assumes wavelengths are in ascending order.
    """

def read_psi_header(file_path: Path) -> dict[str, object]:
    """Parse a PSI-format header file into a flat metadata dictionary.
    The 'WAVELENGTHS' key maps to list[float].
    """

def create_envi_header(file_path: Path, meta: dict[str, object]) -> None:
    """Write an ENVI-standard .hdr file from a PSI metadata dictionary.
    Writes NBANDS, NBITS, LAYOUT, NROWS, NCOLS, WAVELENGTHS fields.
    """

def numpy_to_qpixmap(image: NDArray[uint8]) -> QPixmap:
    """Convert a uint8 RGB array of shape (H, W, 3) to a QPixmap.
    Qt is imported locally; callers in non-Qt contexts should not use this.
    """
```

Note: `find_red_nir_bands` (for NDVI-family indices) has no caller anywhere in `src/` yet — the Visualization tab's index radio buttons (see the old `visualization_panel.py`) were never wired up in the current tab-based controller. This lines up with Jira `LEAF-113`/`LEAF-114` (vegetation index formulas + wiring) being unstarted sprint-1 work.

---

## `ui/viewer.py`

### `HSIViewer(QtWidgets.QGraphicsView)`

Custom interactive view for displaying and annotating hyperspectral images. Decoupled from `MainWindowController` via outbound signals. **Four independent instances now exist** — `viewer`, `calibrationViewer`, `superResViewer`, `classificationViewer` — one per tab, each loaded with the same `rgb_array`/`mask_array`/pixmap on image load. They are not kept in sync with each other after that (e.g. annotating in one does not reflect in another).

#### Signals

```python
spectrumPlotRequested = pyqtSignal(QPointF)        # from context menu
meanIndexRequested    = pyqtSignal(str)            # index name e.g. "NDVI"
cropRequested         = pyqtSignal(QtCore.QRectF)  # from context menu -> Crop drag
```

All three are connected in `MainWindowController._connect_signals` (`_on_spectrum_plot`, `_on_mean_index`, `_on_crop_requested`).

#### Public state (read/write by controller)

```python
rgb:                   Optional[NDArray[uint8]]  # source RGB array, shape (H, W, 3)
mask_array:            Optional[NDArray[uint8]]  # current segmentation mask, shape (H, W)
pixel_value_provider:  Callable[[int, int], Mapping[str, PixelValueEntry]]
    # per-pixel hover readout for the "Show Pixel Values" overlay; the
    # controller assigns its own callback (`_pixel_values_at`) after
    # construction, replacing the no-op default.
```

#### Public API

```python
def has_photo(self) -> bool:
    """True when an image is currently loaded."""

def photo_size(self) -> Optional[QSize]:
    """Pixel dimensions of the currently displayed photo, or None if empty."""

def fit_in_view(self) -> None:
    """Scale the view to fit the current image exactly."""

def set_photo(self, pixmap: Optional[QPixmap] = None) -> None:
    """Load a new image. Clears all annotation state."""

def get_view_state(self) -> Optional[tuple[float, QPointF]]:
    """Return (scale_factor, scene_center) describing the current pan/zoom."""

def set_view_state(self, state: tuple[float, QPointF]) -> None:
    """Apply a (scale_factor, scene_center) pair captured via get_view_state."""

def queue_view_state(self, state: tuple[float, QPointF]) -> None:
    """Apply state now, and again on every resize for a short window
    (covers a tab whose viewport hasn't settled to its final size yet)."""

def draw_circle(self, point: NDArray[uint32], label: int) -> None:
    """Draw a prompt point: green circle for foreground (label=1),
    red for background (label=0)."""
```

#### Key interaction model

| Input | Behaviour |
|-------|-----------|
| Scroll wheel | Zoom in / out, centred on cursor |
| Drag | Pan (ScrollHandDrag mode) |
| Ctrl + left-click | Add foreground point |
| Ctrl + right-click | Add background point |
| Ctrl + release (no Shift) | Re-render the mask overlay |
| Shift held during Ctrl+release | Batch mode — defer mask re-render |
| Shift release | Flush batch, re-render mask overlay |
| Hover (no buttons) | Emits `pixelHovered(row, column)` once per new pixel; drives the Spectrum Inspector's live trace |
| Right-click | Context menu (Spectrum Plot, Index Mean, Show Pixel Values, Crop) — Spectrum Plot pins that pixel in the Spectrum Inspector |
| Right-click menu → Crop, then drag | Crop-selection mode; release emits `cropRequested` |

Point clicks accumulate into `input_points`/`input_labels` and draw markers via `draw_circle`, but nothing in the current codebase runs a segmentation model against them — `mask_array` is only ever set directly by the controller (on image load, on crop, etc.), never produced from these prompts. The click-collection UI is scaffolding for a future model-driven segmentation flow; build that wiring (and any undo/redo or box-prompt support it needs) when the feature actually lands rather than trusting the pre-`LEAF-156` version of this doc.

---

## `ui/main_window.py`

### `MainWindowController(QMainWindow, Ui_MainWindow)`

Application controller. Owns `HSIData` and wires the static tab UI. Inherits the widget layout from `Ui_MainWindow` via `setupUi(self)`.

```python
def __init__(self, *args, **kwargs) -> None:
    """setupUi, create HSIData, configure tab transitions, wire signals."""
```

#### Private methods (current)

```python
def _configure_tabs(self) -> None:
    """Name/style the two QTabWidgets (tabWidget = top-level sections,
    classificationModeTabs = Unsupervised/Supervised) and attach a
    _TabTransitionController to each for the slide/fade animation."""

def _configure_file_menu(self) -> None:
    """Build the File menu (Load Image / Save Image), opened from the left
    stripe's File button (StripeMenuButton). The actions are also added to
    the window so Ctrl+O / Ctrl+S work while the menu is closed."""

def _setup_workflow_panels_form(self) -> None:
    """Run Ui_WorkflowPanels.setupUi and expose its widgets as window
    attributes (like Ui_MainWindow's), before the controllers are built."""

def _configure_status_bar(self) -> None:
    """Add the loaded-file label (imageFilePath: name only, full path in its
    tooltip) and the CPU/RAM monitor to the status bar."""

def _configure_workflow_panels(self) -> None:
    """Build the right workflow stripe that replaces the main tabs (see
    ui/tool_windows.py below): one ToolWindow per WorkflowPanels.ui panel,
    and hide tabWidget's tab bar."""

def _connect_signals(self) -> None:
    """Wires: actionLoadImage -> _load_image, actionSaveImage -> _save_image,
    darkFileButton/referenceFileButton/pushButton -> file pickers.
    Unconditionally disables calibrateButton with a "not implemented yet"
    tooltip — it is never re-enabled, including after image load."""

def _select_dark_file(self) -> None: ...
def _select_reference_file(self) -> None: ...
def _select_groundtruth_file(self) -> None: ...
    # Thin wrappers around _select_supporting_file for each file-picker button.

def _select_supporting_file(self, target_edit: QLineEdit, dialog_title: str) -> None:
    """Open a file dialog, write the chosen path into target_edit."""

def _load_image(self) -> None:
    """Open file dialog, detect PSI vs ENVI format, convert header if needed,
    open via spectral, extract RGB, populate self._hsi_data, then push the
    same pixmap/rgb/mask into all four HSIViewer instances directly and
    enable unsupervisedClassifyButton + pushButton_2 (Supervised Classify)."""

def _save_image(self) -> None:
    """(stub)"""

def _on_spectrum_plot(self, pos: QPointF) -> None:
    """(stub, and currently unreachable — not connected to any signal)"""

def _on_mean_index(self, index_name: str) -> None:
    """(stub, and currently unreachable — not connected to any signal)"""
```

#### `_TabTransitionController(QtCore.QObject)`

New private helper class (not present in the previous design). Gives a `QTabWidget` a sliding underline indicator plus a cross-fade on the incoming page (180ms, `OutCubic` easing). One instance is created per tab widget in `_configure_tabs`; purely cosmetic, holds no application state.

---

## Removed: `ui/panels/` (historical)

The original design swapped a single `FeaturePanel` subclass (`base_panel.py`, plus `calibration_panel.py`/`classification_panel.py`/`super_resolution_panel.py`/`visualization_panel.py`) in and out of a named `panelContainer`, selected via a sidebar + `Functionality` dispatch. After the pivot to the static `QTabWidget` layout (PRs #5–#11), none of these classes were instantiated anywhere — `main_window.py` kept only their imports. That dead code, along with the four per-panel `.ui` files (`src/qt/Visualization.ui`, `Calibration.ui`, `Classification.ui`, `Super-resolution.ui`), has since been deleted; `src/qt/` then contained only `MainWindow.ui` (`WorkflowPanels.ui` was added with the right sidebar).

---

## `ui/tool_windows.py` — tool-window stripes (left and right)

JetBrains-style tool windows. `ToolWindowStripe` is a fixed vertical
`QToolBar` in the left or right toolbar area (`edge=`; outside the dock area,
so it stays on the window edge). Each `ToolWindow` binds a `ToolStripeButton`
to a panel (compact title bar with ⧉ pop out / ⇤ dock and – hide, plus the
content) that lives in exactly one container: a fixed `QDockWidget` beside
its stripe, or a normal top-level `_PoppedOutWindow` with a system title bar.
One tool window is docked per side at a time; popped-out ones are independent.

**Why not Qt floating docks:** a floating/dragged `QDockWidget` is moved by a
client-side mouse grab, which Wayland refuses ("This plugin supports grabbing
the mouse only for popup windows"), leaving the window stuck and drag-redock
broken. The docks are therefore neither movable nor floatable, and pop-out
reparents the panel into a compositor-managed window instead. Closing that
window (system ✕) hides the tool; it reopens popped out, at its last size.

| Button state | Meaning | Look |
|---|---|---|
| Selected | window open — docked **or** popped out | tinted highlight, accent icon |
| Not selected | window hidden | no background (faint tint on hover) |
| Disabled | nothing to show (`ToolWindow.set_available(False, reason)`) | gray icon; tooltip gives the reason |

The **left stripe** has, top to bottom: the logo, the **File** menu button
(`StripeMenuButton`), then:

- **View toggles** — `ViewToggleButton`s for **Super-Resolution** and
  **Calibration**, driven by `resolution_toggle.ResolutionSwitchGroup`. Each
  switches *every* tab between two image versions: highlighted while the
  second version (High Res / After Calibration) is shown, plain for the first,
  gray while there is nothing to compare (tooltip says what to run).
  Clicks go through `MainWindowController._select_canvas_resolution` /
  `_select_canvas_calibration`, so busy-pipeline guards still apply. These
  replaced the per-tab pill rails and the later "Image view" tool window.
  Buttons sit an even `STRIPE_GAP` from the window edge and from the image
  area (pages have no side margins, so this holds in every view; with a panel
  docked, the gap is to the dock).
- **Tool windows** (below a divider) — currently **Spectrum**, the Spectrum
  Inspector (see `docs/model_spectral_analysis_api.md`). Gray until an image
  is loaded; the Spectra list stacks under the plot while docked and sits
  beside it when popped out.

The **right stripe** (`workflowStripe`) replaced the main tab row. Its buttons
are captioned "pills" (`ToolStripeButton(caption=...)`): **Visualize**,
**Super-Res**, **Calibrate**, **Segment**, **Batch**. Each opens a
`ToolSettingsPanel` (a scroll area) around a panel from `WorkflowPanels.ui`
(index radios, Run Super-Resolution, dark/white references, classifier
settings, the whole batch form). `tabWidget` remains as the canvas's page
stack with its tab bar hidden, so the pan/zoom hand-off and fade on view
change are unchanged. The first four panels are bound to a view: when one is
*activated* (`ToolWindow.activated` — opened, its popped-out window
activated, or focus moved into it) the canvas switches to that page, and that
pill gets an accent bar (`set_current`) even while its panel is closed. Batch
has no view and leaves the canvas alone. Visualization opens docked at
startup. The Segmentation class-layer inspector stays beside its canvas.

---

## `ui/theme.py`

New module, applied once from `main.py` via `apply_theme(app: QApplication) -> None`. Loads bundled fonts through `QFontDatabase` and applies a single global QSS stylesheet (`APP_QSS`) covering the main window background, tool-window stripes and panels, tab bars, buttons, and form controls. `main_window.py` has no other styling code — all colours/spacing are centralized here.

---

## `ui/generated/MainWindow.py`

Auto-generated by `pyuic6` from `qt/MainWindow.ui` (and `WorkflowPanels.py` from `qt/WorkflowPanels.ui`). **Never edit manually** — changes will be overwritten on next regeneration.

`Ui_MainWindow` creates the canvas: one `tabWidget` (tab bar hidden at runtime) with four pages — `Visualization`, `SuperResolution`, `Calibration`, `Classification` — each holding only its `HSIViewer` (plus the class-layer inspector on `Classification`). `Ui_WorkflowPanels` creates the sidebar panels' controls listed below. The low/high-resolution choice is the controller's `_high_resolution_selected` flag, set through `_set_high_resolution_selected`; there are no radio buttons for it.

| Attribute | Type | Purpose |
|-----------|------|---------|
| `tabWidget` | `QTabWidget` | Canvas page stack; pages are switched by the right-sidebar panels |
| `viewer` | `HSIViewer` | Visualization view's image display |
| `superResViewer` | `HSIViewer` | Super-Resolution view's image display |
| `calibrationViewer` | `HSIViewer` | Calibration view's image display |
| `darkFileButton` / `darkFileEdit` | `QPushButton` / `QLineEdit` | Dark-frame file picker |
| `referenceFileButton` / `referenceFileEdit` | `QPushButton` / `QLineEdit` | Reference-frame file picker |
| `calibrateButton` | `QPushButton` | Starts/cancels calibration after source and references are selected |
| `classificationViewer` | `HSIViewer` | Segmentation view's image display |
| `classificationModeTabs` | `QTabWidget` | Nested Unsupervised / Supervised sub-tabs |
| `numOfClassesEdit`, `maxIterationsEdit` | `QLineEdit` | Unsupervised tab inputs (K-means params) |
| `unsupervisedClassifyButton` | `QPushButton` | Starts/cancels worker-thread SPy K-means after image load |
| `lineEdit` | `QLineEdit` | Supervised tab — groundtruth file path |
| `comboBox` | `QComboBox` | Supervised tab — Model-backed classifier choice. The Controller populates it from `SupervisedClassifierType` and stores each enum as item data (Gaussian / Mahalanobis). |
| `pushButton` | `QPushButton` | Supervised tab — mask picker with automatic reference-cube pairing |
| `pushButton_2` | `QPushButton` | Runs/cancels Gaussian or Mahalanobis reference-example classification |
| `actionLoadImage` | `QAction` | File dropdown — load |
| `actionSaveImage` | `QAction` | File dropdown — save |

There is no `menubar`/`menuFile` anymore — `MainWindow.ui` no longer declares a `QMenuBar`. `_configure_file_menu` (in `main_window.py`) builds the File `QMenu` from the two actions above at runtime; it opens from the File button on the left tool-window stripe.

Classification layer UI should be backed by one
`core.ClassificationLayerModel` per current classification result. The
Controller owns mutable layer visibility/names and passes only Model-produced
RGB/RGBA arrays or statistics to widgets. The layer Model is invalidated with
the classification result after load, crop, crop undo/redo, or
reclassification. See `docs/model_classification_api.md` for the integration
contract and `docs/classification_layer_api.md` for the detailed PyQt6
handoff.

A fifth top-level tab, `Batch`, hosts an independent multi-file workflow — a
file list (`batchFileList`), stage checkboxes (Calibrate/Super-Resolution/
Classify), an output folder/format, and a results list — owned entirely by
`ui.batch_controller.BatchController` via `ui.batch_worker.BatchWorker` and
`core.BatchService`. It does not share `HSIData`, the four `HSIViewer`
instances, or any other tab's state; see `docs/model_batch_api.md`.

---

## Design Decisions

### Acyclic dependency via signals

`HSIViewer` emits `spectrumPlotRequested(pos)`, `meanIndexRequested(name)`, `cropRequested(rect)` rather than holding a back-reference to the controller. `ui/viewer.py` still has no import from `ui/main_window.py`. All three are connected in `MainWindowController._connect_signals`. A parallel `historyChanged`/`photoClicked` pair existed for an annotation undo/redo mechanism but was never connected to anything; both were removed as dead code in `LEAF-156` along with the mechanism they supported.

### Sidebar + swappable panels → static tabs

The original design swapped a single `FeaturePanel` subclass in and out of a named `panelContainer`, driven by a `Functionality` enum and a sidebar of mode buttons. As of PRs #5–#11, `MainWindow.ui` was rebuilt around a `QTabWidget` with one fixed tab per feature, each holding its own `HSIViewer` and controls directly. This trades the old single-shared-viewer model for four independent viewers (simpler per-tab layout, no dynamic widget teardown/rebuild) at the cost of the four viewers not staying in sync. The now-dead `ui/panels/` classes and their `.ui` files have since been removed (see "Removed: `ui/panels/`" above).

### File menu → ribbon-style corner dropdown (`LEAF-121`)

The `QMenuBar`/`menuFile` row above the tabs has been removed from `MainWindow.ui`; `actionLoadImage`/`actionSaveImage` are still declared as standalone `QAction`s but are no longer attached to a menu bar in the `.ui` file. `MainWindowController._configure_file_menu` builds the `QMenu` from those actions at runtime; it opens from the File button at the top of the left tool-window stripe (the tab row it used to share is gone).

### `HSIData` consolidates scattered state

`HSIData` owns `image_path`, `spectral_obj`, `rgb_array`, `mask_array`, and related metadata. Source loading updates the shared instance in place; feature services read through its lazy API. Temporary calibration and SR outputs each own a separate `HSIData` whose backing files live for the result lifetime.

### `core/` has zero Qt imports

Still true. All functions in `core/hsi_utils.py` work without a running `QApplication`. `numpy_to_qpixmap` imports Qt locally inside the function body. This keeps the module testable in a headless environment.

---

## Visualization Model integration

The production Visualization Model now lives in `src/core/visualization_model.py`
and is re-exported through `core`. File import is implemented by
`src/core/hsi_reader.py`; Model-domain failures live in `src/core/errors.py`.

The team Controller now uses this production path directly:

```
MainWindowController
  -> HSIReader.open(selected_path)
  -> VisualizationService.render(... RGB ...)
  -> existing HSIData.update_from(candidate)
  -> Qt pixmap/viewer update
```

`core.hsi_utils` is the shared low-level compatibility module. It owns PSI
parsing, temporary ENVI adapter generation, and wavelength lookup, so
`HSIReader`, `HSIData`, and legacy Controller helpers use the same rules.
PSI adapters are cached in the operating-system temporary directory; loading
a capture never writes conversion artifacts beside source data.

`HSIData` preserves the mutable fields consumed by the current team Controller
and also exposes lazy targeted reads for feature Models. Controllers should
import from `core`, run disk-reading Model calls in workers, and create Qt or
OpenGL widgets only on the GUI thread. The full contract is documented in
`docs/model_visualization_api.md`.

Local tests, sample captures, and the temporary Model-only test View/Controller
are intentionally ignored and are not part of the shared production tree.

---

## Known gaps

Carried over from the current rubric self-assessment / sprint-1 backlog, listed here because they're structural rather than just "unfinished feature":

- **Perceptron classification is not integrated.** The selector intentionally exposes only the reference-example Gaussian and Mahalanobis workflows connected through `pushButton_2`; SPy's perceptron still requires dimensionality-reduction and training-policy decisions.
- **Full calibrated-cube export is not integrated.** File → Save Image exports the rendered calibrated RGB preview, not the float32 ENVI result.
