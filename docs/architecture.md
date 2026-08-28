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
│   ├── hsi_data.py                HSIData dataclass + enums
│   └── hsi_utils.py               Stateless utility functions
└── ui/                            All Qt-touching code
    ├── theme.py                   App-wide QSS stylesheet + font loading
    ├── viewer.py                  HSIViewer custom QGraphicsView
    ├── main_window.py             MainWindowController (tab-based)
    └── generated/
        └── MainWindow.py          Build artifact — never hand-edited
```

UI files live in `src/qt/` (just `MainWindow.ui` now — it defines every tab's controls directly). Regenerate `ui/generated/MainWindow.py` after editing it:

```bash
python -m PyQt6.uic.pyuic src/qt/MainWindow.ui -o src/ui/generated/MainWindow.py
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
| Right-click | Context menu (Spectrum Plot, Index Mean, Show Pixel Values, Crop) |
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
    """Build a File dropdown (Load Image / Save Image) and install it as
    tabWidget's top-left corner widget, so it sits in the same row as the
    Visualization/Super-Resolution/Calibration/Classification tabs instead
    of a separate QMenuBar row."""

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

The original design swapped a single `FeaturePanel` subclass (`base_panel.py`, plus `calibration_panel.py`/`classification_panel.py`/`super_resolution_panel.py`/`visualization_panel.py`) in and out of a named `panelContainer`, selected via a sidebar + `Functionality` dispatch. After the pivot to the static `QTabWidget` layout (PRs #5–#11), none of these classes were instantiated anywhere — `main_window.py` kept only their imports. That dead code, along with the four per-panel `.ui` files (`src/qt/Visualization.ui`, `Calibration.ui`, `Classification.ui`, `Super-resolution.ui`), has since been deleted; `src/qt/` now contains only `MainWindow.ui`.

---

## `ui/theme.py`

New module, applied once from `main.py` via `apply_theme(app: QApplication) -> None`. Loads bundled fonts through `QFontDatabase` and applies a single global QSS stylesheet (`APP_QSS`) covering the main window background, the (now-removed) `#navigationPanel`/`#panelContainer` selectors left over from the sidebar design, tab bars, buttons, and form controls. `main_window.py` has no other styling code — all colours/spacing are centralized here.

---

## `ui/generated/MainWindow.py`

Auto-generated by `pyuic6` from `qt/MainWindow.ui`. **Never edit manually** — changes will be overwritten on next regeneration.

Defines `Ui_MainWindow` with `setupUi(self)`, which creates all named widgets as instance attributes. Current top-level structure: one `tabWidget` with four tabs — `Visualization`, `SuperResolution`, `Calibration`, `Classification` — each hosting its own `HSIViewer`. There is no sidebar and no `panelContainer` anymore.

| Attribute | Type | Purpose |
|-----------|------|---------|
| `tabWidget` | `QTabWidget` | Top-level section selector (replaces the old sidebar buttons) |
| `viewer` | `HSIViewer` | Visualization tab's image display |
| `imageFilePath` | `QLabel` | Loaded file path, Visualization tab |
| `superResViewer` | `HSIViewer` | Super-Resolution tab's image display |
| `superResFilePath` | `QLabel` | Loaded file path, Super-Resolution tab |
| `calibrationViewer` | `HSIViewer` | Calibration tab's image display |
| `darkFileButton` / `darkFileEdit` | `QPushButton` / `QLineEdit` | Dark-frame file picker |
| `referenceFileButton` / `referenceFileEdit` | `QPushButton` / `QLineEdit` | Reference-frame file picker |
| `calibrateButton` | `QPushButton` | Always disabled — see `_connect_signals` |
| `classificationViewer` | `HSIViewer` | Classification tab's image display |
| `classificationFilePath` | `QLabel` | Loaded file path, Classification tab |
| `classificationModeTabs` | `QTabWidget` | Nested Unsupervised / Supervised sub-tabs |
| `numOfClassesEdit`, `maxIterationsEdit` | `QLineEdit` | Unsupervised tab inputs (K-means params) |
| `unsupervisedClassifyButton` | `QPushButton` | Starts/cancels worker-thread SPy K-means after image load |
| `lineEdit` | `QLineEdit` | Supervised tab — groundtruth file path |
| `comboBox` | `QComboBox` | Supervised tab — Model-backed classifier choice. The Controller populates it from `SupervisedClassifierType` and stores each enum as item data (Gaussian / Mahalanobis). |
| `pushButton` | `QPushButton` | Supervised tab — mask picker with automatic reference-cube pairing |
| `pushButton_2` | `QPushButton` | Runs/cancels Gaussian or Mahalanobis reference-example classification |
| `actionLoadImage` | `QAction` | File dropdown — load |
| `actionSaveImage` | `QAction` | File dropdown — save |

There is no `menubar`/`menuFile` anymore — `MainWindow.ui` no longer declares a `QMenuBar`. `_configure_file_menu` (in `main_window.py`) builds the File `QMenu` from the two actions above at runtime and installs it as a `QToolButton` in `tabWidget`'s top-left corner (see below).

Classification layer UI should be backed by one
`core.ClassificationLayerModel` per current classification result. The
Controller owns mutable layer visibility/names and passes only Model-produced
RGB/RGBA arrays or statistics to widgets. The layer Model is invalidated with
the classification result after load, crop, crop undo/redo, or
reclassification. See `docs/model_classification_api.md` for the integration
contract and `docs/classification_layer_api.md` for the detailed PyQt6
handoff.

---

## Design Decisions

### Acyclic dependency via signals

`HSIViewer` emits `spectrumPlotRequested(pos)`, `meanIndexRequested(name)`, `cropRequested(rect)` rather than holding a back-reference to the controller. `ui/viewer.py` still has no import from `ui/main_window.py`. All three are connected in `MainWindowController._connect_signals`. A parallel `historyChanged`/`photoClicked` pair existed for an annotation undo/redo mechanism but was never connected to anything; both were removed as dead code in `LEAF-156` along with the mechanism they supported.

### Sidebar + swappable panels → static tabs

The original design swapped a single `FeaturePanel` subclass in and out of a named `panelContainer`, driven by a `Functionality` enum and a sidebar of mode buttons. As of PRs #5–#11, `MainWindow.ui` was rebuilt around a `QTabWidget` with one fixed tab per feature, each holding its own `HSIViewer` and controls directly. This trades the old single-shared-viewer model for four independent viewers (simpler per-tab layout, no dynamic widget teardown/rebuild) at the cost of the four viewers not staying in sync. The now-dead `ui/panels/` classes and their `.ui` files have since been removed (see "Removed: `ui/panels/`" above).

### File menu → ribbon-style corner dropdown (`LEAF-121`)

The `QMenuBar`/`menuFile` row above the tabs has been removed from `MainWindow.ui`; `actionLoadImage`/`actionSaveImage` are still declared as standalone `QAction`s but are no longer attached to a menu bar in the `.ui` file. `MainWindowController._configure_file_menu` builds the `QMenu` from those actions at runtime and sets it on a `QToolButton` installed via `tabWidget.setCornerWidget(..., Qt.Corner.TopLeftCorner)`, so "File" reads left-to-right with the other tabs in one row (Word ribbon-style) instead of a separate row above them.

### `HSIData` consolidates scattered state

`HSIData` still owns `image_path`, `spectral_obj`, `rgb_array`, `mask_array`, etc., written only by `_load_image`. In practice `_load_image` currently pushes data into the four `HSIViewer`s directly from its local variables rather than reading back through `self._hsi_data`, so `HSIData` is populated but not yet consumed as the single read path it was designed to be.

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

- **Calibration is permanently disabled.** `calibrateButton` is disabled unconditionally in `_connect_signals` and nothing re-enables it after image load. Tracked by `LEAF-116`.
- **Perceptron classification is not integrated.** The selector intentionally exposes only the reference-example Gaussian and Mahalanobis workflows connected through `pushButton_2`; SPy's perceptron still requires dimensionality-reduction and training-policy decisions.
- **`_save_image` is a stub** with no format decided yet.
