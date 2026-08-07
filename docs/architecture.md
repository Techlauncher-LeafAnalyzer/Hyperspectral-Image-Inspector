# Hyperspectral Image Inspector — Architecture Reference

> **2026-08 update:** the UI pivoted from a sidebar + swappable-panel design to a
> static `QTabWidget` layout (PRs #5–#11). The now-unused `src/ui/panels/*.py`
> classes and their per-panel `.ui` files have since been deleted — see
> "Sidebar + swappable panels → static tabs" under Design Decisions.

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

`HSIViewer` has **no import from `ui.main_window`**. Communication flows upward through Qt signals only — though as of this writing, `MainWindowController` does not connect to any of them (see "Known gaps" below).

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

### `PromptMode`

```python
class PromptMode(Enum):
    POINTS = auto()
    BOXES  = auto()
```

Controls the annotation interaction mode.

---

### `HSIViewer(QtWidgets.QGraphicsView)`

Custom interactive view for displaying and annotating hyperspectral images. Decoupled from `MainWindowController` via outbound signals. **Four independent instances now exist** — `viewer`, `calibrationViewer`, `superResViewer`, `classificationViewer` — one per tab, each loaded with the same `rgb_array`/`mask_array`/pixmap on image load. They are not kept in sync with each other after that (e.g. annotating in one does not reflect in another).

#### Signals

```python
photoClicked          = pyqtSignal(QPointF)       # emitted on click over the photo
historyChanged        = pyqtSignal(bool, bool)    # (can_undo, can_redo)
spectrumPlotRequested = pyqtSignal(QPointF)       # from context menu
meanIndexRequested    = pyqtSignal(str)           # index name e.g. "NDVI"
```

#### Public state (read/write by controller)

```python
rgb:          Optional[NDArray[uint8]]   # source RGB array, shape (H, W, 3)
mask_array:   Optional[NDArray[uint8]]   # current segmentation mask, shape (H, W)
prompt_mode:  PromptMode                 # POINTS or BOXES
is_split:     bool                       # split-view mode flag
```

#### Public API

```python
def has_photo(self) -> bool:
    """True when an image is currently loaded."""

def fit_in_view(self) -> None:
    """Scale the view to fit the current image exactly."""

def set_photo(self, pixmap: Optional[QPixmap] = None) -> None:
    """Load a new image. Clears all annotation state."""

def set_avatar(self, pixmap: QPixmap) -> None:
    """Set the overlay avatar pixmap."""

def set_mask(self, mask: NDArray[uint8]) -> None:
    """Replace the current segmentation mask and re-render the overlay."""

def draw_circle(self, point: NDArray[uint32], label: int) -> None:
    """Draw a prompt point: green circle for foreground (label=1),
    red for background (label=0)."""

def draw_rectangle(self, start: QPointF, end: QPointF) -> None:
    """Draw a blue bounding-box prompt rectangle."""

def undo(self) -> None:
    """Remove the last annotation action. Emits historyChanged."""

def redo(self) -> None:
    """Re-apply the last undone action. Emits historyChanged."""
```

#### Key interaction model

| Input | Behaviour |
|-------|-----------|
| Scroll wheel | Zoom in / out, centred on cursor |
| Drag | Pan (ScrollHandDrag mode) |
| Ctrl + left-click | Add foreground point |
| Ctrl + right-click | Add background point |
| Ctrl + release (no Shift) | Trigger segmentation render |
| Shift held during Ctrl+release | Batch mode — defer segmentation |
| Shift release | Flush batch, trigger segmentation |
| Right-click | Context menu (Spectrum Plot, Clear Selection, Index Mean) |

All of the above still works standalone inside each `HSIViewer`, but `historyChanged`, `spectrumPlotRequested`, and `meanIndexRequested` are emitted into the void — no slot is connected to any of them (see "Known gaps" below). There is also no `actionUndo`/`actionRedo`/`actionClear` in `MainWindow.ui` anymore for `historyChanged` to eventually drive.

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
| `unsupervisedClassifyButton` | `QPushButton` | Enabled once an image is loaded; not yet wired to a handler |
| `lineEdit` | `QLineEdit` | Supervised tab — groundtruth file path |
| `comboBox` | `QComboBox` | Supervised tab — classifier choice (Gaussian / Mahalanobis / Perceptron / TBD) |
| `pushButton` | `QPushButton` | Supervised tab — groundtruth file picker |
| `pushButton_2` | `QPushButton` | Supervised tab — Classify; enabled once an image is loaded, not yet wired |
| `actionLoadImage` | `QAction` | File dropdown — load |
| `actionSaveImage` | `QAction` | File dropdown — save |

There is no `menubar`/`menuFile` anymore — `MainWindow.ui` no longer declares a `QMenuBar`. `_configure_file_menu` (in `main_window.py`) builds the File `QMenu` from the two actions above at runtime and installs it as a `QToolButton` in `tabWidget`'s top-left corner (see below).

---

## Design Decisions

### Acyclic dependency via signals

`HSIViewer` emits `historyChanged(can_undo, can_redo)`, `spectrumPlotRequested(pos)`, `meanIndexRequested(name)` rather than holding a back-reference to the controller. `ui/viewer.py` still has no import from `ui/main_window.py`. The controller side of this contract is currently empty (see below) but the emitting side is intact, so wiring it back up is additive, not a redesign.

### Sidebar + swappable panels → static tabs

The original design swapped a single `FeaturePanel` subclass in and out of a named `panelContainer`, driven by a `Functionality` enum and a sidebar of mode buttons. As of PRs #5–#11, `MainWindow.ui` was rebuilt around a `QTabWidget` with one fixed tab per feature, each holding its own `HSIViewer` and controls directly. This trades the old single-shared-viewer model for four independent viewers (simpler per-tab layout, no dynamic widget teardown/rebuild) at the cost of the four viewers not staying in sync. The now-dead `ui/panels/` classes and their `.ui` files have since been removed (see "Removed: `ui/panels/`" above).

### File menu → ribbon-style corner dropdown (`LEAF-121`)

The `QMenuBar`/`menuFile` row above the tabs has been removed from `MainWindow.ui`; `actionLoadImage`/`actionSaveImage` are still declared as standalone `QAction`s but are no longer attached to a menu bar in the `.ui` file. `MainWindowController._configure_file_menu` builds the `QMenu` from those actions at runtime and sets it on a `QToolButton` installed via `tabWidget.setCornerWidget(..., Qt.Corner.TopLeftCorner)`, so "File" reads left-to-right with the other tabs in one row (Word ribbon-style) instead of a separate row above them.

### `HSIData` consolidates scattered state

`HSIData` still owns `image_path`, `spectral_obj`, `rgb_array`, `mask_array`, etc., written only by `_load_image`. In practice `_load_image` currently pushes data into the four `HSIViewer`s directly from its local variables rather than reading back through `self._hsi_data`, so `HSIData` is populated but not yet consumed as the single read path it was designed to be.

### `core/` has zero Qt imports

Still true. All functions in `core/hsi_utils.py` work without a running `QApplication`. `numpy_to_qpixmap` imports Qt locally inside the function body. This keeps the module testable in a headless environment.

---

## Known gaps

Carried over from the current rubric self-assessment / sprint-1 backlog, listed here because they're structural rather than just "unfinished feature":

- **Viewer signals are unconnected.** `historyChanged`, `spectrumPlotRequested`, and `meanIndexRequested` are emitted by all four `HSIViewer`s but `MainWindowController._connect_signals` does not listen to any of them. `_on_spectrum_plot` and `_on_mean_index` exist as stubs but are unreachable. Tracked by `LEAF-118` (spectrum plot) and the Visualization-tab index wiring under `LEAF-114`.
- **Calibration is permanently disabled.** `calibrateButton` is disabled unconditionally in `_connect_signals` and nothing re-enables it after image load. Tracked by `LEAF-116`.
- **Unsupervised/Supervised classify buttons are enabled but unwired** — `unsupervisedClassifyButton` and `pushButton_2` flip to enabled on image load but have no click handler. Tracked by `LEAF-119`/`LEAF-120`.
- **`_save_image` is a stub** with no format decided yet.
