# Hyperspectral Image Inspector — Architecture Reference

## Package Structure

```
src/
├── main.py                        Entry point (12 lines)
├── core/                          Pure Python — zero Qt imports
│   ├── hsi_data.py                HSIData dataclass + enums
│   └── hsi_utils.py               Stateless utility functions
└── ui/                            All Qt-touching code
    ├── viewer.py                  HSIViewer custom QGraphicsView
    ├── main_window.py             MainWindowController
    ├── generated/
    │   └── MainWindow.py          Build artifact — never hand-edited
    └── panels/
        ├── base_panel.py          FeaturePanel abstract base
        ├── visualization_panel.py
        ├── calibration_panel.py
        ├── classification_panel.py
        └── super_resolution_panel.py
```

UI files live in `src/qt/`. Regenerate `ui/generated/MainWindow.py` after editing `MainWindow.ui`:

```bash
python -m PyQt6.uic.pyuic src/qt/MainWindow.ui -o src/ui/generated/MainWindow.py
```

---

## Dependency Graph

```
main.py
  └─ ui.main_window.MainWindowController
       ├─ ui.generated.MainWindow.Ui_MainWindow   (layout)
       │    └─ ui.viewer.HSIViewer                (custom widget)
       ├─ core.hsi_data.HSIData                   (shared state)
       ├─ core.hsi_utils                          (I/O helpers)
       └─ ui.panels.*Panel                        (swappable feature UIs)
            └─ core.hsi_data.HSIData              (read-only, injected)
```

`HSIViewer` has **no import from `ui.main_window`**. Communication flows upward through Qt signals only.

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

Used as the key in `MainWindowController._panels` to select which `FeaturePanel` subclass to instantiate. Replaces the previous string-literal dispatch.

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

Single source of truth for all loaded image state. Created once in `MainWindowController.__init__` and passed by reference to every `FeaturePanel`. Only `MainWindowController._load_image` writes to it; panels read from it via `self._hsi_data`.

---

## `core/hsi_utils.py`

Stateless, pure functions with no Qt dependency. Safe to call from tests without a display.

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

---

## `ui/viewer.py`

### `PromptMode`

```python
class PromptMode(Enum):
    POINTS = auto()
    BOXES  = auto()
```

Controls the annotation interaction mode. Replaces the previous untyped `self.prompt: int` attribute.

---

### `HSIViewer(QtWidgets.QGraphicsView)`

Custom interactive view for displaying and annotating hyperspectral images. Decoupled from `MainWindowController` via outbound signals.

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

---

## `ui/main_window.py`

### `MainWindowController(QMainWindow, Ui_MainWindow)`

Application controller. Owns `HSIData`, manages panel lifecycle, and connects all signals. Inherits the widget layout from `Ui_MainWindow` via `setupUi(self)`.

```python
def __init__(self, *args, **kwargs) -> None:
    """Sets up UI, creates HSIData, wires all signals, loads default panel."""
```

#### Private methods

```python
def _connect_signals(self) -> None:
    """Wire sidebar buttons → _select_functionality,
    menu actions → _load_image/_save_image,
    viewer signals → handler slots."""

def _select_functionality(self, func: Functionality) -> None:
    """Swap the feature panel in panelContainer.
    Removes and deletes the old panel, constructs the new one,
    calls on_image_loaded() if an image is already loaded."""

def _load_image(self) -> None:
    """Open file dialog, detect PSI vs ENVI format, convert header if needed,
    open via spectral, extract RGB, populate HSIData, update viewer."""

def _save_image(self) -> None:
    """(stub)"""

def _on_history_changed(self, can_undo: bool, can_redo: bool) -> None:
    """Respond to viewer annotation history changes.
    Connect to actionUndo/actionRedo/actionClear once added to MainWindow.ui."""

def _on_spectrum_plot(self, pos: QPointF) -> None:
    """(stub)"""

def _on_mean_index(self, index_name: str) -> None:
    """(stub)"""
```

#### Panel registry

```python
_panels: dict[Functionality, type[FeaturePanel]] = {
    Functionality.VISUALIZATION:    VisualizationPanel,
    Functionality.SUPER_RESOLUTION: SuperResolutionPanel,
    Functionality.CALIBRATION:      CalibrationPanel,
    Functionality.CLASSIFICATION:   ClassificationPanel,
}
```

To add a new feature: create a `FeaturePanel` subclass, add a `.ui` file, and register the pair here.

---

## `ui/panels/base_panel.py`

### `FeaturePanel(QWidget)`

Abstract base for all swappable feature panels.

```python
class FeaturePanel(QWidget):

    def __init__(self, hsi_data: HSIData, parent: Optional[QWidget] = None) -> None:
        """Stores hsi_data reference. Subclasses must call super().__init__ then
        uic.loadUi(self._UI_PATH, self) to inflate their layout onto self."""

    def on_image_loaded(self) -> None:
        """Called by MainWindowController after _load_image() succeeds.
        Minimal implementation: self.setEnabled(True).
        Raises NotImplementedError if not overridden."""

    def reset(self) -> None:
        """Return to default (no-image) state.
        Minimal implementation: self.setEnabled(False).
        Raises NotImplementedError if not overridden."""
```

#### Contract for subclasses

```python
class MyPanel(FeaturePanel):
    # Declare widget attributes for IDE type-checking
    # (uic.loadUi sets them as instance attrs at runtime):
    someButton: QPushButton
    someEdit:   QLineEdit

    _UI_PATH = Path(__file__).parents[2] / "qt" / "MyPanel.ui"

    def __init__(self, hsi_data: HSIData, parent=None) -> None:
        super().__init__(hsi_data, parent)
        uic.loadUi(self._UI_PATH, self)
        # connect internal signals here
        self.setEnabled(False)

    def on_image_loaded(self) -> None:
        self.setEnabled(True)

    def reset(self) -> None:
        self.setEnabled(False)
```

---

## `ui/panels/visualization_panel.py`

### `VisualizationPanel(FeaturePanel)`

Loaded from `qt/Visualization.ui`.

```python
# Widget attributes (injected by uic.loadUi):
radioButton:   QRadioButton   # RGB (default checked)
radioButton_2: QRadioButton   # NDVI
radioButton_3: QRadioButton   # EVI
radioButton_4: QRadioButton   # MCARI
radioButton_5: QRadioButton   # MTVI
radioButton_6: QRadioButton   # OSAVI
radioButton_7: QRadioButton   # PRI
radioButton_8: QRadioButton   # Hypercube

def on_image_loaded(self) -> None: ...   # enables panel
def reset(self) -> None: ...             # checks RGB, disables panel
```

---

## `ui/panels/calibration_panel.py`

### `CalibrationPanel(FeaturePanel)`

Loaded from `qt/Calibration.ui`.

```python
# Widget attributes:
darkFileButton:      QPushButton
darkFileEdit:        QLineEdit
referenceFileButton: QPushButton
referenceFileEdit:   QLineEdit
calibrateButton:     QPushButton

def on_image_loaded(self) -> None: ...   # enables panel
def reset(self) -> None: ...             # clears file edits, disables panel
```

---

## `ui/panels/classification_panel.py`

### `ClassificationPanel(FeaturePanel)`

Loaded from `qt/Classification.ui`. Two-tab layout: Unsupervised and Supervised.

```python
# Unsupervised tab widget attributes:
numOfClassesEdit:           QLineEdit
maxIterationsEdit:          QLineEdit
unsupervisedClassifyButton: QPushButton

# Supervised tab widget attributes:
lineEdit:     QLineEdit    # groundtruth file path
comboBox:     QComboBox    # GaussianClassifier / MahalanobisDistanceClassifier / PerceptronClassifier
pushButton_2: QPushButton  # Classify

def on_image_loaded(self) -> None: ...   # enables panel
def reset(self) -> None: ...             # clears inputs, disables panel
```

---

## `ui/panels/super_resolution_panel.py`

### `SuperResolutionPanel(FeaturePanel)`

Loaded from `qt/Super-resolution.ui`.

```python
# Widget attributes:
superResolutionButton: QPushButton
lowResRadioButton:     QRadioButton
highResRadioButton:    QRadioButton
progressBar:           QProgressBar

def on_image_loaded(self) -> None: ...   # enables panel
def reset(self) -> None: ...             # resets progress bar to 0, disables panel
```

---

## `ui/generated/MainWindow.py`

Auto-generated by `pyuic6` from `qt/MainWindow.ui`. **Never edit manually** — changes will be overwritten on next regeneration.

Defines `Ui_MainWindow` with `setupUi(self)`, which creates all named widgets as instance attributes, including:

| Attribute | Type | Purpose |
|-----------|------|---------|
| `viewer` | `HSIViewer` | Main image display (custom widget) |
| `panelContainer` | `QWidget` | Host for the active `FeaturePanel` |
| `label_2` | `QLabel` | Displays the loaded file path |
| `visualizationButton` | `QPushButton` | Sidebar mode selector |
| `superResolutionButton` | `QPushButton` | Sidebar mode selector |
| `calibrationButton` | `QPushButton` | Sidebar mode selector |
| `classificationButton` | `QPushButton` | Sidebar mode selector |
| `actionLoadImage` | `QAction` | File menu — load |
| `actionSaveImage` | `QAction` | File menu — save |

---

## Design Decisions

### Acyclic dependency via signals

The previous design set `viewer.mainui = self` so the viewer could call `self.mainui.actionUndo.setEnabled(...)`. This created a circular dependency. The new design inverts it: `HSIViewer` emits `historyChanged(can_undo, can_redo)` and `MainWindowController` connects a slot. `ui/viewer.py` has no import from `ui/main_window.py`.

### Named `panelContainer` replaces `itemAt(2)`

The previous code used `self.verticalLayoutBottomRight.itemAt(2).widget()` to find the swappable panel — a silent runtime failure if the layout order changed. `panelContainer` is a named `QWidget` defined in `MainWindow.ui`, so `setupUi` makes it a typed attribute directly accessible as `self.panelContainer`.

### `Functionality` enum replaces string dispatch

`selectFunctionality("Super-resolution")` had no compile-time safety. The `Functionality` enum makes invalid values a `TypeError` at the call site, and the `_panels` dict is exhaustively typed.

### `HSIData` consolidates scattered state

Previously, `image_path` and `hsi` lived in `MainWindowController` while `rgb` and `mask_array` lived in `HSIViewer`. `HSIData` owns all of it. Panels read from `self._hsi_data`; only the controller writes to it on load.

### `core/` has zero Qt imports

All functions in `core/hsi_utils.py` work without a running `QApplication`. `numpy_to_qpixmap` imports Qt locally inside the function body. This keeps the module testable in a headless environment.