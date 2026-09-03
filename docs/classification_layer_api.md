# Classification Layer API — View/Controller Integration Guide

This is the implementation handoff for developers building the classification
layer panel, RGB filtering, and per-class vegetation-index UI. The Model
already provides the scientific and layer operations. View and Controller code
should consume this API rather than interpreting or modifying classification
arrays directly.

The implementation is `src/core/classification_layer_model.py`, exported from
`core`, and has no Qt dependency.

## 1. Capabilities and ownership

After supervised or unsupervised classification, the API can:

- expose one immutable descriptor per class;
- show, hide, solo, or rename classes;
- preserve true RGB pixels while hiding unselected classes;
- return HxWx4 transparent layers for Photoshop-style graphics items;
- calculate NDVI, EVI, MCARI, MTVI, OSAVI, and PRI once per image;
- derive per-class statistics and spatial rasters from that calculation;
- apply class visibility to false-colour index displays;
- reject stale IDs, masks, analyses, and incompatible spatial shapes.

The Model does not create widgets, Qt images, dialogs, signals, or files.

| Owner | Responsibilities |
| --- | --- |
| Model | Class IDs, masks, visibility/name state, compositing, index calculations, statistics, validation. |
| Controller | Object lifecycle, workers, signal routing, error dialogs, cached analyses, conversion between Model arrays and View objects. |
| View | Layer rows, eye/checkbox state, charts/tables, `QImage`/`QPixmap`, interaction and presentation. |

## 2. Public types

Import from `core`, not implementation modules:

```python
from core import (
    CancelledError,
    ClassificationError,
    ClassificationIndexAnalysis,
    ClassificationLayer,
    ClassificationLayerComposite,
    ClassificationLayerModel,
    ClassIndexStatistics,
    VisualizationMode,
    VisualizationRequest,
)
```

| Type | Purpose |
| --- | --- |
| `ClassificationLayerModel` | Controller-owned visibility/name state, compositing, and index entry points. |
| `ClassificationLayer` | Immutable snapshot used to render one layer-panel row. |
| `ClassificationLayerComposite` | Read-only RGB, RGBA, visible-mask, and visible-ID output. |
| `ClassificationIndexAnalysis` | One index result, all class statistics, and on-demand class rasters. |
| `ClassIndexStatistics` | Immutable counts and summary values for one class. |

`ClassificationLayerModel` accepts either
`UnsupervisedClassificationResult` or `SupervisedClassificationResult`.

## 3. Controller state and lifecycle

Recommended Controller state:

```python
self._classification_result = None
self._classification_layers: ClassificationLayerModel | None = None
self._classification_index_analyses: dict[
    VisualizationMode,
    ClassificationIndexAnalysis,
] = {}
```

Accept a completed classification atomically:

```python
def accept_classification_result(self, result) -> None:
    self._classification_result = result
    self._classification_layers = ClassificationLayerModel(result)
    self._classification_index_analyses.clear()
    self._rebuild_class_layer_panel()
    self._refresh_classification_rgb()
```

Always create a new layer Model for a new result. Clear all classification
layer/index state after these events:

| Event | Why it invalidates state |
| --- | --- |
| Load another cube | Masks belong to another image. |
| Crop | Dimensions and pixel coordinates changed. |
| Crop undo/redo | The active spatial extent changed again. |
| Reclassify | IDs, masks, and counts may differ. |
| Close/clear project | Source data are no longer current. |

```python
def _clear_classification_state(self) -> None:
    self._classification_result = None
    self._classification_layers = None
    self._classification_index_analyses.clear()
    self.layerPanel.clear()
```

The Model also validates shapes and raises `ClassificationError` when stale
objects are accidentally combined, but the Controller should clear them first.

## 4. Building the layer panel

`model.layers` returns a stable tuple of `ClassificationLayer` snapshots:

```python
for layer in model.layers:
    print(layer.class_id, layer.name, layer.pixel_count, layer.visible, layer.opacity)
```

`layer.opacity` is a `float` in `[0.0, 1.0]`, defaulting to `1.0` for every
class. It is independent of `visible`: hiding a layer does not reset its
stored opacity, so toggling it back on restores the previous fade.

Never use the widget row as the class ID. K-means IDs are zero-based, while
supervised IDs may be `(1, 2)`, `(2, 7)`, or another positive sequence. Store
`layer.class_id` as Qt item data.

Example with `QListWidget`:

```python
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

CLASS_ID_ROLE = QtCore.Qt.ItemDataRole.UserRole

def _rebuild_class_layer_panel(self) -> None:
    model = self._classification_layers
    self.layerList.blockSignals(True)
    try:
        self.layerList.clear()
        if model is None:
            return
        for layer in model.layers:
            item = QtWidgets.QListWidgetItem(
                f"{layer.name} ({layer.pixel_count:,} px)"
            )
            item.setData(CLASS_ID_ROLE, layer.class_id)
            item.setFlags(
                item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable
            )
            item.setCheckState(
                QtCore.Qt.CheckState.Checked
                if layer.visible
                else QtCore.Qt.CheckState.Unchecked
            )
            self.layerList.addItem(item)
    finally:
        self.layerList.blockSignals(False)
```

Blocking signals avoids recomposing the image for every checkbox while rows
are being rebuilt.

### Toggle one layer

```python
def _on_layer_item_changed(self, item) -> None:
    model = self._classification_layers
    if model is None:
        return
    class_id = int(item.data(CLASS_ID_ROLE))
    visible = item.checkState() == QtCore.Qt.CheckState.Checked
    try:
        model.set_class_visible(class_id, visible)
        self._refresh_classification_rgb()
    except ClassificationError as exc:
        self._show_classification_error(str(exc))
        self._rebuild_class_layer_panel()
```

### Adjust one layer's opacity

```python
def _on_layer_opacity_changed(self, class_id: int, opacity: float) -> None:
    model = self._classification_layers
    if model is None:
        return
    try:
        model.set_class_opacity(class_id, opacity)
        self._refresh_classification_rgb()  # debounce for slider drags
    except ClassificationError as exc:
        self._show_classification_error(str(exc))
```

`compose_display`/`compose_rgb`/`compose_index` blend each class toward the
requested background colour by its opacity, using `visible=False` as
opacity `0.0` for that class regardless of its stored value. A slider drag
should debounce the recomposition call, not the Model update — see
[Performance and memory](#11-performance-and-memory).

Pass `compose_display(..., base_rgb=data.rgb_array)` to reveal the true-colour
image beneath a faded/hidden class instead of a flat background colour —
this is the Photoshop-style "layer over a photo" look. `base_rgb` must match
`image_shape`; fall back to `background_color` when it does not (e.g. the
cube was reclassified but the cached true-colour array has not caught up).

### Solo, show/hide all, rename, and restore

```python
model.show_only(class_id)
model.set_all_visible(True)
model.set_all_visible(False)
model.rename_class(class_id, user_entered_name)
model.set_visible_classes(saved_visible_class_ids)
```

Rebuild rows after solo/show-all/hide-all/rename so widget state reflects the
Model. `set_visible_classes` validates the complete selection before changing
anything. Names are presentation metadata and never change IDs or masks.

## 5. Displaying selected classes as true RGB

Use the cached RGB `VisualizationResult.display_rgb`. Do not pass the coloured
classification label map, because that would display palette colours rather
than the leaf's true RGB appearance.

```python
def _refresh_classification_rgb(self) -> None:
    model = self._classification_layers
    rgb_result = self._visualization_results.get(VisualizationMode.RGB)
    if model is None or rgb_result is None:
        return

    composite = model.compose_rgb(
        rgb_result.display_rgb,
        background_color=(255, 255, 255),
    )
    self.classificationViewer.set_photo(
        hsi_utils.numpy_to_qpixmap(composite.display_rgb)
    )
```

`ClassificationLayerComposite` fields:

| Field | Shape/type | Meaning |
| --- | --- | --- |
| `display_rgb` | HxWx3 `uint8` | True RGB blended toward `base_rgb` (if given) or the requested background colour, by each pixel's class opacity (hidden = opacity 0). |
| `display_rgba` | HxWx4 `uint8` | True RGB with alpha = each pixel's class opacity scaled to 0-255 (0 when hidden). |
| `visible_mask` | HxW `bool` | Union of currently visible class masks, independent of opacity value. |
| `visible_class_ids` | `tuple[int, ...]` | Visibility snapshot used for the composite. |

Returned arrays are read-only. Replace the View image; do not edit arrays.

### One graphics item per class

For a Photoshop-style `QGraphicsPixmapItem` stack:

```python
rgba = model.rgb_layer(rgb_result.display_rgb, class_id)
```

Convert it into a detached Qt image:

```python
rgba = np.ascontiguousarray(rgba)
height, width = rgba.shape[:2]
image = QtGui.QImage(
    rgba.data,
    width,
    height,
    rgba.strides[0],
    QtGui.QImage.Format.Format_RGBA8888,
).copy()
pixmap = QtGui.QPixmap.fromImage(image)
```

`.copy()` detaches `QImage` from the temporary NumPy object's lifetime.

Current masks are mutually exclusive, so graphics-item order does not change
pixels. The API manages visibility and names, not arbitrary z-order or opacity.

## 6. Vegetation-index analysis per class

Supported modes are NDVI, EVI, MCARI, MTVI, OSAVI, and PRI.

### Reuse a cached index

Preferred when the Controller already rendered the index:

```python
cached = self._visualization_results[VisualizationMode.NDVI]
analysis = self._classification_layers.analyze_visualization(cached)
self._classification_index_analyses[VisualizationMode.NDVI] = analysis
```

This does not reread spectral bands or repeat the formula.

### Calculate from `HSIData`

```python
analysis = model.analyze_index(
    hsi_data,
    VisualizationRequest(
        mode=VisualizationMode.NDVI,
        colormap="RdYlGn",
    ),
    progress=worker.emit_progress,
    is_cancelled=worker.is_cancelled,
)
```

`analyze_index` delegates to `VisualizationService`, reads required bands once,
calculates one HxW raster, and partitions it with every class mask. Run this in
a worker because it performs disk I/O. Do not read the same `HSIData` from
another worker concurrently.

### Minimal PyQt6 worker

```python
class ClassificationIndexWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, str)
    completed = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal()

    def __init__(self, layers, data, request):
        super().__init__()
        self.layers = layers
        self.data = data
        self.request = request
        self._cancel_requested = False

    @QtCore.pyqtSlot()
    def run(self):
        try:
            analysis = self.layers.analyze_index(
                self.data,
                self.request,
                progress=self.progress.emit,
                is_cancelled=lambda: self._cancel_requested,
            )
        except CancelledError:
            self.cancelled.emit()
        except ClassificationError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"Unexpected index-analysis failure: {exc}")
        else:
            self.completed.emit(analysis)
        finally:
            self.finished.emit()

    def cancel(self):
        self._cancel_requested = True
```

Move it to a `QThread`, connect all signals before starting, and update widgets
only from GUI-thread slots. Keep the previous analysis visible until success.

## 7. Reading per-class index results

Get summary values without allocating another HxW raster:

```python
stats = analysis.statistics_for_class(class_id)
```

| `ClassIndexStatistics` field | Meaning |
| --- | --- |
| `class_id`, `class_name` | Stable ID and name captured when analysis ran. |
| `pixel_count` | All pixels assigned to the class. |
| `finite_pixel_count` | Pixels with valid finite index values. |
| `image_fraction` | Class pixels divided by total image pixels. |
| `mean`, `median` | Central tendency of finite class values. |
| `standard_deviation` | Population standard deviation. |
| `minimum`, `maximum` | Finite value range. |

Non-finite values are excluded. If a class is empty or has no finite samples,
numeric summaries are `None`; show “No valid pixels” or an em dash, not zero.
`class_name` is a snapshot from analysis time. If a user renames a layer later,
use the current `model.layers` name in the View or recompute the analysis.

### Raw spatial values

```python
masked = analysis.masked_values(class_id)
```

This returns HxW `float32`; outside pixels are `NaN`. A format-specific fill
value can be requested with `fill_value=-9999.0`, but record that sentinel in
export metadata.

### False-colour display

```python
class_rgba = analysis.display_rgba_for_class(class_id)

visible_index = model.compose_index(
    analysis,
    background_color=(255, 255, 255),
)
view.show_rgb(visible_index.display_rgb)
```

`display_rgba_for_class` returns one transparent false-colour layer.
`compose_index` applies current visibility to the full false-colour result and
rejects analyses created by another layer Model.

## 8. Method reference

### `ClassificationLayerModel`

| Member | Use |
| --- | --- |
| `result` | Original classification result. |
| `image_shape` | Required `(height, width)`. |
| `class_ids` | Stable IDs in mask order. |
| `visible_class_ids` | Current visible IDs. |
| `layers` | Immutable View-row snapshots. |
| `mask_for_class(id)` | Read-only HxW binary mask. |
| `rename_class(id, name)` | Change presentation label. |
| `set_class_visible(id, bool)` | Toggle one class. |
| `set_class_opacity(id, float)` | Set one class's blend opacity (0.0-1.0). |
| `set_visible_classes(ids)` | Replace visible selection atomically. |
| `show_only(id)` | Solo one class. |
| `set_all_visible(bool)` | Show/hide all. |
| `visible_mask()` | Read-only HxW visible union. |
| `compose_rgb(rgb, ...)` | True-RGB visibility composite. |
| `compose_display(rgb, ..., base_rgb=None)` | Composite any HxWx3 rendering, optionally over a true-colour base image. |
| `rgb_layer(rgb, id)` | Transparent true-RGB class layer. |
| `analyze_index(data, request, ...)` | Read/calculate/partition an index. |
| `analyze_visualization(result, ...)` | Partition a cached index. |
| `compose_index(analysis, ...)` | Visibility-filtered false-colour display. |

### `ClassificationIndexAnalysis`

| Member | Use |
| --- | --- |
| `visualization` | Values, display, title, bands, wavelengths, and colormap. |
| `statistics` | All class summaries in class order. |
| `class_ids` | Stable IDs for this analysis. |
| `statistics_for_class(id)` | One class summary. |
| `mask_for_class(id)` | Original class mask. |
| `masked_values(id, ...)` | Allocate one class-only analytical raster. |
| `display_rgba_for_class(id)` | Allocate one class-only false-colour RGBA. |

## 9. Saving/restoring layer UI state

Do not duplicate large classification arrays in a lightweight project file.
Save only View metadata keyed by stable class ID:

```python
layer_state = {
    "visible_class_ids": list(model.visible_class_ids),
    "names": {str(layer.class_id): layer.name for layer in model.layers},
}
```

After recomputing classification, restore only matching IDs:

```python
if set(saved_visible_ids).issubset(model.class_ids):
    model.set_visible_classes(saved_visible_ids)

for class_id_text, name in saved_names.items():
    class_id = int(class_id_text)
    if class_id in model.class_ids:
        model.rename_class(class_id, name)
```

K-means numeric IDs are not guaranteed to keep the same semantic meaning
across runs. Persistent semantic labels require saving the result itself or an
explicit class-matching workflow.

### Exporting what the user sees

For an opaque image, pass `composite.display_rgb` to the existing
`VisualizationExportService.save_display` API. This exports the selected true
RGB/index rendering with the chosen background colour.

For a transparent PNG, use `composite.display_rgba`, `rgb_layer`, or
`analysis.display_rgba_for_class` in the View/Controller export path. Preserve
all four channels when constructing `QImage` or a Pillow image. Do not send
RGBA to an export function that validates only HxWx3 RGB.

## 10. Errors and cancellation

Catch `ClassificationError` for expected problems such as:

- unknown/non-integer or duplicate class IDs;
- empty names or non-boolean visibility;
- wrong RGB/index dimensions;
- unsupported analysis modes such as RGB or BAND;
- an analysis belonging to another result;
- overlapping or malformed one-hot masks.

Treat `CancelledError` as neutral: retain the previous image/analysis and do
not show an error dialog. Log unexpected exceptions with a traceback, then
show a short user-facing failure at the Controller boundary.

## 11. Performance and memory

- Visibility changes never read the cube.
- A composite allocates one HxW mask, one HxWx3 image, and one HxWx4 image.
- `rgb_layer`/`display_rgba_for_class` allocate one HxWx4 image per call;
  generate on demand instead of caching every class layer.
- `analyze_index` calculates one full index raster for all classes.
- `masked_values` allocates one HxW float32 array only when requested.
- Never construct a KxHxW float index cube. The API intentionally reuses one
  raster plus existing uint8 masks.
- Index reads belong in a worker. For very large captures, debounce checkbox
  changes or move RGB composition to a short worker if GUI updates stutter.
- Returned arrays are read-only and must not be mutated.

## 12. Extending the API safely

When adapting the layer system, preserve these boundaries:

- Add scientific index formulas and wavelength rules to
  `VisualizationService`, then opt the mode into classification analysis.
  Do not duplicate formulas in a widget or Controller.
- Add persistent layer properties (opacity already lives on
  `ClassificationLayerModel`/`ClassificationLayer`, per `set_class_opacity`
  above) to the Model; do not store the only copy in widget state.
- Keep Qt types/signals in `src/ui`. Public `core` inputs and outputs should
  remain Python values, dataclasses, and NumPy arrays.
- Keep stable class IDs distinct from row position and user-facing names.
- Preserve lazy allocation: do not cache a float raster for every class unless
  a demonstrated workflow needs it.
- Add public types through `src/core/__init__.py` so Controllers depend on the
  stable `core` surface.
- Update local Model tests, this guide, architecture notes, and tracked UI tests
  together when changing a public contract.

## 13. Integration checklist

- [ ] Keep one `ClassificationLayerModel` for the current result.
- [ ] Store `class_id` in every layer row; never infer it from row position.
- [ ] Block signals while rebuilding layer widgets.
- [ ] Pass true RGB—not the class palette—to `compose_rgb`.
- [ ] Use RGBA and detached `QImage.copy()` for graphics-item layers.
- [ ] Reuse cached `VisualizationResult` objects when available.
- [ ] Run `analyze_index` in a worker and serialize `HSIData` reads.
- [ ] Display missing statistics as unavailable, not zero.
- [ ] Clear state after load, crop, undo/redo, and reclassification.
- [ ] Catch `ClassificationError`; treat cancellation as neutral.
- [ ] Do not assume K-means IDs are semantically stable between runs.

## 14. Suggested View/Controller tests

1. Build rows for zero-based K-means and non-zero supervised IDs.
2. Hide one class and verify its RGB pixels use the background.
3. Hide all and verify alpha is zero everywhere.
4. Solo one class and verify only its mask remains visible.
5. Rename a class without changing its ID.
6. Switch RGB/index modes without losing visibility state.
7. Cancel index calculation and retain the previous result.
8. Load/crop/reclassify and verify old rows/analyses are cleared.
9. Show “No valid pixels” for a class without finite index values.
10. Close during a worker and verify clean cancellation.

Model tests remain in the ignored local `tests/` directory. Add tracked UI
tests under `ui_tests/` when the layer panel is implemented.
