# Classification Model API

The classification implementation is a UI-independent MVC Model in
`core/classification_model.py`. It uses Spectral Python (SPy) for K-means and
returns plain dataclasses and NumPy arrays. It imports no Qt types.

## Unsupervised K-means

```python
from core import (
    ClassificationService,
    HSIReader,
    UnsupervisedClassificationRequest,
)

data = HSIReader().open(r"capture.hdr")
service = ClassificationService()
request = UnsupervisedClassificationRequest(
    n_classes=5,
    max_iterations=20,
)

result = service.classify_unsupervised(
    data,
    request,
    progress=controller.on_progress,
    is_cancelled=controller.is_cancelled,
)
```

The result contains:

- `class_map`: zero-based integer class IDs shaped `(height, width)`.
- `one_hot_masks`: binary `uint8` masks shaped `(classes, height, width)`.
  `one_hot_masks[class_id]` is the HxW mask for one class.
- `cluster_centers`: mean spectrum for each class, shaped
  `(classes, selected_bands)`.
- `class_pixel_counts`: number of pixels assigned to each class.
- `band_indices` and `band_wavelengths_nm`: the features used by K-means.
- `iterations_completed`: number of completed SPy K-means iterations.

SPy may produce fewer populated clusters than requested. The result still
contains one mask and one pixel count per requested class; an empty class has
an all-zero mask and count zero. Every image pixel belongs to exactly one mask,
so `result.one_hot_masks.sum(axis=0)` is an all-one HxW array.

## Controller/View integration

`classify_unsupervised` is synchronous and can load substantial data. The
Controller should run it in a worker, bridge progress callbacks to queued Qt
signals, and update widgets only after the result returns to the GUI thread.
The Model does not select colors. The Controller/View can assign a palette to
`class_map`, or composite colors from the masks, without modifying analytical
class data.

Recommended flow:

1. Parse `numOfClassesEdit` and `maxIterationsEdit` as integers.
2. Construct `UnsupervisedClassificationRequest` and catch
   `ClassificationError` to report invalid input without starting a worker.
3. Optionally call `estimate_kmeans_working_bytes` and warn before a large
   allocation. Cropping or selecting fewer bands reduces memory and runtime.
4. Run `classify_unsupervised` in a worker. Do not read the same `HSIData`
   concurrently from multiple workers.
5. Retain the returned result in Controller/session state. Pass `class_map` or
   individual `mask_for_class(class_id)` arrays to the View.
6. Treat `CancelledError` as neutral completion and keep the previous result.
   On `ClassificationError`, show its message and keep the previous result.

The callback executes on the caller's thread. Cancellation is checked before
and after the cube read and between K-means iterations; an individual disk read
or SPy iteration is not interruptible.

### Current PyQt6 integration

`MainWindowController` connects `unsupervisedClassifyButton` to this API. It
uses a `QThread` worker, sends callback progress to the status bar, turns the
Classify button into a cancellation action while processing, and renders a
deterministically colored class map in `classificationViewer`. Analytical
labels and one-hot masks remain unchanged in `_classification_result`; the RGB
palette is View-only state. Loading/cropping is prevented during an active
worker, and any completed result is discarded after a later crop or image
load. Saving while the Classification tab is active saves the colored result.

## Band selection and memory

By default, every spectral band is used. For a reproducible subset, pass
zero-based band indices:

```python
request = UnsupervisedClassificationRequest(
    n_classes=5,
    max_iterations=20,
    band_indices=(0, 20, 40, 60, 80),
)
```

The supplied tray is approximately `1971x500x480`, or about 1.9 GB when loaded
as float32. SPy's vectorized K-means also allocates working arrays. Always run
large classification jobs outside the GUI thread and use
`estimate_kmeans_working_bytes` for preflight UI messaging.

SPy's official K-means API accepts an HxWxBands array and returns an HxW class
map plus KxBands centers. The Model converts that class map into the stable
one-hot contract; Controllers do not call SPy directly.

## Reference-example supervised classification

The user selects only a ground-truth mask. `TrainingPairResolver` searches the
same directory for a header/data pair using these conventions:

- `image_XXX.png` with `image_XXX.hdr` and `image_XXX.bil`;
- `image_XXX_mask.png` with `image_XXX.hdr` and `image_XXX.bil`;
- `image_XXX_mask.png` with `image_XXX_hyperspectral.hdr` and
  `image_XXX_hyperspectral.bil`.

The data extension may also be `.bip`, `.bsq`, `.dat`, `.img`, or `.raw`. A
same-stem `.hdr` is always required. Pair resolution is case-insensitive and
never searches outside the mask directory. The Controller catches
`ClassificationError` and shows the expected names when no complete pair is
found.

```python
from core import (
    ClassificationService,
    HSIReader,
    SupervisedClassificationRequest,
    SupervisedClassifierType,
    TrainingPairResolver,
)

pair = TrainingPairResolver().resolve(mask_path)
training_data = HSIReader().open(pair.cube_path)
target_data = HSIReader().open(current_image_path)

result = ClassificationService().classify_supervised(
    target_data,
    training_data,
    pair.mask_path,
    SupervisedClassificationRequest(
        SupervisedClassifierType.MAHALANOBIS,
    ),
    progress=controller.on_progress,
    is_cancelled=controller.is_cancelled,
)
```

Black/white masks are normalized to `1=target`, `2=background`, and
`0=ignored`. To prevent JPEG compression shades from becoming accidental SPy
classes, values at least 240 become target, values at most 15 become
background, and intermediate values are ignored. Lossless PNG/TIFF masks are
preferred. Exact indexed masks using IDs 0, 1, and 2 are also accepted.

The default supervised request evenly samples at most 24 training wavelengths
and matches them to the target within 0.5 nm. This bounds memory and gives each
class enough observations to estimate covariance on typical one-example
masks. The exact training/target indices and wavelengths are returned for
reproducibility. Gaussian and Mahalanobis are supported; Perceptron is rejected
with a user-facing explanation until its dimensionality-reduction and training
policy is defined.

For MVC integration, populate the View's classifier selector from the Model
enum and retain the enum as item data instead of reconstructing it from the
display label:

```python
classifier_combo.clear()
for classifier in SupervisedClassifierType:
    classifier_combo.addItem(classifier.value, classifier)

request = SupervisedClassificationRequest(classifier_combo.currentData())
```

This currently presents `GaussianClassifier` and
`MahalanobisDistanceClassifier`. Adding a future supported classifier to the
Model enum makes it available to the Controller without maintaining a second
hard-coded GUI list.

`SupervisedClassificationResult.class_map` retains the original positive class
IDs. `one_hot_masks` remains class-first KxHxW; `class_ids[index]` identifies
the class represented by `one_hot_masks[index]`.

## Classification layer API

For the complete View/Controller handoff—including PyQt6 layer widgets,
worker code, RGBA conversion, state persistence, lifecycle rules, performance
guidance, and an integration checklist—see
[`classification_layer_api.md`](classification_layer_api.md).

`ClassificationLayerModel` is the Controller-facing API for treating a
completed classification as independently manageable layers. It accepts both
supervised and unsupervised results and imports no Qt types.

Create a new layer model whenever classification completes:

```python
from core import ClassificationLayerModel

controller.classification_result = result
controller.classification_layers = ClassificationLayerModel(result)

for layer in controller.classification_layers.layers:
    # Build one View row / checkbox / eye icon.
    view.add_layer(
        class_id=layer.class_id,
        label=layer.name,
        pixel_count=layer.pixel_count,
        checked=layer.visible,
    )
```

The Controller must store `class_id` as the row's stable data value. Do not use
the row index: supervised class IDs can be values such as `(1, 2)` or another
non-zero sequence, while unsupervised IDs are zero-based.

### Toggle classes in the RGB image

When an eye/checkbox changes, update Model state and ask it for a new RGB/RGBA
composite:

```python
def on_layer_toggled(class_id: int, checked: bool) -> None:
    layers = controller.classification_layers
    layers.set_class_visible(class_id, checked)

    composite = layers.compose_rgb(
        controller.rgb_result.display_rgb,
        background_color=(255, 255, 255),
    )
    view.show_rgb(composite.display_rgb)
```

`display_rgb` preserves the real RGB pixels of visible classes and fills
hidden pixels with the requested background. It is ready for the existing
NumPy-to-QPixmap conversion. `display_rgba` contains the same visible RGB with
alpha 255 and makes hidden pixels transparent, which is convenient when the
View uses stacked graphics items. `visible_mask` and `visible_class_ids` are
returned with the composite so the Controller does not reconstruct state.

Additional layer-panel operations are:

- `set_class_visible(class_id, visible)` — one eye/checkbox;
- `set_visible_classes(class_ids)` — atomically restore a saved selection;
- `show_only(class_id)` — solo one layer;
- `set_all_visible(True/False)` — show/hide all;
- `rename_class(class_id, name)` — user-facing label only;
- `mask_for_class(class_id)` — original read-only HxW one-hot mask;
- `rgb_layer(rgb, class_id)` — independent HxWx4 transparent true-RGB layer.

These operations never modify `class_map`, `one_hot_masks`, or source RGB.
Layer snapshots and returned arrays should be treated as read-only.

### Calculate vegetation indices per class

The layer Model delegates NDVI, EVI, MCARI, MTVI, OSAVI, and PRI calculations
to `VisualizationService`, so the application has one authoritative set of
wavelength mappings and formulas:

```python
from core import VisualizationMode, VisualizationRequest

analysis = layers.analyze_index(
    hsi_data,
    VisualizationRequest(VisualizationMode.NDVI),
    progress=worker.emit_progress,
    is_cancelled=worker.is_cancelled,
)

for layer in layers.layers:
    stats = analysis.statistics_for_class(layer.class_id)
    view.update_index_row(
        class_id=layer.class_id,
        mean=stats.mean,
        median=stats.median,
        std=stats.standard_deviation,
        minimum=stats.minimum,
        maximum=stats.maximum,
        valid_pixels=stats.finite_pixel_count,
    )
```

`analyze_index` reads the required bands once and then partitions the single
analytical raster using all class masks. Run it in a Controller worker and do
not read the same `HSIData` concurrently from another worker. If the Controller
already cached the matching `VisualizationResult`, avoid another cube read:

```python
analysis = layers.analyze_visualization(cached_ndvi_result)
```

For spatial display or export:

```python
# Raw float32 values inside one class; outside pixels are NaN.
class_values = analysis.masked_values(class_id)

# One false-colour class layer with transparent pixels outside the class.
class_rgba = analysis.display_rgba_for_class(class_id)

# Current visibility applied to the full false-colour index image.
visible_index = layers.compose_index(analysis)
view.show_rgb(visible_index.display_rgb)
```

Statistics ignore non-finite index values and report both total class pixels
and finite pixels. An empty class or a class with no finite index samples uses
`None` for mean/median/standard-deviation/minimum/maximum rather than inventing
a numeric value.

### Lifecycle and invalidation

The Controller must discard both `ClassificationLayerModel` and any
`ClassificationIndexAnalysis` objects after loading a different image,
cropping, undoing/redoing a crop, or completing a new classification. Those
operations change the spatial relationship between the cube and masks. The
Model validates shapes and raises `ClassificationError` if stale objects are
combined accidentally.
