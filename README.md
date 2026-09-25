# Hyperspectral Image Inspector

The production application uses a PyQt6 View/Controller and a UI-independent
Model under `src/core`.

Visualization Model capabilities include lazy ENVI/PSI loading, RGB, single
bands, NDVI, EVI, MCARI, MTVI, OSAVI, PRI, pixel spectra, and renderer-neutral
hypercube payloads. Rendered display arrays can be exported through the
View-neutral visualization export Model. The Classification Model provides
SPy-backed unsupervised K-means and reference-example Gaussian/Mahalanobis
classification with class-first one-hot masks. `ClassificationLayerModel`
adds per-class visibility, true-RGB/transparent composites, and NDVI/EVI/
MCARI/MTVI/OSAVI/PRI statistics and masked rasters for layer-style GUIs.
The Calibration Model averages the middle 21 rows of timestamped dark/bright
frames, aligns them to the source's pre-crop geometry and current crop, performs
bounded-memory reflectance correction, and renders the lazy float32 result
through a cancellable Calibration-tab worker without modifying the source.
Nearby earlier `_calibFrame` pairs are selected automatically on source load,
with manual Dark/Bright replacement always available.

Controllers should depend on the public Model surface:

```python
from core import HSIReader, VisualizationRequest, VisualizationService
```

See [the Visualization Model API](docs/model_visualization_api.md),
[the Calibration Model API](docs/model_calibration_api.md),
[the Classification Model API](docs/model_classification_api.md), the detailed
[Classification Layer View/Controller guide](docs/classification_layer_api.md),
and the [Model development workflow](docs/model-development-workflow.md).

Super-Resolution uses the existing `model/fin_msdformer.pth` checkpoint for
480-band, 2× spatial inference. Install `requirements-sr.txt` in the environment
used to launch the app, load a compatible capture, and click **Run Super-Resolution**.
See [SR setup, model contract, and limitations](docs/super-resolution.md).
