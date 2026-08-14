# Hyperspectral Image Inspector

The production application uses a PyQt6 View/Controller and a UI-independent
Model under `src/core`.

Visualization Model capabilities include lazy ENVI/PSI loading, RGB, single
bands, NDVI, EVI, MCARI, MTVI, OSAVI, PRI, pixel spectra, and renderer-neutral
hypercube payloads. Rendered display arrays can be exported through the
View-neutral visualization export Model.

Controllers should depend on the public Model surface:

```python
from core import HSIReader, VisualizationRequest, VisualizationService
```

See [the Visualization Model API](docs/model_visualization_api.md) and
[Model development workflow](docs/model-development-workflow.md).
