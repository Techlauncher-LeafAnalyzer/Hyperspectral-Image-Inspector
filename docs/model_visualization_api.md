# Visualization Model API

The visualization package is the MVC Model. It contains no Qt imports and can
be used by a PySide6, PyQt6, CLI, test, or batch controller.

Its implementation lives in `core/visualization_model.py`. New feature modules use
the same `<feature>_model.py` convention while `core` re-exports
the stable public API.

## Public types

- `HSIReader`: validates and opens native ENVI or PSI `.hdr`/data pairs with SPy.
- `HSIData`: shared lazy `SpyFile` state plus immutable paths, metadata, shape,
  and wavelength information.
- `VisualizationService`: computes controller-requested visualization data.
- `VisualizationRequest`: mode, optional band index, stretch, and colormap.
- `VisualizationResult`: RGB `uint8` display array, raw index values, actual
  bands/wavelengths, title, range, and colormap.
- `SpectrumResult`: the values and wavelength axis for one pixel.
- `HypercubeData`: RGB top face and raw row/column spectral side slices for a
  view-owned 3D renderer.
- `HypercubeViewData`: aspect-preserving, downsampled real cube surfaces for
  SPy's interactive OpenGL hypercube without loading the full image volume.
- `VisualizationExportService`: validates and atomically saves the exact RGB
  display array selected by a Controller.
- `VisualizationExportRequest`: destination, overwrite permission, and JPEG
  quality supplied after the Controller completes its save-dialog workflow.
- `VisualizationExportResult`: saved path, format, dimensions, size, checksum,
  and whether the selected encoding is lossless.

## Controller integration

```python
from core import (
    HSIReader,
    VisualizationMode,
    VisualizationRequest,
    VisualizationService,
)

data = HSIReader().open(r"capture.hdr")
service = VisualizationService()

result = service.render(
    data,
    VisualizationRequest(VisualizationMode.NDVI),
    progress=controller.on_progress,
    is_cancelled=controller.is_cancelled,
)

# The view converts this NumPy array to its own image type.
view.show_rgb(result.display_rgb)
view.show_value_range(result.value_range)
```

Controller code should import the stable package surface shown above. Do not
import from `visualization_model.py` directly; `core` re-exports
the supported interface so implementation modules can change independently.

### Saving the current displayed view

The Model cannot determine what a Qt widget currently shows. The Controller
must retain the latest displayed RGB array and pass that exact array to the
export service. For an ordinary visualization this is normally
`VisualizationResult.display_rgb` or the matching `HSIData.rgb_array` after a
crop. Do not pass raw cube bands unless they are actually what the View shows.
If annotations, cursor overlays, viewport-only cropping, or zoom must appear in
the saved file, the View must capture its framebuffer and the Controller must
convert that capture to an RGB NumPy array before calling the Model.

```python
from core import (
    VisualizationExportError,
    VisualizationExportRequest,
    VisualizationExportService,
)

exporter = VisualizationExportService()

# output_path comes from the View's save-file dialog. Set overwrite=True only
# after the Controller has asked the user to replace an existing file.
try:
    saved = exporter.save_display(
        current_display_rgb,
        VisualizationExportRequest(output_path, overwrite=user_confirmed),
    )
except VisualizationExportError as error:
    controller.show_export_error(str(error))
else:
    controller.show_export_complete(
        saved.output_path,
        saved.file_size_bytes,
    )
```

Recommended MVC sequence:

1. View emits a Save action; Controller verifies an image is displayed.
2. Controller opens a save dialog using `supported_extensions` and obtains any
   overwrite confirmation. Dialogs and confirmation never belong in Model code.
3. Controller passes the retained `uint8 (height, width, 3)` RGB array and a
   `VisualizationExportRequest` to a worker.
4. Worker catches `VisualizationExportError` and returns either the error text
   or `VisualizationExportResult` through queued signals.
5. Controller updates status/UI on the GUI thread. A failed export leaves both
   the displayed pixels and any pre-existing destination unchanged.

PNG is appended when the selected name has no extension and is recommended for
exact pixel preservation. TIFF and BMP are also lossless; JPEG is intentionally
lossy. For the interactive hypercube, framebuffer capture is a View operation:
the View captures the current camera frame, the Controller converts it to an RGB
NumPy array, and this same Model service persists it.

### Recommended Controller lifecycle

1. Call `HSIReader.open(path)` in a worker when storage may be slow.
2. Open into a temporary `HSIData`; after initial rendering succeeds, call
   `current_data.update_from(candidate)`. This preserves references held by
   existing panels. On `HSIError`, retain the prior dataset and visualization.
3. Build a `VisualizationRequest` from View state and queue one read operation
   at a time for a given `HSIData`.
4. Bridge progress callbacks to the GUI thread with queued signals; callbacks
   execute in the Model's calling/worker thread.
5. Deliver the result to the GUI thread, convert `display_rgb` to the View's
   image type, and retain `values` only if analysis/export requires it.
6. Treat `CancelledError` as neutral completion: clear busy state silently and
   leave the prior image visible.

All model calls are synchronous. The controller owns thread scheduling and
must call long-running model operations from a worker. Progress callbacks use
`(integer_percent, message)`. Cancellation is an optional zero-argument
callable; a true result raises `CancelledError`.

The controller should catch `HSIError` and display its message while retaining
the previous `HSIData` instance. Subclasses distinguish file, header,
wavelength, visualization, and cancellation failures.

The frozen result dataclasses prevent field replacement, but contained NumPy
arrays are still ordinary objects. The Model does not cache returned results.
A Controller may discard them, retain them for export, or copy them when
another component needs mutable ownership.

### Coordinate convention

- Cube/array shapes are `(rows, columns, bands)`.
- Pixel coordinates are zero-based `(row, column)`.
- `display_rgb[row, column]` represents the same source pixel.
- Undo View scaling, scrolling, and letterboxing before calling
  `spectrum(data, row, column)` for a clicked display location.

### Hypercube handoff

Run `prepare_hypercube_view` in a worker because it reads several surfaces.
The returned `HypercubeViewData` can cross a queued signal, but create and
manipulate SPy's OpenGL widget only on the GUI thread. `surface_cube` contains
valid boundary values and a deliberately empty interior; it is a rendering
payload and must not be reused for scientific calculations.

### Error handling map

| Exception | Suggested Controller response |
| --- | --- |
| `HSIFileError` | Show the file/pair problem; retain the current dataset. |
| `HSIHeaderError` | Show the metadata problem; retain the current dataset. |
| `WavelengthError` | Explain or disable the unavailable visualization mode. |
| `VisualizationError` | Keep the prior view and show the request error. |
| `VisualizationExportError` | Show the save/path problem; keep the current view. |
| `CancelledError` | Clear busy state silently; retain the prior view. |
| Unexpected exception | Log its traceback and show a generic internal error. |

## Visualization modes

- RGB uses SPy's `get_rgb` with nearest 660, 550, and 470 nm bands.
- BAND returns a percentile-stretched grayscale band.
- NDVI uses 800 and 670 nm.
- EVI uses 800, 670, and 470 nm.
- MCARI uses 700, 670, and 550 nm.
- MTVI uses 800, 670, and 550 nm.
- OSAVI uses 800 and 670 nm with `L=0.16`.
- PRI uses 531 and 570 nm.

The result records the actual nearest wavelength selected from the cube.
Required bands must be within 15 nm (20 nm for RGB) of their target.

Vegetation-index formulas are scientifically meaningful on calibrated
reflectance. The model can compute them on raw data for inspection and
backward compatibility, but the controller should label the source state and
prefer the future calibration model's output for analysis.

## Memory behavior

Opening an image reads metadata only. RGB and indices use SPy's targeted band
reads instead of loading the full 480-band cube. `HSIData.estimated_float_bytes`
allows a controller to warn before any future algorithm requests a full
float32 load.

`prepare_hypercube_view` reads only four sampled boundary surfaces plus the RGB
top. The Controller creates SPy's `HypercubeWindow` on the GUI thread after
that preparation completes in a worker.
