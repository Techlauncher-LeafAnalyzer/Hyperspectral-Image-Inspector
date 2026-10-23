# Spectral Analysis Model API

Full-spectrum analysis over every band of a cube, complementing the
visualization indices (which each read two or three bands). Implemented in
`core/spectral_analysis.py`; no Qt imports. The Controller is
`ui/spectrum_controller.py`, which owns the `ui/spectrum_inspector.py` window.

## Public types

- `SpectralAnalysisService`: stateless operations below.
- `RegionSpectrum`: `key`, `label`, `pixel_count`, `wavelengths_nm`, and
  per-band `mean`, `std`, `minimum`, `maximum` over one set of pixels.
- `SimilarityResult`: per-pixel spectral `angles` (radians, NaN outside the
  ROI), raw `value_range`, and a `display_rgb` coloured with
  `SIMILARITY_COLORMAP` (`viridis_r`, bright = similar) over the 2nd–98th
  percentile `display_limits`.
- `SpectrumTransform`: `NONE`, `FIRST_DERIVATIVE`, `CONTINUUM_REMOVED`,
  `NORMALIZED` (area under the curve = 1).

## Operations

| Method | Reads | Notes |
|--------|-------|-------|
| `class_spectra(data, class_map, {id: name})` | whole cube, in 32-band chunks | Pixels whose class is not a key (e.g. `-1` outside a polygon crop) are ignored; empty classes omitted. |
| `image_spectrum(data)` | whole cube, chunked | Honors `data.roi_mask`. |
| `spectral_angle_map(data, reference, label)` | whole cube, chunked | Brightness-invariant; raises `VisualizationError` on a band-count mismatch or zero reference. |
| `spectral_angle(a, b)` / `separability(spectra)` | none | Pairwise angle between region means; symmetric, zero diagonal. |
| `transform(wavelengths, values, transform)` | none | Continuum removal divides by the upper convex hull. Wavelengths must ascend. |
| `export_csv(path, wavelengths, {name: values})` | none | One `wavelength_nm` column plus one column per series. |

Peak memory for the chunked scans is `rows × columns × 32` float32 values,
not a full cube copy.

## Controller integration

- Every `HSIViewer` emits `pixelHovered(row, column)` once per new pixel. When
  the inspector is visible and *Follow cursor* is ticked, the Controller reads
  that pixel from the cube the viewer shows (calibrated for the Calibration
  viewer when a result exists, otherwise `_display_data()`).
- Hover reads are skipped — never blocking — while SR, calibration,
  classification or a Hypercube build hold the cube. Explicit actions (pin,
  class spectra, similarity) instead stop the Hypercube worker, read, then
  resume it.
- Full-cube operations run synchronously behind a wait cursor, like the
  cached index renders, so the existing crop/load guards need no new worker.
- The similarity map is tied to the `HSIData` instance it was computed from
  and is discarded on load and crop (crops mutate `HSIData` in place).

## UI entry points

- **Ctrl+Click** any pixel in any viewer pins its spectrum (the cursor is a
  crosshair while Ctrl is held). This replaced the unused Ctrl+left-click
  segmentation prompt marker; Ctrl+right-click still adds a negative prompt.
- Viewer right-click → **Spectrum Analyzer** submenu: *Pin Spectrum Here*
  (shows `Ctrl+Click` in its shortcut column), *Add Image Mean*, *Add Class
  Spectra* (Classification viewer only), *Open Spectrum Inspector*.
- Visualization tab: *Spectrum Inspector* button; *Similarity* mode radio.
- Classification layer panel: classification classes and measured
  annotation regions (rectangles, ellipses, polygons) are listed together as classes,
  each with a selection checkbox and an **Inspect** button.
  - *Class spectra* plots every **selected** class and region
    (`SpectralAnalysisService.masked_spectra`, one chunked pass), replacing
    the previous set. A region's checkbox only selects it for analysis; it
    does not hide the annotation (that stays on the Annotations tab).
  - *Inspect* adds just that class/region to the plot (alongside what is
    already there) and highlights it. Colours: class-layer colours for
    classes, the annotation's stroke colour for regions.
  - *Select all / Deselect all* and the "N of M classes selected" summary
    cover both kinds.
- Pin markers can be **dragged** to another pixel: the pin keeps its number
  and colour, its spectrum updates live during the drag (one read per new
  pixel, skipped while a worker holds the cube) and is re-read on release.
  A drag is refused, and the marker snaps back, while SR/calibration/
  classification runs. Clicking a pin (with or without Ctrl) selects it
  rather than dropping a new pin on top.
- Each pinned spectrum is numbered (`#1 · Pixel (r, c) …`) and marked on
  the image by a numbered circle in its line's colour, at the pixel centre
  and a constant screen size. The selected pin's circle is larger; hidden
  pins have no circle. Circles are drawn only on viewers showing the cube the
  pin was read from (`SpectrumController.refresh_markers`, re-run after each
  viewer refresh), are never part of saved images, and are cleared by a crop
  or new image (numbering restarts at #1 on a new image).
- Inspector **Spectra** list (beside the plot): tick/untick to show or hide a
  series (hidden series are left out of the plot and the CSV export), select
  one — or click its line on the plot — to highlight it, and *Remove* /
  Delete to take it off. *Clear* removes everything.
- Inspector: pin cursor, image mean, display transform, ±1σ, index-band
  markers for the active visualization, reference picker → *Map similarity*,
  Export CSV. The *Separability* tab explains what it shows and how to fill
  it until at least two region spectra are plotted.

## Not implemented

- Background worker for full-cube scans (very large uncropped cubes block
  the GUI for the duration of the read).
- Spectra of annotation regions (they currently report index means only).
- Endmember extraction / unmixing, and saving SAM maps as raw values.
