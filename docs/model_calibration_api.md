# Calibration Model API

`core.CalibrationService` provides dark/white radiometric correction without
Qt dependencies or source-file changes. The UI runs it through the dedicated
`ui.calibration_worker.CalibrationWorker` background boundary.
`ui.calibration_controller.CalibrationController` owns the file pickers,
worker lifecycle, result, and Calibration-tab rendering; `MainWindowController`
only coordinates it with shared image load, crop history, and other features.

## Calculation

For each detector column and spectral band, the Model averages exactly the
middle 21 rows in the dark and bright references. A 50-row frame uses rows
15–35 inclusive (`15:36` in zero-based Python notation). It logically spans
each averaged row to the source capture's original, pre-crop row count, then
applies the source's current cumulative row/column crop before the standard
correction:

```text
reflectance = (source - mean(dark[middle 21], axis=rows))
              / (mean(bright[middle 21], axis=rows)
                 - mean(dark[middle 21], axis=rows))
```

Reference row counts may differ but each must contain at least 21 rows. Their
column and band counts must match the original pre-crop source, and band
wavelengths must agree within the request tolerance (0.5 nm by default). This
policy retains pushbroom detector-column correction while reducing reference
capture noise.

For a 2× Super-Resolution result, pass the current original image as
`reference_source` to `calibrate`. The same 21-row means are cropped to the
original image's detector columns, then linearly interpolated at high-resolution
pixel centres and broadcast across the 2× result. This is an approximation:
the calibration frames were captured at the original resolution. The service
supports this API, but the GUI always calibrates the raw low-resolution source.
Afterward, that calibrated cube becomes the displayed low-resolution image and
the input to Super-Resolution. If SR was run first, starting calibration asks
for confirmation and discards the SR image and its high-resolution derivatives.
There is no calibrated/raw display switch. Changing references clears the
calibration and any SR result derived from it.

Facility calibration names start with `YYYY-MM-DD--HH-MM-SS` and end in
`_calibFrame`. When both selected files follow that convention, the GUI orders
them automatically: the earlier capture is dark and the later capture is
bright. Explicit picker order remains supported for legacy file names.

On source load, `CalibrationFrameResolver` scans only the source folder for
complete `_calibFrame.hdr` or `_calibFrame.json` plus data-file pairs captured
before the source. When both metadata files exist for a frame, `.hdr` takes
priority and the frame is counted only once.
Frames are paired only when captured no more than 60 seconds apart, and the
valid pair with the most recent bright frame is selected. This is a convenience,
not a prerequisite: when no pair is found the fields remain empty, and either
auto-selected path can always be replaced with the Dark/Bright file pickers.

Values are not clipped to `[0, 1]`: negative values and overshoot can be useful
diagnostics and remain part of the analytical result. NaN/infinite input and a
white-minus-dark response at or below the configured minimum fail with a
location-specific `CalibrationError`.

## Usage

```python
from core import CalibrationService, HSIReader

reader = HSIReader()
source = reader.open("capture.hdr")
dark = reader.open("dark.hdr")
white = reader.open("white.hdr")

result = CalibrationService().calibrate(source, dark, white)
reflectance_band = result.data.read_band(0)

# For an SR cube: calibrate(high_res, dark, white, reference_source=source)
```

`CalibrationRequest` configures line chunk size, wavelength tolerance, and
minimum valid detector response. `CalibrationResult` records current and
original source shapes, cumulative crop bounds, reference paths/row counts,
the exact central row ranges, and the finite reflectance range.

## Storage and lifecycle

The source and references stay lazy and unchanged. Expansion uses NumPy
broadcast views, so a 1971-row source does not allocate two full-sized
reference cubes. Calibration reads source data in bounded line chunks and
writes float32 output to a temporary ENVI BIP cube.
Keep the result alive while accessing `result.data`; call `result.cleanup()`
when finished. The application does this when a source/reference changes or
the window closes. Cancellation and failures remove incomplete output.

## GUI behavior

- Loading a timestamped source attempts to populate Dark/Bright automatically;
  Calibrate becomes available once a source plus both reference cubes are set.
- The button becomes Cancel while the worker is active; progress is reported
  in the status bar and other cube readers are paused.
- Success renders calibrated RGB in `calibrationViewer`; spectrum plotting and
  File → Save Image use that calibrated result. All low-resolution views now use
  the calibrated cube; SR uses it as input when subsequently run.
- Starting calibration when SR exists asks whether to discard the high-resolution
  image and its dependent results. No keeps the existing image unchanged; Yes
  discards them and runs calibration on the raw source.
- Cropping from any 2D tab automatically rebuilds an existing calibrated result
  against the new cumulative source bounds. Each completed calibration or SR
  operation clears crop undo/redo history and all prior classification results.
  Loading another source or replacing a reference clears calibration. Errors
  leave the raw source unchanged.
