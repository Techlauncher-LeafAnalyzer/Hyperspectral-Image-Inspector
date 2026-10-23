# Batch Model API

`core.BatchService` runs Calibration / Super-Resolution / Classification over
many files without Qt dependencies. The UI runs it through
`ui.batch_worker.BatchWorker`; `ui.batch_controller.BatchController` owns the
Batch tab's file list, stage options, worker lifecycle, and results list.

## Pipeline

For each input file, in this fixed order:

1. Open the cube (`HSIReader`).
2. Auto-crop — the same Leaf2QR marker detection the single-image load runs
   (`core.marker_crop.find_plus_crop` → `HSIData.crop_polygon`). Files
   without the pattern are left uncropped. Every later stage runs on the
   cropped cube. An RGB preview of the (cropped) unprocessed cube is then
   saved as `<stem>_raw.<ext>` for every file, whichever stages are enabled.
3. Super-Resolution, if enabled — always runs on the uncalibrated source,
   matching the single-image tab.
4. Calibration, if enabled — `CalibrationFrameResolver` auto-resolves that
   file's own nearest `_calibFrame` dark/bright pair by capture timestamp (the
   same discovery the Calibration tab uses on load). When Super-Resolution
   also ran, the raw source is passed as `reference_source` so the 2×
   detector profile is interpolated rather than directly cropped.
5. Render an RGB preview of whatever cube is current.
6. Classification, if enabled — either independent per-file K-means
   (`UnsupervisedClassificationRequest`), or one supervised classifier trained
   once from a single shared ground-truth mask/cube pair
   (`BatchClassificationOptions.training_pair`) and applied to every file.
   Supervised training is calibrated once if any batch target is calibrated,
   reusing the single-image Controller's standardize-features fallback when
   no training calibration frames exist. With
   `BatchClassificationOptions.separate_leaves` (supervised only), each
   file's plant class is split into one `Leaf N` class per clearly separate
   cluster (`core.leaf_separation.separate_leaves`) and the background class
   is drawn black, matching the Classification tab's Separate Leaves button.
   Each leaf's number (the `N` in `Leaf N`) is burned in just beneath it by
   `core.leaf_labels` — the same renderer the Classification canvas and its
   Save Image use.
7. Export the resulting RGB (or colorized class map) as PNG/TIFF/JPEG/BMP,
   named after the input file's stem, into the selected output directory. The
   auto-crop polygon is preserved as alpha for PNG/TIFF; JPEG/BMP have no
   alpha channel and keep the crop's bounding rectangle instead.
8. Summary CSV, if `BatchRequest.export_csv` — `<stem>.csv` in the same
   format as the Classification tab's Export (`export_class_summary_csv`).
   With classification: one row per class with RGB/index means and index
   statistics. Without: one "Whole image" row of means over the ROI. Indices
   the camera's wavelength range cannot compute are omitted.

One file's failure (missing calibration frames, an incompatible SR camera, an
export conflict, ...) is recorded in `BatchResult.file_results` and does not
stop the rest of the batch. `BatchService.run` accepts `progress`/
`is_cancelled` like the other Models; cancellation stops before starting the
next file and is reported via `BatchResult.cancelled`.

## Usage

```python
from core import (
    BatchClassificationMode, BatchClassificationOptions, BatchRequest,
    BatchService, CalibrationRequest, UnsupervisedClassificationRequest,
)

request = BatchRequest(
    input_paths=(Path("a.hdr"), Path("b.hdr")),
    output_directory=Path("out"),
    calibrate=CalibrationRequest(),
    classify=BatchClassificationOptions(
        mode=BatchClassificationMode.UNSUPERVISED,
        unsupervised_request=UnsupervisedClassificationRequest(n_classes=5),
    ),
)
result = BatchService().run(request)
for file_result in result.file_results:
    print(file_result.input_path, file_result.succeeded, file_result.output_path)
```

## Not implemented

- Full processed-cube export (ENVI). Only the rendered RGB/class-map preview
  is saved, matching the single-image Save Image limitation noted in
  `docs/functional-requirements.md`.
- A shared dark/bright reference pair override; batch calibration always
  auto-resolves per file.
