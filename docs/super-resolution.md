# Super-Resolution

## Model provenance and contract

The existing tracked `model/fin_msdformer.pth` is byte-identical to
[`Software/GUI/weights/fin_msdformer.pth`](https://github.com/squashking/TechLauncher-HSISR/blob/98388e034e02e4e049a0816c7c9dcce74aa10683/Software/GUI/weights/fin_msdformer.pth)
in the previous project. SHA-256:
`e00fa2c87cbacc5b60a1752710e49e16f62f465514ec66d53bf2b730ce96ee81`.
It contains `epoch=200` and `model` (883 state-dict tensors, 22,436,417 values),
not a pickled model object. No ONNX files exist in that revision's tracked tree.

The network is MSDformer with 480 input/output bands, 240 features, four
deformable-convolution transformer modules, six attention heads, and groups
of eight bands with two overlapping bands. PixelShuffle produces 2× spatial
upscaling. The checkpoint fixes the band count and scale; it cannot be used
with arbitrary spectral channel counts by changing constructor arguments.

The previous [`tab_super_resolution_controller.py`](https://github.com/squashking/TechLauncher-HSISR/blob/98388e034e02e4e049a0816c7c9dcce74aa10683/Software.new_version/controllers/tab_super_resolution_controller.py)
loads all HSI bands as float32, performs `scipy.ndimage.zoom(band, 2, order=3)`
independently per band, converts both arrays from HWC to NCHW, and calls the
network in evaluation mode. Inputs are `(1,480,H,W)` and `(1,480,2H,2W)`;
output is `(1,480,2H,2W)`. There is no input normalization in this inference
path. Older software used intermediate MAT files for the same operations.
The old export clips negatives and rescales each output band independently to
0–4095 before writing BIL and displaying RGB. We omit that export normalization
to retain predictions in input units and avoid modifying spectral ratios.
Negative predictions are retained; display stretching does not modify the cube.

The adapted architecture is in `src/core/sr/`, with the upstream MPL-2.0 license.
Checkpoint parameter names and mathematical operations are preserved. Native
PyTorch reshape replaces einops; zero-probability DropPath is Identity. Unused
training hooks and statistics helpers are removed. No timm, torchvision,
torchnet, h5py, MAT intermediates, or custom compiled operators are required.

## Installation

```sh
conda activate pytorch_env
python -m pip install -r requirements.txt -r requirements-sr.txt -r requirements-dev.txt
python src/main.py
```

Keep the checkpoint at `model/fin_msdformer.pth`; its existing binary is unchanged.
The path is resolved relative to the repository, not the launch directory.
PyTorch/SciPy are optional and loaded on Run, so non-SR functionality still works
without them. Loading uses `weights_only=True`, strict parameter matching, and
finite-weight validation. It never falls back to unsafe model deserialization.

## Model service

Controllers import `SuperResolutionService`, `SuperResolutionRequest`, and
`SuperResolutionResult` from `core`. `run(data, request, progress=...,
is_cancelled=...)` is synchronous and has no Qt dependency. Serialize reads on
the supplied `HSIData`; cancellation is cooperative between tiles/bands.

The source is the currently loaded cube, including its current crop. All 480
bands and wavelengths remain in order. The service reads tiles lazily, supplies
both model inputs, validates finite values and exact output shape, and writes
float32 predictions to a temporary ENVI BIP memmap. The returned `result.data`
is ordinary `HSIData` usable by visualization and spectrum services. Keep the
result alive while using its data: it owns those temporary files. Source files
are never changed. Spectral metadata is retained; stale spatial/georeferencing
and storage metadata is not copied to the resized cube.

Default tile interiors are 64×64 LR pixels with eight context pixels on each
side. Context predictions are discarded; tile interiors cover every output
pixel exactly once, including odd dimensions and edges. Small inputs run in
one pass. Tiled predictions are approximate: spectral attention/global pooling
and interpolation use tile context, not the whole image. Seams or changes near
tile boundaries are possible. `tile_size` and `context` can be configured through
the service API; increasing tile size improves context at higher memory cost.

CUDA is used when available, otherwise CPU. MPS and mixed precision are not
enabled. CPU runs on large captures can take many minutes. Output disk space is
four times the source float32 cube size (about 7.05 GiB for 1971×500×480).
Processing uses bounded tile working memory; RGB display arrays still scale
with image dimensions. Temporary output is not a permanent saved HSI export.

## Validation

The existing SR tab now runs a `SuperResolutionWorker` (`QThread`). Both inference
and RGB rendering happen off the GUI thread; Qt signals deliver progress,
completion, cancellation, and errors. Qt pixmaps are constructed only on the GUI
thread. Run becomes Cancel while busy. Source loading, crop/undo/redo, and
spectrum disk reads are paused to serialize source access, while tabs, cached
visualization modes, and pan/zoom remain interactive. Closing requests cancellation
and waits asynchronously for the worker to finish.

The original `HSIData` remains unchanged. The SR tab shows RGB from the selected
Original/Processed dataset; other tabs retain original data. Pan/zoom coordinates
are scaled when switching between resolutions. SR pixel RGB, spectrum lookup,
and File → Save Image use the displayed dataset. Save Image exports the preview,
not the complete 480-band cube. Index means remain available on the original
image; crop the original and rerun SR instead of applying HR coordinates to LR.
Loading or cropping/undoing a source invalidates the old SR result. Failures and
cancellation retain the previous successful result.

```sh
OMP_NUM_THREADS=2 python -m pytest -q
OMP_NUM_THREADS=2 python -m pytest -q -m sr_model
python -m compileall -q src ui_tests
git diff --check
```

The normal suite checks reconstruction, all spectral channels, cropped input,
odd dimensions, progress, cancellation cleanup, malformed input/output, and
invalid checkpoints. The `sr_model` test runs actual weights and compares the
service output against a direct network call. It skips explicitly if the local
checkpoint or optional dependencies are absent; synthetic test fixtures are
generated locally and never uploaded.

### Local verification (2026-08-28)

Executed in `pytorch_env` (Python 3.12.12, PyTorch 2.10.0, SciPy 1.17.1) on CPU:

- All 883 checkpoint entries load with strict matching. Actual 5×7, 32×32,
  and 64×64 network inputs yield finite `(1,480,2H,2W)` outputs.
- A real plant ROI from `2025-04-17--12-51-23_round-0_cam-1_tray-1.hdr`, rows
  900:1092 and columns 175:367, runs through nine tiles: 192×192×480 →
  384×384×480 in 24.4 seconds, including preview rendering/validation, with
  `OMP_NUM_THREADS=2`. All wavelength entries are unchanged.
- Input range: 5–2627; prediction range: approximately −42.4–2816.4.
  Approximately 0.007% of predicted values are negative. Mean-spectrum
  correlation with the source is 0.999985; this is a sanity check, not a
  ground-truth quality metric. LR/SR previews show consistent structure and
  sharper leaf boundaries, with some edge ringing.
- Desktop PyQt Run/progress/completion and Original/Processed switching were
  inspected on that crop. Automated Qt tests also verify event-loop ticks during
  actual model execution, preview saving, cancellation, failure recovery,
  stale-result invalidation, and safe shutdown.
- Full suite: 79 passed with the real crop supplied in the ignored test resources
  directory. The existing `.venv` without SR dependencies passes 68 tests with
  11 dependency-dependent SR tests skipped. Compile and diff-whitespace checks pass.
- The full 1971×500 capture has not been run end-to-end. Testing used a real
  crop to exercise tiling without generating a 7 GiB temporary result.

## Scientific limitations

480 matching bands is a shape requirement, not evidence that an arbitrary
sensor matches the training distribution. The checkpoint does not contain the
training wavelength grid or a normalization specification. The previous code
targets APPF data and assumes the supplied band ordering and intensity units.
No new training, spectral interpolation, calibration, or accuracy claim is made.
Higher pixel count does not establish recovered ground-truth detail; quality
should be evaluated against matched LR/HR captures before scientific use.
