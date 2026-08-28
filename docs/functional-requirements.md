# Design Overview &amp; Functional Requirements

This page synthesizes the answers gathered in the [Hyperspectral Image Planning Document](https://leaf2qr.atlassian.net/wiki/spaces/L/pages/16416770) and [Client meeting #8 (7/08)](https://leaf2qr.atlassian.net/wiki/spaces/L/pages/16580683) into a single design overview with testable functional requirements, cross-referenced against `main` at `d2b5345` (28 August 2026). Where the planning document's Phase 3/4 checklists were left blank, this page proposes concrete design decisions rather than leaving them open. See the "Residual Open Questions" section for the short list of items that genuinely still require client input.

## 1. Purpose & Problem Statement

The Hyperspectral Image Inspector is built for APPN facility operators, researchers, and technicians who capture hyperspectral images of plants on a routine (weekly/yearly) basis and need to derive plant-health indicators, for example nitrogen content and other biological markers of disease, from those captures.

The manufacturer's existing tool only performs basic RGB visualization and operating-the-machine functions; it does not compute vegetation indices, and the previous internal attempt at a replacement was unreliable (frequent crashes, especially during super-resolution) and had an unprofessional interface. This project replaces both with a purpose-built, reliable inspection tool.

## 2. System Overview

Data flow: a `.bil` data file plus its `.hdr` header (native ENVI, or PSI-proprietary, auto-converted to ENVI on load) is opened into a single shared `HSIData` state object. From there, the user invokes one of the following functional areas against that shared state:

- Visualization (RGB + vegetation indices + hypercube view)
- Cropping
- Calibration (dark + reference frame correction)
- Segmentation / Classification (unsupervised and supervised)
- Super Resolution (spectral-aware upscaling)

Each of the four tabs owns a reusable `HSIViewer` instance. Visualization, Calibration, and Classification display the shared source cube's selected RGB/index result; Super-Resolution independently displays RGB from either the source or its temporary processed `HSIData` result. The viewer supports pan/zoom, point annotation (with dormant box-prompt support), a per-pixel value overlay, spectrum requests, and rectangular crop requests. Pan/zoom is preserved across same-size mode changes and transferred between 2D tabs with coordinate scaling for the 2× SR result. Dimension-changing crops refit the image, and crop undo/redo is connected at the main-window level.

The current UI is a 4-tab `QTabWidget` (Visualization / Super-Resolution / Calibration / Classification). Visualization switches between its 2D viewer and a dedicated PyQt6/OpenGL Hypercube widget. Hypercube preparation and SR inference/preview rendering have background workers; source import and eager 2D visualization computation remain synchronous. Cropping is available from the 2D viewers' context menus, but is blocked during SR and on the processed SR view. `docs/architecture.md` describes an earlier sidebar/panel-swap design that was superseded by this tab-based layout. Treat that part of the architecture document as historical.

## 3. Functional Requirements

Requirements are written as Given/When/Then acceptance criteria, per the format the planning document itself prescribes. Status reflects implementation and wiring found on the audited `main` revision:

- **Done** — the user-facing path is implemented and connected.
- **Partial** — only part of the acceptance criterion or only some architectural layers exist.
- **UI only / Model only** — the named layer exists but there is no end-to-end workflow.
- **Not started** — no material implementation was found.

A headless `pytest`/`pytest-qt` suite is tracked in `ui_tests/` and runs in GitHub Actions on pushes to `main`, pull requests, and manual dispatches. It covers synthetic-cube import, visualization switching, crop/undo/redo/refitting, display export, context-menu wiring, pan/zoom transfer, Hypercube geometry/controller/worker behavior, and SR lifecycle/inference contracts. Real-capture tests require locally supplied sample files. SR Model tests require optional PyTorch/SciPy dependencies, and the `sr_model` tests additionally use the tracked checkpoint to run real inference and check Qt responsiveness. CI installs only `requirements-dev.txt`, so optional SR dependency tests and real-capture tests skip there. Headless checks do not establish desktop OpenGL-driver compatibility. A **Done** status records code-level implementation and any stated regression coverage, not completed stakeholder validation or scientific-accuracy validation.

### 3.1 Import & Session

| Requirement | Status |
| --- | --- |
| Given a valid `.bil`/`.hdr` pair, when the user selects Load Image, then the cube opens and its RGB composite renders in the viewer within a few seconds. | Partial: file selection and RGB rendering are connected, `--image` supports developer startup loading, and synthetic-cube loading has UI regression coverage. The controller then computes all supported index views synchronously on the GUI thread; the "within a few seconds" condition has not been benchmarked. |
| Given a PSI-format header, when loaded, then it is automatically converted to an ENVI-standard header before opening. | Done: PSI headers are adapted to a temporary ENVI header without modifying the source file. |
| Given a selected image whose paired header is missing or unparseable, when load is attempted, then a clear error dialog appears and the prior session state is preserved, with no crash. | Done: pair resolution, header parsing, wavelength validation, SPy opening, and truncated-data checks raise user-visible errors; shared state is replaced only after import and initial RGB rendering succeed. |
| Given a folder containing multiple `.bil`/`.hdr` pairs, when the user chooses batch import, then all valid pairs queue for sequential loading. | Not started |
| Given a `.bil`/`.hdr` pair dragged onto the main window, when dropped, then it loads exactly as if chosen via File → Load. | Not started |

### 3.2 Visualization

| Requirement | Status |
| --- | --- |
| Given a loaded cube, when the user selects RGB, NDVI, EVI, MCARI, MTVI, OSAVI, or PRI, then the corresponding per-pixel index is computed and rendered as a false-colour image. | Done: the radio buttons select cached Model results in the source-data viewers without resetting pan/zoom for unchanged dimensions. All six formulas exist; SR keeps its own Original/Processed RGB display. Synthetic-cube UI tests cover every mode and optional local-sample tests exercise real captures. |
| Given a loaded cube, when the user selects Hypercube view, then a 3D/stacked representation of the full spectral cube is displayed. | Done: Hypercube selects a PyQt6 OpenGL widget with an RGB top and colorized spectral boundary faces prepared in a background worker. Rotation, pan, zoom, labeled axes, Reset view, and Export current view are wired. The representation uses downsampled surfaces across the cube's spatial/spectral extent, not a full-volume renderer. Geometry, mode wiring, progress, errors, cancellation, and stale-result handling have automated coverage. |
| Given two computed results (e.g. RGB vs NDVI, or pre/post calibration), when the user toggles split-view, then both render side-by-side in the same viewer. | Not started (the viewer already exposes an `is_split` flag intended for this) |
| Given a loaded cube, when the user right-clicks a pixel and selects Spectrum Plot, then a reflectance-vs-wavelength plot for that pixel is displayed. | Done for plotting: the action opens a wavelength/value dialog for the source pixel, or the selected Original/Processed dataset when invoked in SR. It plots stored values without performing calibration; raw input is not automatically reflectance. Reads are paused during SR and serialized with Hypercube preparation. |

Additional implemented inspection aids (not new MVP acceptance criteria): Show Pixel Values displays numeric readings with swatches from each mode's displayed color; the SR viewer supplies only RGB for its selected dataset. Index Mean now opens a non-modal dialog showing the source image's global mean, observed min/max, and colormap gauge instead of a transient status-bar message. It does not compute selection/mask statistics and is unavailable on the processed SR view.

### 3.3 Cropping

| Requirement | Status |
| --- | --- |
| Given a loaded cube and a user-drawn or specified rectangular region, when the user confirms the crop, then a new cube containing only that region's data (all bands) is produced. | Done: context-menu Crop applies a dragged rectangle on mouse release, retains all bands in a lazy sub-image, recomputes source displays/Hypercube, and supports undo/redo. Dimension changes refit the 2D views. Source crop/undo/redo invalidates the old SR result; crop is blocked while SR runs and on the processed SR view. UI tests cover drag signaling, recomputation, refitting, history, degenerate selections, and crop-to-export dimensions. |

### 3.4 Calibration

| Requirement | Status |
| --- | --- |
| Given a raw cube, a selected dark-frame file, and a selected reference (white) frame file, when the user selects Calibrate, then a radiometrically calibrated cube is produced and rendered. | UI only: file pickers work; the Calibrate button is explicitly disabled in code ("not implemented yet"); no algorithm exists. |

### 3.5 Segmentation / Classification

| Requirement | Status |
| --- | --- |
| Given a loaded cube, a target number of classes, and a max-iterations value, when the user runs unsupervised classification, then the image is grouped into that many spectrally-similar segments and rendered as a labeled mask. | UI only: the Classification tab contains class-count and iteration inputs and enables its Classify button after image load, but the button has no handler and no clustering Model exists. |
| Given a loaded cube, a ground-truth file, and a chosen classifier (Gaussian / Mahalanobis Distance / Perceptron), when the user runs supervised classification, then each pixel is labeled according to the trained classifier. | UI only: the ground-truth picker and classifier choices exist and the Classify button is enabled after image load, but the button has no handler and no classifier implementations exist. |
| Given the Segmentation panel is active, when the user Ctrl+clicks to place foreground/background points or Ctrl+drags a box, then those prompts feed the selected classifier as interactive input. | Partial: Ctrl+click point drawing and underlying point/box history methods exist in the viewer, but box mode is not exposed or activated by the UI, viewer undo/redo is not connected, and no classifier consumes the prompts. |

### 3.6 Super Resolution

| Requirement | Status |
| --- | --- |
| Given a loaded cube, when the user selects a target resolution and runs Super Resolution, then an ML-accelerated algorithm upscales the image using all spectral bands (not just RGB), with progress shown via a progress bar. | Partial: Run now performs real MSDformer inference on a background QThread using the tracked checkpoint, with progress, cancellation, error recovery, and Original/Processed RGB comparison. The current model requires exactly 480 bands and produces a fixed 2× spatial result; there is still no user-selectable target resolution or support for arbitrary band counts. The complete output cube retains all 480 bands/wavelengths, rather than upscaling only RGB. |

Current SR implementation boundaries (see [Super-Resolution](super-resolution.md)):

- PyTorch and SciPy are optional dependencies in `requirements-sr.txt`, imported only when needed. Missing dependencies/checkpoints and incompatible input produce actionable errors.
- Tiled inference writes float32 predictions to a temporary ENVI BIP cube, leaving the original file unchanged. The largest documented capture would require about 7 GiB for this temporary output; it is not a permanent HSI export.
- Original/Processed toggles, pixel RGB, spectrum lookup, and Save Image use the displayed SR dataset. Loading/cropping/undoing the source invalidates old SR output; cancellation or failure retains the previous successful result.
- Source load, crop/undo/redo, and spectrum reads are paused during SR to serialize file access. Tabs, cached modes, and pan/zoom remain interactive. Closing requests cancellation and waits asynchronously for SR completion.
- Model tests cover channel-preserving reconstruction, cropped/odd-size inputs, cancellation cleanup, invalid input/output/checkpoints, and actual-checkpoint agreement with a direct network call. The documented local CPU ROI run is not full-capture or ground-truth quality validation; fixed band count alone does not establish sensor/training compatibility, and tiled predictions may have seams.

### 3.7 Analyse (Cross-Cutting)

| Requirement | Status |
| --- | --- |
| Given a chosen calibration file pair, visualization mode, and classifier settings, when the user saves a project, then those paths/settings persist to a lightweight JSON project file (not the raw image data, which can exceed 900 MB per capture). | Not started |
| Given a saved project file, when the user reopens it, then the same file paths and settings are restored (the user must still have the original image files on disk). | Not started |
| Given a computed visualization, calibration, or segmentation result, when the user selects Export, then the result is written as PNG/TIFF (images) or a labeled mask plus a CSV of per-segment statistics (segmentation). | Partial: File → Save Image exports the selected 2D RGB/index display, or the selected Original/Processed RGB preview in SR, as PNG, TIFF, JPEG, or BMP. Hypercube has a separate Export current view control for a PNG/JPEG framebuffer capture; File → Save Image does not capture the 3D view. UI tests verify active-mode/SR preview pixels, cancellation, empty-state handling, and cropped dimensions. Permanent full SR-cube export, calibration export, labeled masks, and per-segment CSV statistics are not implemented. |

## 4. Non-Functional Requirements

| Category | Target | Current state on `main` |
| --- | --- | --- |
| **Reliability** | Zero-crash tolerance on core operations (load, calibrate, visualize, classify, super-resolve). This is the direct pain point named by the client about the previous tool. All I/O and model-inference paths must catch exceptions and surface an error dialog instead of crashing. Session state auto-saves (via the JSON project file) after each significant action so a crash doesn't lose configuration. | Partial: import/spectrum/export errors are surfaced; Hypercube and SR workers report failures and handle cancellation. SR validates checkpoints and finite tensors, cleans incomplete output, preserves prior successful results, and avoids destroying an active SR thread on close. Regression coverage includes these lifecycles, but optional inference dependencies are absent in CI. Session auto-save, calibration, and classification remain unimplemented. |
| **Performance** | Based on the largest real capture on hand (~900 MB, 480 bands, 1971×500 px, VNIR 352–899 nm, 12-bit): long-running operations (load, super-resolution, classification) run off the UI thread with progress feedback. Target: single-tray load completes in under 10 seconds on typical facility hardware. | Partial: Hypercube preparation and SR inference/RGB preview run in background workers with progress, and SR uses bounded working-memory tiles. Import and eager RGB/index rendering still run on the GUI thread; waiting for a Hypercube reader to stop can also block it. A local CPU SR crop measurement is documented, but the full-capture SR run and under-10-second import target remain unverified. |
| **GPU Support** | Super-resolution should auto-detect CUDA and use it when available, with mandatory CPU fallback (no assumption that deployment hardware has a GPU). | Partial validation: the optional PyTorch service selects CUDA when available and CPU otherwise; actual-checkpoint CPU tests exist. CUDA execution has not been validated in the tracked local verification or CI. MPS is not enabled. |
| **Accuracy** | Vegetation indices (NDVI, EVI, MCARI, MTVI, OSAVI, PRI) follow standard published formulas; validated via unit tests against hand-calculated reference values on the sample data already in the repository, within 1% numeric tolerance. | Partial: all six formulas and wavelength-tolerance checks are implemented, and UI tests confirm that each mode renders, but no tracked reference dataset or numerical accuracy tests establish the 1% tolerance. Index results can currently be computed from uncalibrated raw data. |
| **Usability** | Target user is a skilled technical operator, not a programmer. No onboarding flow is required, but every disabled control must show a tooltip explaining why. Keyboard shortcuts and multi-monitor polish are nice-to-haves. | Partial: Calibration has an unavailable tooltip; Hypercube now has interactive camera/reset/export controls; SR offers Run/Cancel, progress, and comparison feedback. Crop refits after resizing, 2D comparison framing is preserved, pixel overlays include color swatches, and index means remain visible in a dialog. Classification buttons can still become enabled without an implemented action, and disabled-control explanations are not yet consistent everywhere. |
| **Compatibility** | Primary target OS is Windows (typical facility workstation), with Linux supported for development. macOS is explicitly out of scope unless requested later. | Partial: headless Ubuntu CI exercises core/UI behavior, while Hypercube requires a working OpenGL context and driver. Local CPU SR/desktop checks are documented but do not establish Windows support. Windows remains the primary target with no tracked CI job or platform-specific acceptance validation. |

## 5. Phase 2 Backlog (Explicitly Deferred, Not Dropped)

The six models above (Visualization, Cropping, Calibration, Segmentation/Classification, Super-Resolution, plus the Hypercube view) are the mandatory MVP scope, per the client's explicit description in meeting #8. The following were only mentioned in the earlier brainstorming stage and are deferred rather than committed to:

- PCA analysis
- Spectral unmixing
- Target detection
- General material identification
- Automated report generation

## 6. Residual Open Questions

The following items still require client input:

- Exact sensor/camera make & model. Headers already self-describe band count and wavelengths, so this is documentation-only and not blocking.
- Availability of a labeled ground-truth dataset to validate supervised-classification accuracy against.
- Whether deployment machines have GPU hardware. This doesn't block development since CPU fallback is mandatory regardless, but it affects whether GPU acceleration is ever exercised in practice.

## 7. Traceability

- [Hyperspectral Image Planning Document](https://leaf2qr.atlassian.net/wiki/spaces/L/pages/16416770): source of Phase 1/2 answers.
- [Client meeting #8 (7/08)](https://leaf2qr.atlassian.net/wiki/spaces/L/pages/16580683): source of the six-model functional description.
- [Hyper Spectral Image MVP](https://leaf2qr.atlassian.net/wiki/spaces/L/pages/15695881): earlier brainstorm and source of the Phase 2 backlog items.

Per the planning document's own instruction, each functional requirement above should still be validated with the stakeholder after implementation and covered by a written test before it is accepted as complete for delivery.
