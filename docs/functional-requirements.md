# Design Overview &amp; Functional Requirements

This page synthesizes the answers gathered in the [Hyperspectral Image Planning Document](https://leaf2qr.atlassian.net/wiki/spaces/L/pages/16416770) and [Client meeting #8 (7/08)](https://leaf2qr.atlassian.net/wiki/spaces/L/pages/16580683) into a single design overview with testable functional requirements, cross-referenced against `main` at `5261ab6` (21 August 2026). Where the planning document's Phase 3/4 checklists were left blank, this page proposes concrete design decisions rather than leaving them open. See the "Residual Open Questions" section for the short list of items that genuinely still require client input.

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

Results render through the reusable `HSIViewer` widget. The four tabs each own a viewer instance, while the controller keeps the loaded cube and visualization results as shared application state. The viewer supports pan/zoom, point/box prompt annotation, a per-pixel value overlay, spectrum requests, and rectangular crop requests. Pan/zoom state is preserved when switching visualization modes and carried to the viewer on a newly selected tab. Crop undo/redo is connected at the main-window level.

The current UI is a 4-tab `QTabWidget` (Visualization / Super-Resolution / Calibration / Classification). Cropping is available from each viewer's styled, accessible context menu; Hypercube has a disabled Visualization control but no connected renderer. `docs/architecture.md` describes an earlier sidebar/panel-swap design that was superseded by this tab-based layout. Treat that part of the architecture document as historical.

## 3. Functional Requirements

Requirements are written as Given/When/Then acceptance criteria, per the format the planning document itself prescribes. Status reflects implementation and wiring found on the audited `main` revision:

- **Done** — the user-facing path is implemented and connected.
- **Partial** — only part of the acceptance criterion or only some architectural layers exist.
- **UI only / Model only** — the named layer exists but there is no end-to-end workflow.
- **Not started** — no material implementation was found.

A headless `pytest`/`pytest-qt` UI suite is tracked in `ui_tests/` and runs in GitHub Actions on pushes to `main`, pull requests, and manual dispatches. It covers synthetic-cube import, visualization switching, crop/undo/redo, display export, viewer context-menu wiring, and pan/zoom transfer. Real-capture tests run when a developer supplies local sample files but are skipped in CI because those large captures are not committed. A **Done** status records code-level implementation and any stated regression coverage, not completed stakeholder validation or scientific-accuracy validation.

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
| Given a loaded cube, when the user selects RGB, NDVI, EVI, MCARI, MTVI, OSAVI, or PRI, then the corresponding per-pixel index is computed and rendered as a false-colour image. | Done: the radio buttons are wired to cached Model results, the six index formulas are implemented, and the selected result is rendered in every viewer without resetting pan/zoom. Synthetic-cube UI tests cover every mode and optional local-sample tests exercise real captures. |
| Given a loaded cube, when the user selects Hypercube view, then a 3D/stacked representation of the full spectral cube is displayed. | Model only: renderer-neutral slice data and a downsampled SPy-compatible surface payload are implemented, but the Hypercube radio button is disabled and no 3D View is connected. |
| Given two computed results (e.g. RGB vs NDVI, or pre/post calibration), when the user toggles split-view, then both render side-by-side in the same viewer. | Not started (the viewer already exposes an `is_split` flag intended for this) |
| Given a loaded cube, when the user right-clicks a pixel and selects Spectrum Plot, then a reflectance-vs-wavelength plot for that pixel is displayed. | Done: the context-menu action reads the selected pixel spectrum and opens a wavelength/value plot dialog. |

### 3.3 Cropping

| Requirement | Status |
| --- | --- |
| Given a loaded cube and a user-drawn or specified rectangular region, when the user confirms the crop, then a new cube containing only that region's data (all bands) is produced. | Done: Crop in the viewer context menu enters a drag-to-select workflow, replaces the current state with a lazy all-band sub-image, recomputes the displayed results, and supports undo/redo. Crop is applied on mouse release rather than through a separate confirmation dialog. UI tests cover drag signaling, recomputation, undo/redo, degenerate selections, and crop-to-export dimensions. |

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
| Given a loaded cube, when the user selects a target resolution and runs Super Resolution, then an ML-accelerated algorithm upscales the image using all spectral bands (not just RGB), with progress shown via a progress bar. | UI demonstration only: the original/processed selector, Run button, and progress bar are connected to a 2.8-second animation after image load. No target-resolution input, ML dependency, inference Model, or high-resolution result exists. |

### 3.7 Analyse (Cross-Cutting)

| Requirement | Status |
| --- | --- |
| Given a chosen calibration file pair, visualization mode, and classifier settings, when the user saves a project, then those paths/settings persist to a lightweight JSON project file (not the raw image data, which can exceed 900 MB per capture). | Not started |
| Given a saved project file, when the user reopens it, then the same file paths and settings are restored (the user must still have the original image files on disk). | Not started |
| Given a computed visualization, calibration, or segmentation result, when the user selects Export, then the result is written as PNG/TIFF (images) or a labeled mask plus a CSV of per-segment statistics (segmentation). | Partial: File → Save Image exports the currently selected RGB/index display as PNG, TIFF, JPEG, or BMP using atomic replacement. UI tests verify active-mode pixels, cancellation, empty-state handling, and cropped dimensions. Calibration export, labeled-mask export, and per-segment CSV statistics do not exist. |

## 4. Non-Functional Requirements

| Category | Target | Current state on `main` |
| --- | --- | --- |
| **Reliability** | Zero-crash tolerance on core operations (load, calibrate, visualize, classify, super-resolve). This is the direct pain point named by the client about the previous tool. All I/O and model-inference paths must catch exceptions and surface an error dialog instead of crashing. Session state auto-saves (via the JSON project file) after each significant action so a crash doesn't lose configuration. | Partial: import, spectrum, and export paths translate known failures into dialogs, and a failed import preserves the previous dataset. A headless UI regression suite runs on Ubuntu in CI for implemented load/visualize/crop/export workflows. Session auto-save is not implemented; the other core operations do not yet exist end to end. |
| **Performance** | Based on the largest real capture on hand (~900 MB, 480 bands, 1971×500 px, VNIR 352–899 nm, 12-bit): long-running operations (load, super-resolution, classification) run off the UI thread with progress feedback. Target: single-tray load completes in under 10 seconds on typical facility hardware. | Not met or unverified: image opening and eager RGB/index rendering are synchronous on the GUI thread, and no benchmark is tracked. |
| **GPU Support** | Super-resolution should auto-detect CUDA and use it when available, with mandatory CPU fallback (no assumption that deployment hardware has a GPU). | Not started: no ML framework is declared in `requirements.txt` and no inference code exists. |
| **Accuracy** | Vegetation indices (NDVI, EVI, MCARI, MTVI, OSAVI, PRI) follow standard published formulas; validated via unit tests against hand-calculated reference values on the sample data already in the repository, within 1% numeric tolerance. | Partial: all six formulas and wavelength-tolerance checks are implemented, and UI tests confirm that each mode renders, but no tracked reference dataset or numerical accuracy tests establish the 1% tolerance. Index results can currently be computed from uncalibrated raw data. |
| **Usability** | Target user is a skilled technical operator, not a programmer. No onboarding flow is required, but every disabled control must show a tooltip explaining why. Keyboard shortcuts and multi-monitor polish are nice-to-haves. | Partial: unavailable Calibration and Hypercube controls explain why they are disabled, crop undo/redo uses standard shortcuts, the context menu has icons and accessible names, and pan/zoom survives mode and tab changes without drift. Classification controls can become enabled without an implemented action, and several workflows still lack completion/error feedback. |
| **Compatibility** | Primary target OS is Windows (typical facility workstation), with Linux supported for development. macOS is explicitly out of scope unless requested later. | Partial: the application uses cross-platform PyQt6 APIs and the UI suite runs headlessly on Ubuntu Linux in GitHub Actions. Windows remains the primary target but has no tracked CI job or platform-specific validation. |

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
