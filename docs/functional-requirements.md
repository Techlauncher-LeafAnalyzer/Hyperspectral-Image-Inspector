# Design Overview &amp; Functional Requirements

This page synthesizes the answers gathered in the [Hyperspectral Image Planning Document](https://leaf2qr.atlassian.net/wiki/spaces/L/pages/16416770) and [Client meeting #8 (7/08)](https://leaf2qr.atlassian.net/wiki/spaces/L/pages/16580683) into a single design overview with testable functional requirements, cross-referenced against the current state of the codebase. Where the planning document's Phase 3/4 checklists were left blank, this page proposes concrete design decisions rather than leaving them open. See the "Residual Open Questions" section for the short list of items that genuinely still require client input.

## 1. Purpose & Problem Statement

The Hyperspectral Image Inspector is built for APPN facility operators, researchers, and technicians who capture hyperspectral images of plants on a routine (weekly/yearly) basis and need to derive plant-health indicators, for example nitrogen content and other biological markers of disease, from those captures.

The manufacturer's existing tool only performs basic RGB visualization and operating-the-machine functions; it does not compute vegetation indices, and the previous internal attempt at a replacement was unreliable (frequent crashes, especially during super-resolution) and had an unprofessional interface. This project replaces both with a purpose-built, reliable inspection tool.

## 2. System Overview

Data flow: a `.bil` data file plus its `.hdr` header (native ENVI, or PSI-proprietary, auto-converted to ENVI on load) is opened into a single shared `HSIData` state object. From there, the user invokes one of six functional models against that shared state:

- Visualization (RGB + vegetation indices + hypercube view)
- Cropping
- Calibration (dark + reference frame correction)
- Segmentation / Classification (unsupervised and supervised)
- Super Resolution (spectral-aware upscaling)

Results render back through the shared `HSIViewer` widget, which already supports pan/zoom, undo/redo, and point/box prompt annotation.

Current UI is a 4-tab `QTabWidget` (Visualization / Super-Resolution / Calibration / Classification). Cropping and the Hypercube view have no dedicated UI yet. `docs/architecture.md` in the repo describes an earlier sidebar/panel-swap design that was superseded by this tab-based layout. Treat it as historical, not current.

## 3. Functional Requirements

Requirements are written as Given/When/Then acceptance criteria, per the format the planning document itself prescribes. Status reflects the current codebase audit.

### 3.1 Import & Session

| Requirement | Status |
| --- | --- |
| Given a valid `.bil`/`.hdr` pair, when the user selects Load Image, then the cube opens and its RGB composite renders in the viewer within a few seconds. | Done |
| Given a PSI-format header, when loaded, then it is automatically converted to an ENVI-standard header before opening. | Done |
| Given a selected image whose paired header is missing or unparseable, when load is attempted, then a clear error dialog appears and the prior session state is preserved, with no crash. | Partial (missing-header case handled; malformed-header case not yet hardened) |
| Given a folder containing multiple `.bil`/`.hdr` pairs, when the user chooses batch import, then all valid pairs queue for sequential loading. | Not started |
| Given a `.bil`/`.hdr` pair dragged onto the main window, when dropped, then it loads exactly as if chosen via File → Load. | Not started |

### 3.2 Visualization

| Requirement | Status |
| --- | --- |
| Given a loaded cube, when the user selects RGB, NDVI, EVI, MCARI, MTVI, OSAVI, or PRI, then the corresponding per-pixel index is computed and rendered as a false-colour image. | UI only: 8 radio buttons exist and are completely unwired; no index-computation functions exist beyond a band-lookup helper that's never called. |
| Given a loaded cube, when the user selects Hypercube view, then a 3D/stacked representation of the full spectral cube is displayed. | Not started |
| Given two computed results (e.g. RGB vs NDVI, or pre/post calibration), when the user toggles split-view, then both render side-by-side in the same viewer. | Not started (the viewer already exposes an `is_split` flag intended for this) |
| Given a loaded cube, when the user right-clicks a pixel and selects Spectrum Plot, then a reflectance-vs-wavelength plot for that pixel is displayed. | Not started (signal + handler stub already exist) |

### 3.3 Cropping

| Requirement | Status |
| --- | --- |
| Given a loaded cube and a user-drawn or specified rectangular region, when the user confirms the crop, then a new cube containing only that region's data (all bands) is produced. | Not started: no UI element exists yet. |

### 3.4 Calibration

| Requirement | Status |
| --- | --- |
| Given a raw cube, a selected dark-frame file, and a selected reference (white) frame file, when the user selects Calibrate, then a radiometrically calibrated cube is produced and rendered. | UI only: file pickers work; the Calibrate button is explicitly disabled in code ("not implemented yet"); no algorithm exists. |

### 3.5 Segmentation / Classification

| Requirement | Status |
| --- | --- |
| Given a loaded cube, a target number of classes, and a max-iterations value, when the user runs unsupervised classification, then the image is grouped into that many spectrally-similar segments and rendered as a labeled mask. | Not started: live Classification tab is a completely empty stub; a fully-designed but disconnected panel exists in orphaned code. |
| Given a loaded cube, a ground-truth file, and a chosen classifier (Gaussian / Mahalanobis Distance / Perceptron), when the user runs supervised classification, then each pixel is labeled according to the trained classifier. | Not started: classifier names exist as UI options only, no implementations. |
| Given the Segmentation panel is active, when the user Ctrl+clicks to place foreground/background points or Ctrl+drags a box, then those prompts feed the selected classifier as interactive input. | Partial: point/box prompt drawing and undo/redo are already implemented in the viewer; nothing yet consumes the prompts to actually segment. |

### 3.6 Super Resolution

| Requirement | Status |
| --- | --- |
| Given a loaded cube, when the user selects a target resolution and runs Super Resolution, then an ML-accelerated algorithm upscales the image using all spectral bands (not just RGB), with progress shown via a progress bar. | UI only: low/high-res selector and progress bar exist and are unwired; `torch` is a declared dependency but no model/inference code exists. |

### 3.7 Analyse (Cross-Cutting)

| Requirement | Status |
| --- | --- |
| Given a chosen calibration file pair, visualization mode, and classifier settings, when the user saves a project, then those paths/settings persist to a lightweight JSON project file (not the raw image data, which can exceed 900 MB per capture). | Not started |
| Given a saved project file, when the user reopens it, then the same file paths and settings are restored (the user must still have the original image files on disk). | Not started |
| Given a computed visualization, calibration, or segmentation result, when the user selects Export, then the result is written as PNG/TIFF (images) or a labeled mask plus a CSV of per-segment statistics (segmentation). | Not started |

## 4. Non-Functional Requirements

| Category | Target |
| --- | --- |
| **Reliability** | Zero-crash tolerance on core operations (load, calibrate, visualize, classify, super-resolve). This is the direct pain point named by the client about the previous tool. All I/O and model-inference paths must catch exceptions and surface an error dialog instead of crashing. Session state auto-saves (via the JSON project file) after each significant action so a crash doesn't lose configuration. |
| **Performance** | Based on the largest real capture on hand (~900 MB, 480 bands, 1971×500 px, VNIR 352–899 nm, 12-bit): long-running operations (load, super-resolution, classification) run off the UI thread with progress feedback. Target: single-tray load completes in under 10 seconds on typical facility hardware. |
| **GPU Support** | `torch` is already a dependency. Super-resolution should auto-detect CUDA and use it when available, with mandatory CPU fallback (no assumption that deployment hardware has a GPU). |
| **Accuracy** | Vegetation indices (NDVI, EVI, MCARI, MTVI, OSAVI, PRI) follow standard published formulas; validated via unit tests against hand-calculated reference values on the sample data already in the repository, within 1% numeric tolerance. |
| **Usability** | Target user is a skilled technical operator, not a programmer. No onboarding flow is required, but every disabled control must show a tooltip explaining why (already the convention used on the Calibrate button). Keyboard shortcuts and multi-monitor polish are nice-to-haves. |
| **Compatibility** | Primary target OS is Windows (typical facility workstation), with Linux supported for development. PyQt6 is cross-platform, so this is low-cost either way. macOS is explicitly out of scope unless requested later. |

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

Per the planning document's own instruction, each functional requirement above should still be validated with the stakeholder after implementation and covered by a written test prior to being marked complete.