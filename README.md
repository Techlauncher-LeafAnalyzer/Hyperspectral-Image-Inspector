<div align="center">

<img src="src/ui/assets/hyperview.png" alt="HyperView logo" width="140">

# HyperView

**A desktop inspector for hyperspectral plant imagery.**

Load, calibrate, super-resolve, classify and analyse hyperspectral captures, and derive plant-health indices, in one reliable app.

[![UI Tests](https://github.com/Techlauncher-LeafAnalyzer/Hyperspectral-Image-Inspector/actions/workflows/ui-tests.yml/badge.svg)](https://github.com/Techlauncher-LeafAnalyzer/Hyperspectral-Image-Inspector/actions/workflows/ui-tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Qt](https://img.shields.io/badge/GUI-PyQt6-41cd52)
![Platforms](https://img.shields.io/badge/platforms-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

[Download](https://github.com/Techlauncher-LeafAnalyzer/Hyperspectral-Image-Inspector/releases) ·
[Documentation](docs/) ·
[Report a bug](https://github.com/Techlauncher-LeafAnalyzer/Hyperspectral-Image-Inspector/issues)

</div>

---

## Table of Contents

- [About](#about)
- [Features](#features)
- [Installation](#installation)
  - [Option 1: Download a release](#option-1-download-a-release-recommended)
  - [Option 2: Run from source](#option-2-run-from-source)
- [Usage](#usage)
- [Building a Binary](#building-a-binary)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## About

HyperView (the Hyperspectral Image Inspector) is built for plant-phenotyping facility operators, researchers and technicians who routinely capture hyperspectral images of plants and need to turn them into actionable plant-health indicators, such as nitrogen content and markers of disease.

It replaces a vendor tool limited to basic RGB visualisation with a purpose-built application that computes vegetation indices, corrects captures against dark and reference frames, and supports segmentation and spectral super-resolution, without sacrificing stability.

## Features

- **Visualization:** RGB composites, single bands, and the NDVI, EVI, MCARI, MTVI, OSAVI and PRI vegetation indices, plus an interactive 3D OpenGL hypercube view.
- **Pixel inspection:** per-pixel spectrum plots, numeric value overlays and index mean/min/max.
- **Calibration:** automatic dark/bright frame discovery with reflectance correction and a one-click **Before / After** comparison.
- **Classification:** unsupervised K-means and supervised (reference-example) classification with per-class layers, visibility and statistics.
- **Super-Resolution:** 2× spatial upscaling of 480-band captures using a spectral-aware model (optional; see [SR setup](docs/super-resolution.md)).
- **Cropping:** rectangle and polygon region-of-interest tools with undo/redo.
- **Export:** save the current rendered view as an image.
- **Formats:** ENVI `.bil`/`.hdr` and PSI headers (converted automatically on load).
- **Live resource monitor:** CPU and RAM usage shown in the header.

## Installation

### Option 1: Download a release (recommended)

No Python required.

1. Go to the [**Releases**](https://github.com/Techlauncher-LeafAnalyzer/Hyperspectral-Image-Inspector/releases) page.
2. Download the archive for your operating system (`HyperView-<os>-<arch>`):
   - **Windows:** `.zip`, then extract and run `HyperView.exe`
   - **macOS:** `.zip`, then extract and open `HyperView.app`
   - **Linux:** `.tar.xz`, then extract and run `./HyperView`
3. Launch the app and use **Load Image** to open a capture.

```sh
# Linux example
tar -xf HyperView-linux-x86_64.tar.xz
./HyperView/HyperView
```

### Option 2: Run from source

**Prerequisites:** Python 3.10 or newer (CI runs 3.14) and Git.

```sh
# 1. Clone the repository
git clone https://github.com/Techlauncher-LeafAnalyzer/Hyperspectral-Image-Inspector.git
cd Hyperspectral-Image-Inspector

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install requirements
pip install -r requirements.txt
pip install -r requirements-sr.txt   # optional: Super-Resolution (PyTorch CPU, SciPy)

# 4. Run
python src/main.py
```

> **Linux note:** if Qt fails to start, install the system libraries PyQt6 expects, e.g. on Debian/Ubuntu: `sudo apt install libegl1 libgl1 libxkbcommon0`.

## Usage

1. Click **Load Image** and select a `.bil` file (its `.hdr` header must sit alongside it).
2. Explore the tabs: **Visualization**, **Super-Resolution**, **Calibration** and **Classification**.
3. Right-click the image for spectrum plots, pixel values, index means and crop tools.
4. Export the displayed result with the save button.

To open an image directly at startup:

```sh
python src/main.py --image path/to/capture.bil
```

## Building a Binary

Packaging must be done on the target operating system:

```sh
python scripts/build_pyinstaller.py
```

The script creates a fresh `build/venv`, installs the runtime, super-resolution and build requirements, and produces:

| OS | Output |
| --- | --- |
| Windows | `dist/HyperView/HyperView.exe` |
| Linux | `dist/HyperView/HyperView` |
| macOS | `dist/HyperView.app` |

plus a `HyperView-<os>-<arch>` archive (`.tar.xz` on Linux, `.zip` on Windows/macOS). Pass `--reuse-venv` to speed up repeated builds.

## Testing

A headless `pytest` / `pytest-qt` suite lives in `ui_tests/` and runs in CI on every push to `main` and every pull request.

```sh
pip install -r requirements-dev.txt
pytest
```

## Project Structure

```text
├── src/
│   ├── main.py        # application entry point
│   ├── core/          # UI-independent Model (loading, indices, calibration, classification, SR)
│   ├── ui/            # PyQt6 Views and Controllers
│   └── qt/            # Qt Designer .ui files
├── docs/              # architecture, API and workflow documentation
├── scripts/           # packaging scripts
└── ui_tests/          # headless UI test suite
```

HyperView follows a Model / View / Controller split: controllers depend only on the public Model surface.

```python
from core import HSIReader, VisualizationRequest, VisualizationService
```

## Documentation

- [Design overview & functional requirements](docs/functional-requirements.md)
- [Architecture](docs/architecture.md)
- [Visualization Model API](docs/model_visualization_api.md)
- [Calibration Model API](docs/model_calibration_api.md)
- [Classification Model API](docs/model_classification_api.md)
- [Classification layer View/Controller guide](docs/classification_layer_api.md)
- [Super-Resolution setup and limitations](docs/super-resolution.md)
- [Model development workflow](docs/model-development-workflow.md)
- [Branding assets](docs/branding.md)

## Contributing

Contributions are welcome.

1. Fork the repository and create a branch (`task/<ticket>-short-description`).
2. Make your changes and add or update tests in `ui_tests/`.
3. Run `pytest` and make sure it passes.
4. Open a pull request against `main`.

Please open an [issue](https://github.com/Techlauncher-LeafAnalyzer/Hyperspectral-Image-Inspector/issues) first for larger changes.

## Acknowledgements

Built by the Techlauncher LeafAnalyzer team. Powered by [PyQt6](https://www.riverbankcomputing.com/software/pyqt/), [Spectral Python](https://www.spectralpython.net/), [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/) and [PyTorch](https://pytorch.org/).
