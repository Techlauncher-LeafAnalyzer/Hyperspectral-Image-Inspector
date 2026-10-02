#!/usr/bin/env bash
# Package the inspector with PyInstaller.
#
# Usage:
#   scripts/build_pyinstaller.sh
#
# Prerequisites (same Python environment):
#   python -m pip install -r requirements.txt -r requirements-sr.txt -r requirements-build.txt
#
# Output:
#   dist/HyperspectralImageInspector/HyperspectralImageInspector   (onedir build)
#   dist/HyperspectralImageInspector-linux-<arch>.tar.gz           (distributable)
#
# Unlike Nuitka, PyInstaller copies torch's existing shared libraries and
# bytecode instead of compiling them to C, so a build takes minutes and no
# torch subsystems need to be excluded to keep compile time down.
#
# onedir rather than --onefile: a onefile binary re-extracts the ~1GB bundle
# to /tmp on every launch, adding several seconds of startup for no benefit
# over shipping the tarball.
#
# glibc: the binary only runs on systems whose glibc is at least as new as
# the build machine's. Build on the oldest distro you need to support (e.g.
# an Ubuntu 22.04 container) rather than a rolling-release host.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

APP_NAME="HyperspectralImageInspector"
PYTHON="${PYTHON:-python}"

if ! "$PYTHON" -c "import torch, scipy" 2>/dev/null; then
  echo "torch/scipy missing: Super-Resolution would be unavailable in the build." >&2
  echo "Run: $PYTHON -m pip install -r requirements-sr.txt" >&2
  exit 1
fi

if [[ ! -f model/fin_msdformer.pth ]]; then
  echo "model/fin_msdformer.pth missing: Super-Resolution needs the checkpoint." >&2
  exit 1
fi

"$PYTHON" -m PyInstaller \
  --noconfirm \
  --clean \
  --windowed \
  --name "$APP_NAME" \
  --paths src \
  `# Collected explicitly: spectral's submodules are partly reached through` \
  `# package __init__ re-exports that static analysis can miss.` \
  --collect-submodules spectral \
  `# PyOpenGL imports its platform backend by name at runtime, and the stock` \
  `# hook only bundles glx on Linux. Wayland sessions select egl, so without` \
  `# it the hypercube tab's initializeGL fails to import OpenGL.GL.` \
  --hidden-import OpenGL.platform.egl \
  `# Resource paths: ui/*.py resolve Path(__file__).parent / "assets", and` \
  `# core/super_resolution_model.py resolves sys._MEIPASS / "model".` \
  --add-data "src/ui/assets:ui/assets" \
  --add-data "model:model" \
  `# Unused GUI toolkits that matplotlib/spectral can optionally import.` \
  --exclude-module tkinter \
  --exclude-module wx \
  --exclude-module PyQt5 \
  --exclude-module PySide2 \
  --exclude-module PySide6 \
  src/main.py

ARCHIVE="dist/${APP_NAME}-linux-$(uname -m).tar.gz"
tar -C dist -czf "$ARCHIVE" "$APP_NAME"

echo
echo "Built dist/$APP_NAME/$APP_NAME"
echo "Archive: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
