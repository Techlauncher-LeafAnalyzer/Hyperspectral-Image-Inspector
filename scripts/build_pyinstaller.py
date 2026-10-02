"""Package the inspector with PyInstaller on Linux or Windows.

Usage (from any directory, with the build environment's Python):
    python scripts/build_pyinstaller.py

Prerequisites (same Python environment):
    python -m pip install -r requirements.txt -r requirements-sr.txt -r requirements-build.txt

Output:
    dist/HyperspectralImageInspector/                  onedir build
    dist/HyperspectralImageInspector-<os>-<arch>.*     distributable (.tar.gz on
                                                       Linux, .zip on Windows)

PyInstaller cannot cross-compile: run this on each target OS to get a build
for it.

Unlike Nuitka, PyInstaller copies torch's existing shared libraries and
bytecode instead of compiling them to C, so a build takes minutes and no
torch subsystems need to be excluded to keep compile time down.

onedir rather than --onefile: a onefile binary re-extracts the ~1GB bundle
to a temp directory on every launch, adding several seconds of startup for
no benefit over shipping the archive.

glibc (Linux only): the binary only runs on systems whose glibc is at least
as new as the build machine's. Build on the oldest distro you need to
support (e.g. an Ubuntu 22.04 container) rather than a rolling-release host.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path

APP_NAME = "HyperspectralImageInspector"
REPO_ROOT = Path(__file__).resolve().parents[1]
IS_WINDOWS = sys.platform == "win32"


def pyinstaller_args() -> list[str]:
    args = [
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name", APP_NAME,
        "--paths", "src",
        # Collected explicitly: spectral's submodules are partly reached
        # through package __init__ re-exports that static analysis can miss.
        "--collect-submodules", "spectral",
        # Resource paths: ui/*.py resolve Path(__file__).parent / "assets", and
        # core/super_resolution_model.py resolves sys._MEIPASS / "model".
        # PyInstaller 6 accepts ":" as the separator on every OS.
        "--add-data", "src/ui/assets:ui/assets",
        "--add-data", "model:model",
    ]
    if not IS_WINDOWS:
        # PyOpenGL imports its platform backend by name at runtime, and the
        # stock hook only bundles glx on Linux. Wayland sessions select egl,
        # so without it the hypercube tab's initializeGL fails to import
        # OpenGL.GL. (On Windows the hook already bundles win32.)
        args += ["--hidden-import", "OpenGL.platform.egl"]
    # Unused GUI toolkits that matplotlib/spectral can optionally import.
    for module in ("tkinter", "wx", "PyQt5", "PySide2", "PySide6"):
        args += ["--exclude-module", module]
    args.append("src/main.py")
    return args


def check_prerequisites() -> None:
    missing = [name for name in ("PyInstaller", "torch", "scipy")
               if importlib.util.find_spec(name) is None]
    if missing:
        sys.exit(
            f"Missing {', '.join(missing)} in {sys.executable}.\n"
            "Run: python -m pip install -r requirements-sr.txt -r requirements-build.txt"
        )
    if not (REPO_ROOT / "model" / "fin_msdformer.pth").is_file():
        sys.exit("model/fin_msdformer.pth missing: Super-Resolution needs the checkpoint.")


def make_archive() -> Path:
    os_name = "windows" if IS_WINDOWS else "linux"
    base = REPO_ROOT / "dist" / f"{APP_NAME}-{os_name}-{platform.machine().lower()}"
    archive = shutil.make_archive(
        str(base), "zip" if IS_WINDOWS else "gztar",
        root_dir=REPO_ROOT / "dist", base_dir=APP_NAME,
    )
    return Path(archive)


def main() -> None:
    os.chdir(REPO_ROOT)
    check_prerequisites()

    import PyInstaller.__main__
    PyInstaller.__main__.run(pyinstaller_args())

    executable = Path("dist", APP_NAME, APP_NAME + (".exe" if IS_WINDOWS else ""))
    archive = make_archive()
    size_mb = archive.stat().st_size / 1024**2
    print(f"\nBuilt {executable}")
    print(f"Archive: {archive.relative_to(REPO_ROOT)} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
