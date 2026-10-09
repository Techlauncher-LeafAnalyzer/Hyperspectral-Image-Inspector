"""Package the inspector with PyInstaller on Linux or Windows.

Usage (from any directory, with any Python 3.10+):
    python scripts/build_pyinstaller.py

Every build starts from a fresh virtual environment (build/venv, deleted and
recreated each run) with only requirements.txt, requirements-sr.txt and
requirements-build.txt installed, so packages that happen to be present in the
developer's own environment never leak into the bundle. Pass --reuse-venv to
skip recreating it when iterating on the script itself.

Output:
    dist/HyperspectralImageInspector/                  onedir build
    dist/HyperspectralImageInspector-<os>-<arch>.*     distributable (.tar.xz on
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

import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

APP_NAME = "HyperspectralImageInspector"
REPO_ROOT = Path(__file__).resolve().parents[1]
IS_WINDOWS = sys.platform == "win32"
VENV_DIR = REPO_ROOT / "build" / "venv"
REQUIREMENTS = ("requirements.txt", "requirements-sr.txt", "requirements-build.txt")


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


def stage(msg: str) -> None:
    print(f"\n==> {msg}", flush=True)


def venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def prepare_venv(reuse: bool) -> Path:
    """Create a clean build environment and install pinned dependencies."""
    if not (REPO_ROOT / "model" / "fin_msdformer.pth").is_file():
        sys.exit("model/fin_msdformer.pth missing: Super-Resolution needs the checkpoint.")
    if reuse and venv_python().is_file():
        return venv_python()
    stage("Creating fresh build venv")
    shutil.rmtree(VENV_DIR, ignore_errors=True)
    subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
    py = str(venv_python())
    stage("Installing dependencies (torch download is the slow part)")
    subprocess.check_call([py, "-m", "pip", "install", "--upgrade", "pip"])
    # pip's cache is kept so repeat builds don't re-download torch.
    cmd = [py, "-m", "pip", "install"]
    for name in REQUIREMENTS:
        cmd += ["-r", name]
    subprocess.check_call(cmd)
    return venv_python()


def make_archive() -> Path:
    """Compress dist/<APP_NAME> with the strongest standard-library codec."""
    os_name = "windows" if IS_WINDOWS else "linux"
    base = REPO_ROOT / "dist" / f"{APP_NAME}-{os_name}-{platform.machine().lower()}"
    src = REPO_ROOT / "dist" / APP_NAME
    if IS_WINDOWS:
        archive = base.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as zf:
            for path in sorted(src.rglob("*")):
                zf.write(path, path.relative_to(src.parent))
    else:
        archive = base.with_suffix(".tar.xz")
        if shutil.which("xz"):
            # Python's lzma module is single-threaded; the xz CLI with -T0
            # uses every core, which is many times faster on a ~1GB bundle.
            tar = subprocess.Popen(
                ["tar", "-C", str(src.parent), "-cf", "-", APP_NAME],
                stdout=subprocess.PIPE)
            with open(archive, "wb") as out:
                subprocess.check_call(["xz", "-T0", "-5", "-c"], stdin=tar.stdout, stdout=out)
            if tar.wait():
                sys.exit("tar failed while creating the archive")
        else:
            with tarfile.open(archive, "w:xz", preset=5) as tf:
                tf.add(src, arcname=APP_NAME)
    return archive


def main() -> None:
    os.chdir(REPO_ROOT)
    py = prepare_venv(reuse="--reuse-venv" in sys.argv[1:])

    stage("Running PyInstaller")
    subprocess.check_call([str(py), "-m", "PyInstaller", *pyinstaller_args()])

    executable = Path("dist", APP_NAME, APP_NAME + (".exe" if IS_WINDOWS else ""))
    stage("Compressing archive")
    archive = make_archive()
    size_mb = archive.stat().st_size / 1024**2
    print(f"\nBuilt {executable}")
    print(f"Archive: {archive.relative_to(REPO_ROOT)} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
