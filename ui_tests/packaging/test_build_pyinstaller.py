from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS app archive round-trip")
def test_macos_archive_preserves_bundle_links_and_executable_mode(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts" / "build_pyinstaller.py"
    spec = importlib.util.spec_from_file_location("build_pyinstaller", script)
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    monkeypatch.setattr(build, "REPO_ROOT", tmp_path)

    app = tmp_path / "dist" / "HyperView.app"
    executable = app / "Contents" / "MacOS" / "HyperView"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    framework = app / "Contents" / "Frameworks" / "Example.framework"
    version = framework / "Versions" / "A"
    version.mkdir(parents=True)
    (version / "Example").write_bytes(b"framework binary")
    (framework / "Versions" / "Current").symlink_to("A", target_is_directory=True)
    (framework / "Example").symlink_to("Versions/Current/Example")
    resources = app / "Contents" / "Resources"
    resources.mkdir()
    (resources / "Example.framework").symlink_to(
        "../Frameworks/Example.framework", target_is_directory=True,
    )

    archive = build.make_archive()
    extracted = tmp_path / "extracted"
    subprocess.check_call(["/usr/bin/ditto", "-x", "-k", str(archive), str(extracted)])
    restored = extracted / app.name
    for original in app.rglob("*"):
        copy = restored / original.relative_to(app)
        if original.is_symlink():
            assert copy.is_symlink()
            assert copy.readlink() == original.readlink()
            assert copy.exists()
        elif original.is_file():
            assert copy.read_bytes() == original.read_bytes()
            assert copy.stat().st_mode & 0o777 == original.stat().st_mode & 0o777
    subprocess.check_call([str(restored / "Contents" / "MacOS" / "HyperView")])
