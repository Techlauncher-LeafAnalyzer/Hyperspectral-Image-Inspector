# Building the HyperView installer

1. On Windows, run `python scripts/build_pyinstaller.py` from the repository to create `dist\HyperView` (containing `HyperView.exe` and `_internal`).
2. Open `inno.iss` in Inno Setup 7.
3. Set `MyAppVersion` near the top to your release version (currently a placeholder: `1.0.0`).
4. Choose **Build > Compile** (Ctrl+F9).
5. Share the installer created in `dist\installer`.

The installer includes the whole PyInstaller bundle, so recipients do not need a separate Python installation. It installs for the current Windows user in `%LOCALAPPDATA%\Programs\HyperView`, adds a Start menu shortcut, offers a desktop shortcut, and registers an uninstaller in Windows Settings. Launching HyperView at the end is optional.

For later releases, replace the `dist\HyperView` folder with your new complete PyInstaller output and update `MyAppVersion`. Keep `AppId` unchanged so Windows recognises the same application. Keep using complete bundles; copying just `HyperView.exe` is insufficient.

Compilation checks the installer script and packages the files. Before sharing a release, run the installer on a clean Windows machine or VM, launch the installed application, and check uninstall. Application functionality and missing system prerequisites need that separate check.

## BIL and HDR defaults

The installer registers HyperView for `.bil` and `.hdr` files for the current Windows user. File extensions are case-insensitive. Opening a file uses `HyperView.exe --image "<file path>"`, including support for paths with spaces.

On the final installer page, select **Choose HyperView as the default for .BIL and .HDR files** to open Windows Default apps settings. Choose HyperView for each extension. On older Windows versions, search for HyperView or the extensions in Default apps. Windows requires the user to confirm this choice; the installer does not override an existing default automatically.

Uninstall removes HyperView's file-type registration. To verify a release, install it, choose both defaults, then double-click a BIL file and its HDR header, including a pair in a folder with spaces. Keep each header with its matching data file.