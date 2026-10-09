import argparse
import sys
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication
from ui.branding import APP_ID, APP_NAME, ICON_PATH
from ui.main_window import MainWindowController
from ui.theme import apply_theme


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} — Hyperspectral Image Inspector")
    parser.add_argument(
        "-i",
        "--image",
        type=Path,
        default=None,
        help=(
            "Path to a hyperspectral image (.bil/.bip/.bsq) to load "
            "automatically on startup. Intended for use as a devtool, "
            "e.g. from a PyCharm Run Configuration."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if sys.platform == "win32":
        # Give source runs and packaged builds the same taskbar identity.
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setDesktopFileName(APP_NAME)
    app.setWindowIcon(QIcon(str(ICON_PATH)))
    apply_theme(app)
    window = MainWindowController()
    window.show()

    if args.image is not None:
        QTimer.singleShot(0, lambda: window.load_image_from_path(args.image))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
