import argparse
import sys
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindowController
from ui.theme import apply_theme


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperspectral Image Inspector")
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

    app = QApplication(sys.argv)
    apply_theme(app)
    window = MainWindowController()
    window.show()

    if args.image is not None:
        QTimer.singleShot(0, lambda: window.load_image_from_path(args.image))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
