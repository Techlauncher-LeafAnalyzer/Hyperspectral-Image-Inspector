"""Export native packaging formats from the approved HyperView PNG.

Run with the app environment's Python: python scripts/export_app_icons.py
This only resamples/encodes the original artwork and preserves its alpha.
"""

from pathlib import Path

from PIL import Image

ASSETS_DIR = Path(__file__).resolve().parents[1] / "src" / "ui" / "assets"


def main() -> None:
    with Image.open(ASSETS_DIR / "hyperview.png") as source:
        image = source.convert("RGBA")
        if image.width != image.height:
            raise ValueError("The application icon must be square.")
        image.save(
            ASSETS_DIR / "hyperview.ico",
            sizes=[(size, size) for size in (16, 24, 32, 48, 64, 128, 256)],
        )
        image.resize((1024, 1024), Image.Resampling.LANCZOS).save(
            ASSETS_DIR / "hyperview.icns"
        )


if __name__ == "__main__":
    main()
