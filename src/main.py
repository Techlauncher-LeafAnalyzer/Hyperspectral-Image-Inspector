import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindowController
from ui.theme import apply_theme


def main() -> None:
    app = QApplication(sys.argv)
    apply_theme(app)
    window = MainWindowController()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
