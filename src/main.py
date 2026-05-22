import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindowController


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindowController()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
