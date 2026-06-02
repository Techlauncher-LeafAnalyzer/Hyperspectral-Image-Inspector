from __future__ import annotations

from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import QApplication


APP_QSS = """
QMainWindow {
    background: #edf2f1;
}

QWidget#centralwidget {
    background: #edf2f1;
}

QWidget#navigationPanel {
    background: #132225;
    border-right: 1px solid #25373a;
}

QFrame#frame {
    background: #f9fbfa;
    border: 1px solid #d8e2df;
    border-radius: 8px;
}

QWidget#panelContainer {
    background: transparent;
}

QLabel {
    color: #263735;
    font-size: 13px;
}

QLabel#label {
    color: #667673;
    font-weight: 600;
}

QLabel#label_2 {
    color: #1f302e;
    background: #edf5f3;
    border: 1px solid #d6e4e0;
    border-radius: 6px;
    padding: 6px 10px;
}

QPushButton {
    background: #ffffff;
    color: #263735;
    border: 1px solid #cddad6;
    border-radius: 6px;
    padding: 7px 12px;
    font-weight: 600;
}

QPushButton:hover {
    background: #f0f7f5;
    border-color: #8fbcb4;
}

QPushButton:pressed {
    background: #dcefeb;
    border-color: #1f9d8a;
}

QPushButton:disabled {
    color: #9ba9a6;
    background: #f0f3f2;
    border-color: #dde5e2;
}

QPushButton[navButton="true"] {
    background: transparent;
    color: #b7c7c4;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 10px 14px;
    text-align: left;
    font-size: 13px;
    font-weight: 650;
}

QPushButton[navButton="true"]:hover {
    background: #1d3235;
    color: #ecf5f2;
    border-color: #2c474a;
}

QPushButton[navButton="true"]:checked {
    background: #e2fbf4;
    color: #0f3733;
    border-color: #8ee0d0;
}

QPushButton[primaryButton="true"] {
    background: #168b7b;
    color: #ffffff;
    border-color: #168b7b;
}

QPushButton[primaryButton="true"]:hover {
    background: #117b6d;
    border-color: #117b6d;
}

QLineEdit, QComboBox {
    background: #ffffff;
    color: #223331;
    border: 1px solid #cddad6;
    border-radius: 6px;
    padding: 7px 9px;
    selection-background-color: #a8e8dc;
}

QLineEdit:focus, QComboBox:focus {
    border-color: #168b7b;
    background: #fbfefd;
}

QComboBox::drop-down {
    width: 28px;
    border: 0;
}

QRadioButton {
    color: #263735;
    spacing: 8px;
    padding: 4px 8px;
}

QRadioButton::indicator {
    width: 15px;
    height: 15px;
}

QRadioButton::indicator:unchecked {
    border: 1px solid #9dafaa;
    border-radius: 8px;
    background: #ffffff;
}

QRadioButton::indicator:checked {
    border: 4px solid #168b7b;
    border-radius: 8px;
    background: #ffffff;
}

QTabWidget::pane {
    border: 1px solid #d8e2df;
    border-radius: 6px;
    top: -1px;
    background: #ffffff;
}

QTabBar::tab {
    background: #e9f0ee;
    color: #536663;
    border: 1px solid #d8e2df;
    border-bottom: 0;
    padding: 8px 14px;
    margin-right: 4px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}

QTabBar::tab:selected {
    background: #ffffff;
    color: #173532;
    border-color: #cbdad6;
}

QProgressBar {
    background: #e4ece9;
    border: 1px solid #d1dedb;
    border-radius: 6px;
    color: #263735;
    text-align: center;
    min-height: 18px;
}

QProgressBar::chunk {
    background: #d99b2b;
    border-radius: 5px;
}

QGraphicsView#viewer {
    background: #e8eeee;
    border: 1px solid #d1dcda;
    border-radius: 8px;
}

QSplitter::handle {
    background: #dbe5e2;
}

QSplitter::handle:horizontal {
    width: 2px;
}

QSplitter::handle:vertical {
    height: 2px;
}

QMenuBar {
    background: #f8fbfa;
    color: #263735;
    border-bottom: 1px solid #d8e2df;
}

QMenuBar::item {
    padding: 4px 10px;
    background: transparent;
}

QMenuBar::item:selected {
    background: #e3efec;
    border-radius: 4px;
}

QMenu {
    background: #ffffff;
    color: #263735;
    border: 1px solid #cddad6;
    padding: 4px;
}

QMenu::item {
    padding: 6px 28px 6px 12px;
    border-radius: 4px;
}

QMenu::item:selected {
    background: #e5f5f1;
}

QStatusBar {
    background: #f8fbfa;
    color: #596b68;
    border-top: 1px solid #d8e2df;
}

QScrollBar:vertical, QScrollBar:horizontal {
    background: #edf2f1;
    border: 0;
}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont))
    app.setStyleSheet(APP_QSS)
