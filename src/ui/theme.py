from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFontDatabase
from PyQt6.QtWidgets import QApplication


APP_QSS = """
QMainWindow {
    background: #edf2f1;
}

QWidget#centralwidget {
    background: #edf2f1;
}

QWidget#navigationPanel {
    background: #102124;
    border-right: 1px solid #203437;
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
    background: #102124;
    color: #d4e1de;
    border: 0;
    border-bottom: 1px solid #1d3336;
    border-radius: 0;
    padding: 0;
    text-align: center;
    font-size: 13px;
    font-weight: 650;
}

QPushButton[navButton="true"]:hover {
    background: #173034;
    color: #f2fbf8;
}

QPushButton[navButton="true"]:checked {
    background: #168b7b;
    color: #ffffff;
    border-bottom: 1px solid #168b7b;
}

QPushButton[navButton="true"]:checked:hover {
    background: #117d70;
    color: #ffffff;
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

QPushButton#unsupervisedClassifyButton:disabled,
QPushButton#pushButton_2:disabled,
QPushButton#runSuperResButton:disabled {
    color: #9ba9a6;
    background: #f0f3f2;
    border-color: #dde5e2;
}

QGroupBox#superResControls {
    background: #f7faf9;
    border: 1px solid #d8e2df;
    border-radius: 6px;
}

QLabel[stepLabel="true"] {
    color: #526662;
    font-size: 12px;
    font-weight: 600;
}

QLabel[flowArrow="true"] {
    color: #8a9c98;
    font-size: 16px;
    font-weight: 600;
    padding: 0 2px;
}

QLabel[statusMessage="true"] {
    color: #5c6f6b;
    background: #edf3f1;
    border: 1px solid #d9e4e1;
    border-radius: 5px;
    padding: 3px 9px;
}

QStackedWidget#superResStatusStack {
    background: transparent;
    border: 0;
}

QPushButton#runSuperResButton {
    padding: 5px 12px;
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

QComboBox::down-arrow {
    image: url(__COMBOBOX_ARROW__);
    width: 10px;
    height: 6px;
}

QRadioButton {
    color: #263735;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    spacing: 8px;
    padding: 4px 8px;
}

QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border: 0;
}

QRadioButton::indicator:unchecked {
    image: url(__RADIO_UNCHECKED__);
}

QRadioButton::indicator:checked {
    image: url(__RADIO_CHECKED__);
}

QRadioButton:hover {
    color: #173f39;
    background: #edf6f4;
    border-color: #d2e5e1;
}

QRadioButton:checked {
    color: #123e37;
    background: #e3f2ef;
    border-color: #a8d2ca;
}

QRadioButton:checked:hover {
    background: #dbeeea;
    border-color: #82bdb2;
}

QRadioButton:focus {
    border-color: #168b7b;
    background: #f2f9f7;
}

QRadioButton:checked:focus {
    background: #dff1ed;
    border-color: #168b7b;
}

QRadioButton:pressed {
    background: #d5ebe6;
}

QTabWidget#tabWidget::pane {
    border: 1px solid #d8e2df;
    border-radius: 8px;
    top: -1px;
    background: #f9fbfa;
}

QTabBar#mainTabBar {
    background: transparent;
}

QTabBar#mainTabBar::tab,
QToolButton#fileMenuButton {
    font-size: 13px;
    font-weight: 600;
}

QTabBar#mainTabBar::tab {
    min-width: 108px;
    min-height: 20px;
    background: transparent;
    color: #536663;
    border: 1px solid transparent;
    border-bottom: 0;
    padding: 9px 16px 10px 16px;
    margin-right: 4px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}

QTabBar#mainTabBar::tab:hover {
    background: #e3eeeb;
    color: #264b46;
}

QTabBar#mainTabBar::tab:selected {
    background: #f9fbfa;
    color: #173532;
    border-color: #cbdad6;
    border-bottom-color: #f9fbfa;
}

QTabBar#mainTabBar:focus::tab:selected {
    border-color: #78afa5;
    border-bottom-color: #f9fbfa;
}

QFrame#tabIndicator {
    background: #168b7b;
    border: 0;
    border-radius: 1px;
}

QToolButton#fileMenuButton {
    min-width: 82px;
    min-height: 20px;
    background: transparent;
    color: #536663;
    border: 1px solid transparent;
    padding: 9px 16px 10px 14px;
    margin-right: 4px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}

QToolButton#fileMenuButton::menu-indicator {
    image: url(__COMBOBOX_ARROW__);
    width: 10px;
    height: 6px;
    subcontrol-origin: padding;
    subcontrol-position: right center;
    right: 8px;
}

QToolButton#fileMenuButton:hover {
    background: #e3eeeb;
    color: #264b46;
}

QToolButton#fileMenuButton:pressed,
QToolButton#fileMenuButton[menuOpen="true"] {
    background: #dcefeb;
    color: #173532;
}

QToolButton#fileMenuButton:disabled {
    background: transparent;
    color: #9ba9a6;
}

QTabWidget#classificationModeTabs::pane {
    border: 1px solid #cbdad6;
    border-radius: 6px;
    top: -1px;
    background: #ffffff;
}

QTabBar#classificationTabBar::tab {
    min-width: 112px;
    background: transparent;
    color: #536663;
    border: 1px solid transparent;
    border-bottom: 0;
    padding: 7px 16px;
    margin-right: 3px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: 600;
}

QTabBar#classificationTabBar::tab:hover {
    background: #dcebe7;
    color: #264b46;
}

QTabBar#classificationTabBar::tab:selected {
    background: #ffffff;
    color: #173532;
    border-color: #cbdad6;
    border-bottom-color: #ffffff;
}

QTabBar#classificationTabBar:focus::tab:selected {
    border-color: #78afa5;
    border-bottom-color: #ffffff;
}

QTabBar#classificationTabBar::tab:disabled {
    background: #f0f3f2;
    color: #9ba9a6;
    border-color: #dde5e2;
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

QGraphicsView#viewer,
QGraphicsView#superResViewer,
QGraphicsView#calibrationViewer,
QGraphicsView#classificationViewer {
    background: #e8eeee;
    border: 1px solid #d1dcda;
    border-radius: 8px;
}

QLabel#pixelValueOverlay {
    background-color: rgba(20, 20, 20, 200);
    color: #f5f5f5;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 4px 6px;
    font-size: 11px;
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

QMenu#fileMenu {
    background: #fbfdfc;
    color: #263735;
    border: 1px solid #bdcec9;
    border-radius: 8px;
    padding: 6px;
}

QMenu#fileMenu::item {
    min-height: 20px;
    padding: 9px 38px 9px 36px;
    margin: 1px 0;
    border: 1px solid transparent;
    border-radius: 5px;
    font-weight: 550;
}

QMenu#fileMenu::item:selected {
    background: #e3f1ee;
    color: #173f39;
    border-color: #cfe3de;
}

QMenu#fileMenu::item:disabled {
    background: transparent;
    color: #9ba9a6;
    border-color: transparent;
}

QMenu#fileMenu::icon {
    left: 10px;
    width: 18px;
    height: 18px;
}

QMenu#viewerContextMenu,
QMenu#viewerIndexMenu {
    background: #fbfdfc;
    color: #263735;
    border: 1px solid #bdcec9;
    border-radius: 9px;
    padding: 7px 6px;
}

QMenu#viewerContextMenu::item,
QMenu#viewerIndexMenu::item {
    min-height: 22px;
    padding: 8px 38px 8px 38px;
    margin: 1px 0;
    border: 1px solid transparent;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
}

QMenu#viewerContextMenu::item:selected,
QMenu#viewerIndexMenu::item:selected {
    background: #e3f1ee;
    color: #173f39;
    border-color: #c9e0db;
}

QMenu#viewerContextMenu::item:pressed,
QMenu#viewerIndexMenu::item:pressed {
    background: #d5eae5;
    color: #123e37;
    border-color: #9fcac1;
}

QMenu#viewerContextMenu::item:checked,
QMenu#viewerIndexMenu::item:checked {
    background: #edf7f4;
    color: #126f62;
    font-weight: 600;
}

QMenu#viewerContextMenu::item:checked:selected,
QMenu#viewerIndexMenu::item:checked:selected {
    background: #d9ede8;
    color: #105f55;
    border-color: #afd2ca;
}

QMenu#viewerContextMenu::item:disabled,
QMenu#viewerIndexMenu::item:disabled {
    background: transparent;
    color: #9ba9a6;
    border-color: transparent;
}

QMenu#viewerContextMenu::separator,
QMenu#viewerIndexMenu::separator {
    height: 1px;
    background: #dbe5e2;
    margin: 6px 10px;
}

QMenu#viewerContextMenu::icon,
QMenu#viewerIndexMenu::icon {
    left: 11px;
    width: 18px;
    height: 18px;
}

QMenu#viewerContextMenu::indicator,
QMenu#viewerIndexMenu::indicator {
    left: 12px;
    width: 16px;
    height: 16px;
}

QMenu#viewerContextMenu::indicator:checked,
QMenu#viewerIndexMenu::indicator:checked {
    image: url(__MENU_CHECK__);
}

QMenu#viewerContextMenu::right-arrow,
QMenu#viewerIndexMenu::right-arrow {
    image: url(__MENU_CHEVRON__);
    width: 6px;
    height: 10px;
    right: 12px;
}

QStatusBar {
    background: #f8fbfa;
    color: #596b68;
    border-top: 1px solid #d8e2df;
}

QDialog#indexMeanDialog {
    background: #ffffff;
}

QLabel#indexMeanHeading {
    color: #123e37;
    font-size: 16px;
    font-weight: 700;
}

QLabel#indexMeanSubtitle {
    color: #667673;
    font-size: 12px;
}

QFrame#indexMeanCard {
    background: #f3faf8;
    border: 1px solid #d8ece7;
    border-radius: 14px;
}

QLabel#indexMeanValue {
    color: #123e37;
    font-size: 42px;
    font-weight: 700;
}

QLabel#indexMeanCaption {
    color: #8a9c98;
    font-size: 11px;
}

QLabel#indexMeanRangeLabel {
    color: #526662;
    font-size: 12px;
    font-weight: 600;
}

QScrollBar:vertical, QScrollBar:horizontal {
    background: #edf2f1;
    border: 0;
}

QWidget#Classification {
    background: #f9fbfa;
}

QLabel#classificationFilePath {
    color: #6a7c78;
    font-size: 12px;
    padding: 1px 2px;
}

QGroupBox#classificationControls {
    background: transparent;
    border: 0;
}

QWidget#classificationLayerPanel {
    background: #f5f9f8;
    border: 1px solid #cfddda;
    border-radius: 10px;
}

QWidget#classificationLayerHeader,
QWidget#classificationLayerHeaderDetails,
QWidget#classificationLayerBody,
QWidget#classificationLayerRows {
    background: transparent;
    border: 0;
}

QLabel#classificationLayerPanelTitle {
    color: #173532;
    font-size: 14px;
    font-weight: 700;
}

QLabel#classificationLayerCount {
    min-width: 20px;
    max-width: 20px;
    min-height: 20px;
    max-height: 20px;
    color: #176d61;
    background: #dcefeb;
    border: 1px solid #c3ded8;
    border-radius: 6px;
    font-size: 10px;
    font-weight: 700;
}

QToolButton#classificationLayerCollapseButton {
    min-width: 22px;
    max-width: 22px;
    min-height: 22px;
    max-height: 22px;
    color: #3e5c57;
    background: #ffffff;
    border: 1px solid #d1dedb;
    border-radius: 7px;
    font-size: 21px;
    font-weight: 500;
    padding: 0 0 2px 0;
}

QToolButton#classificationLayerCollapseButton:hover {
    color: #126f62;
    background: #e5f3f0;
    border-color: #9fcac1;
}

QToolButton#classificationLayerCollapseButton:pressed {
    background: #d5ebe6;
    border-color: #78afa5;
}

QToolButton#classificationLayerCollapseButton:focus {
    border-color: #168b7b;
}

QLabel#classificationLayerSummary {
    color: #71827f;
    font-size: 11px;
    padding: 0 1px;
}

QFrame#classificationLayerControls {
    background: #eaf2f0;
    border: 1px solid #d4e2df;
    border-radius: 8px;
}

QLabel#classificationLayerSectionLabel {
    color: #516662;
    font-size: 11px;
    font-weight: 700;
    padding: 1px 2px 0 2px;
}

QLabel#classificationLayerPanelEmpty {
    color: #849490;
    background: #edf4f2;
    border: 1px dashed #cbdad6;
    border-radius: 8px;
    font-size: 12px;
    padding: 18px 13px;
}

QScrollArea#classificationLayerScrollArea {
    background: transparent;
    border: 0;
}

QScrollArea#classificationLayerScrollArea > QWidget > QWidget {
    background: transparent;
}

QScrollArea#classificationLayerScrollArea QScrollBar:vertical {
    width: 7px;
    background: transparent;
    margin: 1px 0;
}

QScrollArea#classificationLayerScrollArea QScrollBar::handle:vertical {
    min-height: 26px;
    background: #bfceca;
    border-radius: 3px;
}

QScrollArea#classificationLayerScrollArea QScrollBar::handle:vertical:hover {
    background: #93aaa5;
}

QScrollArea#classificationLayerScrollArea QScrollBar::add-line:vertical,
QScrollArea#classificationLayerScrollArea QScrollBar::sub-line:vertical,
QScrollArea#classificationLayerScrollArea QScrollBar::add-page:vertical,
QScrollArea#classificationLayerScrollArea QScrollBar::sub-page:vertical {
    height: 0;
    background: transparent;
}

QWidget#classificationLayerRow {
    background: #ffffff;
    border: 1px solid #d5e2df;
    border-radius: 9px;
}

QWidget#classificationLayerRow:hover {
    background: #fbfefd;
    border-color: #a9ccc5;
}

QWidget#classificationLayerRow[layerVisible="false"] {
    background: #f1f5f4;
    border-color: #dde6e4;
}

QWidget#classificationLayerRow[layerVisible="false"] QLabel#layerNameLabel,
QWidget#classificationLayerRow[layerVisible="false"] QLabel#layerOpacityCaption,
QWidget#classificationLayerRow[layerVisible="false"] QLabel#layerOpacityValueLabel {
    color: #869591;
}

QLabel#layerNameLabel {
    color: #263735;
    font-weight: 650;
    font-size: 12px;
}

QLabel#layerPixelCountLabel {
    color: #82918e;
    font-size: 10px;
}

QLabel#layerOpacityCaption,
QLabel#layerOpacityValueLabel {
    color: #60736f;
    font-size: 10px;
}

QLabel#layerOpacityValueLabel,
QLabel#globalOpacityValueLabel {
    font-weight: 600;
}

QCheckBox#layerVisibilityToggle {
    spacing: 0;
    padding: 1px;
}

QCheckBox#layerVisibilityToggle::indicator {
    width: 18px;
    height: 18px;
    border: 0;
}

QCheckBox#layerVisibilityToggle::indicator:unchecked {
    image: url(__CHECKBOX_UNCHECKED__);
}

QCheckBox#layerVisibilityToggle::indicator:checked {
    image: url(__CHECKBOX_CHECKED__);
}

QCheckBox#layerVisibilityToggle:focus {
    background: #dff1ed;
    border: 1px solid #79b5aa;
    border-radius: 5px;
}

QPushButton#layerToggleAllButton,
QPushButton#layerOutlineToggleButton {
    min-height: 18px;
    color: #254c46;
    background: #ffffff;
    border: 1px solid #c7d8d4;
    border-radius: 6px;
    font-size: 10px;
    font-weight: 650;
    padding: 5px 7px;
}

QPushButton#layerToggleAllButton:hover,
QPushButton#layerOutlineToggleButton:hover {
    color: #126f62;
    background: #f8fcfb;
    border-color: #8fbcb4;
}

QPushButton#layerToggleAllButton:pressed,
QPushButton#layerOutlineToggleButton:pressed {
    background: #dcefeb;
    border-color: #6ba99e;
}

QPushButton#layerOutlineToggleButton:checked {
    background: #168b7b;
    border-color: #168b7b;
    color: #ffffff;
}

QPushButton#layerOutlineToggleButton:checked:hover {
    background: #117b6d;
    border-color: #117b6d;
}

QPushButton#layerToggleAllButton:disabled,
QPushButton#layerOutlineToggleButton:disabled {
    color: #9daba8;
    background: #edf2f1;
    border-color: #d9e3e0;
}

QLabel#globalOpacityLabel {
    color: #526863;
    font-size: 10px;
    font-weight: 650;
}

QLabel#globalOpacityValueLabel {
    color: #526863;
    font-size: 10px;
}

QSlider#layerOpacitySlider,
QSlider#globalOpacitySlider {
    min-height: 14px;
}

QSlider#layerOpacitySlider::groove:horizontal,
QSlider#globalOpacitySlider::groove:horizontal {
    height: 4px;
    background: #d8e3e0;
    border-radius: 2px;
}

QSlider#layerOpacitySlider::sub-page:horizontal,
QSlider#globalOpacitySlider::sub-page:horizontal {
    background: #39a393;
    border-radius: 2px;
}

QSlider#layerOpacitySlider::add-page:horizontal,
QSlider#globalOpacitySlider::add-page:horizontal {
    background: #d8e3e0;
    border-radius: 2px;
}

QSlider#layerOpacitySlider::handle:horizontal,
QSlider#globalOpacitySlider::handle:horizontal {
    width: 13px;
    height: 13px;
    margin: -5px 0;
    background: #ffffff;
    border: 2px solid #168b7b;
    border-radius: 7px;
}

QSlider#layerOpacitySlider::handle:horizontal:hover,
QSlider#globalOpacitySlider::handle:horizontal:hover {
    background: #dff1ed;
    border-color: #0d6a5f;
}
"""


# --- HSIViewer scene styling -------------------------------------------- #
# QGraphicsScene items (the photo, mask overlay, watermark, and annotation
# prompts drawn on HSIViewer) are painted directly and are not reachable by
# QSS, so their colours live here as plain constants instead.
VIEWER_SCENE_BACKGROUND   = QColor(255, 255, 255)
VIEWER_WATERMARK_COLOR    = QColor("#eaecee")
VIEWER_WATERMARK_POINT_SIZE = 45
PROMPT_POSITIVE_COLOR     = Qt.GlobalColor.green
PROMPT_NEGATIVE_COLOR     = Qt.GlobalColor.red
PROMPT_POINT_RADIUS       = 5.0
CROP_OVERLAY_COLOR        = QColor(0, 0, 0, 140)
CROP_SELECTION_COLOR      = Qt.GlobalColor.yellow


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont))
    assets_dir = Path(__file__).parent / "assets"
    combo_box_arrow = assets_dir / "chevron_down.svg"
    radio_unchecked = assets_dir / "radio_unchecked.svg"
    radio_checked = assets_dir / "radio_checked.svg"
    checkbox_unchecked = assets_dir / "checkbox_unchecked.svg"
    checkbox_checked = assets_dir / "checkbox_checked.svg"
    menu_check = assets_dir / "menu_check.svg"
    menu_chevron = assets_dir / "chevron_right.svg"
    app.setStyleSheet(
        APP_QSS.replace("__COMBOBOX_ARROW__", combo_box_arrow.as_posix())
        .replace("__RADIO_UNCHECKED__", radio_unchecked.as_posix())
        .replace("__RADIO_CHECKED__", radio_checked.as_posix())
        .replace("__CHECKBOX_UNCHECKED__", checkbox_unchecked.as_posix())
        .replace("__CHECKBOX_CHECKED__", checkbox_checked.as_posix())
        .replace("__MENU_CHECK__", menu_check.as_posix())
        .replace("__MENU_CHEVRON__", menu_chevron.as_posix())
    )
