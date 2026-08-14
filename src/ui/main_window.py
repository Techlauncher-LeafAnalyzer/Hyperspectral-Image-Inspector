from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QFileDialog, QMessageBox

import core.hsi_utils as hsi_utils
from core import (
    HSIData,
    HSIError,
    HSIReader,
    VisualizationMode,
    VisualizationRequest,
    VisualizationService,
)
from ui.generated.MainWindow import Ui_MainWindow


LOGGER = logging.getLogger(__name__)


class _TabTransitionController(QtCore.QObject):
    """Adds a restrained indicator slide and content fade to a tab widget."""

    _DURATION_MS = 180

    def __init__(self, tab_widget: QtWidgets.QTabWidget) -> None:
        super().__init__(tab_widget)
        self._tab_widget = tab_widget
        self._tab_bar = tab_widget.tabBar()
        self._page_animations: dict[
            QtWidgets.QWidget,
            tuple[QtWidgets.QGraphicsOpacityEffect, QtCore.QPropertyAnimation],
        ] = {}

        self._indicator = QtWidgets.QFrame(self._tab_bar)
        self._indicator.setObjectName("tabIndicator")
        self._indicator.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self._indicator_animation = QtCore.QPropertyAnimation(
            self._indicator,
            b"geometry",
            self,
        )
        self._indicator_animation.setDuration(self._DURATION_MS)
        self._indicator_animation.setEasingCurve(
            QtCore.QEasingCurve.Type.OutCubic
        )

        self._tab_bar.installEventFilter(self)
        self._tab_widget.currentChanged.connect(self._on_current_changed)
        QtCore.QTimer.singleShot(0, self._sync_indicator)

    def eventFilter(
        self,
        watched: QtCore.QObject,
        event: QtCore.QEvent,
    ) -> bool:
        if watched is self._tab_bar and event.type() in {
            QtCore.QEvent.Type.LayoutRequest,
            QtCore.QEvent.Type.Resize,
            QtCore.QEvent.Type.Show,
            QtCore.QEvent.Type.StyleChange,
        }:
            QtCore.QTimer.singleShot(0, self._sync_indicator)
        return super().eventFilter(watched, event)

    def _indicator_rect(self, index: int) -> QtCore.QRect:
        tab_rect = self._tab_bar.tabRect(index)
        inset = min(14, max(6, tab_rect.width() // 8))
        return QtCore.QRect(
            tab_rect.x() + inset,
            max(0, self._tab_bar.height() - 3),
            max(0, tab_rect.width() - (inset * 2)),
            3,
        )

    def _sync_indicator(self) -> None:
        index = self._tab_widget.currentIndex()
        if index < 0:
            self._indicator.hide()
            return

        self._indicator_animation.stop()
        self._indicator.setGeometry(self._indicator_rect(index))
        self._indicator.show()
        self._indicator.raise_()

    def _on_current_changed(self, index: int) -> None:
        if index < 0:
            return

        target_geometry = self._indicator_rect(index)
        self._indicator_animation.stop()
        self._indicator_animation.setStartValue(self._indicator.geometry())
        self._indicator_animation.setEndValue(target_geometry)
        self._indicator_animation.start()
        self._fade_in_page(self._tab_widget.widget(index))

    def _fade_in_page(self, page: QtWidgets.QWidget) -> None:
        effect_animation = self._page_animations.get(page)
        if effect_animation is None:
            effect = QtWidgets.QGraphicsOpacityEffect(page)
            animation = QtCore.QPropertyAnimation(effect, b"opacity", self)
            animation.setDuration(self._DURATION_MS)
            animation.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
            animation.finished.connect(lambda: effect.setEnabled(False))
            page.setGraphicsEffect(effect)
            effect_animation = (effect, animation)
            self._page_animations[page] = effect_animation

        effect, animation = effect_animation
        animation.stop()
        effect.setEnabled(True)
        effect.setOpacity(0.72)
        animation.setStartValue(0.72)
        animation.setEndValue(1.0)
        animation.start()


class MainWindowController(QtWidgets.QMainWindow, Ui_MainWindow):
    """Application controller for the Hyperspectral Image Inspector.

    Inherits the widget layout from ``Ui_MainWindow`` (auto-generated from
    ``qt/MainWindow.ui``) and wires all application logic on top of it.
    State is encapsulated in a single ``HSIData`` instance that is injected
    into each feature panel on construction.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self._hsi_data = HSIData()
        self._hsi_reader = HSIReader()
        self._visualization_service = VisualizationService()
        self._configure_tabs()
        self._configure_file_menu()
        self._connect_signals()

    # ------------------------------------------------------------------ #
    # Private: signal wiring                                               #
    # ------------------------------------------------------------------ #

    def _configure_tabs(self) -> None:
        tab_settings = (
            (self.tabWidget, "mainTabBar", "Application sections"),
            (
                self.classificationModeTabs,
                "classificationTabBar",
                "Classification mode",
            ),
        )
        self._tab_transitions: list[_TabTransitionController] = []

        for tab_widget, object_name, accessible_name in tab_settings:
            tab_bar = tab_widget.tabBar()
            tab_bar.setObjectName(object_name)
            tab_bar.setAccessibleName(accessible_name)
            tab_bar.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            tab_bar.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            tab_bar.setExpanding(False)
            tab_bar.setElideMode(QtCore.Qt.TextElideMode.ElideRight)
            self._tab_transitions.append(_TabTransitionController(tab_widget))

    def _configure_file_menu(self) -> None:
        """Fold the File menu into the main tab row as a ribbon-style dropdown."""
        assets_dir = Path(__file__).parent / "assets"
        self.actionLoadImage.setIcon(
            QtGui.QIcon(str(assets_dir / "folder_open.svg"))
        )
        self.actionLoadImage.setShortcut(
            QtGui.QKeySequence.StandardKey.Open
        )
        self.actionLoadImage.setStatusTip("Open a hyperspectral image")
        self.actionSaveImage.setIcon(
            QtGui.QIcon(str(assets_dir / "save_image.svg"))
        )
        self.actionSaveImage.setShortcut(
            QtGui.QKeySequence.StandardKey.Save
        )
        self.actionSaveImage.setStatusTip("Save the current image")

        self._file_menu = QtWidgets.QMenu(self)
        self._file_menu.setObjectName("fileMenu")
        self._file_menu.setAccessibleName("File actions")
        self._file_menu.setToolTipsVisible(True)
        self._file_menu.setMinimumWidth(220)
        self._file_menu.addAction(self.actionLoadImage)
        self._file_menu.addAction(self.actionSaveImage)

        self._file_menu_button = QtWidgets.QToolButton(self.tabWidget)
        self._file_menu_button.setObjectName("fileMenuButton")
        self._file_menu_button.setText("File")
        self._file_menu_button.setAccessibleName("File menu")
        self._file_menu_button.setAccessibleDescription(
            "Open the menu for loading and saving images"
        )
        self._file_menu_button.setToolTip("File actions")
        self._file_menu_button.setMenu(self._file_menu)
        self._file_menu_button.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup
        )
        self._file_menu_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly
        )
        self._file_menu_button.setCursor(
            QtCore.Qt.CursorShape.PointingHandCursor
        )
        self._file_menu_button.setFocusPolicy(
            QtCore.Qt.FocusPolicy.StrongFocus
        )
        self._file_menu.aboutToShow.connect(
            lambda: self._set_file_menu_open(True)
        )
        self._file_menu.aboutToHide.connect(
            lambda: self._set_file_menu_open(False)
        )
        self.tabWidget.setCornerWidget(
            self._file_menu_button, QtCore.Qt.Corner.TopLeftCorner
        )

    def _set_file_menu_open(self, is_open: bool) -> None:
        """Keep the File trigger visually active while its menu is open."""
        self._file_menu_button.setProperty("menuOpen", is_open)
        style = self._file_menu_button.style()
        style.unpolish(self._file_menu_button)
        style.polish(self._file_menu_button)
        self._file_menu_button.update()

    def _connect_signals(self) -> None:
        self.actionLoadImage.triggered.connect(self._load_image)
        self.actionSaveImage.triggered.connect(self._save_image)
        self.darkFileButton.clicked.connect(self._select_dark_file)
        self.referenceFileButton.clicked.connect(self._select_reference_file)
        self.pushButton.clicked.connect(self._select_groundtruth_file)
        self.calibrateButton.setEnabled(False)
        self.calibrateButton.setToolTip("Calibration is not implemented yet")

    # ------------------------------------------------------------------ #
    # Private: image I/O                                                   #
    # ------------------------------------------------------------------ #

    def _select_dark_file(self) -> None:
        self._select_supporting_file(
            self.darkFileEdit,
            "Open Dark File",
        )

    def _select_reference_file(self) -> None:
        self._select_supporting_file(
            self.referenceFileEdit,
            "Open Reference File",
        )

    def _select_groundtruth_file(self) -> None:
        self._select_supporting_file(
            self.lineEdit,
            "Open Groundtruth File",
        )

    def _select_supporting_file(
        self,
        target_edit: QtWidgets.QLineEdit,
        dialog_title: str,
    ) -> None:
        file_path_str, _ = QFileDialog.getOpenFileName(
            self,
            dialog_title,
            "",
            (
                "Supported Images (*.bil *.bip *.bsq *.png *.jpg *.jpeg "
                "*.tif *.tiff);;All Files (*)"
            ),
        )
        if not file_path_str:
            return

        file_path = Path(file_path_str)
        target_edit.setText(str(file_path))
        target_edit.setToolTip(str(file_path))
        self.statusbar.showMessage(f"Selected {file_path.name}")

    def _load_image(self) -> None:
        image_path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open Hyperspectral Image",
            "",
            "Hyperspectral Images (*.hdr *.bil *.bip *.bsq *.dat *.img *.raw)",
        )
        if not image_path_str:
            return

        selected_path = Path(image_path_str)
        try:
            candidate = self._hsi_reader.open(selected_path)
            result = self._visualization_service.render(
                candidate,
                VisualizationRequest(mode=VisualizationMode.RGB),
            )
        except HSIError as exc:
            QMessageBox.critical(self, "Unable to load image", str(exc))
            self.statusbar.showMessage("Image load failed")
            return
        except Exception as exc:  # Keep Qt's event loop alive for unexpected failures.
            LOGGER.exception("Unexpected hyperspectral import failure")
            QMessageBox.critical(
                self,
                "Unable to load image",
                f"An unexpected error occurred: {exc}",
            )
            self.statusbar.showMessage("Image load failed")
            return

        rgb_array = result.display_rgb
        candidate.rgb_array = rgb_array
        candidate.mask_array = np.zeros(rgb_array.shape[:2], dtype=np.uint8)
        self._hsi_data.update_from(candidate)
        image_path = self._hsi_data.data_path

        loaded_file_text = f"File Loaded: {image_path}"
        self.imageFilePath.setText(loaded_file_text)
        self.imageFilePath.setToolTip(str(image_path))
        self.superResFilePath.setText(loaded_file_text)
        self.superResFilePath.setToolTip(str(image_path))
        self.classificationFilePath.setText(loaded_file_text)
        self.classificationFilePath.setToolTip(str(image_path))
        self.unsupervisedClassifyButton.setEnabled(True)
        self.pushButton_2.setEnabled(True)
        self.statusbar.showMessage(f"Loaded {image_path.name}")
        pixmap = hsi_utils.numpy_to_qpixmap(rgb_array)

        self.viewer.rgb        = rgb_array
        self.viewer.mask_array = self._hsi_data.mask_array
        self.viewer.set_photo(pixmap)

        self.calibrationViewer.rgb        = rgb_array
        self.calibrationViewer.mask_array = self._hsi_data.mask_array
        self.calibrationViewer.set_photo(pixmap)

        self.superResViewer.rgb        = rgb_array
        self.superResViewer.mask_array = self._hsi_data.mask_array
        self.superResViewer.set_photo(pixmap)

        self.classificationViewer.rgb        = rgb_array
        self.classificationViewer.mask_array = self._hsi_data.mask_array
        self.classificationViewer.set_photo(pixmap)

    def _save_image(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Private: viewer signal handlers                                      #
    # ------------------------------------------------------------------ #

    def _on_spectrum_plot(self, pos: QPointF) -> None:
        pass

    def _on_mean_index(self, index_name: str) -> None:
        pass
