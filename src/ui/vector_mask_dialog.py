"""Dialog for converting class regions into vector masks.

Chooses which classes to trace and how closely, like Illustrator's Image
Trace panel; the Controller runs :func:`core.trace_mask` with the result.
"""

from __future__ import annotations

from typing import Optional

from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import Qt

from core import VectorTraceOptions

RGB = tuple[int, int, int]


class VectorMaskDialog(QtWidgets.QDialog):
    """Pick classes and trace options; read them back after ``exec()``."""

    def __init__(
        self,
        classes: tuple[tuple[int, str, RGB, bool], ...],
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """``classes`` holds ``(class_id, name, colour, preferred)`` per class."""

        super().__init__(parent)
        self.setWindowTitle("Edit class boundary")
        defaults = VectorTraceOptions()

        intro = QtWidgets.QLabel(
            "Choose a class, then drag points on its boundary with the Pen. "
            "The edited region is used for index means, spectra and export.",
            self,
        )
        intro.setWordWrap(True)

        self._class_list = QtWidgets.QListWidget(self)
        self._class_list.setAccessibleName("Class to edit")
        for class_id, name, color, preferred in classes:
            item = QtWidgets.QListWidgetItem(name)
            swatch = QtGui.QPixmap(14, 14)
            swatch.fill(QtGui.QColor(*color))
            item.setIcon(QtGui.QIcon(swatch))
            item.setData(Qt.ItemDataRole.UserRole, class_id)
            self._class_list.addItem(item)
            if preferred and self._class_list.currentRow() < 0:
                self._class_list.setCurrentItem(item)
        if self._class_list.currentRow() < 0 and classes:
            self._class_list.setCurrentRow(0)
        self._class_list.setMaximumHeight(160)

        self._tolerance = QtWidgets.QDoubleSpinBox(self)
        self._tolerance.setRange(0.1, 20.0)
        self._tolerance.setSingleStep(0.25)
        self._tolerance.setDecimals(2)
        self._tolerance.setSuffix(" px")
        self._tolerance.setValue(defaults.tolerance)
        self._tolerance.setToolTip(
            "How far the outline may stray from the pixel edge. Higher values "
            "give fewer points and smoother outlines."
        )
        self._corner_angle = QtWidgets.QDoubleSpinBox(self)
        self._corner_angle.setRange(0.0, 180.0)
        self._corner_angle.setDecimals(0)
        self._corner_angle.setSingleStep(5.0)
        self._corner_angle.setSuffix("°")
        self._corner_angle.setValue(defaults.corner_angle)
        self._corner_angle.setToolTip(
            "Turns sharper than this become corner points; gentler ones become "
            "smooth points. 180° makes every point smooth."
        )
        self._min_area = QtWidgets.QSpinBox(self)
        self._min_area.setRange(0, 1_000_000)
        self._min_area.setSuffix(" px")
        self._min_area.setValue(round(defaults.min_area))
        self._min_area.setToolTip(
            "Regions and holes smaller than this are dropped, removing speckle "
            "from noisy classifications."
        )

        options = QtWidgets.QWidget(self)
        form = QtWidgets.QFormLayout(options)
        form.setContentsMargins(0, 0, 0, 0)
        options.hide()
        disclosure = QtWidgets.QCheckBox("Adjust tracing for new boundaries", self)
        disclosure.toggled.connect(options.setVisible)
        form.addRow("Path tolerance", self._tolerance)
        form.addRow("Corners sharper than", self._corner_angle)
        form.addRow("Ignore specks under", self._min_area)

        self._buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self._buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Edit boundary")
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self._class_list.currentRowChanged.connect(self._update_ok)
        self._class_list.itemDoubleClicked.connect(self.accept)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(QtWidgets.QLabel("Class", self))
        layout.addWidget(self._class_list)
        layout.addWidget(disclosure)
        layout.addWidget(options)
        layout.addWidget(self._buttons)
        self._update_ok()

    def selected_class_id(self) -> int:
        return int(self._class_list.currentItem().data(Qt.ItemDataRole.UserRole))

    def options(self) -> VectorTraceOptions:
        return VectorTraceOptions(
            tolerance=self._tolerance.value(),
            corner_angle=self._corner_angle.value(),
            min_area=float(self._min_area.value()),
        )

    def _update_ok(self, *_args: object) -> None:
        self._buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            self._class_list.currentRow() >= 0
        )
