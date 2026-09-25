"""A shared-state resolution switch that floats over image canvases."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from PyQt6 import QtCore, QtGui, QtWidgets


class ResolutionToggle(QtWidgets.QAbstractButton):
    """Pill switch: light 1× original, dark 2× super-resolved."""

    def __init__(self, parent: QtWidgets.QWidget, name: str) -> None:
        super().__init__(parent)
        self.setObjectName(name)
        self.setAccessibleName("Image resolution switch")
        self.setCheckable(True)
        self.setFixedSize(154, 42)
        self.move(14, 14)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.toggled.connect(self._update_hint)
        self._update_hint(False)
        self.hide()

    def _update_hint(self, high_resolution: bool) -> None:
        destination = "original (low-res)" if high_resolution else "Super-Resolution (high-res)"
        self.setToolTip(f"Switch to {destination}")
        self.setAccessibleDescription(f"Switch to {destination}")
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        high = self.isChecked()
        track = QtCore.QRectF(0.5, 0.5, 153, 41)
        painter.setPen(QtGui.QPen(QtGui.QColor("#d6d9de") if not high else QtGui.QColor("#121417")))
        painter.setBrush(QtGui.QColor("#eceef1") if not high else QtGui.QColor("#121417"))
        painter.drawRoundedRect(track, 21, 21)

        circle = QtCore.QRectF(3 if high else 113, 3, 36, 36)
        painter.setPen(QtGui.QPen(QtGui.QColor("#ccd0d6") if not high else QtGui.QColor("#ffffff")))
        painter.setBrush(QtGui.QColor("#ffffff"))
        painter.drawEllipse(circle)

        font = painter.font()
        font.setBold(True)
        font.setPointSize(9)
        painter.setFont(font)
        painter.setPen(QtGui.QColor("#14181d"))
        painter.drawText(circle, QtCore.Qt.AlignmentFlag.AlignCenter, "2×" if high else "1×")
        painter.setPen(QtGui.QColor("#ffffff") if high else QtGui.QColor("#1d2228"))
        label_rect = QtCore.QRectF(44, 0, 106, 42) if high else QtCore.QRectF(5, 0, 104, 42)
        painter.drawText(
            label_rect,
            QtCore.Qt.AlignmentFlag.AlignCenter,
            "HIGH RES" if high else "LOW RES",
        )


class ResolutionSwitchGroup(QtCore.QObject):
    """Keep the three non-SR image canvases on one resolution selection."""

    def __init__(
        self,
        canvases: Iterable[tuple[QtWidgets.QWidget, str]],
        on_clicked: Callable[[bool], None],
        parent: QtCore.QObject,
    ) -> None:
        super().__init__(parent)
        self.switches = tuple(
            ResolutionToggle(canvas, name) for canvas, name in canvases
        )
        for switch in self.switches:
            switch.clicked.connect(on_clicked)

    def sync(self, *, available: bool, high_resolution: bool, enabled: bool) -> None:
        for switch in self.switches:
            switch.setChecked(high_resolution)
            switch.setEnabled(enabled)
            switch.setVisible(available)
            if available:
                switch.raise_()

    def raise_switches(self) -> None:
        for switch in self.switches:
            if switch.isVisible():
                switch.raise_()
