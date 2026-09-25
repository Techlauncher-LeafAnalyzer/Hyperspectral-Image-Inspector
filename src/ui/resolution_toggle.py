"""A shared-state resolution switch that floats over image canvases."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from PyQt6 import QtCore, QtGui, QtWidgets


class ResolutionToggle(QtWidgets.QAbstractButton):
    """Pill switch: light 1× original, dark 2× super-resolved."""

    WIDTH = 138
    HEIGHT = 36
    INSET = 12
    KNOB = 30

    def __init__(self, parent: QtWidgets.QWidget, name: str) -> None:
        super().__init__(parent)
        self.setObjectName(name)
        self.setAccessibleName("Image resolution switch")
        self.setCheckable(True)
        self.setFixedSize(self.WIDTH, self.HEIGHT)
        self.move(self.INSET, self.INSET)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self._progress = 0.0
        self._animation = QtCore.QPropertyAnimation(self, b"progress", self)
        self._animation.setDuration(200)
        self._animation.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
        self.toggled.connect(self._update_hint)
        self._update_hint(False)
        self.hide()

    @QtCore.pyqtProperty(float)
    def progress(self) -> float:
        return self._progress

    @progress.setter
    def progress(self, value: float) -> None:
        self._progress = float(value)
        self.update()

    def _update_hint(self, high_resolution: bool) -> None:
        destination = "original (low-res)" if high_resolution else "Super-Resolution (high-res)"
        self.setToolTip(f"Switch to {destination}")
        self.setAccessibleDescription(f"Switch to {destination}")
        self._animation.stop()
        self._animation.setStartValue(self._progress)
        self._animation.setEndValue(1.0 if high_resolution else 0.0)
        self._animation.start()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        progress = self._progress
        track = QtCore.QRectF(0.5, 0.5, self.WIDTH - 1, self.HEIGHT - 1)
        light = QtGui.QColor("#eceef1")
        dark = QtGui.QColor("#121417")
        def blend(start: QtGui.QColor, end: QtGui.QColor) -> QtGui.QColor:
            return QtGui.QColor(
                *(round(a * (1 - progress) + b * progress)
                  for a, b in zip(start.getRgb()[:3], end.getRgb()[:3]))
            )

        painter.setPen(QtGui.QPen(blend(QtGui.QColor("#d6d9de"), dark)))
        painter.setBrush(blend(light, dark))
        painter.drawRoundedRect(track, self.HEIGHT / 2, self.HEIGHT / 2)

        circle_x = 3 + (self.WIDTH - self.KNOB - 6) * (1 - progress)
        circle = QtCore.QRectF(circle_x, 3, self.KNOB, self.KNOB)
        painter.setPen(QtGui.QPen(blend(QtGui.QColor("#ccd0d6"), QtGui.QColor("#ffffff"))))
        painter.setBrush(QtGui.QColor("#ffffff"))
        painter.drawEllipse(circle)

        font = painter.font()
        font.setBold(True)
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QtGui.QColor("#14181d"))
        painter.setOpacity(1 - progress)
        painter.drawText(circle, QtCore.Qt.AlignmentFlag.AlignCenter, "1×")
        painter.setOpacity(progress)
        painter.drawText(circle, QtCore.Qt.AlignmentFlag.AlignCenter, "2×")
        painter.setOpacity(1 - progress)
        painter.setPen(QtGui.QColor("#1d2228"))
        painter.drawText(
            QtCore.QRectF(3, 0, self.WIDTH - 39, self.HEIGHT),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            "LOW RES",
        )
        painter.setOpacity(progress)
        painter.setPen(QtGui.QColor("#ffffff"))
        painter.drawText(
            QtCore.QRectF(36, 0, self.WIDTH - 40, self.HEIGHT),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            "HIGH RES",
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
            if not switch.isHidden():
                switch.move(switch.INSET, switch.INSET)
                switch.raise_()

    def schedule_raise(self) -> None:
        QtCore.QTimer.singleShot(0, self.raise_switches)
