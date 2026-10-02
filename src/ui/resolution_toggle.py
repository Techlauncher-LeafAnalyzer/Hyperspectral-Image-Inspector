"""A shared-state resolution switch that floats over image canvases."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from PyQt6 import QtCore, QtGui, QtWidgets


class ResolutionToggle(QtWidgets.QAbstractButton):
    """Pill switch: light 1× original, dark 2× super-resolved."""

    WIDTH = 116
    HEIGHT = 36
    INSET = 12
    KNOB = 30

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        name: str,
        *,
        low_label: str = "LOW RES",
        high_label: str = "HIGH RES",
        low_symbol: str = "1×",
        high_symbol: str = "2×",
        width: int = WIDTH,
    ) -> None:
        super().__init__(parent)
        self._low_label = low_label
        self._high_label = high_label
        self._low_symbol = low_symbol
        self._high_symbol = high_symbol
        self.setObjectName(name)
        self.setAccessibleName(f"{low_label} or {high_label} switch")
        self.setCheckable(True)
        self.setFixedSize(width, self.HEIGHT)
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
        destination = self._low_label if high_resolution else self._high_label
        self.setToolTip(f"Switch to {destination}")
        self.setAccessibleDescription(f"Switch to {destination}")
        self._animation.stop()
        target = 1.0 if high_resolution else 0.0
        if self._progress == target:
            return
        self._animation.setStartValue(self._progress)
        self._animation.setEndValue(target)
        self._animation.start()

    def stop_animation(self) -> None:
        self._animation.stop()
        self.progress = 1.0 if self.isChecked() else 0.0

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        self.stop_animation()
        super().hideEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        progress = self._progress
        width = self.width()
        track = QtCore.QRectF(0.5, 0.5, width - 1, self.HEIGHT - 1)
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

        circle_x = 3 + (width - self.KNOB - 6) * (1 - progress)
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
        painter.drawText(circle, QtCore.Qt.AlignmentFlag.AlignCenter, self._low_symbol)
        painter.setOpacity(progress)
        painter.drawText(circle, QtCore.Qt.AlignmentFlag.AlignCenter, self._high_symbol)
        painter.setOpacity(1 - progress)
        painter.setPen(QtGui.QColor("#1d2228"))
        painter.drawText(
            QtCore.QRectF(6, 0, width - self.KNOB - 12, self.HEIGHT),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            self._low_label,
        )
        painter.setOpacity(progress)
        painter.setPen(QtGui.QColor("#ffffff"))
        painter.drawText(
            QtCore.QRectF(self.KNOB + 6, 0, width - self.KNOB - 12, self.HEIGHT),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            self._high_label,
        )


class CalibrationBadge(QtWidgets.QLabel):
    """Noninteractive status tag for the currently displayed calibrated cube."""

    def __init__(self, canvas: QtWidgets.QWidget, name: str) -> None:
        super().__init__("Calibrated", canvas)
        self.setObjectName(name)
        self.setAccessibleName("Calibrated image")
        self.setToolTip("The displayed hyperspectral image is calibrated")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(28)
        self.setStyleSheet(
            "color: #166534; background: #dcfce7; border: 1px solid #bbf7d0; "
            "border-radius: 13px; padding: 0 10px; font-size: 12px; font-weight: 600;"
        )
        self.adjustSize()
        self.hide()


class ResolutionSwitchGroup(QtCore.QObject):
    """Shared resolution selection and fixed-position calibration status tags."""

    def __init__(
        self,
        canvases: Iterable[tuple[QtWidgets.QWidget, str]],
        on_clicked: Callable[[bool], None],
        parent: QtCore.QObject,
        *,
        badge_only_canvases: Iterable[tuple[QtWidgets.QWidget, str]] = (),
    ) -> None:
        super().__init__(parent)
        canvases = tuple(canvases)
        self.switches = tuple(
            ResolutionToggle(canvas, name) for canvas, name in canvases
        )
        self.calibration_badges = tuple(
            CalibrationBadge(canvas, name.replace("ResolutionSwitch", "CalibrationBadge"))
            for canvas, name in canvases
        ) + tuple(CalibrationBadge(canvas, name) for canvas, name in badge_only_canvases)
        for switch in self.switches:
            switch.clicked.connect(on_clicked)
        self._raise_timer = QtCore.QTimer(self)
        self._raise_timer.setSingleShot(True)
        self._raise_timer.timeout.connect(self.raise_switches)

    def stop(self) -> None:
        self._raise_timer.stop()
        for switch in self.switches:
            switch.stop_animation()

    def sync(self, *, available: bool, high_resolution: bool, enabled: bool) -> None:
        for switch in self.switches:
            switch.setChecked(high_resolution)
            switch.setEnabled(enabled)
            switch.setVisible(available)
            if available:
                switch.raise_()
        self.raise_switches()

    def set_calibrated(self, calibrated: bool) -> None:
        for badge in self.calibration_badges:
            badge.setVisible(calibrated)
        self.raise_switches()

    @QtCore.pyqtSlot()
    def raise_switches(self) -> None:
        for switch in self.switches:
            if not switch.isHidden():
                switch.move(switch.INSET, switch.INSET)
                switch.raise_()
        for index, badge in enumerate(self.calibration_badges):
            if badge.isHidden():
                continue
            x = ResolutionToggle.INSET
            if index < len(self.switches) and not self.switches[index].isHidden():
                x += self.switches[index].width() + 8
            y = ResolutionToggle.INSET + (ResolutionToggle.HEIGHT - badge.height()) // 2
            badge.move(x, y)
            badge.raise_()

    def schedule_raise(self) -> None:
        self._raise_timer.start(0)
