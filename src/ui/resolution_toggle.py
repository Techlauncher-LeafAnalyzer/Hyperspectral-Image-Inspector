"""Consistent, animated image-state selectors floating over the canvases."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from PyQt6 import QtCore, QtGui, QtWidgets


class ResolutionToggle(QtWidgets.QAbstractButton):
    """Two visible choices with a sliding selection and direct mouse/keyboard input."""

    WIDTH = 208
    HEIGHT = 36
    INSET = 12

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        name: str,
        *,
        low_label: str = "Low Res",
        high_label: str = "High Res",
        width: int = WIDTH,
    ) -> None:
        super().__init__(parent)
        self._low_label = low_label
        self._high_label = high_label
        self._mouse_selection: bool | None = None
        self._hovered_segment: bool | None = None
        self.setObjectName(name)
        self.setAccessibleName(f"{low_label} or {high_label}")
        self.setCheckable(True)
        self.setFixedSize(width, self.HEIGHT)
        self.move(self.INSET, self.INSET)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self._progress = 0.0
        self._animation = QtCore.QPropertyAnimation(self, b"progress", self)
        self._animation.setDuration(180)
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

    def _update_hint(self, checked: bool) -> None:
        selected = self._high_label if checked else self._low_label
        destination = self._low_label if checked else self._high_label
        self.setToolTip(f"Viewing {selected}. Select {destination} to compare.")
        self.setAccessibleDescription(
            f"Viewing {selected}. Use Left or Right to select, or Space to switch."
        )
        self._animation.stop()
        target = 1.0 if checked else 0.0
        if self._progress == target:
            return
        self._animation.setStartValue(self._progress)
        self._animation.setEndValue(target)
        self._animation.start()

    def nextCheckState(self) -> None:
        self.setChecked(
            not self.isChecked() if self._mouse_selection is None else self._mouse_selection
        )

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._mouse_selection = event.position().x() >= self.width() / 2
        super().mouseReleaseEvent(event)
        self._mouse_selection = None

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        self._hovered_segment = event.position().x() >= self.width() / 2
        self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._hovered_segment = None
        self.update()
        super().leaveEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Right):
            checked = event.key() == QtCore.Qt.Key.Key_Right
            self.setChecked(checked)
            self.clicked.emit(checked)
            event.accept()
            return
        super().keyPressEvent(event)

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
        enabled = self.isEnabled()
        track = QtCore.QRectF(0.5, 0.5, self.width() - 1, self.HEIGHT - 1)
        painter.setPen(QtGui.QPen(QtGui.QColor("#cddad6")))
        painter.setBrush(QtGui.QColor("#f0f5f3"))
        painter.drawRoundedRect(track, 8, 8)
        segment_width = (self.width() - 8) / 2
        if enabled and self._hovered_segment is not None:
            hover = QtCore.QRectF(
                4 + segment_width * self._hovered_segment, 4,
                segment_width, self.HEIGHT - 8,
            )
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor("#dcebe6"))
            painter.drawRoundedRect(hover, 5, 5)
        selected = QtCore.QRectF(
            4 + segment_width * self._progress, 4, segment_width, self.HEIGHT - 8
        )
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(
            "#cddad6" if not enabled else "#116f62" if self.isDown() else "#168b7b"
        ))
        painter.drawRoundedRect(selected, 5, 5)
        font = painter.font()
        font.setPointSize(9)
        font.setWeight(QtGui.QFont.Weight.DemiBold)
        painter.setFont(font)
        for high, label in ((False, self._low_label), (True, self._high_label)):
            painter.setPen(QtGui.QColor(
                "#758680" if not enabled else "#ffffff" if high == self.isChecked()
                else "#405b53"
            ))
            painter.drawText(
                QtCore.QRectF(4 + segment_width * high, 4, segment_width, self.HEIGHT - 8),
                QtCore.Qt.AlignmentFlag.AlignCenter, label,
            )
        if self.hasFocus():
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.setPen(QtGui.QPen(QtGui.QColor("#116f62"), 2))
            painter.drawRoundedRect(track.adjusted(1, 1, -1, -1), 7, 7)


class CalibrationToggle(ResolutionToggle):
    """The same selector for cached raw and calibrated images."""

    def __init__(self, canvas: QtWidgets.QWidget, name: str) -> None:
        super().__init__(
            canvas, name, low_label="Before Calibration", high_label="After Calibration",
            width=328,
        )


class ResolutionSwitchGroup(QtCore.QObject):
    """Synchronize independent resolution/calibration choices across every page."""

    def __init__(
        self,
        canvases: Iterable[tuple[QtWidgets.QWidget, str]],
        on_clicked: Callable[[bool], None],
        parent: QtCore.QObject,
        *,
        on_calibration_clicked: Callable[[bool], None],
        calibration_only_canvases: Iterable[tuple[QtWidgets.QWidget, str]] = (),
    ) -> None:
        super().__init__(parent)
        canvases = tuple(canvases)
        self._canvas_switches = tuple(ResolutionToggle(canvas, name) for canvas, name in canvases)
        self.switches = self._canvas_switches
        self.calibration_switches = tuple(
            CalibrationToggle(canvas, name.replace("ResolutionSwitch", "CalibrationSwitch"))
            for canvas, name in canvases
        ) + tuple(CalibrationToggle(canvas, name) for canvas, name in calibration_only_canvases)
        for switch in self.switches:
            switch.clicked.connect(on_clicked)
        for switch in self.calibration_switches:
            switch.clicked.connect(on_calibration_clicked)
            switch.parentWidget().installEventFilter(self)
        self._raise_timer = QtCore.QTimer(self)
        self._raise_timer.setSingleShot(True)
        self._raise_timer.timeout.connect(self.raise_switches)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.Resize:
            self.raise_switches()
        return super().eventFilter(watched, event)

    def stop(self) -> None:
        self._raise_timer.stop()
        for switch in self.switches + self.calibration_switches:
            switch.stop_animation()

    def sync(self, *, available: bool, high_resolution: bool, enabled: bool) -> None:
        for switch in self.switches:
            switch.setChecked(high_resolution)
            switch.setEnabled(enabled and available)
            switch.setVisible(available)
        self.raise_switches()

    def sync_calibration(
        self, *, available: bool, calibrated: bool, enabled: bool,
        current_available: bool,
    ) -> None:
        for switch in self.calibration_switches:
            switch.setChecked(calibrated)
            switch.setEnabled(enabled and current_available)
            switch.setVisible(available)
            if available and not current_available:
                switch.setToolTip("No calibrated result at this resolution. Run Calibration to compare.")
        self.raise_switches()

    @QtCore.pyqtSlot()
    def raise_switches(self) -> None:
        for switch in self._canvas_switches:
            if not switch.isHidden():
                switch.move(switch.INSET, switch.INSET)
                switch.raise_()
        for index, switch in enumerate(self.calibration_switches):
            if switch.isHidden():
                continue
            x, y = switch.INSET, switch.INSET
            if index < len(self._canvas_switches):
                resolution = self._canvas_switches[index]
                if not resolution.isHidden():
                    next_x = x + resolution.width() + 8
                    if next_x + switch.width() + switch.INSET <= switch.parentWidget().width():
                        x = next_x
                    else:
                        y += switch.HEIGHT + 8
            switch.move(x, y)
            switch.raise_()

    def schedule_raise(self) -> None:
        self._raise_timer.start(0)
