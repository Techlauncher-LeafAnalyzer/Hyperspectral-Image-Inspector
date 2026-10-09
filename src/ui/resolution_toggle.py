"""Pill selectors and a dedicated image-state rail beside each canvas."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from PyQt6 import QtCore, QtGui, QtWidgets
from ui.theme import INSPECTOR_PANEL_WIDTH


class ResolutionToggle(QtWidgets.QAbstractButton):
    """Two visible choices with a sliding selection and direct mouse/keyboard input."""

    WIDTH = 208
    HEIGHT = 40
    INSET = 12

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        name: str,
        *,
        low_label: str = "Low Res",
        high_label: str = "High Res",
        display_labels: tuple[str, str] | None = None,
    ) -> None:
        super().__init__(parent)
        self._low_label = low_label
        self._high_label = high_label
        self._display_labels = display_labels or (low_label, high_label)
        self._mouse_selection: bool | None = None
        self._hovered_segment: bool | None = None
        self._keyboard_focus = False
        self.setObjectName(name)
        self.setAccessibleName(f"{low_label} or {high_label}")
        self.setCheckable(True)
        self.setFixedHeight(self.HEIGHT)
        self.setMinimumWidth(180)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.resize(self.WIDTH, self.HEIGHT)
        self.move(self.INSET, self.INSET)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self._progress = 0.0
        self._animation = QtCore.QPropertyAnimation(self, b"progress", self)
        self._animation.setDuration(220)
        self._animation.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
        self._hover_progress = 0.0
        self._hover_animation = QtCore.QPropertyAnimation(self, b"hoverProgress", self)
        self._hover_animation.setDuration(140)
        self._hover_animation.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._hover_animation.finished.connect(self._on_hover_finished)
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

    @QtCore.pyqtProperty(float)
    def hoverProgress(self) -> float:
        return self._hover_progress

    @hoverProgress.setter
    def hoverProgress(self, value: float) -> None:
        self._hover_progress = float(value)
        self.update()

    def _animate_hover(self, segment: bool | None) -> None:
        if segment is not None:
            self._hovered_segment = segment
        self._hover_animation.stop()
        self._hover_animation.setStartValue(self._hover_progress)
        self._hover_animation.setEndValue(0.0 if segment is None else 1.0)
        self._hover_animation.start()

    def _on_hover_finished(self) -> None:
        if self._hover_progress == 0.0:
            self._hovered_segment = None
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

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(self.WIDTH, self.HEIGHT)

    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:
        self._keyboard_focus = event.reason() in (
            QtCore.Qt.FocusReason.TabFocusReason,
            QtCore.Qt.FocusReason.BacktabFocusReason,
            QtCore.Qt.FocusReason.ShortcutFocusReason,
        )
        super().focusInEvent(event)
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self._keyboard_focus = False
        super().mousePressEvent(event)
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._mouse_selection = event.position().x() >= self.width() / 2
        super().mouseReleaseEvent(event)
        self._mouse_selection = None

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        segment = event.position().x() >= self.width() / 2
        if segment != self._hovered_segment or self._hover_animation.endValue() != 1.0:
            self._animate_hover(segment)
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._animate_hover(None)
        super().leaveEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        self._keyboard_focus = True
        self.update()
        if event.key() in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Right):
            checked = event.key() == QtCore.Qt.Key.Key_Right
            self.setChecked(checked)
            self.clicked.emit(checked)
            event.accept()
            return
        super().keyPressEvent(event)

    def stop_animation(self) -> None:
        self._animation.stop()
        self._hover_animation.stop()
        self._hovered_segment = None
        self.hoverProgress = 0.0
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
        painter.setPen(QtGui.QPen(QtGui.QColor("#dce5e0")))
        painter.setBrush(QtGui.QColor("#edf2ef"))
        painter.drawRoundedRect(track, track.height() / 2, track.height() / 2)
        segment_width = (self.width() - 8) / 2
        if enabled and self._hovered_segment is not None:
            hover = QtCore.QRectF(
                4 + segment_width * self._hovered_segment, 4,
                segment_width, self.HEIGHT - 8,
            )
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            color = QtGui.QColor("#dce7e0")
            color.setAlphaF(self._hover_progress)
            painter.setBrush(color)
            painter.drawRoundedRect(hover, hover.height() / 2, hover.height() / 2)
        selected = QtCore.QRectF(
            4 + segment_width * self._progress, 4, segment_width, self.HEIGHT - 8
        )
        # A small tinted shadow gives the selected capsule depth without
        # covering the image or introducing a heavyweight graphics effect.
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(28, 65, 47, 15 if enabled else 0))
        painter.drawRoundedRect(selected.translated(0, 1.5), 16, 16)
        fill = QtGui.QColor("#247b64")
        if not enabled:
            fill = QtGui.QColor("#e5ebe7")
        elif self.isDown():
            fill = QtGui.QColor("#1c6753")
        elif self._hovered_segment == self.isChecked():
            hovered = QtGui.QColor("#2b856c")
            fill = QtGui.QColor(*(
                round(start * (1 - self._hover_progress) + end * self._hover_progress)
                for start, end in zip(fill.getRgb()[:3], hovered.getRgb()[:3])
            ))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(fill)
        painter.drawRoundedRect(selected, selected.height() / 2, selected.height() / 2)
        font = painter.font()
        font.setPointSize(10)
        font.setWeight(QtGui.QFont.Weight.DemiBold)
        painter.setFont(font)
        for high, label in enumerate(self._display_labels):
            weight = self._progress if high else 1 - self._progress
            inactive = QtGui.QColor("#66786e" if enabled else "#9aa69f")
            active = QtGui.QColor("#ffffff" if enabled else "#73877a")
            color = QtGui.QColor(*(
                round(start * (1 - weight) + end * weight)
                for start, end in zip(inactive.getRgb()[:3], active.getRgb()[:3])
            ))
            painter.setPen(color)
            painter.drawText(
                QtCore.QRectF(4 + segment_width * high, 4, segment_width, self.HEIGHT - 8),
                QtCore.Qt.AlignmentFlag.AlignCenter, label,
            )
        if self.hasFocus() and self._keyboard_focus:
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.setPen(QtGui.QPen(QtGui.QColor("#4d9272"), 2))
            painter.drawRoundedRect(track.adjusted(1, 1, -1, -1), 18, 18)


class CalibrationToggle(ResolutionToggle):
    """The same selector for cached raw and calibrated images."""

    def __init__(self, canvas: QtWidgets.QWidget, name: str) -> None:
        super().__init__(
            canvas, name, low_label="Before Calibration", high_label="After Calibration",
            display_labels=("Before", "After"),
        )


class ImageStatePanel(QtWidgets.QFrame):
    """A quiet, fixed-width rail that reserves space beside the image."""

    def __init__(self, parent: QtWidgets.QWidget, name: str) -> None:
        super().__init__(parent)
        self.setObjectName("imageStatePanel")
        self.setAccessibleName("Image version controls")
        self.setFixedWidth(INSPECTOR_PANEL_WIDTH)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 20, 12, 20)
        layout.setSpacing(24)
        title = QtWidgets.QLabel("Image view", self)
        title.setObjectName("imageStateTitle")
        layout.addWidget(title)
        self.resolution_switch = ResolutionToggle(self, name)
        self.calibration_switch = CalibrationToggle(
            self, name.replace("ResolutionSwitch", "CalibrationSwitch")
        )
        self.resolution_section = self._section("Resolution", self.resolution_switch)
        self.calibration_section = self._section("Calibration", self.calibration_switch)
        layout.addWidget(self.resolution_section)
        layout.addWidget(self.calibration_section)
        layout.addStretch(1)
        self.resolution_section.hide()
        self.calibration_section.hide()
        self.hide()

    def _section(self, title: str, switch: ResolutionToggle) -> QtWidgets.QWidget:
        section = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        label = QtWidgets.QLabel(title, section)
        label.setObjectName("imageStateLabel")
        label.setBuddy(switch)
        layout.addWidget(label)
        layout.addWidget(switch)
        return section

    def sync_visibility(self) -> None:
        self.resolution_section.setVisible(not self.resolution_switch.isHidden())
        self.calibration_section.setVisible(not self.calibration_switch.isHidden())
        self.setVisible(
            not self.resolution_switch.isHidden() or not self.calibration_switch.isHidden()
        )


class ResolutionSwitchGroup(QtCore.QObject):
    """Synchronize independent image-state choices in each page's left rail."""

    def __init__(
        self,
        panels: Iterable[ImageStatePanel],
        on_clicked: Callable[[bool], None],
        parent: QtCore.QObject,
        *,
        on_calibration_clicked: Callable[[bool], None],
    ) -> None:
        super().__init__(parent)
        self.panels = tuple(panels)
        self.switches = tuple(panel.resolution_switch for panel in self.panels)
        self.calibration_switches = tuple(panel.calibration_switch for panel in self.panels)
        for switch in self.switches:
            switch.clicked.connect(on_clicked)
        for switch in self.calibration_switches:
            switch.clicked.connect(on_calibration_clicked)
        self._raise_timer = QtCore.QTimer(self)
        self._raise_timer.setSingleShot(True)
        self._raise_timer.timeout.connect(self.raise_switches)

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
        # Qt layouts own placement; version controls never sit over image pixels.
        for panel in self.panels:
            panel.sync_visibility()

    def schedule_raise(self) -> None:
        self._raise_timer.start(0)
