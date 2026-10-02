from __future__ import annotations

from PyQt6 import QtCore, QtWidgets


class _PageFade(QtWidgets.QGraphicsOpacityEffect):
    """Keep the animation and its completion slot in the effect's lifetime."""

    def __init__(self, page: QtWidgets.QWidget, duration: int) -> None:
        super().__init__(page)
        self.animation = QtCore.QPropertyAnimation(self, b"opacity", self)
        self.animation.setDuration(duration)
        self.animation.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self.animation.finished.connect(self._finish)

    @QtCore.pyqtSlot()
    def _finish(self) -> None:
        self.setEnabled(False)


class TabTransitionHandler(QtCore.QObject):
    """Adds a restrained indicator slide and content fade to a tab widget."""

    _DURATION_MS = 180

    def __init__(self, tab_widget: QtWidgets.QTabWidget) -> None:
        super().__init__(tab_widget)
        self._tab_widget = tab_widget
        self._tab_bar = tab_widget.tabBar()
        self._indicator = QtWidgets.QFrame(self._tab_bar)
        self._indicator.setObjectName("tabIndicator")
        self._indicator.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self._indicator_animation = QtCore.QPropertyAnimation(
            self._indicator,
            b"geometry",
            self._indicator,
        )
        self._indicator_animation.setDuration(self._DURATION_MS)
        self._indicator_animation.setEasingCurve(
            QtCore.QEasingCurve.Type.OutCubic
        )

        self._tab_bar.installEventFilter(self)
        self._tab_widget.currentChanged.connect(self._on_current_changed)
        self._sync_timer = QtCore.QTimer(self)
        self._sync_timer.setSingleShot(True)
        self._sync_timer.timeout.connect(self._sync_indicator)
        self._sync_timer.start(0)

    def stop(self) -> None:
        """Quiesce transitions before the window's deferred deletion."""
        self._sync_timer.stop()
        self._indicator_animation.stop()
        for index in range(self._tab_widget.count()):
            effect = self._tab_widget.widget(index).graphicsEffect()
            if isinstance(effect, _PageFade):
                effect.animation.stop()
                effect.setEnabled(False)

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
            self._sync_timer.start(0)
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

    @QtCore.pyqtSlot()
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
        effect = page.graphicsEffect()
        if not isinstance(effect, _PageFade):
            effect = _PageFade(page, self._DURATION_MS)
            page.setGraphicsEffect(effect)

        animation = effect.animation
        animation.stop()
        effect.setEnabled(True)
        effect.setOpacity(0.72)
        animation.setStartValue(0.72)
        animation.setEndValue(1.0)
        animation.start()
