"""JetBrains-style tool windows: edge stripes of icon buttons plus dock panels.

Each tool window is a :class:`QDockWidget` docked beside its stripe -- the
left stripe holds image-view toggles and analysis tools, the right stripe
holds the workflow panels (Visualization, Super-Resolution, ...) that replace
the old main tabs. A :class:`ToolWindowStripe` is a narrow bar of icon
buttons (squares on the left, captioned "pills" on the right):

* **selected** -- the window is open, docked or popped out -- the button gets
  a tinted highlight background;
* **not selected** -- the window is hidden -- the button has no background;
* **disabled** -- the tool has nothing to show yet -- the icon is drawn gray.

Like an IDE, one *docked* tool window shows per side: opening another hides
the one docked there. Popped-out windows are independent of that rule.

Each tool has a compact header with "pop out"/"dock" and "hide" buttons,
mirroring an IDE tool window. Popping out moves the panel into a normal
top-level window with a system title bar instead of floating the dock: Qt's
floating docks are moved by a client-side mouse grab, which Wayland refuses.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets, sip
from PyQt6.QtCore import Qt

from ui.theme import (
    TOOL_STRIPE_DISABLED_COLOR,
    TOOL_STRIPE_HOVER_COLOR,
    TOOL_STRIPE_ICON_COLOR,
    TOOL_STRIPE_SELECTED_COLOR,
    TOOL_STRIPE_SELECTED_ICON_COLOR,
)

# Paints a tool's icon into ``rect`` with the given pen/brush colour.
IconPainter = Callable[[QtGui.QPainter, QtCore.QRectF, QtGui.QColor], None]

# Logical-pixel gap between a stripe button and each side of the stripe. Qt
# scales logical pixels by the screen's device pixel ratio, so the gap grows
# with display scaling like the rest of the UI.
STRIPE_GAP = 5
# Button side in logical pixels; scales with display scaling like the gap.
STRIPE_BUTTON_SIDE = 46
# Captioned pills are wider so a one-word caption fits under the icon.
PILL_BUTTON_WIDTH = 58
PILL_BUTTON_HEIGHT = 54
# Share of a button's side the icon occupies.
_ICON_FRACTION = 0.56
_CAPTION_POINT_SIZE = 7.5
# Workflow settings panels open at this width and shrink to the minimum.
SETTINGS_PANEL_WIDTH = 330
SETTINGS_PANEL_MIN_WIDTH = 260
# Width of the accent bar that marks the pill whose view the canvas shows.
_CURRENT_MARKER_WIDTH = 3


def paint_super_resolution_icon(
    painter: QtGui.QPainter, rect: QtCore.QRectF, color: QtGui.QColor
) -> None:
    """A coarse 2x2 grid growing into a fine 4x4 grid: 2x spatial resolution."""

    unit = rect.width() / 18
    pen = QtGui.QPen(color, 1.4)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    small = QtCore.QRectF(rect.left() + 1.5 * unit, rect.top() + 9 * unit, 7 * unit, 7 * unit)
    painter.drawRect(small)
    painter.drawLine(QtCore.QLineF(small.center().x(), small.top(), small.center().x(), small.bottom()))
    painter.drawLine(QtCore.QLineF(small.left(), small.center().y(), small.right(), small.center().y()))
    large = QtCore.QRectF(rect.left() + 7.5 * unit, rect.top() + 1.5 * unit, 9 * unit, 9 * unit)
    fill = QtGui.QColor(color)
    fill.setAlphaF(0.18)
    painter.setBrush(fill)
    painter.drawRect(large)
    thin = QtGui.QPen(color, 0.9)
    painter.setPen(thin)
    for step in (1, 2, 3):
        offset = large.width() * step / 4
        painter.drawLine(QtCore.QLineF(large.left() + offset, large.top(), large.left() + offset, large.bottom()))
        painter.drawLine(QtCore.QLineF(large.left(), large.top() + offset, large.right(), large.top() + offset))


def paint_calibration_icon(
    painter: QtGui.QPainter, rect: QtCore.QRectF, color: QtGui.QColor
) -> None:
    """Half-dark, half-bright disc: dark/white-reference calibration."""

    unit = rect.width() / 18
    disc = QtCore.QRectF(rect.left() + 2.5 * unit, rect.top() + 2.5 * unit, 13 * unit, 13 * unit)
    # Left half filled (dark reference), right half open (bright reference).
    half = QtGui.QPainterPath()
    half.moveTo(disc.center())
    half.arcTo(disc, 90, 180)
    half.closeSubpath()
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)
    painter.drawPath(half)
    painter.setPen(QtGui.QPen(color, 1.5))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(disc)


def paint_spectrum_icon(painter: QtGui.QPainter, rect: QtCore.QRectF, color: QtGui.QColor) -> None:
    """Axes with a spectral curve, matching the context menu's Spectrum icon."""

    unit = rect.width() / 18
    pen = QtGui.QPen(color, 1.4)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    axes = QtGui.QPainterPath()
    axes.moveTo(rect.left() + 3 * unit, rect.top() + 3 * unit)
    axes.lineTo(rect.left() + 3 * unit, rect.top() + 15 * unit)
    axes.lineTo(rect.left() + 15.5 * unit, rect.top() + 15 * unit)
    painter.drawPath(axes)
    pen.setWidthF(1.8)
    painter.setPen(pen)
    curve = QtGui.QPainterPath()
    points = ((4.5, 12.5), (7.2, 9.0), (9.4, 10.8), (12.8, 5.2), (15.0, 7.0))
    curve.moveTo(rect.left() + points[0][0] * unit, rect.top() + points[0][1] * unit)
    for x, y in points[1:]:
        curve.lineTo(rect.left() + x * unit, rect.top() + y * unit)
    painter.drawPath(curve)


def paint_visualization_icon(
    painter: QtGui.QPainter, rect: QtCore.QRectF, color: QtGui.QColor
) -> None:
    """A 3x3 tile map in graded tints: a per-pixel index rendering."""

    unit = rect.width() / 18
    tile = 4.4 * unit
    gap = 0.9 * unit
    origin = QtCore.QPointF(rect.left() + 1.9 * unit, rect.top() + 1.9 * unit)
    # Diagonal gradient of opacities reads as a smooth index ramp.
    alphas = ((0.95, 0.7, 0.45), (0.7, 0.45, 0.25), (0.45, 0.25, 0.12))
    painter.setPen(Qt.PenStyle.NoPen)
    for row, row_alphas in enumerate(alphas):
        for column, alpha in enumerate(row_alphas):
            fill = QtGui.QColor(color)
            fill.setAlphaF(alpha)
            painter.setBrush(fill)
            painter.drawRoundedRect(
                QtCore.QRectF(
                    origin.x() + column * (tile + gap), origin.y() + row * (tile + gap),
                    tile, tile,
                ),
                0.8 * unit, 0.8 * unit,
            )


def paint_segmentation_icon(
    painter: QtGui.QPainter, rect: QtCore.QRectF, color: QtGui.QColor
) -> None:
    """A leaf split along its midrib, one half filled: a classified region."""

    unit = rect.width() / 18
    tip = QtCore.QPointF(rect.left() + 15.5 * unit, rect.top() + 2.5 * unit)
    base = QtCore.QPointF(rect.left() + 3.0 * unit, rect.top() + 15.0 * unit)
    upper = QtCore.QPointF(rect.left() + 3.0 * unit, rect.top() + 3.0 * unit)
    lower = QtCore.QPointF(rect.left() + 15.5 * unit, rect.top() + 15.0 * unit)
    half = QtGui.QPainterPath(base)
    half.quadTo(upper, tip)
    half.closeSubpath()
    fill = QtGui.QColor(color)
    fill.setAlphaF(0.35)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(fill)
    painter.drawPath(half)
    outline = QtGui.QPainterPath(base)
    outline.quadTo(upper, tip)
    outline.quadTo(lower, base)
    pen = QtGui.QPen(color, 1.5)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(outline)
    painter.drawLine(QtCore.QLineF(base, tip))
    # The stem.
    painter.drawLine(QtCore.QLineF(base, QtCore.QPointF(base.x() - 1.2 * unit, base.y() + 1.2 * unit)))


def paint_batch_icon(painter: QtGui.QPainter, rect: QtCore.QRectF, color: QtGui.QColor) -> None:
    """Three stacked sheets: the same pipeline applied to many files."""

    unit = rect.width() / 18
    pen = QtGui.QPen(color, 1.4)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    front = QtCore.QRectF(rect.left() + 2.0 * unit, rect.top() + 7.0 * unit, 9.5 * unit, 8.5 * unit)
    fill = QtGui.QColor(color)
    fill.setAlphaF(0.22)
    painter.setBrush(fill)
    painter.drawRoundedRect(front, 1.2 * unit, 1.2 * unit)
    # Sheets behind: only the edges peeking out above and to the right.
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for step in (1, 2):
        offset = 2.3 * unit * step
        edge = QtGui.QPainterPath()
        edge.moveTo(front.left() + offset, front.top() - offset + 2.3 * unit)
        edge.lineTo(front.left() + offset, front.top() - offset)
        edge.lineTo(front.right() + offset, front.top() - offset)
        edge.lineTo(front.right() + offset, front.bottom() - offset)
        edge.lineTo(front.right() + offset - 2.3 * unit, front.bottom() - offset)
        painter.drawPath(edge)


def paint_file_icon(painter: QtGui.QPainter, rect: QtCore.QRectF, color: QtGui.QColor) -> None:
    """An open folder: the File menu (open and save images)."""

    unit = rect.width() / 18
    pen = QtGui.QPen(color, 1.5)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    back = QtGui.QPainterPath()
    back.moveTo(rect.left() + 2.0 * unit, rect.top() + 14.5 * unit)
    back.lineTo(rect.left() + 2.0 * unit, rect.top() + 3.5 * unit)
    back.lineTo(rect.left() + 6.5 * unit, rect.top() + 3.5 * unit)
    back.lineTo(rect.left() + 8.0 * unit, rect.top() + 5.5 * unit)
    back.lineTo(rect.left() + 14.5 * unit, rect.top() + 5.5 * unit)
    back.lineTo(rect.left() + 14.5 * unit, rect.top() + 8.0 * unit)
    painter.drawPath(back)
    front = QtGui.QPainterPath()
    front.moveTo(rect.left() + 2.0 * unit, rect.top() + 14.5 * unit)
    front.lineTo(rect.left() + 4.8 * unit, rect.top() + 8.0 * unit)
    front.lineTo(rect.left() + 16.5 * unit, rect.top() + 8.0 * unit)
    front.lineTo(rect.left() + 13.7 * unit, rect.top() + 14.5 * unit)
    front.closeSubpath()
    fill = QtGui.QColor(color)
    fill.setAlphaF(0.2)
    painter.setBrush(fill)
    painter.drawPath(front)


class ToolStripeButton(QtWidgets.QAbstractButton):
    """Checkable stripe button that paints its own icon by state.

    A plain button is a square icon. Given a ``caption`` it becomes a taller
    "pill" with the caption under the icon, like an IDE stripe with tool
    window names shown. A pill can also be marked *current*
    (:meth:`set_current`): an accent bar on its inner edge shows which view
    the canvas is displaying, even while the pill's panel is closed.
    """

    def __init__(
        self,
        icon_painter: IconPainter,
        title: str,
        parent: QtWidgets.QWidget | None = None,
        *,
        caption: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self._icon_painter = icon_painter
        self._caption = caption
        self._hovered = False
        self._current = False
        # The side facing the image area, where the current marker goes.
        self.marker_edge = Qt.Edge.LeftEdge
        self.setCheckable(True)
        self.set_side(STRIPE_BUTTON_SIDE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setAccessibleName(title)
        self.setText(title)
        self.setToolTip(title)

    @property
    def caption(self) -> Optional[str]:
        return self._caption

    def set_side(self, side: int) -> None:
        """Make the button a ``side`` x ``side`` square (or the pill size)."""

        if self._caption:
            self.setFixedSize(PILL_BUTTON_WIDTH, PILL_BUTTON_HEIGHT)
        else:
            self.setFixedSize(side, side)

    def is_current(self) -> bool:
        return self._current

    def set_current(self, current: bool) -> None:
        """Mark this pill as the view the canvas currently shows."""

        if current != self._current:
            self._current = current
            self.update()

    def sizeHint(self) -> QtCore.QSize:
        return self.size()

    def enterEvent(self, event: QtGui.QEnterEvent) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def icon_color(self) -> QtGui.QColor:
        """The colour the icon is drawn in for the current state."""

        if not self.isEnabled():
            return QtGui.QColor(TOOL_STRIPE_DISABLED_COLOR)
        if self.isChecked():
            return QtGui.QColor(TOOL_STRIPE_SELECTED_ICON_COLOR)
        return QtGui.QColor(TOOL_STRIPE_ICON_COLOR)

    def background_color(self) -> Optional[QtGui.QColor]:
        """Highlight when selected, a faint hover tint, otherwise none."""

        if not self.isEnabled():
            return None
        if self.isChecked():
            return QtGui.QColor(TOOL_STRIPE_SELECTED_COLOR)
        if self._hovered or self.isDown():
            return QtGui.QColor(TOOL_STRIPE_HOVER_COLOR)
        return None

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        bounds = QtCore.QRectF(self.rect()).adjusted(0.75, 0.75, -0.75, -0.75)
        radius = bounds.width() * 0.2
        background = self.background_color()
        if background is not None:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(background)
            painter.drawRoundedRect(bounds, radius, radius)
        if self.hasFocus():
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QtGui.QPen(QtGui.QColor(TOOL_STRIPE_SELECTED_ICON_COLOR), 1.5))
            painter.drawRoundedRect(bounds, radius, radius)
        if self._current and self.isEnabled():
            self._paint_current_marker(painter, bounds)
        if not self._caption:
            side = round(bounds.width() * _ICON_FRACTION)
            icon = QtCore.QRectF(0, 0, side, side)
            icon.moveCenter(bounds.center())
            self._icon_painter(painter, icon, self.icon_color())
            return
        # Pill: icon in the upper part, caption beneath it.
        font = QtGui.QFont(self.font())
        font.setPointSizeF(_CAPTION_POINT_SIZE)
        font.setWeight(QtGui.QFont.Weight.DemiBold)
        metrics = QtGui.QFontMetricsF(font)
        caption_height = metrics.height()
        side = round(STRIPE_BUTTON_SIDE * 0.5)
        top = bounds.top() + (bounds.height() - side - caption_height - 2) / 2
        icon = QtCore.QRectF(bounds.center().x() - side / 2, top, side, side)
        self._icon_painter(painter, icon, self.icon_color())
        text_rect = QtCore.QRectF(bounds.left() + 2, icon.bottom() + 2, bounds.width() - 4, caption_height)
        painter.setFont(font)
        painter.setPen(self.icon_color())
        painter.drawText(
            text_rect, Qt.AlignmentFlag.AlignCenter,
            metrics.elidedText(self._caption, Qt.TextElideMode.ElideRight, text_rect.width()),
        )

    def _paint_current_marker(self, painter: QtGui.QPainter, bounds: QtCore.QRectF) -> None:
        length = bounds.height() * 0.42
        x = bounds.left() if self.marker_edge == Qt.Edge.LeftEdge else bounds.right() - _CURRENT_MARKER_WIDTH
        marker = QtCore.QRectF(x, bounds.center().y() - length / 2, _CURRENT_MARKER_WIDTH, length)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(TOOL_STRIPE_SELECTED_ICON_COLOR))
        painter.drawRoundedRect(marker, _CURRENT_MARKER_WIDTH / 2, _CURRENT_MARKER_WIDTH / 2)


class StripeMenuButton(ToolStripeButton):
    """Non-checkable stripe button that opens a menu beside the stripe.

    Highlighted like a selected tool while its menu is open.
    """

    def __init__(
        self,
        icon_painter: IconPainter,
        title: str,
        menu: QtWidgets.QMenu,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(icon_painter, title, parent)
        self.setCheckable(False)
        self.menu = menu
        self._menu_open = False
        menu.aboutToShow.connect(lambda: self._set_menu_open(True))
        menu.aboutToHide.connect(lambda: self._set_menu_open(False))
        self.clicked.connect(self.show_menu)

    def is_menu_open(self) -> bool:
        return self._menu_open

    def show_menu(self) -> None:
        # Open beside the stripe rather than over the button.
        self.menu.popup(self.mapToGlobal(QtCore.QPoint(self.width() + STRIPE_GAP, 0)))

    def _set_menu_open(self, is_open: bool) -> None:
        self._menu_open = is_open
        self.update()

    def background_color(self) -> Optional[QtGui.QColor]:
        if self._menu_open and self.isEnabled():
            return QtGui.QColor(TOOL_STRIPE_SELECTED_COLOR)
        return super().background_color()

    def icon_color(self) -> QtGui.QColor:
        if self._menu_open and self.isEnabled():
            return QtGui.QColor(TOOL_STRIPE_SELECTED_ICON_COLOR)
        return super().icon_color()


class ToolWindowStripe(QtWidgets.QToolBar):
    """A vertical bar on one window edge: logo, toggles, tool-window buttons.

    A fixed toolbar in the left or right toolbar area, which Qt places
    outside the dock area, so the stripe stays on the very edge with docks
    beside it. Its single child, a :class:`QFrame`, lays everything out by
    hand, top to bottom: the app logo (left stripe only), then buttons with
    optional dividers between groups.

    Buttons keep :data:`STRIPE_GAP` from the window edge. The inner gap is
    measured to the image area: the main window reports the page pane's own
    border/padding with :meth:`set_inner_inset`, and the stripe shrinks its
    inner margin by that much so the visible gap on both sides stays equal.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        edge: Qt.Edge = Qt.Edge.LeftEdge,
        title: str = "Tool windows",
        object_name: str = "toolWindowStripe",
        show_logo: bool = True,
    ) -> None:
        super().__init__(title, parent)
        self.tool_windows: list["ToolWindow"] = []
        self.edge = edge
        self.setObjectName(object_name)
        self.setAccessibleName(title)
        self.setOrientation(Qt.Orientation.Vertical)
        self.setMovable(False)
        self.setFloatable(False)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        self._body = QtWidgets.QFrame(self)
        self._body.setObjectName("toolWindowStripeBody")
        self._layout = QtWidgets.QVBoxLayout(self._body)
        self._layout.setSpacing(STRIPE_GAP)
        self.logo = QtWidgets.QLabel(self._body)
        self.logo.setObjectName("toolWindowStripeLogo")
        self.logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.logo.setAccessibleName("HyperView")
        self.logo.setVisible(show_logo)
        self._logo_pixmap: Optional[QtGui.QPixmap] = None
        if show_logo:
            self._layout.addWidget(self.logo, 0, Qt.AlignmentFlag.AlignHCenter)
            # Space between the logo row and the first button.
            self._layout.addSpacing(STRIPE_GAP * 2)
        self._buttons: list[ToolStripeButton] = []
        self._dividers: list[QtWidgets.QFrame] = []
        self._layout.addStretch(1)
        self.addWidget(self._body)
        self._inner_inset = 0
        self._apply_geometry()

    # ------------------------------------------------------------------ #

    @property
    def toolbar_area(self) -> Qt.ToolBarArea:
        if self.edge == Qt.Edge.RightEdge:
            return Qt.ToolBarArea.RightToolBarArea
        return Qt.ToolBarArea.LeftToolBarArea

    @property
    def dock_area(self) -> Qt.DockWidgetArea:
        """The dock area beside this stripe, where its tool windows dock."""

        if self.edge == Qt.Edge.RightEdge:
            return Qt.DockWidgetArea.RightDockWidgetArea
        return Qt.DockWidgetArea.LeftDockWidgetArea

    def set_logo(self, path: str) -> None:
        """Show the image at ``path``, cropped to its opaque content."""

        image = QtGui.QImage(path)
        if image.isNull():
            return
        if image.hasAlphaChannel():
            image = image.copy(_opaque_bounds(image))
        self._logo_pixmap = QtGui.QPixmap.fromImage(image)
        self._apply_geometry()

    def add_button(self, button: ToolStripeButton) -> None:
        button.setParent(self._body)
        # The current-view marker faces the image area.
        button.marker_edge = (
            Qt.Edge.LeftEdge if self.edge == Qt.Edge.RightEdge else Qt.Edge.RightEdge
        )
        self._layout.insertWidget(self._layout.count() - 1, button, 0, Qt.AlignmentFlag.AlignHCenter)
        self._buttons.append(button)
        self._apply_geometry()

    def add_divider(self) -> None:
        divider = QtWidgets.QFrame(self._body)
        divider.setObjectName("toolWindowStripeDivider")
        divider.setFixedHeight(1)
        self._layout.insertWidget(self._layout.count() - 1, divider)
        self._dividers.append(divider)
        self._apply_geometry()

    def set_inner_inset(self, inset: int) -> None:
        """Account for ``inset`` px between the stripe and the image area."""

        inset = min(max(inset, 0), STRIPE_GAP)
        if inset != self._inner_inset:
            self._inner_inset = inset
            self._apply_geometry()

    def stripe_width(self) -> int:
        """Window edge to the stripe's inner edge."""

        return STRIPE_GAP + self.button_width() + STRIPE_GAP - self._inner_inset

    def button_side(self) -> int:
        return STRIPE_BUTTON_SIDE

    def button_width(self) -> int:
        """Width of the widest button: pills are wider than plain squares."""

        return max([STRIPE_BUTTON_SIDE, *(button.width() for button in self._buttons)])

    def _apply_geometry(self) -> None:
        side = self.button_side()
        for button in self._buttons:
            button.set_side(side)
        width = self.stripe_width()
        self.setFixedWidth(width)
        self._body.setFixedWidth(width)
        # The logo is a little smaller than a button.
        logo_side = max(round(side * 0.82), 12)
        self.logo.setFixedSize(logo_side, logo_side)
        if self._logo_pixmap is not None:
            ratio = self.devicePixelRatioF()
            pixmap = self._logo_pixmap.scaled(
                round(logo_side * ratio), round(logo_side * ratio),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            pixmap.setDevicePixelRatio(ratio)
            self.logo.setPixmap(pixmap)
        inner = STRIPE_GAP - self._inner_inset
        if self.edge == Qt.Edge.RightEdge:
            self._layout.setContentsMargins(inner, STRIPE_GAP, STRIPE_GAP, STRIPE_GAP)
        else:
            self._layout.setContentsMargins(STRIPE_GAP, STRIPE_GAP, inner, STRIPE_GAP)
        for divider in self._dividers:
            divider.setFixedWidth(self.button_width())


def _opaque_bounds(image: QtGui.QImage) -> QtCore.QRect:
    """Bounding rectangle of pixels that are not fully transparent."""

    alpha = image.convertToFormat(QtGui.QImage.Format.Format_Alpha8)
    stride, height = alpha.bytesPerLine(), alpha.height()
    pixels = np.frombuffer(
        alpha.constBits().asstring(stride * height), dtype=np.uint8
    ).reshape(height, stride)[:, :alpha.width()]
    rows = np.flatnonzero(pixels.any(axis=1))
    columns = np.flatnonzero(pixels.any(axis=0))
    if rows.size == 0:
        return image.rect()
    return QtCore.QRect(
        int(columns[0]), int(rows[0]),
        int(columns[-1] - columns[0] + 1), int(rows[-1] - rows[0] + 1),
    )


class ViewToggleButton(ToolStripeButton):
    """Stripe button that switches every tab between two image versions.

    Follows the tool-window highlight rules: highlighted while the *second*
    version (High Res / After Calibration) is shown, plain for the first,
    gray when there is no second version to switch to. Clicking emits
    ``clicked(checked)`` with the version requested; the Controller decides
    whether to honour it and calls :meth:`set_state` with the result.
    """

    def __init__(
        self,
        icon_painter: IconPainter,
        object_name: str,
        *,
        off_label: str,
        on_label: str,
        unavailable_hint: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(icon_painter, on_label, parent)
        self.setObjectName(object_name)
        self._off_label = off_label
        self._on_label = on_label
        self._unavailable_hint = unavailable_hint
        self._unavailable_reason: Optional[str] = None
        self.setAccessibleName(f"{off_label} or {on_label}")
        self.toggled.connect(lambda _checked: self._update_hint())
        self._update_hint()

    def set_state(self, *, checked: bool, available: bool, enabled: bool = True,
                  reason: Optional[str] = None) -> None:
        """Reflect the app's state without emitting ``clicked``."""

        self.blockSignals(True)
        self.setChecked(checked)
        self.blockSignals(False)
        self._unavailable_reason = None if available else (reason or self._unavailable_hint)
        self.setEnabled(available and enabled)
        self._update_hint()
        self.update()

    def _update_hint(self) -> None:
        if self._unavailable_reason is not None:
            self.setToolTip(self._unavailable_reason)
            self.setAccessibleDescription(self._unavailable_reason)
            return
        shown = self._on_label if self.isChecked() else self._off_label
        other = self._off_label if self.isChecked() else self._on_label
        hint = f"Viewing {shown}. Click to view {other} on every tab."
        self.setToolTip(hint)
        self.setAccessibleDescription(hint)


class _ToolTitleBar(QtWidgets.QFrame):
    """Compact tool-window header: title, pop out/dock, hide.

    Buttons only -- the bar is deliberately not a drag handle. Dragging a
    dock out relies on a mouse grab that Wayland refuses ("This plugin
    supports grabbing the mouse only for popup windows"), so pop-out is an
    explicit button that reparents into a normal, compositor-managed window.
    """

    def __init__(self, title: str, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("toolWindowTitleBar")
        self._title = QtWidgets.QLabel(title, self)
        self._title.setObjectName("toolWindowTitle")
        self.float_button = self._make_button("toolWindowFloatButton")
        self.hide_button = self._make_button("toolWindowHideButton")
        self.hide_button.setToolTip("Hide")
        self.hide_button.setAccessibleName("Hide tool window")
        self.hide_button.setText("–")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 4, 4)
        layout.setSpacing(2)
        layout.addWidget(self._title, 1)
        layout.addWidget(self.float_button)
        layout.addWidget(self.hide_button)
        self.set_floating(False)

    def _make_button(self, name: str) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton(self)
        button.setObjectName(name)
        button.setAutoRaise(True)
        button.setFixedSize(22, 22)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    def set_floating(self, floating: bool) -> None:
        # ⧉ pops out; ⇤ returns the window to the sidebar.
        self.float_button.setText("⇤" if floating else "⧉")
        action = "Dock to the sidebar" if floating else "Pop out into a window"
        self.float_button.setToolTip(action)
        self.float_button.setAccessibleName(action)


class _PoppedOutWindow(QtWidgets.QWidget):
    """Normal top-level window hosting a popped-out tool.

    Uses the system title bar so the window manager moves and resizes it
    (no client-side mouse grab). Closing it hides the tool.
    """

    # Emitted once the window is actually hidden (system close, or code).
    hidden = QtCore.pyqtSignal()
    # Emitted when the window becomes the active window.
    activated = QtCore.pyqtSignal()

    def __init__(self, title: str, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setObjectName("toolWindowPopOut")
        self.setWindowTitle(title)
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

    def host(self, widget: QtWidgets.QWidget) -> None:
        self._layout.addWidget(widget)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # Remember the size the user chose before the window goes away.
        self.last_size = self.size()
        super().closeEvent(event)

    def changeEvent(self, event: QtCore.QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.ActivationChange and self.isActiveWindow():
            self.activated.emit()

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        # closeEvent fires before the window is hidden; by hideEvent
        # isHidden() is already true, so listeners see the final state.
        super().hideEvent(event)
        if not event.spontaneous():
            self.hidden.emit()


class ToolWindow(QtCore.QObject):
    """One stripe button bound to one panel, docked or popped out.

    The panel (title bar + content) lives in exactly one container at a time:
    the :class:`QDockWidget` beside its stripe, or a :class:`_PoppedOutWindow`.
    """

    # Emitted after the panel moves between the sidebar and its own window.
    floatingChanged = QtCore.pyqtSignal(bool)
    # Emitted when the user turns to the tool: it is opened, its popped-out
    # window is activated, or focus moves into its panel.
    activated = QtCore.pyqtSignal()

    def __init__(
        self,
        main_window: QtWidgets.QMainWindow,
        stripe: ToolWindowStripe,
        *,
        object_name: str,
        title: str,
        icon_painter: IconPainter,
        content: QtWidgets.QWidget,
        tooltip: str,
        caption: Optional[str] = None,
        popped_size: QtCore.QSize = QtCore.QSize(820, 560),
    ) -> None:
        super().__init__(main_window)
        self._main_window = main_window
        self._floating = False
        self._popped_size: Optional[QtCore.QSize] = None
        self._default_popped_size = popped_size

        # Title bar + content travel together between the two containers.
        self.panel = QtWidgets.QWidget()
        self.panel.setObjectName(f"{object_name}Panel")
        panel_layout = QtWidgets.QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)
        self.title_bar = _ToolTitleBar(title, self.panel)
        panel_layout.addWidget(self.title_bar)
        panel_layout.addWidget(content, 1)
        self.title_bar.float_button.clicked.connect(lambda: self.set_floating(not self._floating))
        self.title_bar.hide_button.clicked.connect(self.hide)

        self.dock = QtWidgets.QDockWidget(title, main_window)
        self.dock.setObjectName(f"{object_name}Dock")
        # Not movable/floatable: Qt's drag-to-float needs a mouse grab that
        # Wayland refuses. Pop-out goes through set_floating() instead.
        self.dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.dock.setAllowedAreas(stripe.dock_area)
        # An empty title-bar widget suppresses Qt's own (draggable) header.
        self.dock.setTitleBarWidget(QtWidgets.QWidget(self.dock))
        self.dock.setWidget(self.panel)
        main_window.addDockWidget(stripe.dock_area, self.dock)
        self.dock.hide()

        self.window = _PoppedOutWindow(title, main_window)
        self.window.setObjectName(f"{object_name}PopOut")
        self.window.hidden.connect(self._on_window_hidden)
        self.window.activated.connect(self.activated)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.focusChanged.connect(self._on_focus_changed)

        self.button = ToolStripeButton(icon_painter, title, stripe, caption=caption)
        self.button.setObjectName(f"{object_name}StripeButton")
        self._tooltip = tooltip
        self.button.setToolTip(tooltip)
        stripe.add_button(self.button)
        self._stripe = stripe
        stripe.tool_windows.append(self)
        # Button → window: the button is the user's switch for the tool.
        self.button.toggled.connect(self._on_button_toggled)
        # Dock → button, for anything that hides the dock behind our back.
        self.dock.visibilityChanged.connect(self._sync_button)

    # ------------------------------------------------------------------ #

    @property
    def floating(self) -> bool:
        """Whether the tool lives in its own popped-out window."""

        return self._floating

    def is_open(self) -> bool:
        """Open means shown in the sidebar or in its popped-out window."""

        container = self.window if self._floating else self.dock
        return not container.isHidden()

    def show(self) -> None:
        if not self.button.isEnabled():
            return
        if self._floating:
            self.window.show()
            self.window.raise_()
            self.window.activateWindow()
        else:
            self._hide_docked_siblings()
            self.dock.show()
            self.dock.raise_()
        # visibilityChanged doesn't fire while the main window is hidden.
        self._sync_button()
        self.activated.emit()

    def hide(self) -> None:
        if self._floating:
            self._remember_popped_size()
            self.window.hide()
        self.dock.hide()
        self._sync_button()

    def set_floating(self, floating: bool) -> None:
        """Move the panel between the sidebar and its own window.

        Popping out or docking is an explicit request to see the tool there,
        so it is shown afterwards even if it was hidden.
        """

        if floating == self._floating:
            return
        if floating:
            self.dock.hide()
            self.dock.setWidget(None)
            self.window.host(self.panel)
            self.panel.show()
            self._floating = True
            self.window.resize(self._popped_size or self._default_popped_size)
        else:
            self._remember_popped_size()
            self.window.hide()
            self.dock.setWidget(self.panel)
            self.panel.show()
            self._floating = False
        self.title_bar.set_floating(floating)
        self.floatingChanged.emit(floating)
        self.show()

    def set_available(self, available: bool, reason: str = "") -> None:
        """Gray the stripe icon (and hide the window) when there's nothing to show."""

        self.button.setEnabled(available)
        self.button.setToolTip(self._tooltip if available or not reason else reason)
        if not available:
            self.hide()

    def _on_focus_changed(self, _old, now) -> None:
        if now is None or sip.isdeleted(self.panel):
            return
        if self.panel.isAncestorOf(now):
            self.activated.emit()

    def _remember_popped_size(self) -> None:
        if not self._floating:
            return
        if self.window.isVisible():
            self._popped_size = self.window.size()
        elif getattr(self.window, "last_size", None) is not None:
            self._popped_size = self.window.last_size

    def _hide_docked_siblings(self) -> None:
        for other in self._stripe.tool_windows:
            if other is not self and not other.floating:
                other.hide()

    def _on_window_hidden(self) -> None:
        # The system close button hides the tool; it stays popped out for
        # the next time it is opened, like an IDE's floating tool window.
        self._sync_button()

    def _on_button_toggled(self, checked: bool) -> None:
        if checked:
            self.show()
        else:
            self.hide()

    def _sync_button(self, *_args) -> None:
        open_ = self.is_open()
        if self.button.isChecked() != open_:
            self.button.blockSignals(True)
            self.button.setChecked(open_)
            self.button.blockSignals(False)
            self.button.update()


class ToolSettingsPanel(QtWidgets.QScrollArea):
    """Scrollable host for a workflow panel built in ``WorkflowPanels.ui``.

    Horizontal scrolling is off, so the panel never gets narrower than its
    content; vertically it scrolls when the window is short.
    """

    def __init__(self, content: QtWidgets.QWidget, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(f"{content.objectName()}Scroll")
        self.setProperty("toolSettings", True)
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.body = content
        content.setProperty("toolSettingsBody", True)
        self.setWidget(content)

    def sizeHint(self) -> QtCore.QSize:
        # The dock opens at this width; the user can drag it wider or narrower.
        width = max(SETTINGS_PANEL_WIDTH, self.minimumSizeHint().width())
        return QtCore.QSize(width, super().sizeHint().height())

    def minimumSizeHint(self) -> QtCore.QSize:
        # Never narrower than the content: horizontal scrolling is off, so a
        # narrower panel would clip it. Leave room for the vertical bar.
        content = self.body.minimumSizeHint().width()
        scrollbar = self.verticalScrollBar().sizeHint().width()
        return QtCore.QSize(
            max(SETTINGS_PANEL_MIN_WIDTH, content + scrollbar), super().minimumSizeHint().height()
        )
