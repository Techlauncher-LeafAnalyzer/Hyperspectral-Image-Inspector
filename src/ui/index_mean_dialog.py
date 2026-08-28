"""Popup surfacing a vegetation index's mean value with range context.

Replaces the transient status-bar message: the mean is shown as a hero
figure alongside the observed min/max, plotted on a gradient gauge drawn
from the same colormap used to render that index in the viewer.
"""

from __future__ import annotations

from matplotlib import colormaps
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

_FULL_NAMES: dict[str, str] = {
    "NDVI": "Normalized Difference Vegetation Index",
    "EVI": "Enhanced Vegetation Index",
    "MCARI": "Modified Chlorophyll Absorption Ratio Index",
    "MTVI": "Modified Triangular Vegetation Index",
    "OSAVI": "Optimized Soil-Adjusted Vegetation Index",
    "PRI": "Photochemical Reflectance Index",
}

_GAUGE_MARKER_DIAMETER = 16
_GAUGE_BAR_HEIGHT = 12


class _RangeGauge(QtWidgets.QWidget):
    """A gradient bar marking where a value sits within an observed range."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._stops: list[tuple[float, QtGui.QColor]] = []
        self._fraction = 0.5
        self.setMinimumHeight(_GAUGE_MARKER_DIAMETER + 6)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

    def set_value(self, colormap_name: str | None, fraction: float) -> None:
        cmap = colormaps.get(colormap_name or "viridis", colormaps["viridis"])
        self._stops = [
            (position, QtGui.QColor.fromRgbF(*cmap(position)[:3]))
            for position in (step / 8 for step in range(9))
        ]
        self._fraction = min(max(fraction, 0.0), 1.0)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt override)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        margin = _GAUGE_MARKER_DIAMETER / 2
        bar_top = (self.height() - _GAUGE_BAR_HEIGHT) / 2
        bar_rect = QtCore.QRectF(
            margin, bar_top, max(self.width() - 2 * margin, 1), _GAUGE_BAR_HEIGHT
        )

        gradient = QtGui.QLinearGradient(bar_rect.topLeft(), bar_rect.topRight())
        for position, color in self._stops:
            gradient.setColorAt(position, color)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QBrush(gradient))
        painter.drawRoundedRect(bar_rect, _GAUGE_BAR_HEIGHT / 2, _GAUGE_BAR_HEIGHT / 2)

        marker_center = QtCore.QPointF(
            bar_rect.left() + self._fraction * bar_rect.width(), bar_rect.center().y()
        )
        painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 2.5))
        painter.setBrush(QtGui.QColor("#123e37"))
        painter.drawEllipse(marker_center, margin - 1, margin - 1)
        painter.end()


class IndexMeanDialog(QtWidgets.QDialog):
    """Non-modal popup presenting a vegetation index's mean value."""

    def __init__(
        self,
        index_name: str,
        mean_value: float,
        value_range: tuple[float, float],
        colormap: str | None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setObjectName("indexMeanDialog")
        self.setWindowTitle(f"{index_name} Index Mean")
        self.setFixedWidth(380)
        self.setSizeGripEnabled(False)

        icon_dir = QtCore.QFileInfo(__file__).absoluteDir().filePath("assets")
        icon_label = QtWidgets.QLabel()
        icon_label.setPixmap(QtGui.QIcon(f"{icon_dir}/index_mean.svg").pixmap(20, 20))

        heading = QtWidgets.QLabel(index_name)
        heading.setObjectName("indexMeanHeading")

        heading_row = QtWidgets.QHBoxLayout()
        heading_row.setSpacing(8)
        heading_row.addWidget(icon_label)
        heading_row.addWidget(heading)
        heading_row.addStretch(1)

        subtitle = QtWidgets.QLabel(_FULL_NAMES.get(index_name, "Vegetation index"))
        subtitle.setObjectName("indexMeanSubtitle")
        subtitle.setWordWrap(True)

        card = QtWidgets.QFrame()
        card.setObjectName("indexMeanCard")
        value_label = QtWidgets.QLabel(f"{mean_value:.4f}")
        value_label.setObjectName("indexMeanValue")
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_caption = QtWidgets.QLabel("Mean value across the image")
        value_caption.setObjectName("indexMeanCaption")
        value_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(20, 18, 20, 18)
        card_layout.setSpacing(2)
        card_layout.addWidget(value_label)
        card_layout.addWidget(value_caption)

        low, high = value_range
        span = high - low
        fraction = 0.5 if span <= 0 else (mean_value - low) / span

        gauge = _RangeGauge()
        gauge.set_value(colormap, fraction)

        range_row = QtWidgets.QHBoxLayout()
        low_label = QtWidgets.QLabel(f"{low:.3f}")
        low_label.setObjectName("indexMeanRangeLabel")
        high_label = QtWidgets.QLabel(f"{high:.3f}")
        high_label.setObjectName("indexMeanRangeLabel")
        high_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        range_row.addWidget(low_label)
        range_row.addWidget(high_label)

        range_caption = QtWidgets.QLabel("Observed minimum and maximum")
        range_caption.setObjectName("indexMeanCaption")
        range_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)

        close_button = QtWidgets.QPushButton("Close")
        close_button.setProperty("primaryButton", True)
        close_button.clicked.connect(self.close)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 18)
        layout.setSpacing(6)
        layout.addLayout(heading_row)
        layout.addWidget(subtitle)
        layout.addSpacing(16)
        layout.addWidget(card)
        layout.addSpacing(18)
        layout.addWidget(gauge)
        layout.addLayout(range_row)
        layout.addWidget(range_caption)
        layout.addSpacing(12)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)
