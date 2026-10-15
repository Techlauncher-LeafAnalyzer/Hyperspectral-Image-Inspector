"""Turn annotation layers into Qt graphics items, and burn them into pixels.

Shared by the viewer (live display) and image export so a saved image is drawn
by exactly the code that draws the screen.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsRectItem

from core import AnnotationKind, AnnotationLayer


def make_annotation_item(layer: AnnotationLayer) -> QtWidgets.QGraphicsItem:
    style = layer.style
    fill = QColor(*style.fill)
    if layer.kind is AnnotationKind.TEXT:
        item = QtWidgets.QGraphicsSimpleTextItem(layer.text)
        font = item.font()
        font.setPixelSize(style.font_size)
        item.setFont(font)
        item.setBrush(QBrush(fill))
        if style.stroke_width > 0:
            item.setPen(QPen(QColor(*style.stroke), min(style.stroke_width, 4) / 2))
        else:
            item.setPen(QPen(Qt.PenStyle.NoPen))
        item.setPos(layer.x, layer.y)
        item.setOpacity(style.opacity)
        return item
    if layer.kind is AnnotationKind.ARROW:
        return make_arrow_item(layer)
    rect = QtCore.QRectF(layer.x, layer.y, layer.width, layer.height)
    item = (
        QtWidgets.QGraphicsEllipseItem(rect)
        if layer.kind is AnnotationKind.ELLIPSE
        else QGraphicsRectItem(rect)
    )
    if style.stroke_width > 0:
        pen = QPen(QColor(*style.stroke), style.stroke_width)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    else:
        pen = QPen(Qt.PenStyle.NoPen)
    item.setPen(pen)
    item.setBrush(QBrush(fill) if style.fill_enabled else QBrush(Qt.BrushStyle.NoBrush))
    item.setOpacity(style.opacity)
    return item

def make_arrow_item(layer: AnnotationLayer) -> QtWidgets.QGraphicsItem:
    """Build a shaft and filled head, drawn in the stroke colour."""

    style = layer.style
    color = QColor(*style.stroke)
    width = max(style.stroke_width, 1.0)
    tail = QPointF(layer.x, layer.y)
    tip = QPointF(layer.x + layer.width, layer.y + layer.height)
    line = QtCore.QLineF(tail, tip)
    head_length = min(max(width * 4, 12.0), max(line.length() * 0.6, 1.0))
    angle = math.radians(line.angle())  # Qt's angle is counter-clockwise, y up
    direction = QPointF(math.cos(angle), -math.sin(angle))
    normal = QPointF(-direction.y(), direction.x())
    base = tip - direction * head_length
    half = head_length * 0.4
    path = QPainterPath(tail)
    path.lineTo(base)
    path.moveTo(tip)
    path.lineTo(base + normal * half)
    path.lineTo(base - normal * half)
    path.closeSubpath()
    item = QtWidgets.QGraphicsPathItem(path)
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.FlatCap)
    pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    item.setPen(pen)
    item.setBrush(QBrush(color))
    item.setOpacity(style.opacity)
    return item


def render_annotations(
    rgb: NDArray[np.uint8], layers: Iterable[AnnotationLayer]
) -> NDArray[np.uint8]:
    """Return a copy of ``rgb`` with the visible ``layers`` drawn on top.

    Layers are in stacking order (bottom first) and use image-pixel
    coordinates. Nothing is drawn for hidden layers, and selection outlines and
    handles are never included. Returns ``rgb`` itself when nothing is visible.
    """

    visible = [layer for layer in layers if layer.visible]
    if not visible:
        return rgb
    height, width = rgb.shape[:2]
    source = np.ascontiguousarray(rgb, dtype=np.uint8)
    image = QtGui.QImage(
        source.data, width, height, source.strides[0], QtGui.QImage.Format.Format_RGB888
    ).copy()  # detach from the numpy buffer before painting
    scene = QtWidgets.QGraphicsScene()
    scene.setSceneRect(0, 0, width, height)
    for order, layer in enumerate(visible):
        item = make_annotation_item(layer)
        item.setZValue(order)
        scene.addItem(item)
    painter = QPainter(image)
    painter.setRenderHints(
        QPainter.RenderHint.Antialiasing
        | QPainter.RenderHint.TextAntialiasing
        | QPainter.RenderHint.SmoothPixmapTransform
    )
    scene.render(
        painter, QtCore.QRectF(0, 0, width, height), QtCore.QRectF(0, 0, width, height)
    )
    painter.end()
    rows = np.frombuffer(image.constBits().asstring(image.sizeInBytes()), dtype=np.uint8)
    rows = rows.reshape(height, image.bytesPerLine())
    return np.ascontiguousarray(rows[:, : width * 3].reshape(height, width, 3))
