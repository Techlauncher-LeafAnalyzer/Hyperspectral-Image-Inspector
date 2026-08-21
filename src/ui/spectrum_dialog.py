"""View for a single-pixel spectral reflectance plot.

Takes a ``core.SpectrumResult`` produced by ``VisualizationService.spectrum``
and renders wavelength vs. value with Matplotlib's Qt backend. Contains no
domain logic; the Model has already selected the pixel and its spectrum.
"""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from core import SpectrumResult


class SpectrumDialog(QtWidgets.QDialog):
    """Modal dialog plotting reflectance/intensity across wavelength."""

    def __init__(
        self,
        result: SpectrumResult,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Spectrum at row {result.row}, column {result.column}")
        self.resize(640, 420)

        figure = Figure(figsize=(6, 4))
        canvas = FigureCanvasQTAgg(figure)
        axes = figure.add_subplot(111)
        axes.plot(result.wavelengths_nm, result.values, linewidth=1.2)
        axes.set_xlabel("Wavelength (nm)")
        axes.set_ylabel("Value")
        axes.set_title(f"Pixel ({result.row}, {result.column})")
        axes.grid(True, alpha=0.3)
        figure.tight_layout()

        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.accept)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(canvas)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)
