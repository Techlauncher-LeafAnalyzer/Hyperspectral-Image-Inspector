"""Live, non-modal spectrum plot shared by every image viewer.

Replaces the one-shot modal spectrum dialog. The window plots up to three
kinds of series over wavelength:

* a *live* trace that follows the pixel under the cursor in any viewer;
* *pinned* pixel spectra, added with Ctrl+Click on any image, the viewer's
  right-click "Spectrum Analyzer" menu, or the "Pin cursor" button;
* *region* spectra -- per-class or whole-image means with a ±1σ band --
  supplied by the Controller from :class:`core.SpectralAnalysisService`.

Every pinned or region series is listed beside the plot, where it can be
shown/hidden, selected (highlighted on the plot) and removed.

Contains no domain logic: transforms, similarity and export delegate to the
Model; this widget only owns what is drawn and emits requests for the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal

from ui.viewer import PIN_SPECTRUM_SHORTCUT
from core import (
    RegionSpectrum,
    SpectralAnalysisService,
    SpectrumResult,
    SpectrumTransform,
    VisualizationError,
)

RGBColor = tuple[int, int, int]

_LIVE_COLOR = "#222222"
_PIN_COLORS = colormaps["tab10"].colors
_MARKER_COLOR = "#8a8a8a"
_SERIES_KEY_ROLE = Qt.ItemDataRole.UserRole

_SEPARABILITY_HELP = (
    "<p><b>What this shows</b><br>"
    "The spectral angle, in degrees, between the mean spectra of every pair of "
    "plotted regions (classes or the image mean). It compares spectral "
    "<i>shape</i> across all bands and ignores overall brightness: larger "
    "angles mean the regions are easier to tell apart, and anything under "
    "about 2° is nearly identical.</p>"
    "<p><b>How to fill it</b><br>"
    "Plot at least two region spectra: run a classification, then press "
    "<i>Class spectra</i> in the Classification layer panel (only the "
    "selected classes are added), or right-click an image and choose "
    "<i>Spectrum Analyzer → Add Class Spectra</i>. <i>Image mean</i> adds "
    "the whole image as another region.</p>"
)


@dataclass(frozen=True, slots=True)
class PinnedSpectrum:
    """One pixel spectrum kept on the plot until removed.

    ``number`` is shown in the label and on the image marker, so a circle on
    the canvas can be matched to its line. ``source`` is the cube the pixel
    was read from (opaque here); markers are drawn only on viewers showing it.
    """

    label: str
    wavelengths_nm: np.ndarray
    values: np.ndarray
    color: tuple[float, float, float]
    number: int = 0
    row: int = 0
    column: int = 0
    source: object = None


@dataclass(frozen=True, slots=True)
class PinMarker:
    """What a viewer needs to draw one pin's circle."""

    key: str
    number: int
    row: int
    column: int
    color: tuple[float, float, float]
    selected: bool
    source: object


@dataclass(slots=True)
class _Series:
    """A listed series: a pinned pixel or a region mean, plus its list state."""

    key: str
    color: tuple[float, float, float]
    pin: Optional[PinnedSpectrum] = None
    region: Optional[RegionSpectrum] = None
    visible: bool = True

    @property
    def label(self) -> str:
        return self.pin.label if self.pin is not None else self.region.label

    @property
    def wavelengths_nm(self) -> np.ndarray:
        return self.pin.wavelengths_nm if self.pin is not None else self.region.wavelengths_nm

    @property
    def values(self) -> np.ndarray:
        return self.pin.values if self.pin is not None else self.region.mean

    @property
    def legend_label(self) -> str:
        if self.region is not None:
            return f"{self.region.label} (n={self.region.pixel_count:,})"
        return self.pin.label


_MIN_CANVAS_WIDTH = 160
_MIN_CANVAS_HEIGHT = 120


class _SpectrumCanvas(FigureCanvasQTAgg):
    """Canvas that re-fits the axes margins whenever it is resized."""

    def __init__(self, figure: Figure, on_resize) -> None:
        super().__init__(figure)
        self._on_resize = on_resize

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        # Matplotlib already queued a draw for the new size; refit the
        # margins for it and repaint with them.
        self._on_resize()
        self.draw_idle()


def _mpl_color(color: RGBColor) -> tuple[float, float, float]:
    return tuple(channel / 255 for channel in color)


class SpectrumInspector(QtWidgets.QWidget):
    """Tool window plotting live, pinned and region spectra."""

    # (reference label, reference values): map the spectral angle to it.
    similarityRequested = pyqtSignal(str, object)
    imageMeanRequested = pyqtSignal()
    # Visible pins, their numbers or the selection changed: redraw markers.
    pinsChanged = pyqtSignal()

    def __init__(
        self,
        analysis: SpectralAnalysisService,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        # Plain widget: SpectrumController hosts it in a dockable tool window.
        super().__init__(parent)
        self.setObjectName("spectrumInspector")
        self.setWindowTitle("Spectrum Inspector")
        self._analysis = analysis
        self._live: Optional[SpectrumResult] = None
        self._live_source = ""
        # Pinned and region series in list order, keyed by a stable ID.
        self._series_by_key: dict[str, _Series] = {}
        # Numbers pins #1, #2, ...; restarts when the image changes.
        self._pin_counter = 0
        self._selected_key: Optional[str] = None
        self._lines: dict[object, str] = {}
        self._markers: dict[str, float] = {}
        self._marker_title = ""

        # Fixed pixel margins, recomputed on resize. Matplotlib's tight and
        # constrained layouts both warn ("axes sizes collapsed to zero") while
        # the canvas is tiny -- e.g. a hidden or still-laying-out dock.
        self._figure = Figure(figsize=(6, 4))
        self._canvas = _SpectrumCanvas(self._figure, self._fit_axes)
        self._canvas.setObjectName("spectrumInspectorCanvas")
        self._canvas.setMinimumSize(_MIN_CANVAS_WIDTH, _MIN_CANVAS_HEIGHT)
        self._axes = self._figure.add_subplot(111)
        self._live_line = None
        self._canvas.mpl_connect("pick_event", self._on_line_picked)

        self._status_label = QtWidgets.QLabel(self._idle_hint(), self)
        self._status_label.setObjectName("spectrumInspectorStatus")
        self._status_label.setWordWrap(True)

        self._series_list = QtWidgets.QListWidget(self)
        self._series_list.setObjectName("spectrumSeriesList")
        self._series_list.setAccessibleName("Plotted spectra")
        self._series_list.setToolTip(
            "Tick to show or hide a spectrum; select one to highlight it "
            "(or click its line on the plot); Delete removes it"
        )
        self._series_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._series_list.itemChanged.connect(self._on_series_item_changed)
        self._series_list.itemSelectionChanged.connect(self._on_series_selection_changed)
        delete_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Delete), self._series_list
        )
        delete_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        delete_shortcut.activated.connect(self.remove_selected)

        self._remove_button = QtWidgets.QPushButton("Remove", self)
        self._remove_button.setObjectName("spectrumRemoveButton")
        self._remove_button.setToolTip("Remove the selected spectra from the plot (Delete)")
        self._remove_button.clicked.connect(self.remove_selected)
        self._remove_button.setEnabled(False)

        self._series_empty_label = QtWidgets.QLabel(
            f"No pinned spectra yet.\n{PIN_SPECTRUM_SHORTCUT} a pixel in any "
            "image, or right-click → Spectrum Analyzer.",
            self,
        )
        self._series_empty_label.setObjectName("spectrumSeriesEmpty")
        self._series_empty_label.setWordWrap(True)
        self._series_empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        series_panel = QtWidgets.QWidget(self)
        series_panel.setObjectName("spectrumSeriesPanel")
        series_layout = QtWidgets.QVBoxLayout(series_panel)
        series_layout.setContentsMargins(0, 0, 0, 0)
        series_layout.addWidget(QtWidgets.QLabel("Spectra", self))
        series_layout.addWidget(self._series_empty_label)
        series_layout.addWidget(self._series_list, 1)
        series_layout.addWidget(self._remove_button)

        self._follow_check = QtWidgets.QCheckBox("Follow cursor", self)
        self._follow_check.setObjectName("spectrumFollowCheck")
        self._follow_check.setChecked(True)
        self._follow_check.setToolTip("Update the live trace as the cursor moves over any image")
        self._follow_check.toggled.connect(self._on_follow_toggled)

        self._transform_combo = QtWidgets.QComboBox(self)
        self._transform_combo.setObjectName("spectrumTransformCombo")
        self._transform_combo.setToolTip(
            "Derivatives sharpen absorption edges; continuum removal isolates "
            "absorption features; area normalisation removes brightness"
        )
        for transform in SpectrumTransform:
            self._transform_combo.addItem(transform.value, transform)
        self._transform_combo.currentIndexChanged.connect(lambda _index: self._redraw())

        self._spread_check = QtWidgets.QCheckBox("±1σ", self)
        self._spread_check.setObjectName("spectrumSpreadCheck")
        self._spread_check.setChecked(True)
        self._spread_check.setToolTip("Shade one standard deviation around region means")
        self._spread_check.toggled.connect(lambda _checked: self._redraw())

        self._markers_check = QtWidgets.QCheckBox("Index bands", self)
        self._markers_check.setObjectName("spectrumMarkersCheck")
        self._markers_check.setChecked(True)
        self._markers_check.setToolTip(
            "Mark the bands used by the visualization shown in the Visualization tab"
        )
        self._markers_check.toggled.connect(lambda _checked: self._redraw())

        self._pin_button = QtWidgets.QPushButton("Pin cursor", self)
        self._pin_button.setObjectName("spectrumPinButton")
        self._pin_button.setToolTip(
            f"Keep the live spectrum on the plot for comparison. Faster: "
            f"{PIN_SPECTRUM_SHORTCUT} the pixel directly in the image."
        )
        self._pin_button.clicked.connect(self.pin_live)

        self._image_mean_button = QtWidgets.QPushButton("Image mean", self)
        self._image_mean_button.setObjectName("spectrumImageMeanButton")
        self._image_mean_button.setToolTip("Add the mean spectrum of the whole region of interest")
        self._image_mean_button.clicked.connect(self.imageMeanRequested)

        self._clear_button = QtWidgets.QPushButton("Clear", self)
        self._clear_button.setObjectName("spectrumClearButton")
        self._clear_button.setToolTip("Remove every pinned and region spectrum")
        self._clear_button.clicked.connect(self.clear)

        self._export_button = QtWidgets.QPushButton("Export CSV", self)
        self._export_button.setObjectName("spectrumExportButton")
        self._export_button.setToolTip("Save every shown spectrum, as displayed, to a CSV file")
        self._export_button.clicked.connect(self._on_export_clicked)

        self._reference_combo = QtWidgets.QComboBox(self)
        self._reference_combo.setObjectName("spectrumReferenceCombo")
        self._reference_combo.setToolTip("Spectrum to compare every pixel against")
        self._similarity_button = QtWidgets.QPushButton("Map similarity", self)
        self._similarity_button.setObjectName("spectrumSimilarityButton")
        self._similarity_button.setToolTip(
            "Show each pixel's spectral angle to the reference in the "
            "Visualization tab's Similarity mode (brightness-independent)"
        )
        self._similarity_button.clicked.connect(self._on_similarity_clicked)

        self._separability_table = QtWidgets.QTableWidget(self)
        self._separability_table.setObjectName("spectrumSeparabilityTable")
        self._separability_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        separability_hint = QtWidgets.QLabel(
            "Spectral angle (degrees) between region means. Larger angles mean "
            "classes are easier to tell apart; under ~2° they are nearly identical.",
            self,
        )
        separability_hint.setWordWrap(True)
        self._separability_help = QtWidgets.QLabel(_SEPARABILITY_HELP, self)
        self._separability_help.setObjectName("spectrumSeparabilityHelp")
        self._separability_help.setTextFormat(Qt.TextFormat.RichText)
        self._separability_help.setWordWrap(True)
        self._separability_help.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self._separability_help.setContentsMargins(12, 12, 12, 12)
        self._separability_data = QtWidgets.QWidget(self)
        data_layout = QtWidgets.QVBoxLayout(self._separability_data)
        data_layout.setContentsMargins(0, 0, 0, 0)
        data_layout.addWidget(separability_hint)
        data_layout.addWidget(self._separability_table, 1)
        self._separability_stack = QtWidgets.QStackedWidget(self)
        self._separability_stack.setObjectName("spectrumSeparabilityStack")
        self._separability_stack.addWidget(self._separability_help)
        self._separability_stack.addWidget(self._separability_data)
        separability_page = self._separability_stack

        self._tabs = QtWidgets.QTabWidget(self)
        self._tabs.setObjectName("spectrumInspectorTabs")
        plot_page = QtWidgets.QSplitter(Qt.Orientation.Horizontal, self)
        self._plot_splitter = plot_page
        plot_page.setObjectName("spectrumPlotSplitter")
        plot_page.addWidget(self._canvas)
        plot_page.addWidget(series_panel)
        plot_page.setStretchFactor(0, 1)
        plot_page.setSizes([560, 200])
        self._tabs.addTab(plot_page, "Plot")
        self._tabs.addTab(separability_page, "Separability")

        # Grids rather than single rows, so the controls fit a narrow sidebar
        # dock as well as a wide popped-out window.
        display_row = QtWidgets.QGridLayout()
        display_row.setHorizontalSpacing(8)
        display_row.addWidget(self._follow_check, 0, 0)
        display_row.addWidget(self._spread_check, 0, 1)
        display_row.addWidget(self._markers_check, 0, 2)
        display_row.addWidget(QtWidgets.QLabel("Display", self), 1, 0)
        display_row.addWidget(self._transform_combo, 1, 1, 1, 2)
        display_row.setColumnStretch(3, 1)

        action_row = QtWidgets.QGridLayout()
        action_row.setHorizontalSpacing(6)
        for column, button in enumerate(
            (self._pin_button, self._image_mean_button, self._clear_button, self._export_button)
        ):
            action_row.addWidget(button, 0, column)
        self._reference_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._reference_combo.setMinimumContentsLength(12)
        reference_row = QtWidgets.QHBoxLayout()
        reference_row.addWidget(QtWidgets.QLabel("Reference", self))
        reference_row.addWidget(self._reference_combo, 1)
        reference_row.addWidget(self._similarity_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addLayout(display_row)
        layout.addWidget(self._tabs, 1)
        layout.addWidget(self._status_label)
        layout.addLayout(action_row)
        layout.addLayout(reference_row)
        self.setMinimumWidth(340)

        self._refresh_series_list()
        self._refresh_separability()
        self._redraw()

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def is_following(self) -> bool:
        """Whether "Follow cursor" is ticked; the Controller also checks the
        hosting tool window is open."""

        return self._follow_check.isChecked()

    def set_compact(self, compact: bool) -> None:
        """Stack the Spectra list under the plot in a narrow sidebar dock."""

        self._plot_splitter.setOrientation(
            Qt.Orientation.Vertical if compact else Qt.Orientation.Horizontal
        )

    def set_live(self, result: Optional[SpectrumResult], source: str = "") -> None:
        """Replace the live trace with the spectrum under the cursor."""

        self._live = result
        self._live_source = source
        if self._shown_live() is None:
            # Not following: keep the spectrum for when following resumes,
            # but draw no cursor line.
            self._update_pin_button()
            if self._live_line is not None:
                self._redraw()
            return
        self._status_label.setText(
            f"Cursor: row {result.row}, column {result.column}"
            + (f" · {source}" if source else "")
        )
        self._update_pin_button()
        if self._live_line is None or self._needs_full_redraw():
            self._redraw()
            return
        # Fast path for cursor motion: move the existing line only.
        self._live_line.set_data(result.wavelengths_nm, self._transformed(
            result.wavelengths_nm, result.values
        ))
        self._axes.relim()
        self._axes.autoscale_view()
        self._draw()

    def set_status(self, message: str) -> None:
        self._status_label.setText(message)

    def pin_live(self) -> None:
        live = self._shown_live()
        if live is not None:
            self.add_pin(live, self._live_source)

    def add_pin(
        self, result: SpectrumResult, source: str = "", data_source: object = None
    ) -> None:
        """Keep ``result`` on the plot under a ``#N · (row, column)`` label.

        ``data_source`` identifies the cube the pixel came from, so its image
        marker is drawn only on viewers showing that cube.
        """

        number = self._pin_counter + 1
        label = (
            f"#{number} · Pixel ({result.row}, {result.column})"
            + (f" {source}" if source else "")
        )
        color = tuple(_PIN_COLORS[self._pin_counter % len(_PIN_COLORS)])
        key = f"pin:{self._pin_counter}"
        self._pin_counter += 1
        pin = PinnedSpectrum(
            label, result.wavelengths_nm, result.values, color,
            number=number, row=result.row, column=result.column, source=data_source,
        )
        self._series_by_key[key] = _Series(key, color, pin=pin)
        self._selected_key = key
        self._series_changed()

    def move_pin(self, key: str, result: SpectrumResult, *, final: bool = True) -> None:
        """Re-point pin ``key`` at ``result``'s pixel, keeping its number and colour.

        ``final=False`` is a live update during a drag: only the plot is
        redrawn; the list, references and markers refresh on the final move.
        """

        series = self._series_by_key.get(key)
        if series is None or series.pin is None:
            return
        pin = series.pin
        source = pin.label.split(")", 1)[1] if ")" in pin.label else ""
        series.pin = PinnedSpectrum(
            f"#{pin.number} · Pixel ({result.row}, {result.column}){source}",
            result.wavelengths_nm, result.values, pin.color,
            number=pin.number, row=result.row, column=result.column, source=pin.source,
        )
        if final:
            self._selected_key = key
            self._series_changed()
        else:
            self._redraw()

    def pinned_spectra(self) -> tuple[PinnedSpectrum, ...]:
        return tuple(s.pin for s in self._series_by_key.values() if s.pin is not None)

    def pin_keys(self) -> dict[str, PinnedSpectrum]:
        return {key: s.pin for key, s in self._series_by_key.items() if s.pin is not None}

    def pin_markers(self) -> tuple[PinMarker, ...]:
        """Image markers for every *shown* pin, in list order."""

        return tuple(
            PinMarker(
                s.key, s.pin.number, s.pin.row, s.pin.column, s.color,
                s.key == self._selected_key, s.pin.source,
            )
            for s in self._series_by_key.values()
            if s.pin is not None and s.visible
        )

    def set_region_spectra(
        self,
        spectra: Sequence[RegionSpectrum],
        colors: Mapping[str, RGBColor] | None = None,
        *,
        replace_prefix: str | None = None,
    ) -> None:
        """Add (or replace) region mean spectra, keyed by ``RegionSpectrum.key``.

        ``replace_prefix`` first drops existing regions whose key starts with
        it, so re-sending the class spectra replaces rather than accumulates.
        """

        if replace_prefix is not None:
            self._series_by_key = {
                key: series for key, series in self._series_by_key.items()
                if series.region is None or not key.startswith(replace_prefix)
            }
        colors = colors or {}
        for index, spectrum in enumerate(spectra):
            color = colors.get(spectrum.key)
            self._series_by_key[spectrum.key] = _Series(
                spectrum.key,
                _mpl_color(color) if color is not None
                else tuple(_PIN_COLORS[(index + 3) % len(_PIN_COLORS)]),
                region=spectrum,
            )
        self._series_changed()

    def remove_regions_with_prefix(self, prefixes: tuple[str, ...]) -> None:
        """Drop region series whose key starts with any of ``prefixes``."""

        self.remove_series([
            key for key, series in self._series_by_key.items()
            if series.region is not None and key.startswith(prefixes)
        ])

    def region_spectra(self) -> tuple[RegionSpectrum, ...]:
        return tuple(s.region for s in self._series_by_key.values() if s.region is not None)

    def series_labels(self) -> tuple[str, ...]:
        """Labels of every listed spectrum, in list order."""

        return tuple(series.label for series in self._series_by_key.values())

    def set_series_visible(self, key: str, visible: bool) -> None:
        series = self._series_by_key.get(key)
        if series is None or series.visible == visible:
            return
        series.visible = visible
        self._series_changed()

    def select_series(self, key: Optional[str]) -> None:
        """Highlight one listed spectrum (``None`` clears the highlight)."""

        self._selected_key = key if key in self._series_by_key else None
        self._sync_list_selection()
        self._redraw()
        self.pinsChanged.emit()

    def selected_series(self) -> Optional[str]:
        return self._selected_key

    def remove_series(self, keys: Sequence[str]) -> None:
        removed = [key for key in keys if self._series_by_key.pop(key, None) is not None]
        if not removed:
            return
        if self._selected_key in removed:
            self._selected_key = None
        self._series_changed()

    def remove_selected(self) -> None:
        keys = [item.data(_SERIES_KEY_ROLE) for item in self._series_list.selectedItems()]
        if not keys and self._selected_key is not None:
            keys = [self._selected_key]
        self.remove_series(keys)

    def set_band_markers(self, markers: Mapping[str, float], title: str = "") -> None:
        """Mark the wavelengths the active visualization reads."""

        self._markers = dict(markers)
        self._marker_title = title
        self._redraw()

    def clear(self) -> None:
        self._series_by_key.clear()
        self._selected_key = None
        self._series_changed()

    def reset_for_new_image(self) -> None:
        """Drop everything tied to the previous cube's geometry or bands."""

        self._live = None
        self._pin_counter = 0
        self.clear()
        self._status_label.setText(self._idle_hint())

    def current_transform(self) -> SpectrumTransform:
        return self._transform_combo.currentData()

    def set_transform(self, transform: SpectrumTransform) -> None:
        self._transform_combo.setCurrentIndex(self._transform_combo.findData(transform))

    # ------------------------------------------------------------------ #
    # Private                                                            #
    # ------------------------------------------------------------------ #

    def _transformed(self, wavelengths: np.ndarray, values: np.ndarray) -> np.ndarray:
        return self._analysis.transform(wavelengths, values, self.current_transform())

    def _needs_full_redraw(self) -> bool:
        # The live trace's wavelength grid changed (e.g. a different viewer).
        x = self._live_line.get_xdata()
        live = self._shown_live()
        return live is not None and len(x) != len(live.wavelengths_nm)

    @staticmethod
    def _idle_hint() -> str:
        return (
            f"Hover over an image to see its spectrum · {PIN_SPECTRUM_SHORTCUT} "
            "a pixel to pin it"
        )

    def _visible_series(self) -> list[_Series]:
        return [series for series in self._series_by_key.values() if series.visible]

    def _series(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Every shown series, transformed as displayed, keyed by label."""

        series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for item in self._visible_series():
            series[item.label] = (
                item.wavelengths_nm, self._transformed(item.wavelengths_nm, item.values)
            )
        live = self._shown_live()
        if live is not None:
            series["Cursor"] = (
                live.wavelengths_nm, self._transformed(live.wavelengths_nm, live.values),
            )
        return series

    def _series_changed(self) -> None:
        """Rebuild everything derived from the listed series."""

        self._refresh_series_list()
        self._refresh_references()
        self._refresh_separability()
        self._redraw()
        self.pinsChanged.emit()

    def _redraw(self) -> None:
        axes = self._axes
        axes.clear()
        self._lines = {}
        transform = self.current_transform()
        show_spread = self._spread_check.isChecked() and transform is SpectrumTransform.NONE
        selected = self._selected_key
        for series in self._visible_series():
            x = series.wavelengths_nm
            highlighted = series.key == selected
            dimmed = selected is not None and not highlighted
            is_region = series.region is not None
            (line,) = axes.plot(
                x, self._transformed(x, series.values), color=series.color,
                linewidth=(3.0 if highlighted else 1.6 if is_region else 1.2),
                alpha=0.35 if dimmed else 1.0, label=series.legend_label,
                zorder=3 if highlighted else 2, picker=True, pickradius=5,
            )
            self._lines[line] = series.key
            if is_region and show_spread:
                spectrum = series.region
                axes.fill_between(
                    x, spectrum.mean - spectrum.std, spectrum.mean + spectrum.std,
                    color=series.color, alpha=0.06 if dimmed else 0.15, linewidth=0,
                )
        self._live_line = None
        live = self._shown_live()
        if live is not None:
            (self._live_line,) = axes.plot(
                live.wavelengths_nm,
                self._transformed(live.wavelengths_nm, live.values),
                color=_LIVE_COLOR, linewidth=1.0, linestyle="--", label="Cursor",
            )
        if self._markers_check.isChecked() and self._markers:
            for name, wavelength in self._markers.items():
                axes.axvline(wavelength, color=_MARKER_COLOR, linewidth=0.8, linestyle=":")
                axes.annotate(
                    f"{name} {wavelength:g}", xy=(wavelength, 1.0),
                    xycoords=("data", "axes fraction"), xytext=(2, -10),
                    textcoords="offset points", fontsize=7, color=_MARKER_COLOR,
                    rotation=90, va="top",
                )
        axes.set_xlabel("Wavelength (nm)")
        axes.set_ylabel(
            "Value" if transform is SpectrumTransform.NONE else transform.value
        )
        if self._marker_title and self._markers_check.isChecked() and self._markers:
            axes.set_title(f"Bands: {self._marker_title}", fontsize=9)
        axes.grid(True, alpha=0.3)
        if axes.get_legend_handles_labels()[0]:
            axes.legend(fontsize=7, loc="best")
        else:
            axes.text(
                0.5, 0.5,
                f"Hover over an image to see its spectrum\n"
                f"{PIN_SPECTRUM_SHORTCUT} a pixel to pin it for comparison",
                transform=axes.transAxes, ha="center", va="center", color=_MARKER_COLOR,
            )
        self._fit_axes()
        self._draw()
        self._export_button.setEnabled(bool(self._visible_series() or live))

    def _draw(self) -> None:
        """Queue a repaint, skipped while the canvas is too small to hold axes."""

        if self._canvas.width() >= _MIN_CANVAS_WIDTH and self._canvas.height() >= _MIN_CANVAS_HEIGHT:
            self._canvas.draw_idle()

    def _fit_axes(self) -> None:
        """Place the axes with fixed pixel margins for the labels.

        Pixel (not fractional) margins keep tick labels and the title readable
        at every size; the margins are clamped so the axes never invert.
        """

        width = max(self._canvas.width(), 1)
        height = max(self._canvas.height(), 1)
        left, right = 56 / width, 1 - 12 / width
        bottom = 44 / height
        top = 1 - (28 if self._axes.get_title() else 12) / height
        if right - left < 0.1 or top - bottom < 0.1:
            return  # Too small to lay out; _draw skips painting too.
        self._figure.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    def _refresh_series_list(self) -> None:
        widget = self._series_list
        widget.blockSignals(True)
        widget.clear()
        for series in self._series_by_key.values():
            item = QtWidgets.QListWidgetItem(series.legend_label)
            item.setData(_SERIES_KEY_ROLE, series.key)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable
            )
            item.setCheckState(
                Qt.CheckState.Checked if series.visible else Qt.CheckState.Unchecked
            )
            swatch = QtGui.QPixmap(12, 12)
            swatch.fill(QtGui.QColor.fromRgbF(*series.color))
            item.setIcon(QtGui.QIcon(swatch))
            widget.addItem(item)
        widget.blockSignals(False)
        self._sync_list_selection()
        has_series = bool(self._series_by_key)
        self._series_empty_label.setVisible(not has_series)
        self._series_list.setVisible(has_series)
        self._clear_button.setEnabled(has_series)

    def _sync_list_selection(self) -> None:
        widget = self._series_list
        widget.blockSignals(True)
        for row in range(widget.count()):
            item = widget.item(row)
            item.setSelected(item.data(_SERIES_KEY_ROLE) == self._selected_key)
        widget.blockSignals(False)
        self._remove_button.setEnabled(self._selected_key is not None)

    def _on_series_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        self.set_series_visible(
            item.data(_SERIES_KEY_ROLE), item.checkState() == Qt.CheckState.Checked
        )

    def _on_series_selection_changed(self) -> None:
        items = self._series_list.selectedItems()
        self._remove_button.setEnabled(bool(items))
        # Highlight the most recently selected spectrum on the plot.
        key = items[-1].data(_SERIES_KEY_ROLE) if items else None
        if key != self._selected_key:
            self._selected_key = key
            self._redraw()
            self.pinsChanged.emit()

    def _on_line_picked(self, event) -> None:
        key = self._lines.get(event.artist)
        if key is not None:
            self.select_series(None if key == self._selected_key else key)

    def _refresh_references(self) -> None:
        current = self._reference_combo.currentText()
        self._reference_combo.clear()
        for series in self._series_by_key.values():
            self._reference_combo.addItem(series.label, series.values)
        index = self._reference_combo.findText(current)
        if index >= 0:
            self._reference_combo.setCurrentIndex(index)
        self._similarity_button.setEnabled(self._reference_combo.count() > 0)

    def _refresh_separability(self) -> None:
        spectra = self.region_spectra()
        table = self._separability_table
        table.clear()
        table.setRowCount(len(spectra))
        table.setColumnCount(len(spectra))
        labels = [spectrum.label for spectrum in spectra]
        table.setHorizontalHeaderLabels(labels)
        table.setVerticalHeaderLabels(labels)
        self._separability_stack.setCurrentWidget(
            self._separability_data if len(spectra) >= 2 else self._separability_help
        )
        if len(spectra) < 2:
            return
        matrix = np.degrees(self._analysis.separability(spectra))
        for row in range(len(spectra)):
            for column in range(len(spectra)):
                item = QtWidgets.QTableWidgetItem(
                    "—" if row == column else f"{matrix[row, column]:.2f}"
                )
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row, column, item)
        table.resizeColumnsToContents()

    def _shown_live(self) -> Optional[SpectrumResult]:
        """The cursor spectrum to draw/export/pin: none while not following."""

        return self._live if self._follow_check.isChecked() else None

    def _update_pin_button(self) -> None:
        self._pin_button.setEnabled(self._shown_live() is not None)

    def _on_follow_toggled(self, checked: bool) -> None:
        if checked:
            live = self._live
            self._status_label.setText(
                f"Cursor: row {live.row}, column {live.column}"
                + (f" · {self._live_source}" if self._live_source else "")
                if live is not None else self._idle_hint()
            )
        else:
            self._status_label.setText(
                "Cursor line hidden; tick Follow cursor to show it again"
            )
        self._update_pin_button()
        self._redraw()

    def _on_similarity_clicked(self) -> None:
        index = self._reference_combo.currentIndex()
        if index < 0:
            return
        self.similarityRequested.emit(
            self._reference_combo.itemText(index), self._reference_combo.itemData(index)
        )

    def _on_export_clicked(self) -> None:
        series = self._series()
        if not series:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Spectra", "spectra.csv", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        # Columns share one wavelength axis; skip any series on another grid
        # (e.g. a pin taken before switching to a cube with other bands).
        wavelengths = next(iter(series.values()))[0]
        shared = {
            name: values for name, (x, values) in series.items()
            if len(x) == len(wavelengths) and np.allclose(x, wavelengths)
        }
        try:
            self._analysis.export_csv(path, wavelengths, shared)
        except VisualizationError as exc:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))
            return
        skipped = len(series) - len(shared)
        self._status_label.setText(
            f"Exported {len(shared)} spectra to {path}"
            + (f" ({skipped} on a different band grid skipped)" if skipped else "")
        )
