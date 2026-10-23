"""Controller tying the live Spectrum Inspector to every image viewer.

Owns the :class:`SpectrumInspector` window and the cube reads behind it:

* cursor hover in any viewer feeds the live trace (one read per new pixel);
* Ctrl+Click, or the viewer's right-click "Spectrum Analyzer → Pin Spectrum
  Here", pins that pixel and opens the window;
* per-class and whole-image mean spectra, and Spectral Angle Mapper maps,
  are computed with :class:`core.SpectralAnalysisService` on request.

Every read goes through ``can_read`` first, so nothing touches the cube while
SR, calibration, classification or a Hypercube build hold it.
"""

from __future__ import annotations

import logging
from typing import Callable, Mapping, Optional

import numpy as np
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt

from core import (
    HSIData,
    HSIError,
    RegionSpectrum,
    SimilarityResult,
    SpectralAnalysisService,
    SpectrumResult,
    VisualizationResult,
    VisualizationService,
)
from ui.spectrum_inspector import SpectrumInspector
from ui.viewer import HSIViewer, PinMarkerSpec

LOGGER = logging.getLogger(__name__)

RGBColor = tuple[int, int, int]
# (data, {key: region}) for classes and annotation regions; each region has
# .label, .mask, .color and .selected (see classification_controller).
AnalysisRegions = tuple[HSIData, Mapping[str, object]]


class SpectrumController(QtCore.QObject):
    """Route viewer hover/pin events and analysis requests to the inspector."""

    # Emitted after a new similarity map is ready for the Visualization tab.
    similarityReady = QtCore.pyqtSignal()

    def __init__(
        self,
        *,
        visualization_service: VisualizationService,
        analysis_service: SpectralAnalysisService,
        data_for_viewer: Callable[[HSIViewer], Optional[HSIData]],
        display_data: Callable[[], HSIData],
        blocked_reason: Callable[[], Optional[str]],
        hypercube_busy: Callable[[], bool],
        stop_hypercube: Callable[[], None],
        resume_hypercube: Callable[[], None],
        analysis_regions: Callable[[], Optional[AnalysisRegions]],
        viewer_names: Mapping[HSIViewer, str],
        statusbar: QtWidgets.QStatusBar,
        parent_widget: QtWidgets.QWidget,
    ) -> None:
        super().__init__(parent_widget)
        self._visualization = visualization_service
        self._analysis = analysis_service
        self._data_for_viewer = data_for_viewer
        self._display_data = display_data
        self._blocked_reason = blocked_reason
        self._hypercube_busy = hypercube_busy
        self._stop_hypercube = stop_hypercube
        self._resume_hypercube = resume_hypercube
        self._analysis_regions = analysis_regions
        self._viewer_names = dict(viewer_names)
        self._statusbar = statusbar
        self._parent = parent_widget
        self._similarity: Optional[tuple[HSIData, SimilarityResult]] = None
        self._tool_window = None

        self.inspector = SpectrumInspector(analysis_service, parent_widget)
        self.inspector.similarityRequested.connect(self._on_similarity_requested)
        self.inspector.imageMeanRequested.connect(self.show_image_mean)
        self.inspector.pinsChanged.connect(self.refresh_markers)

        for viewer in self._viewer_names:
            viewer.pixelHovered.connect(
                lambda row, column, viewer=viewer: self._on_pixel_hovered(viewer, row, column)
            )
            viewer.pinDragged.connect(
                lambda key, row, column, viewer=viewer:
                    self._move_pin(viewer, key, row, column, final=False)
            )
            viewer.pinMoved.connect(
                lambda key, row, column, viewer=viewer:
                    self._move_pin(viewer, key, row, column, final=True)
            )
            viewer.pinPicked.connect(self.inspector.select_series)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def set_tool_window(self, tool_window) -> None:
        """Host the inspector in ``tool_window`` (a :class:`ToolWindow`)."""

        self._tool_window = tool_window

    def is_open(self) -> bool:
        """Whether the inspector's tool window is open (docked or popped out)."""

        if self._tool_window is not None:
            return self._tool_window.is_open()
        return self.inspector.isVisible()

    def show(self) -> None:
        if self._tool_window is not None:
            self._tool_window.show()
        else:
            self.inspector.show()

    def pin(self, viewer: HSIViewer, row: int, column: int) -> Optional[SpectrumResult]:
        """Read, pin and show the spectrum at ``(row, column)`` in ``viewer``.

        A pin is an explicit request, so unlike hover it waits for any
        Hypercube build rather than skipping the read.
        """

        reason = self._blocked_reason()
        if reason is not None:
            self._statusbar.showMessage(reason, 5000)
            return None
        data = self._data_for_viewer(viewer)
        if data is None or not (0 <= row < data.rows and 0 <= column < data.columns):
            return None
        # A hypercube worker still in flight reads the same underlying
        # SpyFile handle; serialize this read behind it rather than risking
        # a concurrent read of a non-thread-safe file object.
        self._stop_hypercube()
        try:
            result = self._visualization.spectrum(data, row, column)
        except HSIError as exc:
            QtWidgets.QMessageBox.critical(self._parent, "Unable to plot spectrum", str(exc))
            return None
        finally:
            self._resume_hypercube()
        self.inspector.add_pin(result, self._viewer_names.get(viewer, ""), data)
        self.inspector.set_live(result, self._viewer_names.get(viewer, ""))
        self.show()
        self._statusbar.showMessage(
            f"Pinned pixel ({row}, {column}) in the Spectrum Inspector "
            "· Ctrl+Click another pixel to compare",
            5000,
        )
        return result

    def refresh_markers(self) -> None:
        """Draw each shown pin's circle on every viewer displaying its cube.

        Call after any viewer's displayed cube may have changed (resolution or
        calibration toggles, crops); pins from another cube are not drawn, so
        a low-resolution pin never lands on the 2x Super-Resolution image.
        """

        markers = self.inspector.pin_markers()
        for viewer in self._viewer_names:
            data = self._data_for_viewer(viewer)
            viewer.set_pin_markers(tuple(
                PinMarkerSpec(
                    marker.row, marker.column, marker.number,
                    tuple(round(channel * 255) for channel in marker.color),
                    marker.selected, marker.key,
                )
                for marker in markers
                if data is not None and marker.source is data
            ))

    def set_visualization(self, result: Optional[VisualizationResult]) -> None:
        """Mark the bands read by the visualization now shown."""

        if result is None:
            self.inspector.set_band_markers({})
            return
        self.inspector.set_band_markers(dict(result.band_wavelengths_nm), result.title)

    def show_class_spectra(self) -> None:
        """Plot the mean ±1σ spectrum of every *selected* class and region.

        Classification classes and measured annotation regions (rectangles,
        ellipses) are treated alike; their rows' checkboxes choose which are
        plotted, and re-plotting replaces the previous set.
        """

        summary = self._analysis_regions()
        if summary is None:
            self._statusbar.showMessage(
                "Run a classification or draw a measured rectangle/ellipse first", 5000
            )
            return
        data, regions = summary
        selected = {key: region for key, region in regions.items() if region.selected}
        if not selected:
            self._statusbar.showMessage(
                "Nothing is selected; tick the classes or regions to plot in the layer panel",
                5000,
            )
            return
        spectra = self._compute_regions(data, selected)
        if spectra is None:
            return
        self.inspector.remove_regions_with_prefix(("class:", "annotation:"))
        self.inspector.set_region_spectra(
            spectra, {key: region.color for key, region in selected.items()}
        )
        self.show()
        self._statusbar.showMessage(
            f"Plotted {len(spectra)} selected class and region spectra", 5000
        )

    def inspect_region(self, key: str) -> None:
        """Add one class or region (``"class:3"``, ``"annotation:7"``) to the plot."""

        summary = self._analysis_regions()
        if summary is None or key not in summary[1]:
            return
        data, regions = summary
        region = regions[key]
        spectra = self._compute_regions(data, {key: region})
        if not spectra:
            self._statusbar.showMessage(f"{region.label} has no pixels to inspect", 5000)
            return
        self.inspector.set_region_spectra(spectra, {key: region.color})
        self.inspector.select_series(key)
        self.show()
        self._statusbar.showMessage(f"Added {region.label} to the Spectrum Inspector", 5000)

    def _compute_regions(self, data: HSIData, regions: Mapping[str, object]):
        return self._run_full_cube(
            "Computing spectra…",
            lambda: self._analysis.masked_spectra(
                data, {key: (region.label, region.mask) for key, region in regions.items()}
            ),
        )

    def show_image_mean(self) -> None:
        data = self._display_data()
        if not data.is_loaded():
            return
        spectrum = self._run_full_cube(
            "Computing image mean spectrum…", lambda: self._analysis.image_spectrum(data)
        )
        if spectrum is not None:
            self.inspector.set_region_spectra((spectrum,))
            self.show()

    def similarity_for(self, data: HSIData) -> Optional[SimilarityResult]:
        """Return the similarity map computed from ``data``, if still current."""

        if self._similarity is None or self._similarity[0] is not data:
            return None
        return self._similarity[1]

    def similarity_at(self, data: HSIData, row: int, column: int) -> Optional[float]:
        """Return the spectral angle (degrees) at a pixel of the current map."""

        similarity = self.similarity_for(data)
        if similarity is None:
            return None
        height, width = similarity.angles.shape
        if not (0 <= row < height and 0 <= column < width):
            return None
        angle = float(similarity.angles[row, column])
        return float(np.degrees(angle)) if np.isfinite(angle) else None

    def image_changed(self) -> None:
        """Drop spectra and maps tied to the previous cube."""

        self._similarity = None
        self.inspector.reset_for_new_image()

    def geometry_changed(self) -> None:
        """Drop the similarity map after a crop; ``HSIData`` is updated in place,
        so identity alone can't tell the map is stale."""

        self._similarity = None
        # HSIData is cropped in place, so a pin's source identity survives a
        # crop while its coordinates no longer match: drop the pins.
        self.inspector.remove_series(list(self.inspector.pin_keys()))

    def shutdown(self) -> None:
        if self._tool_window is not None:
            self._tool_window.hide()

    # ------------------------------------------------------------------ #
    # Private                                                            #
    # ------------------------------------------------------------------ #

    def _move_pin(
        self, viewer: HSIViewer, key: str, row: int, column: int, *, final: bool
    ) -> None:
        """Re-read a dragged pin's spectrum at its new pixel.

        Live drag updates skip the read while a worker holds the cube; the
        final move waits for a Hypercube build, like any explicit pin.
        """

        if self._blocked_reason() is not None:
            if final:
                self._statusbar.showMessage(self._blocked_reason(), 5000)
                self.refresh_markers()  # snap the marker back
            return
        if not final and self._hypercube_busy():
            return
        data = self._data_for_viewer(viewer)
        if data is None or not (0 <= row < data.rows and 0 <= column < data.columns):
            return
        if final:
            self._stop_hypercube()
        try:
            result = self._visualization.spectrum(data, row, column)
        except HSIError as exc:
            LOGGER.info("Pin move read failed: %s", exc)
            return
        finally:
            if final:
                self._resume_hypercube()
        self.inspector.move_pin(key, result, final=final)
        if final:
            self._statusbar.showMessage(f"Moved pin to pixel ({row}, {column})", 4000)

    def _on_pixel_hovered(self, viewer: HSIViewer, row: int, column: int) -> None:
        if not (self.is_open() and self.inspector.is_following()):
            return
        if self._blocked_reason() is not None:
            return
        if self._hypercube_busy():
            # Hover must never stall the cursor waiting on a worker.
            self.inspector.set_status("Live trace paused while the hypercube preview builds")
            return
        data = self._data_for_viewer(viewer)
        if data is None or not (0 <= row < data.rows and 0 <= column < data.columns):
            return
        try:
            result = self._visualization.spectrum(data, row, column)
        except HSIError as exc:
            LOGGER.info("Live spectrum read failed: %s", exc)
            return
        source = self._viewer_names.get(viewer, "")
        angle = self.similarity_at(data, row, column)
        if angle is not None:
            source += f" · {angle:.2f}° to {self._similarity[1].reference_label}"
        self.inspector.set_live(result, source)

    def _on_similarity_requested(self, label: str, reference: object) -> None:
        data = self._display_data()
        if not data.is_loaded():
            return
        values = np.asarray(reference)
        if values.size != data.bands:
            QtWidgets.QMessageBox.information(
                self._parent,
                "Reference unavailable",
                f"“{label}” has {values.size} bands but the shown image has "
                f"{data.bands}; pin a spectrum from this image instead.",
            )
            return
        result = self._run_full_cube(
            f"Mapping similarity to {label}…",
            lambda: self._analysis.spectral_angle_map(data, values, label),
        )
        if result is None:
            return
        self._similarity = (data, result)
        low, high = (np.degrees(limit) for limit in result.display_limits)
        self._statusbar.showMessage(
            f"Similarity to {label}: bright = similar ({low:.1f}°–{high:.1f}° shown)", 8000
        )
        self.similarityReady.emit()

    def _run_full_cube(self, message: str, compute: Callable[[], object]):
        """Run a whole-cube read with the GUI's guards, cursor and errors."""

        reason = self._blocked_reason()
        if reason is not None:
            self._statusbar.showMessage(reason, 5000)
            return None
        self._stop_hypercube()
        self._statusbar.showMessage(message)
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            return compute()
        except HSIError as exc:
            QtWidgets.QMessageBox.critical(self._parent, "Spectral analysis failed", str(exc))
            return None
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._statusbar.clearMessage()
            self._resume_hypercube()
