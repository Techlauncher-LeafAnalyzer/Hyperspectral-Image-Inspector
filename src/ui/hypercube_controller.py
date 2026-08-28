from __future__ import annotations

import dataclasses
import logging
from typing import Optional

from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QRadioButton, QStackedWidget, QStatusBar

from core import HSIData, HypercubeViewData, VisualizationService
from ui.hypercube_widget import HypercubeWidget
from ui.hypercube_worker import HypercubeWorker

LOGGER = logging.getLogger(__name__)


class HypercubeController(QObject):
    """Owns the HyperCube mode's worker lifecycle and display state.

    Wires the mode-select radio button, the visualization tab's stacked
    viewer/cube pages, and the cube widget itself to a background
    ``HypercubeWorker``. Kept separate from ``MainWindowController`` so all
    hypercube-specific state (generation counter, cached result/error,
    worker lifecycle) lives in one place rather than mixed into the rest of
    the application's wiring.
    """

    def __init__(
        self,
        mode_button: QRadioButton,
        stack: QStackedWidget,
        widget: HypercubeWidget,
        statusbar: QStatusBar,
        service: VisualizationService,
    ) -> None:
        super().__init__()
        self._mode_button = mode_button
        self._stack = stack
        self._widget = widget
        self._statusbar = statusbar
        self._service = service

        self._view_data: Optional[HypercubeViewData] = None
        self._error: Optional[str] = None
        self._worker: Optional[HypercubeWorker] = None
        self._generation = 0
        self._loaded = False

        mode_button.toggled.connect(self._on_mode_toggled)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def refresh(self, hsi_data: HSIData) -> None:
        """Recompute the cube for the current dataset.

        Call after every load and every crop. Cancels any worker still in
        flight (bounded by a blocking wait, so a rapid second load/crop can't
        leave two readers on the same underlying SpyFile handle --
        ``HSIData.crop()`` wraps rather than replaces ``spectral_obj``)
        before starting a fresh one on a decoupled snapshot of the current
        field values.
        """
        self._loaded = hsi_data.is_loaded()
        self._view_data = None
        self._error = None
        self.stop_and_wait()

        if not self._loaded:
            self._refresh_display()
            return

        self._generation += 1
        generation = self._generation
        snapshot = dataclasses.replace(hsi_data)
        worker = HypercubeWorker(self._service, snapshot)
        worker.progress.connect(
            lambda value, message, gen=generation: self._on_progress(gen, value, message)
        )
        worker.finished.connect(
            lambda result, gen=generation: self._on_finished(gen, result)
        )
        worker.failed.connect(
            lambda message, gen=generation: self._on_failed(gen, message)
        )
        self._worker = worker
        if self._mode_button.isChecked():
            self._refresh_display()
        worker.start()

    def stop_and_wait(self) -> None:
        """Cancel and block until any in-flight worker's thread has stopped.

        Call before any other synchronous read of the same SpyFile handle
        (e.g. a spectrum-plot click) so it can't race the worker's read.
        """
        worker = self._worker
        self._worker = None
        if worker is None:
            return
        # Ignore progress/results already queued by the cancelled reader.
        self._generation += 1
        try:
            worker.cancel()
            worker._thread.wait()
        except RuntimeError:
            pass  # The worker already finished and deleted itself.

    def resume(self, hsi_data: HSIData) -> None:
        """Resume an interrupted build after another feature releases the source.

        Cached results/errors remain valid when the source has not changed;
        only an unfinished build needs to be restarted.
        """
        if self._worker is None and self._view_data is None and self._error is None:
            self.refresh(hsi_data)

    def shutdown(self) -> None:
        """Detach any in-flight worker before the owning window closes."""
        self.stop_and_wait()

    # ------------------------------------------------------------------ #
    # Private: signal handlers                                            #
    # ------------------------------------------------------------------ #

    def _on_mode_toggled(self, checked: bool) -> None:
        if not checked:
            return
        self._stack.setCurrentWidget(self._widget)
        self._refresh_display()

    def _on_progress(self, generation: int, _value: int, message: str) -> None:
        if generation != self._generation:
            return
        self._statusbar.showMessage(f"Hypercube: {message}")

    def _on_finished(self, generation: int, result: HypercubeViewData) -> None:
        if generation != self._generation:
            return
        self._view_data = result
        self._error = None
        if self._mode_button.isChecked():
            self._refresh_display()

    def _on_failed(self, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        self._view_data = None
        self._error = message
        LOGGER.info("Hypercube unavailable: %s", message)
        if self._mode_button.isChecked():
            self._refresh_display()

    def _refresh_display(self) -> None:
        if self._error is not None:
            self._widget.set_data(None)
            self._widget.set_status_message(self._error)
        elif self._view_data is not None:
            self._widget.set_data(self._view_data)
            self._widget.set_status_message(None)
        elif not self._loaded:
            self._widget.set_data(None)
            self._widget.set_status_message("Load an image to view its hypercube.")
        else:
            self._widget.set_data(None)
            self._widget.set_status_message("Computing hypercube…")
