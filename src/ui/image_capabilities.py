"""Metadata-only camera capability updates for visualization and SR controls."""

from collections.abc import Iterable

from PyQt6.QtWidgets import QAbstractButton, QPushButton

from core import HSIData, SuperResolutionService, VisualizationMode, VisualizationService


class ImageCapabilityController:
    """Keep camera-dependent UI state outside the main window."""

    def __init__(self, visualization_service: VisualizationService,
                 sr_service: SuperResolutionService, sr_button: QPushButton) -> None:
        self._visualization = visualization_service
        self._sr_service = sr_service
        self._sr_button = sr_button

    def refresh_visualizations(
        self, data: HSIData, mode_buttons: Iterable[tuple[QAbstractButton, VisualizationMode]]
    ) -> None:
        for button, mode in mode_buttons:
            reason = self._visualization.unavailable_reason(data, mode)
            button.setEnabled(reason is None)
            button.setToolTip(reason or (
                "RGB preview (first/middle/last bands if visible RGB is unavailable)"
                if mode is VisualizationMode.RGB else f"View {mode.value}"
            ))

    def refresh_super_resolution(self, data: HSIData) -> None:
        reason = self._sr_service.compatibility_error(data)
        self._sr_button.setEnabled(reason is None)
        self._sr_button.setText(
            "Image incompatible with SR" if data.is_loaded() and reason
            else "Run Super-Resolution"
        )
        self._sr_button.setToolTip(reason or
                                  "Run the 480-band MSDformer model at 2× spatial resolution")
