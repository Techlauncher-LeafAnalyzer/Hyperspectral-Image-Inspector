"""Metadata-only camera capability updates for visualization and SR controls."""

from collections.abc import Iterable

from PyQt6.QtWidgets import QAbstractButton, QPushButton

from core import (
    HSIData,
    OPTIONAL_VISUALIZATION_MODES,
    SuperResolutionService,
    VisualizationMode,
    VisualizationService,
)


_MODE_TOOLTIPS = {
    VisualizationMode.RGB: "RGB preview (false-colour bands if visible RGB is unavailable)",
    VisualizationMode.CUSTOM: "Your own index formula built from any bands; use ✎ Formula to change it",
    VisualizationMode.BRIGHTNESS: (
        "Spectral brightness: every pixel's spectrum summed over all bands and scaled 0–1"
    ),
}


class ImageCapabilityController:
    """Keep camera-dependent UI state outside the main window."""

    def __init__(self, visualization_service: VisualizationService,
                 sr_service: SuperResolutionService, sr_button: QPushButton) -> None:
        self._visualization = visualization_service
        self._sr_service = sr_service
        self._sr_button = sr_button

    def refresh_visualizations(
        self, data: HSIData, mode_buttons: Iterable[tuple[QAbstractButton, VisualizationMode]]
    ) -> dict[VisualizationMode, str | None]:
        """Update mode button state and return each mode's availability.

        Callers that also need per-mode availability (e.g. to gate rendering)
        should reuse the returned mapping instead of calling
        ``unavailable_reason`` again for the same data/mode pairs.
        """
        reasons: dict[VisualizationMode, str | None] = {}
        for button, mode in mode_buttons:
            reason = self._visualization.unavailable_reason(data, mode)
            reasons[mode] = reason
            button.setEnabled(reason is None)
            if mode in OPTIONAL_VISUALIZATION_MODES:
                button.setVisible(reason is None)
            button.setToolTip(reason or _MODE_TOOLTIPS.get(mode, f"View {mode.value}"))
        return reasons

    def refresh_super_resolution(self, data: HSIData) -> None:
        reason = self._sr_service.compatibility_error(data)
        self._sr_button.setEnabled(reason is None)
        self._sr_button.setText(
            "Image incompatible with SR" if data.is_loaded() and reason
            else "Run Super-Resolution"
        )
        self._sr_button.setToolTip(reason or
                                  "Run the 480-band MSDformer model at 2× spatial resolution")
