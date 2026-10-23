"""Resolution and calibration choices, shown as left-stripe toggle icons.

Both choices are global -- every tab shows the same dataset -- so each is a
single :class:`ui.tool_windows.ViewToggleButton` in the left tool-window
stripe, rather than a control repeated on every page:

* Super-Resolution: highlighted while viewing the 2x result (High Res);
* Calibration: highlighted while viewing calibrated reflectance (After).

A button is gray while its comparison does not exist yet.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6 import QtCore

from ui.tool_windows import ViewToggleButton


class ResolutionSwitchGroup(QtCore.QObject):
    """Keep the stripe's resolution/calibration toggles in step with the app.

    ``switches`` and ``calibration_switches`` are one-element tuples, kept as
    tuples so callers can treat "every control showing this choice" alike.
    """

    def __init__(
        self,
        resolution_button: ViewToggleButton,
        calibration_button: ViewToggleButton,
        on_clicked: Callable[[bool], None],
        parent: QtCore.QObject,
        *,
        on_calibration_clicked: Callable[[bool], None],
    ) -> None:
        super().__init__(parent)
        self.switches = (resolution_button,)
        self.calibration_switches = (calibration_button,)
        resolution_button.clicked.connect(on_clicked)
        calibration_button.clicked.connect(on_calibration_clicked)

    def stop(self) -> None:
        """Nothing animates any more; kept for the window's shutdown path."""

    def sync(self, *, available: bool, high_resolution: bool, enabled: bool) -> None:
        for switch in self.switches:
            switch.set_state(
                checked=high_resolution and available, available=available, enabled=enabled,
            )

    def sync_calibration(
        self, *, available: bool, calibrated: bool, enabled: bool,
        current_available: bool,
    ) -> None:
        reason = (
            "No calibrated result at this resolution. Run Calibration to compare."
            if available and not current_available else None
        )
        for switch in self.calibration_switches:
            switch.set_state(
                checked=calibrated, available=available and current_available,
                enabled=enabled, reason=reason,
            )

    def set_enabled(self, enabled: bool) -> None:
        """Lock both toggles while a worker holds the cube."""

        for switch in self.switches + self.calibration_switches:
            if not enabled:
                switch.setEnabled(False)
