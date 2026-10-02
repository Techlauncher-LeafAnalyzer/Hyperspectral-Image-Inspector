"""Compact, nonblocking system-resource monitor for the application header."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import psutil
from PyQt6 import QtCore, QtGui, QtWidgets


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResourceUsage:
    cpu_percent: float
    program_ram_used: int
    system_ram_used: int
    system_ram_total: int


class SystemUsageSampler:
    """Read system CPU/RAM and this process's resident memory without sleeping."""

    def __init__(self) -> None:
        self._process = psutil.Process()

    def prime(self) -> None:
        # Discard psutil's first CPU value; the next read measures since this one.
        psutil.cpu_percent(interval=None)

    def sample(self) -> ResourceUsage:
        cpu = psutil.cpu_percent(interval=None)
        program_ram = self._process.memory_info().rss
        memory = psutil.virtual_memory()
        return ResourceUsage(cpu, program_ram, memory.total - memory.available, memory.total)


class ResourceUsageWidget(QtWidgets.QWidget):
    """System CPU percentage and program RAM in MB, sampled while visible."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        sampler: SystemUsageSampler | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("resourceUsageWidget")
        self.setAccessibleName("System CPU and program RAM usage")
        self._sampler = sampler if sampler is not None else SystemUsageSampler()
        self.cpu_label = QtWidgets.QLabel("CPU —", self)
        self.ram_label = QtWidgets.QLabel("RAM —", self)
        self.cpu_label.setObjectName("cpuUsageLabel")
        self.ram_label.setObjectName("ramUsageLabel")
        self.ram_label.setAccessibleName("Program RAM usage")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(6)
        self.cpu_label.setFixedWidth(64)
        self.ram_label.setFixedWidth(112)
        for label in (self.cpu_label, self.ram_label):
            label.setFixedHeight(28)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
        self.setFixedSize(198, 40)
        self.setStyleSheet(
            "QWidget#resourceUsageWidget { background: transparent; }"
            "QLabel { color: #49635d; background: #e2ece9; border: 1px solid #cdded8; "
            "border-radius: 6px; padding: 4px 2px; font-size: 12px; font-weight: 600; }"
        )
        self.cpu_label.setToolTip("System-wide CPU usage; waiting for the first sample")
        self.ram_label.setToolTip("Program RAM usage; waiting for the first sample")
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self.refresh)

    def sizeHint(self) -> QtCore.QSize:
        # QTabWidget places corner widgets using their hint, not their fixed size.
        return self.size()

    @QtCore.pyqtSlot()
    def refresh(self) -> None:
        try:
            usage = self._sampler.sample()
        except (OSError, RuntimeError, psutil.Error):
            self._show_unavailable()
            return
        self.cpu_label.setText(f"CPU {usage.cpu_percent:.0f}%")
        program_mb = usage.program_ram_used / 1_000_000
        system_mb = usage.system_ram_used / 1_000_000
        total_mb = usage.system_ram_total / 1_000_000
        self.ram_label.setText(f"RAM {program_mb:.0f} MB")
        self.cpu_label.setToolTip(f"System-wide CPU usage: {usage.cpu_percent:.1f}%")
        self.ram_label.setToolTip(
            f"Program RAM (resident memory): {program_mb:.0f} MB\n"
            f"System-wide RAM: {system_mb:.0f} MB in use / {total_mb:.0f} MB total"
        )

    def _show_unavailable(self) -> None:
        LOGGER.debug("System resource usage unavailable", exc_info=True)
        for label, resource, scope in (
            (self.cpu_label, "CPU", "System-wide"),
            (self.ram_label, "RAM", "Program"),
        ):
            label.setText(f"{resource} —")
            label.setToolTip(f"{scope} {resource} usage unavailable; retrying shortly")

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        try:
            self._sampler.prime()
        except (OSError, RuntimeError, psutil.Error):
            self._show_unavailable()
        self._timer.start()

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        self.stop()
        super().hideEvent(event)

    def stop(self) -> None:
        self._timer.stop()
