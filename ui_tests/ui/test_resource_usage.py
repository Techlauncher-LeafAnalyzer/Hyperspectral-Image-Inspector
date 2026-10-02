from types import SimpleNamespace

import pytest
from PyQt6 import QtCore

from ui import resource_usage
from ui.resource_usage import ResourceUsage, ResourceUsageWidget, SystemUsageSampler


class FakeSampler:
    def __init__(self):
        self.primed = 0
        self.samples = 0
        self.fail = False

    def prime(self):
        self.primed += 1
        if self.fail:
            raise OSError("Monitoring unavailable")

    def sample(self):
        self.samples += 1
        if self.fail:
            raise OSError("Monitoring unavailable")
        return ResourceUsage(23.4, 512_000_000, 10 * 1024 ** 3, 16 * 1024 ** 3)


def test_sampler_uses_system_cpu_and_current_process_rss(monkeypatch):
    intervals = []
    process_calls = []
    process_memory = SimpleNamespace(rss=640_000_000, vms=8_000_000_000)

    def current_process():
        process_calls.append(True)
        return SimpleNamespace(memory_info=lambda: process_memory)

    def cpu_percent(*, interval):
        intervals.append(interval)
        return 42.5

    monkeypatch.setattr(resource_usage.psutil, "cpu_percent", cpu_percent)
    monkeypatch.setattr(resource_usage.psutil, "Process", current_process)
    monkeypatch.setattr(resource_usage.psutil, "virtual_memory", lambda: SimpleNamespace(
        total=16 * 1024 ** 3, available=4 * 1024 ** 3
    ))
    sampler = SystemUsageSampler()
    sampler.prime()
    assert sampler.sample() == ResourceUsage(42.5, 640_000_000, 12 * 1024 ** 3,
                                            16 * 1024 ** 3)
    assert intervals == [None, None]
    process_memory.rss = 700_000_000
    assert sampler.sample().program_ram_used == 700_000_000
    assert process_calls == [True]


def test_usage_labels_and_tooltips_update_without_changing_size(qtbot):
    widget = ResourceUsageWidget(sampler=FakeSampler())
    qtbot.addWidget(widget)
    initial_size = widget.size()
    assert widget.sizeHint() == initial_size
    assert widget.cpu_label.text() == "CPU —"
    assert widget.ram_label.text() == "RAM —"
    widget.refresh()
    assert widget.cpu_label.text() == "CPU 23%"
    assert widget.ram_label.text() == "RAM 512 MB"
    assert "System-wide" in widget.cpu_label.toolTip()
    assert "10737 MB in use / 17180 MB total" in widget.ram_label.toolTip()
    assert "Program RAM (resident memory): 512 MB" in widget.ram_label.toolTip()
    assert "System-wide RAM" in widget.ram_label.toolTip()
    assert "%" not in widget.ram_label.toolTip()
    assert widget.size() == initial_size


@pytest.mark.parametrize("used_bytes,expected", [
    (0, "RAM 0 MB"),
    (512_000_000, "RAM 512 MB"),
    (128 * 1024 ** 3, "RAM 137439 MB"),
])
def test_ram_displays_program_megabytes_not_system_memory(qtbot, used_bytes, expected):
    sampler = FakeSampler()
    sampler.sample = lambda: ResourceUsage(100, used_bytes, 192 * 1024 ** 3,
                                           256 * 1024 ** 3)
    widget = ResourceUsageWidget(sampler=sampler)
    qtbot.addWidget(widget)
    widget.refresh()
    assert widget.ram_label.text() == expected
    assert widget.cpu_label.text() == "CPU 100%"
    assert "%" not in widget.ram_label.toolTip()


def test_system_ram_changes_only_update_tooltip(qtbot):
    sampler = FakeSampler()
    widget = ResourceUsageWidget(sampler=sampler)
    qtbot.addWidget(widget)
    widget.refresh()
    initial_label = widget.ram_label.text()
    initial_tooltip = widget.ram_label.toolTip()
    sampler.sample = lambda: ResourceUsage(23.4, 512_000_000, 12 * 1024 ** 3,
                                           16 * 1024 ** 3)
    widget.refresh()
    assert widget.ram_label.text() == initial_label
    assert widget.ram_label.toolTip() != initial_tooltip
    assert "12885 MB in use" in widget.ram_label.toolTip()


@pytest.mark.parametrize("error_type", [OSError, RuntimeError, resource_usage.psutil.Error])
def test_sampling_failure_displays_unavailable_and_recovers(qtbot, error_type):
    sampler = FakeSampler()
    widget = ResourceUsageWidget(sampler=sampler)
    qtbot.addWidget(widget)
    original_sample = sampler.sample

    def fail():
        raise error_type("Monitoring unavailable")

    sampler.sample = fail
    widget.refresh()
    assert widget.cpu_label.text() == "CPU —"
    assert widget.ram_label.text() == "RAM —"
    assert "retrying" in widget.ram_label.toolTip()
    sampler.sample = original_sample
    widget.refresh()
    assert widget.cpu_label.text() == "CPU 23%"


def test_timer_only_samples_while_visible_and_reprimes_on_show(qtbot):
    sampler = FakeSampler()
    widget = ResourceUsageWidget(sampler=sampler)
    qtbot.addWidget(widget)
    assert widget._timer.interval() == 1000
    assert not widget._timer.isActive()
    widget._timer.setInterval(10)
    widget.show()
    assert sampler.primed == 1
    qtbot.waitUntil(lambda: sampler.samples > 0)
    widget.hide()
    assert not widget._timer.isActive()
    sampler.fail = True
    widget.show()
    assert sampler.primed == 2
    assert widget.cpu_label.text() == "CPU —"
    assert widget._timer.isActive()
    sampler.fail = False
    qtbot.waitUntil(lambda: widget.cpu_label.text() == "CPU 23%")
    widget.stop()
    assert not widget._timer.isActive()


def test_monitor_stays_in_header_across_pages_and_stops_on_close(window, qtbot):
    widget = window.tabWidget.cornerWidget(QtCore.Qt.Corner.TopRightCorner)
    assert widget is window._resource_usage
    window.resize(960, 760)
    window.show()
    qtbot.waitExposed(window)
    header_pos = widget.pos()
    for page in (window.Calibration, window.Classification, window.SuperResolution,
                 window.Visualization):
        window.tabWidget.setCurrentWidget(page)
        assert widget.isVisible()
        assert widget.pos() == header_pos
        assert widget._timer.isActive()
        assert widget.geometry().left() > window.tabWidget.tabBar().geometry().right()
    window.resize(window.width() + 200, window.height() + 40)
    qtbot.waitUntil(lambda: widget.pos().x() > header_pos.x())
    assert widget.pos().y() == header_pos.y()
    window.close()
    assert not widget._timer.isActive()
