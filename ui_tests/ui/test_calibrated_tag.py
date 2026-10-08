import numpy as np
from PyQt6 import QtCore
from spectral.io import envi


def test_calibration_switch_shows_on_all_pages_and_clears_with_reference_or_source(
    loaded_window, file_dialog, tmp_path, qtbot, synthetic_cube_path
):
    window = loaded_window
    window.show()
    qtbot.waitExposed(window)
    badges = window._resolution_switches.calibration_switches
    assert len(badges) == 4
    assert all(badge.isHidden() for badge in badges)
    source = window._hsi_data
    for name, value in (("dark", 0), ("bright", 2)):
        path = tmp_path / f"{name}.hdr"
        envi.save_image(str(path), np.full((50, source.columns, source.bands), value,
                                          dtype=np.float32), ext=".bip", interleave="bip",
                        metadata={"wavelength": source.wavelengths})
        file_dialog.open_return = (str(path), "")
        (window.darkFileButton if name == "dark" else window.referenceFileButton).click()
    window.calibrateButton.click()
    qtbot.waitUntil(lambda: not window._calibration_controller.is_running())
    for page, badge in zip((window.Visualization, window.Calibration, window.Classification,
                           window.SuperResolution), badges):
        window.tabWidget.setCurrentWidget(page)
        qtbot.waitUntil(lambda: badge.isVisible())
        assert badge.pos() == QtCore.QPoint(12, 12)
        assert badge.isChecked()
    window.referenceFileButton.click()
    assert all(badge.isHidden() for badge in badges)
    window.calibrateButton.click()
    qtbot.waitUntil(lambda: not window._calibration_controller.is_running())
    assert all(not badge.isHidden() for badge in badges)
    window.load_image_from_path(synthetic_cube_path)
    assert all(badge.isHidden() for badge in badges)
