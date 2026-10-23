"""Batch tab: select multiple inputs, optionally run Calibrate/SR/Classify, save."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from core import (
    BatchClassificationMode,
    BatchClassificationOptions,
    BatchRequest,
    BatchResult,
    BatchService,
    CalibrationRequest,
    ClassificationError,
    HSIError,
    HSI_FILE_FILTER,
    ImageExportFormat,
    SuperResolutionRequest,
    SupervisedClassificationRequest,
    SupervisedClassifierType,
    TrainingPairResolver,
    UnsupervisedClassificationRequest,
)
from core.hsi_reader import DATA_EXTENSIONS, METADATA_EXTENSIONS
from ui.batch_worker import BatchWorker


LOGGER = logging.getLogger(__name__)

_FORMAT_LABELS: tuple[tuple[ImageExportFormat, str], ...] = (
    (ImageExportFormat.PNG, "PNG"),
    (ImageExportFormat.TIFF, "TIFF"),
    (ImageExportFormat.JPEG, "JPEG"),
    (ImageExportFormat.BMP, "BMP"),
)


def _discover_metadata_files(directory: Path):
    """Yield each capture's metadata file directly under ``directory``.

    Calibration-frame files (same-stem ``*_calibFrame``) are excluded: they
    are reference inputs auto-resolved per source file, not batch targets of
    their own. ``.hdr`` is preferred over a same-stem ``.json`` sidecar, as
    elsewhere in the app.
    """

    try:
        entries = sorted(path for path in directory.iterdir() if path.is_file())
    except OSError:
        return
    seen_stems: set[str] = set()
    for path in entries:
        if path.suffix.casefold() not in METADATA_EXTENSIONS:
            continue
        stem = path.stem
        if stem.casefold().endswith("_calibframe"):
            continue
        if stem.casefold() in seen_stems:
            continue
        has_data = any(
            (directory / f"{stem}{extension}").is_file() for extension in DATA_EXTENSIONS
        )
        if not has_data:
            continue
        seen_stems.add(stem.casefold())
        yield path


class BatchController(QtCore.QObject):
    """Owns the Batch tab's file list, stage options, worker, and results."""

    readyToClose = QtCore.pyqtSignal()

    def __init__(
        self,
        *,
        file_list: QtWidgets.QListWidget,
        results_list: QtWidgets.QListWidget,
        add_files_button: QtWidgets.QPushButton,
        add_folder_button: QtWidgets.QPushButton,
        remove_button: QtWidgets.QPushButton,
        clear_button: QtWidgets.QPushButton,
        calibrate_check: QtWidgets.QCheckBox,
        super_resolve_check: QtWidgets.QCheckBox,
        classify_check: QtWidgets.QCheckBox,
        classify_mode_tabs: QtWidgets.QTabWidget,
        num_classes_edit: QtWidgets.QLineEdit,
        max_iterations_edit: QtWidgets.QLineEdit,
        groundtruth_button: QtWidgets.QPushButton,
        groundtruth_edit: QtWidgets.QLineEdit,
        training_cube_button: QtWidgets.QPushButton,
        training_cube_edit: QtWidgets.QLineEdit,
        classifier_combo: QtWidgets.QComboBox,
        output_dir_button: QtWidgets.QPushButton,
        output_dir_edit: QtWidgets.QLineEdit,
        format_combo: QtWidgets.QComboBox,
        overwrite_check: QtWidgets.QCheckBox,
        export_csv_check: QtWidgets.QCheckBox,
        run_button: QtWidgets.QPushButton,
        progress_bar: QtWidgets.QProgressBar,
        status_label: QtWidgets.QLabel,
        parent_widget: QtWidgets.QWidget,
        service: BatchService | None = None,
        training_pair_resolver: TrainingPairResolver | None = None,
    ) -> None:
        super().__init__(parent_widget)
        self._file_list = file_list
        self._results_list = results_list
        self._add_files_button = add_files_button
        self._add_folder_button = add_folder_button
        self._remove_button = remove_button
        self._clear_button = clear_button
        self._calibrate_check = calibrate_check
        self._super_resolve_check = super_resolve_check
        self._classify_check = classify_check
        self._classify_mode_tabs = classify_mode_tabs
        self._num_classes_edit = num_classes_edit
        self._max_iterations_edit = max_iterations_edit
        self._groundtruth_button = groundtruth_button
        self._groundtruth_edit = groundtruth_edit
        self._training_cube_button = training_cube_button
        self._training_cube_edit = training_cube_edit
        self._classifier_combo = classifier_combo
        self._output_dir_button = output_dir_button
        self._output_dir_edit = output_dir_edit
        self._format_combo = format_combo
        self._overwrite_check = overwrite_check
        self._export_csv_check = export_csv_check
        self._run_button = run_button
        self._progress_bar = progress_bar
        self._status_label = status_label
        self._parent = parent_widget
        self._service = service or BatchService()
        self._training_pair_resolver = training_pair_resolver or TrainingPairResolver()

        self._input_paths: list[Path] = []
        self._output_directory: Optional[Path] = None
        self._training_mask_path: Optional[Path] = None
        self._training_cube_path: Optional[Path] = None
        self._manual_training_cube_path: Optional[Path] = None
        self._worker: Optional[BatchWorker] = None
        self._close_after_batch = False

        self._configure_controls()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def is_running(self) -> bool:
        return self._worker is not None

    def request_close(self) -> bool:
        """Cancel active work and tell the caller whether close must be deferred."""

        if not self.is_running():
            return False
        self._close_after_batch = True
        self._cancel()
        return True

    # ------------------------------------------------------------------ #
    # Private: initial wiring                                             #
    # ------------------------------------------------------------------ #

    def _configure_controls(self) -> None:
        self._add_files_button.clicked.connect(self._add_files)
        self._add_folder_button.clicked.connect(self._add_folder)
        self._remove_button.clicked.connect(self._remove_selected)
        self._clear_button.clicked.connect(self._clear_files)
        self._calibrate_check.toggled.connect(lambda _checked: self._update_ready())
        self._super_resolve_check.toggled.connect(lambda _checked: self._update_ready())
        self._classify_check.toggled.connect(self._on_classify_toggled)
        self._classify_mode_tabs.currentChanged.connect(lambda _index: self._update_ready())
        self._groundtruth_button.clicked.connect(self._select_groundtruth)
        self._training_cube_button.clicked.connect(self._select_training_cube)
        self._output_dir_button.clicked.connect(self._select_output_directory)
        self._run_button.clicked.connect(self._run_or_cancel)

        self._num_classes_edit.setValidator(QtGui.QIntValidator(2, 65535, self))
        self._max_iterations_edit.setValidator(QtGui.QIntValidator(1, 10000, self))
        self._num_classes_edit.setText("5")
        self._max_iterations_edit.setText("20")

        self._classifier_combo.clear()
        classifier_labels = {
            SupervisedClassifierType.GAUSSIAN: "Gaussian",
            SupervisedClassifierType.MAHALANOBIS: "Mahalanobis distance",
        }
        for classifier in SupervisedClassifierType:
            self._classifier_combo.addItem(
                classifier_labels.get(classifier, classifier.value), classifier
            )

        self._format_combo.clear()
        for export_format, label in _FORMAT_LABELS:
            self._format_combo.addItem(label, export_format)

        self._classify_mode_tabs.setEnabled(False)
        self._update_ready()

    # ------------------------------------------------------------------ #
    # Private: file list management                                       #
    # ------------------------------------------------------------------ #

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self._parent, "Select Hyperspectral Images", "", HSI_FILE_FILTER
        )
        if not paths:
            return
        self._add_paths(Path(path) for path in paths)

    def _add_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self._parent, "Select Folder of Hyperspectral Images"
        )
        if not directory:
            return
        self._add_paths(_discover_metadata_files(Path(directory)))

    def _add_paths(self, paths) -> None:
        existing = set(self._input_paths)
        added = 0
        for path in paths:
            resolved = Path(path).expanduser().resolve()
            if resolved in existing:
                continue
            existing.add(resolved)
            self._input_paths.append(resolved)
            item = QtWidgets.QListWidgetItem(resolved.name)
            item.setToolTip(str(resolved))
            self._file_list.addItem(item)
            added += 1
        if added:
            self._status_label.setText(f"{len(self._input_paths)} file(s) selected")
        self._update_ready()

    def _remove_selected(self) -> None:
        for item in self._file_list.selectedItems():
            row = self._file_list.row(item)
            self._file_list.takeItem(row)
            del self._input_paths[row]
        self._update_ready()

    def _clear_files(self) -> None:
        self._file_list.clear()
        self._input_paths.clear()
        self._results_list.clear()
        self._update_ready()

    # ------------------------------------------------------------------ #
    # Private: stage options                                              #
    # ------------------------------------------------------------------ #

    def _on_classify_toggled(self, checked: bool) -> None:
        self._classify_mode_tabs.setEnabled(checked)
        self._update_ready()

    def _select_groundtruth(self) -> None:
        mask_path_str, _ = QFileDialog.getOpenFileName(
            self._parent,
            "Open Ground-truth Mask",
            "",
            "Ground-truth Masks (*.png *.tif *.tiff *.bmp *.jpg *.jpeg);;All Files (*)",
        )
        if not mask_path_str:
            return
        try:
            mask_path = self._training_pair_resolver.validate_mask_path(mask_path_str)
        except ClassificationError as exc:
            QMessageBox.critical(self._parent, "Invalid ground-truth mask", str(exc))
            return
        self._training_mask_path = mask_path
        self._groundtruth_edit.setText(str(mask_path))
        self._groundtruth_edit.setToolTip(str(mask_path))
        if self._manual_training_cube_path is not None:
            try:
                pair = self._training_pair_resolver.resolve_manual(
                    mask_path, self._manual_training_cube_path
                )
            except ClassificationError:
                self._training_cube_path = None
            else:
                self._set_training_cube(pair.cube_path)
                self._update_ready()
                return
        try:
            pair = self._training_pair_resolver.resolve(mask_path)
        except ClassificationError:
            self._training_cube_path = None
            self._training_cube_edit.clear()
        else:
            self._set_training_cube(pair.cube_path)
        self._update_ready()

    def _select_training_cube(self) -> None:
        cube_path_str, _ = QFileDialog.getOpenFileName(
            self._parent, "Open Training Hyperspectral Image", "", HSI_FILE_FILTER
        )
        if not cube_path_str:
            return
        try:
            cube_path = self._training_pair_resolver.validate_cube_path(cube_path_str)
        except ClassificationError as exc:
            QMessageBox.critical(self._parent, "Invalid hyperspectral image", str(exc))
            return
        self._manual_training_cube_path = cube_path
        self._set_training_cube(cube_path)
        self._update_ready()

    def _set_training_cube(self, cube_path: Path) -> None:
        self._training_cube_path = cube_path
        self._training_cube_edit.setText(str(cube_path))
        self._training_cube_edit.setToolTip(str(cube_path))

    def _select_output_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self._parent, "Select Output Folder")
        if not directory:
            return
        self._output_directory = Path(directory)
        self._output_dir_edit.setText(directory)
        self._output_dir_edit.setToolTip(directory)
        self._update_ready()

    # ------------------------------------------------------------------ #
    # Private: run / cancel                                               #
    # ------------------------------------------------------------------ #

    def _update_ready(self) -> None:
        if self.is_running():
            return
        ready = bool(self._input_paths) and self._output_directory is not None
        if self._classify_check.isChecked() and self._classify_mode_tabs.currentIndex() == 1:
            ready = ready and self._training_mask_path is not None and self._training_cube_path is not None
        self._run_button.setEnabled(ready)

    def _run_or_cancel(self) -> None:
        if self.is_running():
            self._cancel()
            return
        try:
            request = self._build_request()
        except (ClassificationError, HSIError) as exc:
            QMessageBox.critical(self._parent, "Invalid batch settings", str(exc))
            return
        except ValueError:
            QMessageBox.critical(
                self._parent,
                "Invalid batch settings",
                "Enter whole numbers for classes and maximum iterations.",
            )
            return

        self._results_list.clear()
        self._progress_bar.setValue(0)
        self._run_button.setText("Cancel")
        self._run_button.setToolTip("Cancel after the current file completes")
        self._set_controls_enabled(False)
        self._status_label.setText("Starting batch processing…")

        worker = BatchWorker(self._service, request, parent=self)
        self._worker = worker
        worker.progress.connect(self._on_progress)
        worker.result_ready.connect(self._on_result)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _build_request(self) -> BatchRequest:
        if not self._input_paths:
            raise ClassificationError("Select at least one file to process.")
        if self._output_directory is None:
            raise ClassificationError("Select an output folder.")

        calibrate = CalibrationRequest() if self._calibrate_check.isChecked() else None
        super_resolve = (
            SuperResolutionRequest() if self._super_resolve_check.isChecked() else None
        )

        classify: BatchClassificationOptions | None = None
        if self._classify_check.isChecked():
            if self._classify_mode_tabs.currentIndex() == 0:
                classify = BatchClassificationOptions(
                    mode=BatchClassificationMode.UNSUPERVISED,
                    unsupervised_request=UnsupervisedClassificationRequest(
                        n_classes=int(self._num_classes_edit.text()),
                        max_iterations=int(self._max_iterations_edit.text()),
                    ),
                )
            else:
                if self._training_mask_path is None or self._training_cube_path is None:
                    raise ClassificationError(
                        "Select a ground-truth mask and its paired hyperspectral image."
                    )
                classifier = self._classifier_combo.currentData()
                if not isinstance(classifier, SupervisedClassifierType):
                    raise ClassificationError("Select a supported supervised classifier.")
                pair = self._training_pair_resolver.resolve_manual(
                    self._training_mask_path, self._training_cube_path
                )
                classify = BatchClassificationOptions(
                    mode=BatchClassificationMode.SUPERVISED,
                    supervised_request=SupervisedClassificationRequest(classifier),
                    training_pair=pair,
                )

        export_format = self._format_combo.currentData()
        if not isinstance(export_format, ImageExportFormat):
            export_format = ImageExportFormat.PNG

        return BatchRequest(
            input_paths=tuple(self._input_paths),
            output_directory=self._output_directory,
            calibrate=calibrate,
            super_resolve=super_resolve,
            classify=classify,
            export_format=export_format,
            overwrite=self._overwrite_check.isChecked(),
            export_csv=self._export_csv_check.isChecked(),
        )

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.requestInterruption()
            self._run_button.setEnabled(False)
            self._run_button.setText("Cancelling…")
            self._status_label.setText("Cancelling after the current file…")

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self._add_files_button,
            self._add_folder_button,
            self._remove_button,
            self._clear_button,
            self._calibrate_check,
            self._super_resolve_check,
            self._classify_check,
            self._output_dir_button,
            self._groundtruth_button,
            self._training_cube_button,
            self._format_combo,
            self._overwrite_check,
            self._export_csv_check,
        ):
            widget.setEnabled(enabled)
        self._classify_mode_tabs.setEnabled(enabled and self._classify_check.isChecked())

    @QtCore.pyqtSlot(int, str)
    def _on_progress(self, value: int, message: str) -> None:
        self._progress_bar.setValue(value)
        self._status_label.setText(message)

    @QtCore.pyqtSlot(object)
    def _on_result(self, result: BatchResult) -> None:
        for file_result in result.file_results:
            if file_result.succeeded:
                text = (
                    f"✓ {file_result.input_path.name} → {file_result.output_path.name}"
                    f", {file_result.raw_output_path.name}"
                )
                if file_result.csv_output_path is not None:
                    text += f", {file_result.csv_output_path.name}"
                if file_result.auto_cropped:
                    text += " (auto-cropped)"
            else:
                text = f"✗ {file_result.input_path.name}: {file_result.error}"
            item = QtWidgets.QListWidgetItem(text)
            if not file_result.succeeded:
                item.setForeground(QtGui.QColor("#c0392b"))
            self._results_list.addItem(item)
        if result.cancelled:
            self._status_label.setText("Batch cancelled")
        else:
            self._status_label.setText(
                f"Batch complete: {result.succeeded_count} succeeded, "
                f"{result.failed_count} failed"
            )

    @QtCore.pyqtSlot(str)
    def _on_failed(self, message: str) -> None:
        QMessageBox.critical(self._parent, "Batch processing failed", message)
        self._status_label.setText("Batch processing failed")

    @QtCore.pyqtSlot()
    def _on_finished(self) -> None:
        self._worker = None
        self._set_controls_enabled(True)
        self._run_button.setText("Run Batch")
        self._run_button.setToolTip("")
        self._update_ready()
        if self._close_after_batch:
            self._close_after_batch = False
            QtCore.QTimer.singleShot(0, self.readyToClose.emit)
