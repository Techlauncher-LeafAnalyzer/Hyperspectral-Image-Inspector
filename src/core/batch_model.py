"""UI-independent batch pipeline over multiple hyperspectral captures.

Chains the same per-file Calibration / Super-Resolution / Classification
Models already used for one capture -- in that fixed order -- then exports a
rendered RGB (or colorized class map) preview for each input. A Controller
runs :meth:`BatchService.run` on a worker thread exactly like the other
long-running Models, forwarding ``progress``/``is_cancelled`` the same way.

One file failing (missing calibration frames, an incompatible SR camera, an
export conflict, ...) does not abort the rest of the batch; its failure is
recorded in the returned :class:`BatchResult` and processing continues.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .calibration_model import (
    CalibrationFrameResolver,
    CalibrationRequest,
    CalibrationResult,
    CalibrationService,
)
from .classification_model import (
    ClassificationService,
    SupervisedClassificationRequest,
    TrainingFilePair,
    UnsupervisedClassificationRequest,
)
from .classification_export_model import ExportRegion, export_class_summary_csv
from .classification_layer_model import ClassificationLayerModel, ClassIndexStatistics
from .leaf_labels import draw_leaf_labels, place_leaf_labels
from .leaf_separation import separate_leaves
from .errors import (
    BatchError,
    CalibrationError,
    CancelledError,
    HSIError,
    VisualizationError,
    WavelengthError,
)
from .hsi_data import HSIData
from .hsi_reader import HSIReader
from .marker_crop import find_plus_crop
from .super_resolution_model import SuperResolutionRequest, SuperResolutionService
from .visualization_export_model import (
    ImageExportFormat,
    VisualizationExportRequest,
    VisualizationExportService,
)
from .visualization_model import VisualizationMode, VisualizationRequest, VisualizationService


ProgressCallback = Callable[[int, str], None]
CancellationCheck = Callable[[], bool]

_EXPORT_EXTENSIONS = {
    ImageExportFormat.PNG: ".png",
    ImageExportFormat.JPEG: ".jpg",
    ImageExportFormat.TIFF: ".tif",
    ImageExportFormat.BMP: ".bmp",
}

# Same visualizations the single-image tabs compute, so batch CSVs share columns.
_SUMMARY_MODES = (
    VisualizationMode.RGB,
    VisualizationMode.NDVI,
    VisualizationMode.EVI,
    VisualizationMode.MCARI,
    VisualizationMode.MTVI,
    VisualizationMode.OSAVI,
    VisualizationMode.PRI,
    VisualizationMode.NDWI,
    VisualizationMode.NDMI,
)


class BatchClassificationMode(StrEnum):
    """Which classification workflow a batch run applies to every file."""

    UNSUPERVISED = "unsupervised"
    SUPERVISED = "supervised"


@dataclass(frozen=True, slots=True)
class BatchClassificationOptions:
    """Classification settings shared across every file in a batch run.

    Unsupervised K-means is independent per file. Supervised classification
    trains once per file from the same ``training_pair`` -- there is no
    single-example training workflow that spans multiple target cubes.

    ``separate_leaves`` (supervised only) splits each file's plant class into
    one class per clearly separate leaf, as the Classification tab's
    Separate Leaves button does.
    """

    mode: BatchClassificationMode
    unsupervised_request: UnsupervisedClassificationRequest | None = None
    supervised_request: SupervisedClassificationRequest | None = None
    training_pair: TrainingFilePair | None = None
    separate_leaves: bool = False

    def __post_init__(self) -> None:
        if self.mode is BatchClassificationMode.UNSUPERVISED:
            if self.unsupervised_request is None:
                raise BatchError(
                    "Unsupervised batch classification requires an "
                    "UnsupervisedClassificationRequest."
                )
        elif self.mode is BatchClassificationMode.SUPERVISED:
            if self.supervised_request is None or self.training_pair is None:
                raise BatchError(
                    "Supervised batch classification requires a "
                    "SupervisedClassificationRequest and a resolved training pair."
                )
        else:
            raise BatchError(f"Unsupported batch classification mode: {self.mode}")
        if self.separate_leaves and self.mode is not BatchClassificationMode.SUPERVISED:
            raise BatchError("Leaf separation requires supervised classification.")


@dataclass(frozen=True, slots=True)
class BatchRequest:
    """One batch-processing command: inputs, enabled stages, and output format.

    Each of ``calibrate``/``super_resolve``/``classify`` is independently
    optional; the pipeline always runs in that fixed order, then renders and
    exports whatever the last enabled stage produced (plain RGB when none are
    enabled). Calibration auto-resolves each file's own dark/bright reference
    pair by capture timestamp, the same discovery :class:`CalibrationFrameResolver`
    uses for a single image.
    """

    input_paths: tuple[Path, ...]
    output_directory: Path
    calibrate: CalibrationRequest | None = None
    super_resolve: SuperResolutionRequest | None = None
    classify: BatchClassificationOptions | None = None
    export_format: ImageExportFormat = ImageExportFormat.PNG
    overwrite: bool = False
    export_csv: bool = False

    def __post_init__(self) -> None:
        paths = tuple(Path(path) for path in self.input_paths)
        if not paths:
            raise BatchError("Select at least one file to process.")
        object.__setattr__(self, "input_paths", paths)
        object.__setattr__(self, "output_directory", Path(self.output_directory))


@dataclass(frozen=True, slots=True)
class BatchFileResult:
    """One input's outcome: either saved previews or a recorded failure."""

    input_path: Path
    output_path: Path | None
    error: str | None = None
    raw_output_path: Path | None = None
    csv_output_path: Path | None = None
    auto_cropped: bool = False
    calibrated: bool = False
    super_resolved: bool = False
    classified: bool = False

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Outcome of an entire batch run."""

    file_results: tuple[BatchFileResult, ...]
    cancelled: bool = False

    @property
    def succeeded_count(self) -> int:
        return sum(1 for result in self.file_results if result.succeeded)

    @property
    def failed_count(self) -> int:
        return sum(1 for result in self.file_results if not result.succeeded)


@dataclass
class _TrainingContext:
    """Shared supervised-training cube kept open for the whole batch run.

    Holds one uncalibrated training cube plus, lazily, one calibrated version
    per request (``standardize_features`` toggled) -- used when a file's
    target cube is itself calibrated, matching the single-image Controller's
    ``_calibrate_training`` policy of putting both cubes in the same units.
    """

    data: HSIData
    request: SupervisedClassificationRequest
    mask_path: Path
    _calibrated: (
        tuple[HSIData, CalibrationResult | None, SupervisedClassificationRequest] | None
    ) = None

    def for_target(
        self, *, target_is_calibrated: bool
    ) -> tuple[HSIData, SupervisedClassificationRequest]:
        if not target_is_calibrated:
            return self.data, self.request
        if self._calibrated is None:
            self._calibrated = self._calibrate()
        data, _result, request = self._calibrated
        return data, request

    def _calibrate(
        self,
    ) -> tuple[HSIData, CalibrationResult | None, SupervisedClassificationRequest]:
        fallback = replace(self.request, standardize_features=True)
        pair = CalibrationFrameResolver().resolve(self.data.source_path)
        if pair is None:
            return self.data, None, fallback
        reader = HSIReader()
        dark = bright = None
        try:
            dark = reader.open(pair.dark_path)
            bright = reader.open(pair.bright_path)
            result = CalibrationService().calibrate(self.data, dark, bright)
        except CalibrationError:
            return self.data, None, fallback
        finally:
            for frame in (dark, bright):
                if frame is not None:
                    frame.close()
        return result.data, result, self.request

    def close(self) -> None:
        if self._calibrated is not None:
            _data, result, _request = self._calibrated
            if result is not None:
                result.cleanup()
        self.data.close()


class BatchService:
    """Run Calibration / Super-Resolution / Classification over many files."""

    def __init__(
        self,
        *,
        reader: HSIReader | None = None,
        frame_resolver: CalibrationFrameResolver | None = None,
        calibration_service: CalibrationService | None = None,
        super_resolution_service: SuperResolutionService | None = None,
        classification_service: ClassificationService | None = None,
        visualization_service: VisualizationService | None = None,
        export_service: VisualizationExportService | None = None,
    ) -> None:
        self._reader = reader or HSIReader()
        self._frame_resolver = frame_resolver or CalibrationFrameResolver()
        self._calibration_service = calibration_service or CalibrationService()
        self._super_resolution_service = super_resolution_service or SuperResolutionService()
        self._classification_service = classification_service or ClassificationService()
        self._visualization_service = visualization_service or VisualizationService()
        self._export_service = export_service or VisualizationExportService()

    def run(
        self,
        request: BatchRequest,
        *,
        progress: ProgressCallback | None = None,
        is_cancelled: CancellationCheck | None = None,
    ) -> BatchResult:
        try:
            request.output_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise BatchError(
                f"Could not create output directory {request.output_directory}: {exc}"
            ) from exc

        training_context: _TrainingContext | None = None
        if (
            request.classify is not None
            and request.classify.mode is BatchClassificationMode.SUPERVISED
        ):
            training_context = self._open_training_context(request.classify)

        total = len(request.input_paths)
        results: list[BatchFileResult] = []
        cancelled = False
        try:
            for index, path in enumerate(request.input_paths):
                if is_cancelled is not None and is_cancelled():
                    cancelled = True
                    break

                def file_progress(
                    value: int, message: str, *, index: int = index, path: Path = path
                ) -> None:
                    overall = round((index + value / 100) / total * 100)
                    self._emit(progress, overall, f"{path.name}: {message}")

                try:
                    result = self._process_one(
                        path,
                        request,
                        training_context,
                        progress=file_progress,
                        is_cancelled=is_cancelled,
                    )
                except CancelledError:
                    cancelled = True
                    break
                except HSIError as exc:
                    results.append(BatchFileResult(path, None, error=str(exc)))
                except Exception as exc:  # Keep one bad file from aborting the batch.
                    results.append(
                        BatchFileResult(path, None, error=f"Unexpected error: {exc}")
                    )
                else:
                    results.append(result)
        finally:
            if training_context is not None:
                training_context.close()

        self._emit(
            progress,
            100,
            "Batch cancelled" if cancelled else "Batch processing complete",
        )
        return BatchResult(tuple(results), cancelled=cancelled)

    def _process_one(
        self,
        path: Path,
        request: BatchRequest,
        training_context: _TrainingContext | None,
        *,
        progress: ProgressCallback,
        is_cancelled: CancellationCheck | None,
    ) -> BatchFileResult:
        stages: list[tuple[str, int]] = [("open", 5), ("auto_crop", 10)]
        if request.super_resolve is not None:
            stages.append(("super_resolve", 30))
        if request.calibrate is not None:
            stages.append(("calibrate", 30))
        if request.super_resolve is not None or request.calibrate is not None:
            stages.append(("render", 10))
        if request.classify is not None:
            stages.append(("classify", 20))
        if request.export_csv:
            stages.append(("csv", 15))
        stages.append(("export", 5))
        total_weight = sum(weight for _, weight in stages)
        offset = 0

        def stage_progress(name: str) -> ProgressCallback:
            nonlocal offset
            weight = next(w for stage_name, w in stages if stage_name == name)
            base = offset
            offset += weight

            def callback(value: int, message: str) -> None:
                self._emit(
                    progress,
                    min(99, round((base + weight * value / 100) / total_weight * 100)),
                    message,
                )

            return callback

        self._check_cancelled(is_cancelled)
        stage_progress("open")(0, "Opening image")
        data = self._reader.open(path)
        calibration_result: CalibrationResult | None = None
        sr_result = None
        calibrated = False
        super_resolved = False
        classified = False
        classification_result = None
        leaf_names: dict[int, str] | None = None
        leaf_numbers: dict[int, int] = {}
        fixed_colors: dict[int, tuple[int, int, int]] = {}
        csv_saved: Path | None = None
        extension =_EXPORT_EXTENSIONS[request.export_format]
        try:
            self._check_cancelled(is_cancelled)
            crop_progress = stage_progress("auto_crop")
            crop_progress(0, "Detecting Leaf2QR pattern")
            auto_cropped = self._auto_crop(data)
            crop_progress(50, "Saving raw preview")
            raw_display = self._visualization_service.render(
                data, VisualizationRequest(VisualizationMode.RGB)
            ).display_rgb
            raw_saved = self._save(
                raw_display,
                request.output_directory / f"{path.stem}_raw{extension}",
                request,
                data.roi_mask,
            )
            crop_progress(100, "Saved raw preview")

            current = data
            # Super-Resolution always runs on the uncalibrated source (as in
            # the single-image workflow); calibration runs afterward against
            # whichever cube is current, passing the raw source as
            # ``reference_source`` so a 2x SR result gets its detector
            # profile interpolated instead of directly cropped.
            if request.super_resolve is not None:
                self._check_cancelled(is_cancelled)
                sr_result = self._super_resolution_service.run(
                    current,
                    request.super_resolve,
                    progress=stage_progress("super_resolve"),
                    is_cancelled=is_cancelled,
                )
                current = sr_result.data
                super_resolved = True

            if request.calibrate is not None:
                self._check_cancelled(is_cancelled)
                pair = self._frame_resolver.resolve(path)
                if pair is None:
                    raise BatchError(
                        f"No calibration frame pair was found for {path.name}."
                    )
                dark = bright = None
                try:
                    dark = self._reader.open(pair.dark_path)
                    bright = self._reader.open(pair.bright_path)
                    calibration_result = self._calibration_service.calibrate(
                        current,
                        dark,
                        bright,
                        request.calibrate,
                        progress=stage_progress("calibrate"),
                        is_cancelled=is_cancelled,
                        reference_source=data if super_resolved else None,
                    )
                finally:
                    if dark is not None:
                        dark.close()
                    if bright is not None:
                        bright.close()
                current = calibration_result.data
                calibrated = True

            self._check_cancelled(is_cancelled)
            if current is data:
                display_rgb = raw_display
            else:
                display_rgb = self._visualization_service.render(
                    current,
                    VisualizationRequest(VisualizationMode.RGB),
                    progress=stage_progress("render"),
                    is_cancelled=is_cancelled,
                ).display_rgb

            if request.classify is not None:
                self._check_cancelled(is_cancelled)
                classify_progress = stage_progress("classify")
                if request.classify.mode is BatchClassificationMode.UNSUPERVISED:
                    assert request.classify.unsupervised_request is not None
                    classification_result = self._classification_service.classify_unsupervised(
                        current,
                        request.classify.unsupervised_request,
                        progress=classify_progress,
                        is_cancelled=is_cancelled,
                    )
                    class_ids: Sequence[int] = range(classification_result.n_classes)
                else:
                    assert training_context is not None
                    training_data, training_request = training_context.for_target(
                        target_is_calibrated=calibrated
                    )
                    classification_result = self._classification_service.classify_supervised(
                        current,
                        training_data,
                        training_context.mask_path,
                        training_request,
                        progress=classify_progress,
                        is_cancelled=is_cancelled,
                    )
                    if request.classify.separate_leaves:
                        separation = separate_leaves(classification_result)
                        classification_result = separation.result
                        leaf_names = separation.leaf_names
                        leaf_numbers = separation.leaf_numbers
                        fixed_colors = separation.fixed_colors
                    class_ids = classification_result.class_ids
                display_rgb = _colorize_class_map(
                    classification_result.class_map, class_ids, fixed_colors
                )
                if leaf_numbers:
                    display_rgb = draw_leaf_labels(
                        display_rgb,
                        place_leaf_labels(classification_result.class_map, leaf_numbers),
                        colors=_classification_palette(class_ids, fixed_colors),
                    )
                classified = True

            if request.export_csv:
                self._check_cancelled(is_cancelled)
                csv_progress = stage_progress("csv")
                csv_progress(0, "Writing summary CSV")
                csv_saved = self._export_summary_csv(
                    current,
                    classification_result,
                    request.output_directory / f"{path.stem}.csv",
                    overwrite=request.overwrite,
                    class_names=leaf_names,
                )
                csv_progress(100, "Saved summary CSV")

            self._check_cancelled(is_cancelled)
            stage_progress("export")(0, "Saving preview")
            saved = self._save(
                display_rgb,
                request.output_directory / f"{path.stem}{extension}",
                request,
                current.roi_mask,
            )
            stage_progress("export")(100, "Saved")
            return BatchFileResult(
                input_path=path,
                output_path=saved,
                raw_output_path=raw_saved,
                csv_output_path=csv_saved,
                auto_cropped=auto_cropped,
                calibrated=calibrated,
                super_resolved=super_resolved,
                classified=classified,
            )
        finally:
            if sr_result is not None:
                sr_result.data.close()
            if calibration_result is not None:
                calibration_result.cleanup()
            data.close()

    def _auto_crop(self, data: HSIData) -> bool:
        """Polygon-crop to the Leaf2QR markers, as the single-image load does."""

        rgb = self._visualization_service.render(
            data, VisualizationRequest(VisualizationMode.RGB)
        ).display_rgb
        data.rgb_array = rgb
        data.mask_array = np.zeros(rgb.shape[:2], dtype=np.uint8)
        polygon = find_plus_crop(rgb)
        return polygon is not None and data.crop_polygon(polygon) is not None

    def _export_summary_csv(
        self,
        data: HSIData,
        classification_result,
        output_path: Path,
        *,
        overwrite: bool,
        class_names: dict[int, str] | None = None,
    ) -> Path:
        """Write the Classification tab's summary CSV for one processed cube.

        With a classification, one row per class (means plus index statistics);
        without one, a single whole-image row (within the ROI) of index means.
        """

        if output_path.exists() and not overwrite:
            raise BatchError(f"Export destination already exists: {output_path}")
        visualizations = {}
        for mode in _SUMMARY_MODES:
            try:
                visualizations[mode] = self._visualization_service.render(
                    data, VisualizationRequest(mode)
                )
            except (VisualizationError, WavelengthError):
                continue  # Camera lacks the bands for this index.
        rows, columns = data.rows, data.columns
        if classification_result is not None:
            layers = ClassificationLayerModel(classification_result, names=class_names)
            means: dict[int, dict[str, object]] = {
                layer.class_id: {} for layer in layers.layers
            }
            statistics: dict[int, dict[str, ClassIndexStatistics]] = {}
            for mode, visualization in visualizations.items():
                for class_id, mean in layers.visualization_means(visualization).items():
                    means[class_id][mode.value] = mean.value
                for class_id, stats in layers.visualization_statistics(
                    visualization
                ).items():
                    statistics.setdefault(class_id, {})[mode.value] = stats
            return export_class_summary_csv(
                output_path, layers.layers, means, rows * columns, statistics
            )

        keep = (
            np.ones((rows, columns), dtype=bool)
            if data.roi_mask is None
            else np.asarray(data.roi_mask, dtype=bool)
        )
        image_means: dict[str, object] = {}
        for mode, visualization in visualizations.items():
            if visualization.values is None:
                image_means[mode.value] = tuple(
                    float(channel)
                    for channel in visualization.display_rgb[keep].mean(
                        axis=0, dtype=np.float64
                    )
                )
                continue
            values = np.asarray(visualization.values)[keep]
            finite = values[np.isfinite(values)]
            if finite.size:
                image_means[mode.value] = float(np.mean(finite, dtype=np.float64))
        region = ExportRegion("Whole image", int(np.count_nonzero(keep)), image_means)
        return export_class_summary_csv(
            output_path, (), {}, rows * columns, regions=(region,)
        )

    def _save(
        self,
        display_rgb: np.ndarray,
        output_path: Path,
        request: BatchRequest,
        roi_mask: np.ndarray | None,
    ) -> Path:
        # JPEG/BMP have no alpha, so they keep the crop's bounding rectangle
        # rather than failing every auto-cropped file.
        alpha = (
            roi_mask
            if request.export_format in (ImageExportFormat.PNG, ImageExportFormat.TIFF)
            else None
        )
        return self._export_service.save_display(
            display_rgb,
            VisualizationExportRequest(output_path, overwrite=request.overwrite),
            alpha_mask=alpha,
        ).output_path

    def _open_training_context(
        self, classify: BatchClassificationOptions
    ) -> _TrainingContext:
        assert classify.training_pair is not None and classify.supervised_request is not None
        training_data = self._reader.open(classify.training_pair.cube_path)
        return _TrainingContext(
            data=training_data,
            request=classify.supervised_request,
            mask_path=classify.training_pair.mask_path,
        )

    @staticmethod
    def _check_cancelled(check: CancellationCheck | None) -> None:
        if check is not None and check():
            raise CancelledError("Batch processing cancelled.")

    @staticmethod
    def _emit(progress: ProgressCallback | None, value: int, message: str) -> None:
        if progress is not None:
            progress(value, message)


def _classification_palette(
    class_ids: Sequence[int],
    fixed_colors: Mapping[int, tuple[int, int, int]] | None = None,
) -> dict[int, tuple[int, int, int]]:
    """Return a stable, evenly spaced display colour per class, Qt-free.

    Matches ``ui.classification_colors.classification_palette``'s HSV formula
    (including ``fixed_colors`` handling) so a batch-colorized class map
    looks the same as the single-image tab.
    """

    fixed_colors = fixed_colors or {}
    ordered_ids = tuple(int(class_id) for class_id in class_ids)
    hued_ids = [class_id for class_id in ordered_ids if class_id not in fixed_colors]
    class_count = max(len(hued_ids), 1)
    palette: dict[int, tuple[int, int, int]] = {}
    for color_index, class_id in enumerate(hued_ids):
        red, green, blue = colorsys.hsv_to_rgb(color_index / class_count, 0.72, 0.92)
        palette[class_id] = (round(red * 255), round(green * 255), round(blue * 255))
    return {
        class_id: tuple(fixed_colors[class_id]) if class_id in fixed_colors else palette[class_id]
        for class_id in ordered_ids
    }


def _colorize_class_map(
    class_map: np.ndarray,
    class_ids: Sequence[int],
    fixed_colors: Mapping[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    display_rgb = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for class_id, color in _classification_palette(class_ids, fixed_colors).items():
        display_rgb[class_map == class_id] = color
    return display_rgb
