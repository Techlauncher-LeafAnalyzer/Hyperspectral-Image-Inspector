"""SPy-backed classification Model and Controller-facing data contracts.

This module deliberately contains no Qt imports. Classification is synchronous
and potentially expensive; a Controller must invoke it on a worker thread and
deliver the returned NumPy arrays to the View on the GUI thread.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, UnidentifiedImageError
from spectral import (
    GaussianClassifier,
    MahalanobisDistanceClassifier,
    create_training_classes,
    kmeans as spy_kmeans,
)

from .errors import CancelledError, ClassificationError
from .hsi_data import HSIData


ProgressCallback = Callable[[int, str], None]
CancellationCheck = Callable[[], bool]
SUPPORTED_CUBE_EXTENSIONS = (".bil", ".bip", ".bsq", ".dat", ".img", ".raw")
SUPPORTED_MASK_EXTENSIONS = (".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg")


class SupervisedClassifierType(StrEnum):
    """SPy supervised classifiers exposed through the stable Model API."""

    GAUSSIAN = "GaussianClassifier"
    MAHALANOBIS = "MahalanobisDistanceClassifier"


@dataclass(frozen=True, slots=True)
class TrainingFilePair:
    """Resolved one-example training mask and hyperspectral cube paths."""

    mask_path: Path
    cube_path: Path
    naming_convention: str


class TrainingPairResolver:
    """Resolve a training cube from the single mask selected by a user.

    Supported file-name conventions, for any supported cube/mask extension:

    - ``image_XXX.png`` with ``image_XXX.bil`` (same stem)
    - ``image_XXX_mask.png`` with ``image_XXX_mask.bil`` (same stem)
    - ``image_XXX_mask.png`` with ``image_XXX.bil`` (mask suffix removed)
    - ``image_XXX_mask.png`` with ``image_XXX_hyperspectral.bil``

    The cube data file must have a same-stem ``.hdr`` beside it. Resolution is
    case-insensitive and never searches outside the selected mask directory.
    """

    def resolve(self, mask_path: str | Path) -> TrainingFilePair:
        mask = Path(mask_path).expanduser().resolve()
        if not mask.is_file():
            raise ClassificationError(f"Ground-truth mask does not exist: {mask}")
        if mask.suffix.casefold() not in SUPPORTED_MASK_EXTENSIONS:
            raise ClassificationError(
                "Ground-truth mask must be PNG, TIFF, BMP, or JPEG."
            )

        stem = mask.stem
        if stem.casefold().endswith("_mask"):
            base_stem = stem[:-5]
        else:
            base_stem = stem
        raw_candidates = (
            (stem, "same file stem"),
            (base_stem, "mask suffix removed"),
            (f"{base_stem}_hyperspectral", "_mask/_hyperspectral suffixes"),
        )
        candidates = tuple(
            candidate
            for index, candidate in enumerate(raw_candidates)
            if candidate[0].casefold()
            not in {item[0].casefold() for item in raw_candidates[:index]}
        )
        directory_files = {
            path.name.casefold(): path
            for path in mask.parent.iterdir()
            if path.is_file()
        }
        incomplete: list[str] = []
        for candidate_stem, convention in candidates:
            header = directory_files.get(f"{candidate_stem}.hdr".casefold())
            data = next(
                (
                    directory_files.get(
                        f"{candidate_stem}{extension}".casefold()
                    )
                    for extension in SUPPORTED_CUBE_EXTENSIONS
                    if directory_files.get(
                        f"{candidate_stem}{extension}".casefold()
                    )
                    is not None
                ),
                None,
            )
            if header is not None and data is not None:
                return TrainingFilePair(mask, data, convention)
            if header is not None or data is not None:
                missing = "data file" if data is None else ".hdr file"
                incomplete.append(f"{candidate_stem}: missing {missing}")

        expected = (
            f"{base_stem}.hdr plus {base_stem}.bil (or another supported "
            f"cube extension), or {base_stem}_hyperspectral.hdr plus "
            f"{base_stem}_hyperspectral.bil"
        )
        detail = (
            f" Found an incomplete pair ({'; '.join(incomplete)})."
            if incomplete
            else ""
        )
        raise ClassificationError(
            f"No hyperspectral cube matches mask '{mask.name}'. Expected {expected}."
            f"{detail}"
        )


@dataclass(frozen=True, slots=True)
class SupervisedClassificationRequest:
    """Configuration for one-example SPy supervised classification.

    When no explicit band indices are supplied, ``max_features`` evenly
    samples the training wavelength range. This bounds memory and covariance
    requirements for the application's 480-band captures while recording the
    exact selected features in the result.
    """

    classifier: SupervisedClassifierType | str
    band_indices: tuple[int, ...] | None = None
    max_features: int = 24
    wavelength_tolerance_nm: float = 0.5

    def __post_init__(self) -> None:
        try:
            classifier = SupervisedClassifierType(self.classifier)
        except ValueError as exc:
            raise ClassificationError(
                f"Unsupported supervised classifier: {self.classifier}"
            ) from exc
        object.__setattr__(self, "classifier", classifier)
        if isinstance(self.max_features, bool) or not isinstance(
            self.max_features, Integral
        ):
            raise ClassificationError("Maximum feature count must be an integer.")
        if self.max_features < 1:
            raise ClassificationError("Maximum feature count must be at least one.")
        object.__setattr__(self, "max_features", int(self.max_features))
        if self.wavelength_tolerance_nm < 0:
            raise ClassificationError("Wavelength tolerance cannot be negative.")
        if self.band_indices is not None:
            normalized = tuple(self.band_indices)
            if not normalized:
                raise ClassificationError("At least one training band is required.")
            if any(
                isinstance(index, bool) or not isinstance(index, Integral)
                for index in normalized
            ):
                raise ClassificationError("Training band indices must be integers.")
            normalized = tuple(int(index) for index in normalized)
            if len(set(normalized)) != len(normalized):
                raise ClassificationError("Training band indices must be unique.")
            object.__setattr__(self, "band_indices", normalized)


@dataclass(frozen=True, slots=True)
class SupervisedClassificationResult:
    """One-example classification result with class-first one-hot masks."""

    class_map: np.ndarray
    one_hot_masks: np.ndarray
    class_ids: tuple[int, ...]
    class_pixel_counts: np.ndarray
    training_sample_counts: np.ndarray
    classifier: SupervisedClassifierType
    training_band_indices: tuple[int, ...]
    target_band_indices: tuple[int, ...]
    band_wavelengths_nm: np.ndarray
    training_mask_path: Path
    training_cube_path: Path

    @property
    def n_classes(self) -> int:
        return len(self.class_ids)

    def mask_for_class(self, class_id: int) -> np.ndarray:
        """Return an HxW mask by its original positive training class ID."""

        try:
            index = self.class_ids.index(int(class_id))
        except ValueError as exc:
            raise ClassificationError(
                f"Unknown supervised class ID: {class_id}"
            ) from exc
        return self.one_hot_masks[index]


def load_binary_training_mask(
    mask_path: str | Path,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    """Load a target/background mask as ``0=ignored, 1=target, 2=background``.

    Exact indexed masks containing IDs 0, 1, and 2 are preserved. Ordinary
    black/white masks, including JPEGs, are normalized conservatively:
    grayscale values >= 240 are target, <= 15 are background, and compression
    values between those bounds are ignored rather than becoming false classes.
    """

    path = Path(mask_path).expanduser().resolve()
    try:
        with Image.open(path) as image:
            grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    except (OSError, UnidentifiedImageError) as exc:
        raise ClassificationError(f"Could not read ground-truth mask: {exc}") from exc
    if grayscale.shape != expected_shape:
        raise ClassificationError(
            f"Mask shape {grayscale.shape} does not match training cube shape "
            f"{expected_shape}."
        )

    unique = set(int(value) for value in np.unique(grayscale))
    if unique.issubset({0, 1}) and 1 in unique:
        labels = np.where(grayscale == 1, 1, 2).astype(np.int16)
    elif unique.issubset({0, 1, 2}) and 1 in unique and 2 in unique:
        labels = grayscale.astype(np.int16)
    else:
        labels = np.zeros(grayscale.shape, dtype=np.int16)
        labels[grayscale >= 240] = 1
        labels[grayscale <= 15] = 2
    if not np.any(labels == 1) or not np.any(labels == 2):
        raise ClassificationError(
            "The training mask must contain both target and background pixels."
        )
    labels.setflags(write=False)
    return labels


@dataclass(frozen=True, slots=True)
class UnsupervisedClassificationRequest:
    """Parameters for SPy's K-means unsupervised classification.

    ``n_classes`` and ``max_iterations`` map directly to SPy's ``nclusters``
    and ``max_iterations`` arguments. ``band_indices=None`` uses every band.
    A Controller may supply a reproducible subset for faster/lower-memory
    exploratory classification of very large cubes.
    """

    n_classes: int
    max_iterations: int = 20
    band_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.n_classes, bool) or not isinstance(self.n_classes, Integral):
            raise ClassificationError("Number of classes must be an integer.")
        if self.n_classes < 2:
            raise ClassificationError("K-means requires at least two classes.")
        if isinstance(self.max_iterations, bool) or not isinstance(
            self.max_iterations, Integral
        ):
            raise ClassificationError("Maximum iterations must be an integer.")
        if self.max_iterations < 1:
            raise ClassificationError("Maximum iterations must be at least one.")
        object.__setattr__(self, "n_classes", int(self.n_classes))
        object.__setattr__(self, "max_iterations", int(self.max_iterations))
        if self.band_indices is not None:
            normalized = tuple(self.band_indices)
            if not normalized:
                raise ClassificationError(
                    "At least one classification band is required."
                )
            if any(
                isinstance(index, bool) or not isinstance(index, Integral)
                for index in normalized
            ):
                raise ClassificationError(
                    "Classification band indices must be integers."
                )
            normalized = tuple(int(index) for index in normalized)
            if len(set(normalized)) != len(normalized):
                raise ClassificationError("Classification band indices must be unique.")
            object.__setattr__(self, "band_indices", normalized)


@dataclass(frozen=True, slots=True)
class UnsupervisedClassificationResult:
    """View-neutral K-means result.

    ``class_map`` has shape ``(H, W)`` and stores zero-based class IDs.
    ``one_hot_masks`` has class-first shape ``(K, H, W)`` and dtype ``uint8``;
    therefore ``one_hot_masks[class_id]`` is the requested HxW binary mask.
    Empty SPy clusters are retained as all-zero masks so the first dimension
    always matches the requested class count.

    ``cluster_centers`` has shape ``(K, selected_bands)``. Band metadata makes
    the centers reproducible and allows a Controller to plot their spectra.
    """

    class_map: np.ndarray
    one_hot_masks: np.ndarray
    cluster_centers: np.ndarray
    class_pixel_counts: np.ndarray
    band_indices: tuple[int, ...]
    band_wavelengths_nm: np.ndarray
    iterations_completed: int

    @property
    def n_classes(self) -> int:
        """Return the requested number of classes, including empty classes."""

        return int(self.one_hot_masks.shape[0])

    def mask_for_class(self, class_id: int) -> np.ndarray:
        """Return the HxW one-hot mask for a zero-based class ID."""

        if not 0 <= class_id < self.n_classes:
            raise ClassificationError(
                f"Class {class_id} is outside the valid range 0..{self.n_classes - 1}."
            )
        return self.one_hot_masks[class_id]


class ClassificationService:
    """Stateless MVC Model service for hyperspectral classification."""

    def estimate_kmeans_working_bytes(
        self,
        data: HSIData,
        request: UnsupervisedClassificationRequest,
    ) -> int:
        """Estimate peak bytes used by SPy's in-memory K-means implementation.

        This is a conservative planning estimate, not an allocation guarantee.
        Controllers can compare it with available memory before starting a
        worker and offer cropping or a smaller band subset when appropriate.
        """

        band_indices = self._resolve_band_indices(data, request.band_indices)
        pixels = data.rows * data.columns
        input_bytes = pixels * len(band_indices) * np.dtype(np.float32).itemsize
        difference_bytes = pixels * len(band_indices) * np.dtype(np.float64).itemsize
        distance_bytes = pixels * request.n_classes * np.dtype(np.float64).itemsize
        one_hot_bytes = pixels * request.n_classes * np.dtype(np.uint8).itemsize
        return input_bytes + difference_bytes + distance_bytes + one_hot_bytes

    def classify_unsupervised(
        self,
        data: HSIData,
        request: UnsupervisedClassificationRequest,
        *,
        progress: ProgressCallback | None = None,
        is_cancelled: CancellationCheck | None = None,
    ) -> UnsupervisedClassificationResult:
        """Group every pixel by spectral similarity using SPy K-means.

        The method loads the selected bands into memory because SPy's official
        guidance notes that iterative algorithms are substantially faster this
        way. It then calls :func:`spectral.kmeans` and converts SPy's HxW class
        map to the requested class-first one-hot representation.

        Run this method in a Controller worker. Progress callbacks execute on
        that worker thread. Cancellation is checked before/after the cube read
        and between K-means iterations; disk reads and individual SPy
        iterations cannot be interrupted safely.

        Raises:
            ClassificationError: Parameters, cube values, or SPy output are invalid.
            CancelledError: The Controller requested cancellation.
        """

        band_indices = self._resolve_band_indices(data, request.band_indices)
        # A polygon crop clusters only its region, so that is the pixel budget.
        pixels = (
            int(np.count_nonzero(data.roi_mask))
            if data.roi_mask is not None
            else data.rows * data.columns
        )
        if request.n_classes > pixels:
            raise ClassificationError(
                f"Number of classes ({request.n_classes}) cannot exceed the "
                f"number of image pixels ({pixels})."
            )

        self._check_cancelled(is_cancelled)
        self._emit(progress, 0, "Preparing K-means classification")
        try:
            cube = data.read_bands(band_indices)
        except (CancelledError, ClassificationError):
            raise
        except MemoryError as exc:
            raise ClassificationError(
                "Not enough memory to load the selected classification bands; "
                "crop the image or select fewer bands."
            ) from exc
        except Exception as exc:
            raise ClassificationError(
                f"Could not read classification data: {exc}"
            ) from exc

        # K-means is a global fit, so excluded pixels would pull the cluster
        # centres and can claim whole classes of their own. Cluster the region
        # of interest alone, as an (N, 1, bands) pseudo-image SPy accepts, and
        # scatter the labels back into the full frame afterwards.
        region = data.masked(cube)
        cluster_input = (
            region.select()[:, None, :] if region.is_masked else cube
        )

        self._check_cancelled(is_cancelled)
        self._emit(progress, 10, "Running SPy K-means")
        iteration_count = 0

        def compare_iterations(_previous: np.ndarray, _current: np.ndarray) -> bool:
            nonlocal iteration_count
            iteration_count += 1
            self._check_cancelled(is_cancelled)
            completed = min(
                90,
                10 + round(80 * iteration_count / request.max_iterations),
            )
            self._emit(
                progress,
                completed,
                f"K-means iteration {iteration_count} complete",
            )
            return False

        try:
            class_map, centers = spy_kmeans(
                cluster_input,
                nclusters=request.n_classes,
                max_iterations=request.max_iterations,
                compare=compare_iterations,
            )
        except (CancelledError, ClassificationError):
            raise
        except MemoryError as exc:
            raise ClassificationError(
                "Not enough memory to run K-means; crop the image or select "
                "fewer bands."
            ) from exc
        except Exception as exc:
            raise ClassificationError(f"SPy K-means failed: {exc}") from exc

        self._check_cancelled(is_cancelled)
        class_map = np.asarray(class_map)
        centers = np.asarray(centers, dtype=np.float32)
        self._validate_spy_result(
            class_map,
            centers,
            rows=cluster_input.shape[0],
            columns=cluster_input.shape[1],
            n_classes=request.n_classes,
            n_bands=len(band_indices),
        )
        class_map = class_map.astype(np.int32, copy=False)
        labels = class_map.reshape(-1)
        if region.is_masked:
            # -1 marks "outside the region": it matches no class ID, so every
            # one-hot layer is zero there. ClassificationLayerModel already
            # permits unclaimed pixels (it requires layers not to overlap,
            # not to cover the frame).
            class_map = region.scatter(labels, -1).astype(np.int32, copy=False)

        class_ids = np.arange(request.n_classes, dtype=np.int32)[:, None, None]
        one_hot_masks = (class_map[None, :, :] == class_ids).astype(np.uint8)
        class_pixel_counts = np.bincount(
            labels, minlength=request.n_classes
        ).astype(np.int64, copy=False)
        wavelengths = data.wavelengths_nm[np.asarray(band_indices, dtype=np.intp)]

        for array in (
            class_map,
            one_hot_masks,
            centers,
            class_pixel_counts,
            wavelengths,
        ):
            array.setflags(write=False)

        self._emit(progress, 100, "K-means classification complete")
        return UnsupervisedClassificationResult(
            class_map=class_map,
            one_hot_masks=one_hot_masks,
            cluster_centers=centers,
            class_pixel_counts=class_pixel_counts,
            band_indices=band_indices,
            band_wavelengths_nm=wavelengths,
            iterations_completed=iteration_count,
        )

    def classify_supervised(
        self,
        target_data: HSIData,
        training_data: HSIData,
        training_mask_path: str | Path,
        request: SupervisedClassificationRequest,
        *,
        progress: ProgressCallback | None = None,
        is_cancelled: CancellationCheck | None = None,
    ) -> SupervisedClassificationResult:
        """Train on one labeled reference cube and classify the target cube.

        The training mask is binary target/background data. A conservative
        feature subset is used by default because Gaussian-family covariance
        classifiers are poorly conditioned and memory-intensive on the full
        480-band captures. Training and target bands are matched by wavelength,
        so spatial dimensions may differ while spectral meaning remains aligned.

        Run this method in a Controller worker. SPy's training and full-image
        classification calls cannot be interrupted internally; cancellation is
        checked between the documented stages.
        """

        self._check_cancelled(is_cancelled)
        self._emit(progress, 0, "Validating supervised training example")
        training_indices = self._supervised_training_bands(training_data, request)
        target_indices = self._matching_target_bands(
            training_data,
            target_data,
            training_indices,
            request.wavelength_tolerance_nm,
        )
        labels = load_binary_training_mask(
            training_mask_path,
            (training_data.rows, training_data.columns),
        )
        class_ids = tuple(int(value) for value in np.unique(labels) if value > 0)
        training_counts = np.asarray(
            [np.count_nonzero(labels == class_id) for class_id in class_ids],
            dtype=np.int64,
        )
        insufficient = [
            (class_id, int(count))
            for class_id, count in zip(class_ids, training_counts, strict=True)
            if count < len(training_indices)
        ]
        if insufficient:
            details = ", ".join(
                f"class {class_id}: {count} pixels"
                for class_id, count in insufficient
            )
            raise ClassificationError(
                f"Each training class needs at least {len(training_indices)} "
                f"pixels for the selected features ({details})."
            )

        self._check_cancelled(is_cancelled)
        self._emit(progress, 10, "Reading supervised training spectra")
        try:
            training_cube = training_data.read_bands(training_indices)
        except MemoryError as exc:
            raise ClassificationError(
                "Not enough memory to load the supervised training features."
            ) from exc
        except Exception as exc:
            raise ClassificationError(f"Could not read training cube: {exc}") from exc
        if not np.all(np.isfinite(training_cube)):
            raise ClassificationError("Training cube contains non-finite values.")

        self._check_cancelled(is_cancelled)
        self._emit(progress, 30, f"Training SPy {request.classifier.value}")
        try:
            classes = create_training_classes(training_cube, labels)
            classifier_class = {
                SupervisedClassifierType.GAUSSIAN: GaussianClassifier,
                SupervisedClassifierType.MAHALANOBIS: MahalanobisDistanceClassifier,
            }[request.classifier]
            classifier = classifier_class(classes)
            trained_ids = tuple(
                int(training_class.index)
                for training_class in classifier.classes
            )
        except Exception as exc:
            raise ClassificationError(
                f"SPy could not train {request.classifier.value}: {exc}"
            ) from exc
        if trained_ids != class_ids:
            omitted = sorted(set(class_ids) - set(trained_ids))
            raise ClassificationError(
                f"SPy omitted training classes {omitted}; provide more labeled "
                "pixels or select fewer spectral features."
            )

        self._check_cancelled(is_cancelled)
        self._emit(progress, 50, "Reading target image features")
        try:
            target_cube = target_data.read_bands(target_indices)
        except MemoryError as exc:
            raise ClassificationError(
                "Not enough memory to load the target classification features."
            ) from exc
        except Exception as exc:
            raise ClassificationError(f"Could not read target cube: {exc}") from exc
        if not np.all(np.isfinite(target_cube)):
            raise ClassificationError("Target cube contains non-finite values.")

        self._check_cancelled(is_cancelled)
        self._emit(progress, 65, f"Classifying with {request.classifier.value}")
        try:
            class_map = np.asarray(classifier.classify_image(target_cube))
        except MemoryError as exc:
            raise ClassificationError(
                "Not enough memory to classify the target image; crop it first."
            ) from exc
        except Exception as exc:
            raise ClassificationError(
                f"SPy {request.classifier.value} classification failed: {exc}"
            ) from exc
        if class_map.shape != (target_data.rows, target_data.columns):
            raise ClassificationError(
                f"SPy returned class map shape {class_map.shape}; expected "
                f"{(target_data.rows, target_data.columns)}."
            )
        if not np.issubdtype(class_map.dtype, np.integer):
            raise ClassificationError("SPy returned non-integer supervised labels.")
        unexpected = sorted(
            set(int(value) for value in np.unique(class_map)) - set(class_ids)
        )
        if unexpected:
            raise ClassificationError(
                f"SPy returned unknown supervised class IDs: {unexpected}."
            )

        self._check_cancelled(is_cancelled)
        self._emit(progress, 90, "Building supervised one-hot masks")
        class_map = class_map.astype(np.int32, copy=False)
        if target_data.roi_mask is not None:
            # Gaussian/Mahalanobis label each pixel independently, so unlike
            # K-means the excluded pixels never influenced the model; blanking
            # them afterwards is both correct and cheaper than pre-selecting.
            class_map = np.where(target_data.roi_mask, class_map, -1).astype(
                np.int32, copy=False
            )
        id_array = np.asarray(class_ids, dtype=np.int32)[:, None, None]
        one_hot_masks = (class_map[None, :, :] == id_array).astype(np.uint8)
        class_counts = np.asarray(
            [np.count_nonzero(class_map == class_id) for class_id in class_ids],
            dtype=np.int64,
        )
        wavelengths = training_data.wavelengths_nm[
            np.asarray(training_indices, dtype=np.intp)
        ]
        for array in (
            class_map,
            one_hot_masks,
            class_counts,
            training_counts,
            wavelengths,
        ):
            array.setflags(write=False)

        self._emit(progress, 100, "Supervised classification complete")
        return SupervisedClassificationResult(
            class_map=class_map,
            one_hot_masks=one_hot_masks,
            class_ids=class_ids,
            class_pixel_counts=class_counts,
            training_sample_counts=training_counts,
            classifier=request.classifier,
            training_band_indices=training_indices,
            target_band_indices=target_indices,
            band_wavelengths_nm=wavelengths,
            training_mask_path=Path(training_mask_path).expanduser().resolve(),
            training_cube_path=training_data.source_path,
        )

    @staticmethod
    def _supervised_training_bands(
        training_data: HSIData,
        request: SupervisedClassificationRequest,
    ) -> tuple[int, ...]:
        if request.band_indices is not None:
            indices = tuple(request.band_indices)
        else:
            feature_count = min(training_data.bands, request.max_features)
            indices = tuple(
                int(index)
                for index in np.linspace(
                    0,
                    training_data.bands - 1,
                    feature_count,
                    dtype=np.intp,
                )
            )
        if min(indices) < 0 or max(indices) >= training_data.bands:
            raise ClassificationError(
                f"Training bands must lie in 0..{training_data.bands - 1}."
            )
        return indices

    @staticmethod
    def _matching_target_bands(
        training_data: HSIData,
        target_data: HSIData,
        training_indices: tuple[int, ...],
        tolerance_nm: float,
    ) -> tuple[int, ...]:
        training_wavelengths = training_data.wavelengths_nm[
            np.asarray(training_indices, dtype=np.intp)
        ]
        target_wavelengths = target_data.wavelengths_nm
        target_indices: list[int] = []
        for wavelength in training_wavelengths:
            index = int(np.argmin(np.abs(target_wavelengths - wavelength)))
            difference = abs(float(target_wavelengths[index] - wavelength))
            if difference > tolerance_nm:
                raise ClassificationError(
                    f"Target cube has no band within {tolerance_nm:g} nm of "
                    f"training wavelength {wavelength:g} nm."
                )
            target_indices.append(index)
        if len(set(target_indices)) != len(target_indices):
            raise ClassificationError(
                "Multiple training features map to the same target wavelength band."
            )
        return tuple(target_indices)

    @staticmethod
    def _resolve_band_indices(
        data: HSIData,
        requested: tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        indices = tuple(range(data.bands)) if requested is None else tuple(requested)
        if not indices:
            raise ClassificationError("The loaded cube contains no spectral bands.")
        if min(indices) < 0 or max(indices) >= data.bands:
            raise ClassificationError(
                f"Classification bands must lie in 0..{data.bands - 1}."
            )
        return indices

    @staticmethod
    def _validate_spy_result(
        class_map: np.ndarray,
        centers: np.ndarray,
        *,
        rows: int,
        columns: int,
        n_classes: int,
        n_bands: int,
    ) -> None:
        if class_map.shape != (rows, columns):
            raise ClassificationError(
                f"SPy returned class map shape {class_map.shape}; expected "
                f"{(rows, columns)}."
            )
        if centers.shape != (n_classes, n_bands):
            raise ClassificationError(
                f"SPy returned center shape {centers.shape}; expected "
                f"{(n_classes, n_bands)}."
            )
        if not np.issubdtype(class_map.dtype, np.integer):
            raise ClassificationError("SPy returned non-integer class labels.")
        if class_map.size and (
            int(class_map.min()) < 0 or int(class_map.max()) >= n_classes
        ):
            raise ClassificationError("SPy returned an out-of-range class label.")
        if not np.all(np.isfinite(centers)):
            raise ClassificationError("SPy returned non-finite cluster centers.")

    @staticmethod
    def _emit(callback: ProgressCallback | None, value: int, message: str) -> None:
        if callback is not None:
            callback(value, message)

    @staticmethod
    def _check_cancelled(callback: CancellationCheck | None) -> None:
        if callback is not None and callback():
            raise CancelledError("Classification was cancelled.")
