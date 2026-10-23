"""Split a supervised "Plant" class into one class per clearly separate leaf.

Supervised ground-truth masks only distinguish plant from background, so a
capture of several leaves comes back as one plant class. Leaves that sit apart
from each other form separate connected regions of that class; this module
turns each sizeable region into its own class so the layer panel can show,
hide, rename and export every leaf individually.

A small morphological opening runs before labelling so leaves joined only by
a thin bridge of pixels (touching tips, a stray misclassified line) still come
apart. Pixels the opening removed are then handed back to the nearest leaf in
the same original region, so leaf edges are not eroded in the final result.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy import ndimage as ndi

from .classification_model import SupervisedClassificationResult
from .errors import ClassificationError

PLANT_CLASS_ID = 1
BACKGROUND_CLASS_ID = 2
# Once the plant is split into leaves, background is drawn black so the
# coloured leaves stand out instead of competing with a background hue.
SEPARATED_BACKGROUND_COLOR = (0, 0, 0)

# Regions smaller than this share of the plant (or the absolute floor) are
# treated as noise and left in the original plant class.
_DEFAULT_MIN_AREA_FRACTION = 0.01
_DEFAULT_MIN_AREA_PIXELS = 25
_DEFAULT_OPENING_ITERATIONS = 2


@dataclass(frozen=True, slots=True)
class LeafSeparationResult:
    """The re-labelled result, a display name per leaf, and pinned colours.

    ``fixed_colors`` maps class IDs that must not take a palette hue to their
    display colour -- the background class, drawn black.
    """

    result: SupervisedClassificationResult
    leaf_names: dict[int, str]
    fixed_colors: dict[int, tuple[int, int, int]] = field(default_factory=dict)

    @property
    def leaf_count(self) -> int:
        return len(self.leaf_names)

    @property
    def leaf_numbers(self) -> dict[int, int]:
        """Each leaf class ID mapped to the number in its "Leaf N" name."""

        return {
            class_id: index
            for index, class_id in enumerate(self.leaf_names, start=1)
        }


def separate_leaves(
    result: SupervisedClassificationResult,
    *,
    plant_class_id: int = PLANT_CLASS_ID,
    min_area_fraction: float = _DEFAULT_MIN_AREA_FRACTION,
    min_area_pixels: int = _DEFAULT_MIN_AREA_PIXELS,
    opening_iterations: int = _DEFAULT_OPENING_ITERATIONS,
) -> LeafSeparationResult:
    """Return ``result`` with each clear plant cluster moved into its own class.

    Leaf classes get new IDs above every existing class ID and come first in
    ``class_ids``. Plant pixels that belong to no leaf (small fragments) stay
    in ``plant_class_id``, which is dropped entirely when nothing is left in
    it. Other classes and out-of-region (``-1``) pixels are untouched.
    """

    if plant_class_id not in result.class_ids:
        raise ClassificationError(
            f"The classification has no plant class (ID {plant_class_id}) to separate."
        )
    if not 0.0 <= min_area_fraction < 1.0:
        raise ClassificationError("Minimum leaf area fraction must be in [0, 1).")
    if opening_iterations < 0:
        raise ClassificationError("Opening iterations cannot be negative.")

    plant = result.class_map == plant_class_id
    plant_pixels = int(np.count_nonzero(plant))
    if plant_pixels == 0:
        raise ClassificationError("The plant class is empty; there are no leaves to separate.")

    leaf_labels = _label_leaves(plant, opening_iterations)
    min_area = max(int(min_area_pixels), int(np.ceil(plant_pixels * min_area_fraction)))
    areas = np.bincount(leaf_labels.ravel())
    # Largest first, so "Leaf 1" is the most prominent leaf.
    kept = [
        label
        for label in np.argsort(areas[1:])[::-1] + 1
        if areas[label] >= min_area
    ]
    if not kept:
        raise ClassificationError("No clear leaf clusters were found in the plant class.")

    first_id = max(result.class_ids) + 1
    leaf_ids = tuple(range(first_id, first_id + len(kept)))
    class_map = np.array(result.class_map, dtype=np.int32, copy=True)
    for leaf_id, label in zip(leaf_ids, kept, strict=True):
        class_map[leaf_labels == label] = leaf_id

    leftover = int(np.count_nonzero(class_map == plant_class_id))
    remaining_ids = tuple(
        class_id
        for class_id in result.class_ids
        if class_id != plant_class_id or leftover
    )
    class_ids = leaf_ids + remaining_ids

    id_array = np.asarray(class_ids, dtype=np.int32)[:, None, None]
    one_hot_masks = (class_map[None, :, :] == id_array).astype(np.uint8)
    class_counts = np.asarray(
        [np.count_nonzero(class_map == class_id) for class_id in class_ids],
        dtype=np.int64,
    )
    # Leaves were not separate training classes, so they had no samples.
    original_samples = dict(
        zip(result.class_ids, np.asarray(result.training_sample_counts).tolist())
    )
    training_counts = np.asarray(
        [original_samples.get(class_id, 0) for class_id in class_ids], dtype=np.int64
    )
    for array in (class_map, one_hot_masks, class_counts, training_counts):
        array.setflags(write=False)

    separated = replace(
        result,
        class_map=class_map,
        one_hot_masks=one_hot_masks,
        class_ids=class_ids,
        class_pixel_counts=class_counts,
        training_sample_counts=training_counts,
    )
    names = {leaf_id: f"Leaf {index}" for index, leaf_id in enumerate(leaf_ids, start=1)}
    fixed_colors = (
        {BACKGROUND_CLASS_ID: SEPARATED_BACKGROUND_COLOR}
        if BACKGROUND_CLASS_ID in class_ids
        else {}
    )
    return LeafSeparationResult(
        result=separated, leaf_names=names, fixed_colors=fixed_colors
    )


def _label_leaves(plant: np.ndarray, opening_iterations: int) -> np.ndarray:
    """Label plant pixels by leaf, cutting thin bridges between leaves."""

    structure = ndi.generate_binary_structure(2, 1)
    original_labels, _ = ndi.label(plant, structure=structure)
    if opening_iterations == 0:
        return original_labels

    opened = ndi.binary_opening(plant, structure=structure, iterations=opening_iterations)
    labels, count = ndi.label(opened, structure=structure)
    if count == 0:
        # Every region was thinner than the opening; fall back to plain labels.
        return original_labels

    # Give each pixel the opening removed the label of its nearest opened
    # pixel, but only when that pixel lies in the same original region, so a
    # removed bridge never joins two separate regions back together.
    nearest = ndi.distance_transform_edt(
        labels == 0, return_distances=False, return_indices=True
    )
    nearest_labels = labels[nearest[0], nearest[1]]
    nearest_regions = original_labels[nearest[0], nearest[1]]
    restored = plant & (labels == 0) & (nearest_regions == original_labels)
    labels[restored] = nearest_labels[restored]
    return labels
