"""Splitting a supervised plant class into one class per separate leaf."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core import ClassificationError, ClassificationLayerModel, separate_leaves
from core.classification_model import (
    SupervisedClassificationResult,
    SupervisedClassifierType,
)


def supervised_result(class_map: np.ndarray) -> SupervisedClassificationResult:
    class_map = class_map.astype(np.int32)
    class_ids = (1, 2)
    masks = (class_map[None] == np.asarray(class_ids)[:, None, None]).astype(np.uint8)
    return SupervisedClassificationResult(
        class_map=class_map,
        one_hot_masks=masks,
        class_ids=class_ids,
        class_pixel_counts=masks.reshape(2, -1).sum(axis=1),
        training_sample_counts=np.asarray([10, 20]),
        classifier=SupervisedClassifierType.GAUSSIAN,
        training_band_indices=(0,),
        target_band_indices=(0,),
        band_wavelengths_nm=np.asarray([550.0]),
        training_mask_path=Path("mask.png"),
        training_cube_path=Path("cube.img"),
    )


def three_leaf_map() -> np.ndarray:
    class_map = np.full((40, 60), 2)
    class_map[5:25, 5:25] = 1  # 400 px, largest
    class_map[5:20, 35:50] = 1  # 225 px
    class_map[30:38, 10:20] = 1  # 80 px
    class_map[35, 50] = 1  # one-pixel speck: noise, not a leaf
    return class_map


def test_each_separate_cluster_becomes_its_own_leaf_class():
    separation = separate_leaves(supervised_result(three_leaf_map()))
    result = separation.result

    assert separation.leaf_count == 3
    assert result.class_ids == (3, 4, 5, 1, 2)
    assert separation.leaf_names == {3: "Leaf 1", 4: "Leaf 2", 5: "Leaf 3"}
    # Largest leaf first, and opening must not erode leaf edges.
    assert result.class_pixel_counts.tolist() == [400, 225, 80, 1, 40 * 60 - 706]
    assert result.class_map[35, 50] == 1
    assert result.training_sample_counts.tolist() == [0, 0, 0, 10, 20]
    assert result.one_hot_masks.shape == (5, 40, 60)
    assert np.all(result.one_hot_masks.sum(axis=0) == 1)
    ClassificationLayerModel(result, names=separation.leaf_names)


def test_empty_plant_class_is_dropped_when_every_pixel_is_a_leaf():
    class_map = three_leaf_map()
    class_map[35, 50] = 2

    result = separate_leaves(supervised_result(class_map)).result

    assert result.class_ids == (3, 4, 5, 2)


def test_thin_bridge_between_leaves_is_cut():
    class_map = np.full((30, 50), 2)
    class_map[5:25, 5:20] = 1
    class_map[5:25, 30:45] = 1
    class_map[15, 20:30] = 1  # one-pixel-wide bridge joining the two leaves

    separation = separate_leaves(supervised_result(class_map))

    assert separation.leaf_count == 2
    counts = separation.result.class_pixel_counts.tolist()
    # Every plant pixel, bridge included, ends up in one of the two leaves.
    assert sum(counts[:2]) == 2 * 300 + 10
    assert 1 not in separation.result.class_ids


def test_out_of_region_pixels_are_left_alone():
    class_map = three_leaf_map()
    class_map[:, 55:] = -1

    result = separate_leaves(supervised_result(class_map)).result

    assert np.all(result.class_map[:, 55:] == -1)


def test_missing_or_empty_plant_class_is_reported():
    with pytest.raises(ClassificationError, match="empty"):
        separate_leaves(supervised_result(np.full((10, 10), 2)))
    with pytest.raises(ClassificationError, match="plant class"):
        separate_leaves(supervised_result(three_leaf_map()), plant_class_id=7)
