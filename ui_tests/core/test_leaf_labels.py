from __future__ import annotations

import numpy as np

from core import draw_leaf_labels, place_leaf_labels


def _two_leaves():
    class_map = np.full((60, 60), 2, dtype=np.int32)
    class_map[5:20, 5:25] = 10   # Leaf 1
    class_map[45:60, 30:50] = 11  # Leaf 2 touches the bottom edge
    return class_map


def test_labels_sit_below_each_leaf_centred_on_its_columns():
    labels = {label.class_id: label for label in place_leaf_labels(_two_leaves(), {10: 1, 11: 2})}
    assert labels[10].text == "1" and labels[10].x == 14  # mean of columns 5..24
    assert labels[10].y > 19  # beneath the leaf's lowest row
    # No room under a leaf on the bottom edge: the label moves up to fit.
    assert labels[11].text == "2" and labels[11].y < 60


def test_draw_burns_labels_only_for_visible_classes():
    class_map = _two_leaves()
    labels = place_leaf_labels(class_map, {10: 1, 11: 2})
    rgb = np.zeros((60, 60, 3), dtype=np.uint8)
    drawn = draw_leaf_labels(rgb, labels)
    assert drawn is not rgb and not rgb.any()  # input untouched
    below_leaf_1 = drawn[20:40, 5:25]
    assert (below_leaf_1.min(axis=2) > 150).any()  # light text pixels
    only_leaf_2 = draw_leaf_labels(rgb, labels, visible_ids=(11,))
    assert not only_leaf_2[20:40, 5:25].any()
    assert draw_leaf_labels(rgb, labels, visible_ids=()) is rgb
