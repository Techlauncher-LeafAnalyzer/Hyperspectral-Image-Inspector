from __future__ import annotations

import numpy as np
import pytest

from core.roi import Masked, combine, polygon_mask


def test_polygon_mask_selects_pixels_by_centre():
    # A triangle over the lower-left half of a 4x4 frame.
    mask = polygon_mask([(0, 0), (4, 0), (0, 4)], (4, 4))

    assert mask.shape == (4, 4)
    assert mask.dtype == np.bool_
    # Pixel (0,0)'s centre (0.5,0.5) is inside; (3,3)'s centre is well outside.
    assert mask[0, 0]
    assert not mask[3, 3]
    # The hypotenuse runs corner to corner, so roughly half the pixels survive.
    assert 4 <= mask.sum() <= 12


def test_polygon_mask_rejects_degenerate_input():
    with pytest.raises(ValueError):
        polygon_mask([(0, 0), (4, 0)], (4, 4))
    with pytest.raises(ValueError):
        polygon_mask([(0, 0), (4, 0), (0, 4)], (0, 4))


def test_combine_treats_none_as_everything_selected():
    mask = np.array([[True, False], [True, True]])

    assert combine(None, None) is None
    assert np.array_equal(combine(None, mask), mask)
    assert np.array_equal(combine(mask, None), mask)
    assert np.array_equal(
        combine(mask, np.array([[True, True], [False, True]])),
        np.array([[True, False], [False, True]]),
    )


def test_combine_rejects_mismatched_frames():
    with pytest.raises(ValueError):
        combine(np.ones((2, 2), bool), np.ones((3, 3), bool))


# --- monad laws ------------------------------------------------------- #
# The container is only useful if map/bind compose predictably, so pin the
# three laws rather than trusting the implementation by inspection.

VALUES = np.arange(4, dtype=np.float32).reshape(2, 2)
MASK_A = np.array([[True, True], [False, True]])
MASK_B = np.array([[True, False], [True, True]])


def _lift(values):
    return Masked(values * 2, MASK_B)


def test_left_identity():
    # unit(x).bind(f) == f(x)
    left, right = Masked.unit(VALUES).bind(_lift), _lift(VALUES)

    assert np.array_equal(left.value, right.value)
    assert np.array_equal(left.mask, right.mask)


def test_right_identity():
    # m.bind(unit) == m
    original = Masked(VALUES, MASK_A)
    result = original.bind(Masked.unit)

    assert np.array_equal(result.value, original.value)
    assert np.array_equal(result.mask, original.mask)


def test_associativity():
    # m.bind(f).bind(g) == m.bind(lambda x: f(x).bind(g))
    original = Masked(VALUES, MASK_A)
    add_one = lambda values: Masked(values + 1, None)

    left = original.bind(_lift).bind(add_one)
    right = original.bind(lambda values: _lift(values).bind(add_one))

    assert np.array_equal(left.value, right.value)
    assert np.array_equal(left.mask, right.mask)


def test_map_preserves_the_mask():
    result = Masked(VALUES, MASK_A).map(lambda values: values + 10)

    assert np.array_equal(result.value, VALUES + 10)
    assert result.mask is MASK_A


def test_bind_intersects_masks():
    result = Masked(VALUES, MASK_A).bind(_lift)

    assert np.array_equal(result.mask, np.logical_and(MASK_A, MASK_B))


def test_select_and_scatter_round_trip():
    container = Masked(VALUES, MASK_A)

    restored = container.scatter(container.select(), -1.0)

    assert np.array_equal(restored[MASK_A], VALUES[MASK_A])
    assert restored[~MASK_A] == pytest.approx(-1.0)


def test_select_flattens_a_cube_to_pixels_by_bands():
    cube = np.arange(12, dtype=np.float32).reshape(2, 2, 3)

    assert Masked(cube, MASK_A).select().shape == (3, 3)


def test_fill_marks_excluded_pixels_without_touching_the_rest():
    filled = Masked(VALUES, MASK_A).fill()

    assert np.isnan(filled[~MASK_A]).all()
    assert np.array_equal(filled[MASK_A], VALUES[MASK_A])


def test_unmasked_container_is_a_passthrough():
    container = Masked.unit(VALUES)

    assert not container.is_masked
    assert container.count == 4
    assert np.array_equal(container.fill(), VALUES)
    assert np.array_equal(container.select(), VALUES.reshape(-1))
