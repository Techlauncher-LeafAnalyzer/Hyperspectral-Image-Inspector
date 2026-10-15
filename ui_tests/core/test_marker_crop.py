import numpy as np

from core.marker_crop import detect_markers, find_plus_crop, plus_polygon
from core.roi import polygon_mask


def _scene(tags, shape=(600, 200)):
    img = np.full((*shape, 3), 255, np.uint8)
    for left, top, size in tags:
        img[top : top + size, left : left + size] = 40
        img[top + 8 : top + size - 8 : 2, left + 8 : left + size - 8 : 3] = 255
    return img


TAGS = [(30, 100, 40), (130, 100, 40), (30, 400, 40), (130, 400, 40)]


def test_detects_four_tags():
    layout = detect_markers(_scene(TAGS))
    assert layout is not None
    assert (layout.top_left.left, layout.top_left.top) == (30, 100)
    assert (layout.bottom_right.right, layout.bottom_right.bottom) == (170, 440)


def test_plus_polygon_excludes_tags_and_keeps_centre():
    img = _scene(TAGS)
    poly = find_plus_crop(img)
    mask = polygon_mask(np.asarray(poly, float) , img.shape[:2])
    assert mask[300, 100]
    assert mask[250, 50] and mask[250, 150]  # arms between the tags
    for left, top, size in TAGS:
        assert not mask[top + size // 2, left + size // 2]


def test_no_pattern_returns_none():
    assert find_plus_crop(np.full((600, 200, 3), 255, np.uint8)) is None
    assert find_plus_crop(_scene(TAGS[:3])) is None


def test_plus_polygon_has_twelve_vertices():
    assert len(plus_polygon(detect_markers(_scene(TAGS)))) == 12
