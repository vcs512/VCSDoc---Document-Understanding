"""Tests for the BBox schema."""

import pytest

from src.schemas.bbox import BBox


def test_from_quad_computes_min_max() -> None:
    """A quad of four points should yield the enclosing min/max box."""
    quad = {
        "x1": 10,
        "y1": 20,
        "x2": 30,
        "y2": 20,
        "x3": 30,
        "y3": 40,
        "x4": 10,
        "y4": 40,
    }
    bbox = BBox.from_quad(quad)
    assert bbox.x1 == 10
    assert bbox.y1 == 20
    assert bbox.x2 == 30
    assert bbox.y2 == 40


def test_area() -> None:
    """Area should be width times height of the box."""
    bbox = BBox(x1=0, y1=0, x2=10, y2=5)
    assert bbox.area == 50.0


def test_iou_identical_boxes() -> None:
    """Identical boxes should have an IoU of 1.0."""
    a = BBox(x1=0, y1=0, x2=10, y2=10)
    b = BBox(x1=0, y1=0, x2=10, y2=10)
    assert a.iou(b) == pytest.approx(1.0)


def test_iou_adjacent_boxes() -> None:
    """Boxes sharing only a border should have an IoU of 0.0."""
    a = BBox(x1=0, y1=0, x2=10, y2=10)
    b = BBox(x1=10, y1=0, x2=20, y2=10)
    assert a.iou(b) == pytest.approx(0.0)


def test_iou_partial_overlap() -> None:
    """Half-overlapping boxes should yield an IoU of one third."""
    a = BBox(x1=0, y1=0, x2=10, y2=10)
    b = BBox(x1=5, y1=0, x2=15, y2=10)
    assert a.iou(b) == pytest.approx(1 / 3)


def test_iou_nested_box() -> None:
    """A fully nested box should have an IoU equal to the small/union area."""
    outer = BBox(x1=0, y1=0, x2=10, y2=10)
    inner = BBox(x1=2, y1=2, x2=5, y2=5)
    assert outer.iou(inner) == pytest.approx(9 / 100)
