# tests/shape/test_boolean.py
# Tests for Union, Intersection, Difference bounds(), in particular
# Intersection.bounds() on disjoint operands.

import math

import pytest
import torch

from metashapes.shape import Union, Intersection, Difference
from metashapes.shape.base import EMPTY_BOUNDS, is_empty_bounds
from metashapes.shape.primitives.quads import Rectangle


def _rect(size=(1.0, 1.0), center=(0.0, 0.0)):
    return Rectangle(center=torch.tensor(center), size=torch.tensor(size))


class TestIsEmptyBounds:
    def test_normal_box_is_not_empty(self):
        assert not is_empty_bounds(((0.0, 0.0), (1.0, 1.0)))

    def test_zero_area_box_is_not_empty(self):
        # A degenerate but non-inverted box (xmin == xmax) is still a valid,
        # zero-area extent, not the "no extent at all" empty sentinel.
        assert not is_empty_bounds(((1.0, 1.0), (1.0, 1.0)))

    def test_inverted_box_is_empty(self):
        assert is_empty_bounds(((4.5, 4.5), (0.5, 0.5)))

    def test_canonical_sentinel_is_empty(self):
        assert is_empty_bounds(EMPTY_BOUNDS)


class TestIntersectionBounds:
    def test_disjoint_shapes_give_canonical_empty_bounds(self):
        # Naive max(mins)/min(maxs) inverts for disjoint operands
        # (xmin > xmax, ymin > ymax); bounds() must return the canonical
        # empty sentinel instead.
        i = Intersection(_rect(center=(0.0, 0.0)), _rect(center=(5.0, 5.0)))
        assert i.bounds() == EMPTY_BOUNDS

    def test_overlapping_shapes_give_real_bounds(self):
        i = Intersection(
            _rect(size=(2.0, 2.0), center=(0.0, 0.0)),
            _rect(size=(2.0, 2.0), center=(1.0, 1.0)),
        )
        (x0, y0), (x1, y1) = i.bounds()
        assert not is_empty_bounds(((x0, y0), (x1, y1)))
        assert x0 <= x1 and y0 <= y1

    def test_touching_shapes_are_not_empty(self):
        # Edges exactly touching: max(mins) == min(maxs), a valid
        # zero-width (not inverted) box.
        i = Intersection(
            _rect(size=(1.0, 1.0), center=(0.0, 0.0)),
            _rect(size=(1.0, 1.0), center=(1.0, 0.0)),
        )
        assert not is_empty_bounds(i.bounds())


class TestUnionBoundsWithEmptyChild:
    def test_empty_left_operand_yields_right_bounds(self):
        empty = Intersection(_rect(center=(0.0, 0.0)), _rect(center=(5.0, 5.0)))
        rect = _rect(size=(0.4, 0.6), center=(2.0, 2.0))
        assert Union(empty, rect).bounds() == rect.bounds()

    def test_empty_right_operand_yields_left_bounds(self):
        empty = Intersection(_rect(center=(0.0, 0.0)), _rect(center=(5.0, 5.0)))
        rect = _rect(size=(0.4, 0.6), center=(2.0, 2.0))
        assert Union(rect, empty).bounds() == rect.bounds()

    def test_both_empty_stays_empty(self):
        e1 = Intersection(_rect(center=(0.0, 0.0)), _rect(center=(5.0, 5.0)))
        e2 = Intersection(_rect(center=(0.0, 0.0)), _rect(center=(8.0, 8.0)))
        assert Union(e1, e2).bounds() == EMPTY_BOUNDS


class TestDifferenceBounds:
    def test_returns_left_bounds(self):
        left = _rect(size=(1.0, 1.0), center=(0.0, 0.0))
        right = _rect(size=(0.3, 0.3), center=(0.0, 0.0))
        assert Difference(left, right).bounds() == left.bounds()
