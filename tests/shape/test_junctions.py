# tests/shape/test_junctions.py

import pytest
import torch
from metashapes.shape.primitives.junctions import Cross, TShape
from .conftest import assert_inside, assert_outside, assert_round_trip, assert_bounds_contain, sdf_at


# ---------------------------------------------------------------------------
# Cross
# ---------------------------------------------------------------------------

class TestCross:
    def _basic(self):
        return Cross(center=[0.0, 0.0], length=1.0, width=0.3)

    def test_center_is_inside(self):
        assert_inside(self._basic(), [(0.0, 0.0)])

    def test_arm_tips_inside(self):
        # points along the arms (inside width, inside length)
        c = self._basic()
        assert_inside(c, [(0.4, 0.0), (-0.4, 0.0), (0.0, 0.4), (0.0, -0.4)])

    def test_corners_outside(self):
        c = self._basic()
        # outside all arms
        assert_outside(c, [(0.4, 0.4), (-0.4, 0.4), (0.4, -0.4)])

    def test_tip_boundary(self):
        c = self._basic()
        # point exactly at tip of horizontal arm
        d = sdf_at(c, 0.5, 0.0)
        assert abs(d) < 1e-4

    def test_offset_center(self):
        c = Cross(center=[1.0, -1.0], length=0.8, width=0.2)
        assert_inside(c, [(1.0, -1.0)])
        assert_outside(c, [(0.0, 0.0)])

    def test_rotated(self):
        c = Cross(center=[0.0, 0.0], length=1.0, width=0.3, angle=45.0)
        assert_inside(c, [(0.0, 0.0)])

    def test_outer_corner_radius(self):
        c = Cross(center=[0.0, 0.0], length=1.0, width=0.3, outer_corner_radius=0.05)
        assert_inside(c, [(0.0, 0.0)])

    def test_inner_corner_radius(self):
        c = Cross(
            center=[0.0, 0.0], length=1.0, width=0.3,
            outer_corner_radius=0.04, inner_corner_radius=0.05
        )
        assert_inside(c, [(0.0, 0.0)])

    def test_bounds(self):
        c = self._basic()
        (x0, y0), (x1, y1) = c.bounds()
        assert x0 == pytest.approx(-0.5, abs=1e-5)
        assert x1 == pytest.approx( 0.5, abs=1e-5)
        assert y0 == pytest.approx(-0.5, abs=1e-5)
        assert y1 == pytest.approx( 0.5, abs=1e-5)

    def test_bounds_contain_interior(self):
        c = Cross(center=[0.5, -0.5], length=0.8, width=0.2, angle=20.0)
        assert_bounds_contain(c, [(0.5, -0.5)])

    def test_width_exceeds_length_raises(self):
        with pytest.raises(ValueError):
            Cross(center=[0.0, 0.0], length=0.5, width=0.8)

    def test_outer_radius_too_large_raises(self):
        with pytest.raises(ValueError):
            Cross(center=[0.0, 0.0], length=1.0, width=0.3, outer_corner_radius=0.2)

    def test_round_trip(self):
        c = Cross(
            center=[0.1, -0.1], length=0.9, width=0.25, angle=10.0,
            outer_corner_radius=0.03, inner_corner_radius=0.04
        )
        assert_round_trip(c)

    def test_round_trip_no_rounding(self):
        c = Cross(center=[0.0, 0.0], length=1.0, width=0.3)
        assert_round_trip(c)


# ---------------------------------------------------------------------------
# TShape
# ---------------------------------------------------------------------------

class TestTShape:
    def _basic(self):
        return TShape(center=[0.0, 0.0], length=1.0, width=0.3)

    def test_center_is_inside(self):
        assert_inside(self._basic(), [(0.0, 0.0)])

    def test_stem_is_inside(self):
        t = self._basic()
        assert_inside(t, [(0.0, -0.2), (0.0, 0.2)])

    def test_bar_tip_is_inside(self):
        t = self._basic()
        assert_inside(t, [(0.4, 0.2)])

    def test_below_stem_is_outside(self):
        t = self._basic()
        assert_outside(t, [(0.0, -0.6)])  # below the stem

    def test_top_bar_corners_outside(self):
        t = self._basic()
        # outside the T shape
        assert_outside(t, [(0.4, -0.1), (-0.4, -0.1)])

    def test_offset_center(self):
        t = TShape(center=[2.0, 1.0], length=0.8, width=0.2)
        assert_inside(t, [(2.0, 1.0)])
        assert_outside(t, [(0.0, 0.0)])

    def test_rotated(self):
        t = TShape(center=[0.0, 0.0], length=1.0, width=0.3, angle=90.0)
        assert_inside(t, [(0.0, 0.0)])

    def test_outer_corner_radius(self):
        t = TShape(center=[0.0, 0.0], length=1.0, width=0.3, outer_corner_radius=0.05)
        assert_inside(t, [(0.0, 0.0)])

    def test_inner_corner_radius(self):
        t = TShape(
            center=[0.0, 0.0], length=1.0, width=0.3,
            outer_corner_radius=0.04, inner_corner_radius=0.05
        )
        assert_inside(t, [(0.0, 0.0)])

    def test_width_exceeds_length_raises(self):
        with pytest.raises(ValueError):
            TShape(center=[0.0, 0.0], length=0.5, width=0.8)

    def test_bounds_contain_interior(self):
        t = TShape(center=[0.0, 0.0], length=1.0, width=0.3)
        assert_bounds_contain(t, [(0.0, 0.0), (0.3, 0.3)])

    def test_round_trip(self):
        t = TShape(
            center=[-0.1, 0.2], length=0.8, width=0.2, angle=5.0,
            outer_corner_radius=0.03, inner_corner_radius=0.04
        )
        assert_round_trip(t)

    def test_round_trip_no_rounding(self):
        t = TShape(center=[0.0, 0.0], length=1.0, width=0.3)
        assert_round_trip(t)
