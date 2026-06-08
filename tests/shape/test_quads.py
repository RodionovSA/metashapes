# tests/shape/test_quads.py

import math
import pytest
import torch
from metashapes.shape import Rectangle, ConvexQuad, IsoscelesTrapezoid
from .conftest import assert_inside, assert_outside, assert_round_trip, assert_bounds_contain


# ---------------------------------------------------------------------------
# Rectangle
# ---------------------------------------------------------------------------

class TestRectangle:
    def test_center_is_inside(self):
        r = Rectangle(center=[0.0, 0.0], size=[1.0, 0.5])
        assert_inside(r, [(0.0, 0.0)])

    def test_far_point_is_outside(self):
        r = Rectangle(center=[0.0, 0.0], size=[1.0, 0.5])
        assert_outside(r, [(2.0, 0.0), (0.0, 2.0), (-2.0, -2.0)])

    def test_sdf_value_at_boundary(self):
        r = Rectangle(center=[0.0, 0.0], size=[2.0, 2.0])
        # point exactly on the right edge
        d = r.sdf(torch.tensor(1.0), torch.tensor(0.0)).item()
        assert abs(d) < 1e-5

    def test_negative_interior_sdf(self):
        r = Rectangle(center=[0.0, 0.0], size=[2.0, 2.0])
        d = r.sdf(torch.tensor(0.0), torch.tensor(0.0)).item()
        assert d == pytest.approx(-1.0, abs=1e-5)

    def test_offset_center(self):
        r = Rectangle(center=[1.0, -1.0], size=[0.4, 0.4])
        assert_inside(r, [(1.0, -1.0)])
        assert_outside(r, [(0.0, 0.0)])

    def test_rotated(self):
        # 45-degree rotated square: becomes a diamond; corners at sqrt(2)/2 ≈ 0.707 from center.
        # Edge condition: |x|+|y| = sqrt(2)/2, so (0.5, 0.3) with sum 0.8 is outside.
        r = Rectangle(center=[0.0, 0.0], size=[1.0, 1.0], angle=45.0)
        assert_inside(r, [(0.0, 0.0), (0.6, 0.0)])  # (0.6,0): |0.6|+0 < 0.707 → inside
        assert_outside(r, [(0.5, 0.3), (0.0, 0.8)])

    def test_corner_radius(self):
        r = Rectangle(center=[0.0, 0.0], size=[1.0, 1.0], corner_radius=0.1)
        assert_inside(r, [(0.0, 0.0)])
        # exact corner of the un-rounded box is now slightly outside
        d_corner = r.sdf(torch.tensor(0.5), torch.tensor(0.5)).item()
        assert d_corner > 0

    def test_bounds(self):
        r = Rectangle(center=[1.0, 2.0], size=[2.0, 1.0])
        (x0, y0), (x1, y1) = r.bounds()
        assert x0 == pytest.approx(0.0, abs=1e-5)
        assert x1 == pytest.approx(2.0, abs=1e-5)
        assert y0 == pytest.approx(1.5, abs=1e-5)
        assert y1 == pytest.approx(2.5, abs=1e-5)

    def test_bounds_contain_interior(self):
        r = Rectangle(center=[0.0, 0.0], size=[1.0, 0.6], angle=30.0)
        assert_bounds_contain(r, [(0.0, 0.0)])

    def test_invalid_size(self):
        with pytest.raises(ValueError):
            Rectangle(center=[0.0, 0.0], size=[0.0, 1.0])

    def test_round_trip(self):
        r = Rectangle(center=[0.3, -0.2], size=[0.8, 0.4], angle=15.0, corner_radius=0.05)
        assert_round_trip(r)


# ---------------------------------------------------------------------------
# ConvexQuad
# ---------------------------------------------------------------------------

class TestConvexQuad:
    def _parallelogram(self):
        return ConvexQuad(
            center=[0.0, 0.0],
            u=[0.4, 0.0],
            v=[0.0, 0.3],
        )

    def test_center_is_inside(self):
        assert_inside(self._parallelogram(), [(0.0, 0.0)])

    def test_far_point_is_outside(self):
        assert_outside(self._parallelogram(), [(2.0, 0.0), (0.0, 2.0)])

    def test_asymmetric_quad(self):
        q = ConvexQuad(center=[0.0, 0.0], u=[0.4, 0.0], v=[0.0, 0.3], alpha=0.2, beta=0.1)
        assert_inside(q, [(0.0, 0.0)])

    def test_corner_radius(self):
        q = ConvexQuad(
            center=[0.0, 0.0], u=[0.4, 0.0], v=[0.0, 0.3], corner_radius=0.05
        )
        assert_inside(q, [(0.0, 0.0)])

    def test_rotated(self):
        q = ConvexQuad(center=[0.0, 0.0], u=[0.4, 0.0], v=[0.0, 0.3], angle=45.0)
        assert_inside(q, [(0.0, 0.0)])

    def test_bounds(self):
        q = self._parallelogram()
        (x0, y0), (x1, y1) = q.bounds()
        assert x0 < 0 and x1 > 0
        assert y0 < 0 and y1 > 0

    def test_collinear_uv_raises(self):
        with pytest.raises(ValueError):
            ConvexQuad(center=[0.0, 0.0], u=[1.0, 0.0], v=[2.0, 0.0])

    def test_round_trip(self):
        q = ConvexQuad(
            center=[0.1, -0.1], u=[0.3, 0.0], v=[0.0, 0.25],
            alpha=0.1, beta=0.05, angle=10.0, corner_radius=0.02
        )
        assert_round_trip(q)


# ---------------------------------------------------------------------------
# IsoscelesTrapezoid
# ---------------------------------------------------------------------------

class TestIsoscelesTrapezoid:
    def test_center_is_inside(self):
        t = IsoscelesTrapezoid(
            center=[0.0, 0.0], bottom_width=1.0, top_width=0.6, height=0.8
        )
        assert_inside(t, [(0.0, 0.0)])

    def test_far_point_is_outside(self):
        t = IsoscelesTrapezoid(
            center=[0.0, 0.0], bottom_width=1.0, top_width=0.6, height=0.8
        )
        assert_outside(t, [(2.0, 0.0), (0.0, 2.0)])

    def test_rectangular_degenerate(self):
        # equal widths → effectively a rectangle
        t = IsoscelesTrapezoid(
            center=[0.0, 0.0], bottom_width=1.0, top_width=1.0, height=0.6
        )
        assert_inside(t, [(0.0, 0.0)])
        assert_outside(t, [(0.6, 0.0), (0.0, 0.4)])

    def test_rotated(self):
        t = IsoscelesTrapezoid(
            center=[0.0, 0.0], bottom_width=1.0, top_width=0.5, height=0.8, angle=90.0
        )
        assert_inside(t, [(0.0, 0.0)])

    def test_corner_radius(self):
        t = IsoscelesTrapezoid(
            center=[0.0, 0.0], bottom_width=1.0, top_width=0.6, height=0.8, corner_radius=0.05
        )
        assert_inside(t, [(0.0, 0.0)])

    def test_bounds_contain_interior(self):
        t = IsoscelesTrapezoid(
            center=[0.5, 0.5], bottom_width=0.8, top_width=0.4, height=0.6
        )
        assert_bounds_contain(t, [(0.5, 0.5)])

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError):
            IsoscelesTrapezoid(center=[0.0, 0.0], bottom_width=0.0, top_width=0.5, height=0.8)
        with pytest.raises(ValueError):
            IsoscelesTrapezoid(center=[0.0, 0.0], bottom_width=0.5, top_width=0.0, height=0.8)
        with pytest.raises(ValueError):
            IsoscelesTrapezoid(center=[0.0, 0.0], bottom_width=0.5, top_width=0.4, height=0.0)

    def test_round_trip(self):
        t = IsoscelesTrapezoid(
            center=[0.1, 0.2], bottom_width=0.9, top_width=0.5, height=0.7,
            angle=20.0, corner_radius=0.04
        )
        assert_round_trip(t)
