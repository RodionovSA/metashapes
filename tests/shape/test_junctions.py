# tests/shape/test_junctions.py

import pytest
import torch
from metashapes.shape.primitives.junctions import Cross, TShape
from .conftest import (
    assert_inside, assert_outside, assert_round_trip, assert_bounds_contain, sdf_at,
    assert_dtype_device_flow, assert_direct_call_dtype_promotion,
    assert_gradients_finite, assert_gradients_finite_at,
)


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

    def test_zero_inner_radius_sdf_finite_and_correct(self):
        # The patch is always computed, including at ri=0 where it's a
        # provable no-op; pin the result against a wide grid.
        c = Cross(center=[0.05, -0.1], length=1.1, width=0.35, angle=15.0,
                  outer_corner_radius=0.02)
        x = torch.linspace(-1.0, 1.0, 60)
        y = torch.linspace(-1.0, 1.0, 60)
        X, Y = torch.meshgrid(x, y, indexing="xy")
        d = c.sdf(X, Y)
        assert torch.isfinite(d).all()
        assert_inside(c, [(0.05, -0.1)])
        assert_outside(c, [(3.0, 3.0)])

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

    def test_zero_inner_radius_sdf_finite_and_correct(self):
        # Same check as Cross above.
        t = TShape(center=[0.05, -0.1], length=1.1, width=0.35, angle=15.0,
                  outer_corner_radius=0.02)
        x = torch.linspace(-1.0, 1.0, 60)
        y = torch.linspace(-1.0, 1.0, 60)
        X, Y = torch.meshgrid(x, y, indexing="xy")
        d = t.sdf(X, Y)
        assert torch.isfinite(d).all()
        assert_inside(t, [(0.05, -0.1)])
        assert_outside(t, [(3.0, 3.0)])

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


# ---------------------------------------------------------------------------
# Cross/TShape delegate to the shared _sdf_rounded_box helper
# ---------------------------------------------------------------------------

class TestSharedRoundedBoxHelper:
    """Cross and TShape both call shape/utils.py's _sdf_rounded_box. These
    checks pin the outer-box component (d_base's building blocks) to the
    shared helper directly, so a future edit that reintroduces a diverging
    inline copy would be caught here."""

    def test_cross_outer_box_matches_shared_helper(self):
        from metashapes.shape.utils import _sdf_rounded_box, _to_local_coords

        c = Cross(center=[0.1, -0.2], length=1.2, width=0.4, angle=20.0,
                   outer_corner_radius=0.05)
        x = torch.linspace(-1.0, 1.0, 30)
        y = torch.linspace(-1.0, 1.0, 30)
        X, Y = torch.meshgrid(x, y, indexing="xy")

        cx, cy = c.center[0], c.center[1]
        bx, by = 0.5 * c.length, 0.5 * c.width
        x_local, y_local = _to_local_coords(X, Y, cx, cy, c.angle)
        dh = _sdf_rounded_box(x_local, y_local, bx, by, c.outer_corner_radius)
        dv = _sdf_rounded_box(x_local, y_local, by, bx, c.outer_corner_radius)
        expected_d_base = torch.minimum(dh, dv)

        # With inner_corner_radius=0 (default), the patch is a no-op, so
        # sdf() should be exactly d_base.
        assert torch.equal(c.sdf(X, Y), expected_d_base)

    def test_tshape_outer_box_matches_shared_helper(self):
        from metashapes.shape.utils import _sdf_rounded_box, _to_local_coords

        t = TShape(center=[0.1, -0.2], length=1.2, width=0.4, angle=20.0,
                   outer_corner_radius=0.05)
        x = torch.linspace(-1.0, 1.0, 30)
        y = torch.linspace(-1.0, 1.0, 30)
        X, Y = torch.meshgrid(x, y, indexing="xy")

        cx, cy = t.center[0], t.center[1]
        bx, by = 0.5 * t.length, 0.5 * t.width
        x_local, y_local = _to_local_coords(X, Y, cx, cy, t.angle)
        d_top = _sdf_rounded_box(x_local, y_local - (bx - by), bx, by, t.outer_corner_radius)
        d_stem = _sdf_rounded_box(x_local, y_local, by, bx, t.outer_corner_radius)
        y_cut = bx - 2.0 * by
        d_top_half = torch.maximum(d_top, -(y_local - y_cut))
        expected_d_base = torch.minimum(d_top_half, d_stem)

        assert torch.equal(t.sdf(X, Y), expected_d_base)


# ---------------------------------------------------------------------------
# dtype / device / gradient flow
# ---------------------------------------------------------------------------

class TestCrossDtypeDeviceGrad:
    def test_dtype_device_flow(self):
        c = Cross(center=[0.1, -0.1], length=1.0, width=0.3, angle=10.0,
                  outer_corner_radius=0.02, inner_corner_radius=0.02)
        assert_dtype_device_flow(c)

    def test_direct_call_dtype_promotion(self):
        c = Cross(center=[0.0, 0.0], length=1.0, width=0.3)
        assert_direct_call_dtype_promotion(c)

    def test_integer_query_does_not_crash(self):
        c = Cross(center=[0.0, 0.0], length=1.0, width=0.3)
        out = c.sdf(torch.tensor(0), torch.tensor(0))
        assert torch.isfinite(out)

    def test_gradients_finite_generic_point(self):
        c = Cross(
            center=torch.nn.Parameter(torch.tensor([0.1, -0.1])),
            length=torch.nn.Parameter(torch.tensor(1.0)),
            width=torch.nn.Parameter(torch.tensor(0.3)),
            angle=torch.nn.Parameter(torch.tensor(10.0)),
            outer_corner_radius=torch.nn.Parameter(torch.tensor(0.02)),
            inner_corner_radius=torch.nn.Parameter(torch.tensor(0.02)),
        )
        assert_gradients_finite(
            c, ["center", "length", "width", "angle", "outer_corner_radius", "inner_corner_radius"]
        )

    def test_gradients_finite_at_zero_inner_corner_radius(self):
        # The always-compute-the-patch path must not introduce a NaN
        # gradient at ri=0.
        c = Cross(
            center=torch.tensor([0.0, 0.0]),
            length=torch.tensor(1.0), width=torch.tensor(0.3),
            inner_corner_radius=torch.nn.Parameter(torch.tensor(0.0)),
        )
        assert_gradients_finite(c, ["inner_corner_radius"])

    def test_gradients_finite_at_width_equals_length(self):
        # width == length is deliberately permissive; confirm it also
        # evaluates cleanly with finite gradients, not just "doesn't raise."
        c = Cross(
            center=torch.tensor([0.0, 0.0]),
            length=torch.nn.Parameter(torch.tensor(0.5)),
            width=torch.nn.Parameter(torch.tensor(0.5)),
        )
        assert_gradients_finite_at(c, ["length", "width"], 0.1, 0.1)

    def test_broadcast_x_y_different_shapes(self):
        c = Cross(center=[0.0, 0.0], length=1.0, width=0.3)
        x = torch.linspace(-1, 1, 5).reshape(1, 5)
        y = torch.linspace(-1, 1, 7).reshape(7, 1)
        out = c.sdf(x, y)
        assert out.shape == (7, 5)


class TestTShapeDtypeDeviceGrad:
    def test_dtype_device_flow(self):
        t = TShape(center=[0.1, -0.1], length=1.0, width=0.3, angle=10.0,
                   outer_corner_radius=0.02, inner_corner_radius=0.02)
        assert_dtype_device_flow(t)

    def test_direct_call_dtype_promotion(self):
        t = TShape(center=[0.0, 0.0], length=1.0, width=0.3)
        assert_direct_call_dtype_promotion(t)

    def test_integer_query_does_not_crash(self):
        t = TShape(center=[0.0, 0.0], length=1.0, width=0.3)
        out = t.sdf(torch.tensor(0), torch.tensor(0))
        assert torch.isfinite(out)

    def test_gradients_finite_generic_point(self):
        t = TShape(
            center=torch.nn.Parameter(torch.tensor([0.1, -0.1])),
            length=torch.nn.Parameter(torch.tensor(1.0)),
            width=torch.nn.Parameter(torch.tensor(0.3)),
            angle=torch.nn.Parameter(torch.tensor(10.0)),
            outer_corner_radius=torch.nn.Parameter(torch.tensor(0.02)),
            inner_corner_radius=torch.nn.Parameter(torch.tensor(0.02)),
        )
        assert_gradients_finite(
            t, ["center", "length", "width", "angle", "outer_corner_radius", "inner_corner_radius"]
        )

    def test_gradients_finite_at_zero_inner_corner_radius(self):
        t = TShape(
            center=torch.tensor([0.0, 0.0]),
            length=torch.tensor(1.0), width=torch.tensor(0.3),
            inner_corner_radius=torch.nn.Parameter(torch.tensor(0.0)),
        )
        assert_gradients_finite(t, ["inner_corner_radius"])

    def test_gradients_finite_at_width_equals_length(self):
        t = TShape(
            center=torch.tensor([0.0, 0.0]),
            length=torch.nn.Parameter(torch.tensor(0.5)),
            width=torch.nn.Parameter(torch.tensor(0.5)),
        )
        assert_gradients_finite_at(t, ["length", "width"], 0.1, 0.1)

    def test_broadcast_x_y_different_shapes(self):
        t = TShape(center=[0.0, 0.0], length=1.0, width=0.3)
        x = torch.linspace(-1, 1, 5).reshape(1, 5)
        y = torch.linspace(-1, 1, 7).reshape(7, 1)
        out = t.sdf(x, y)
        assert out.shape == (7, 5)
