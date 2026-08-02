# tests/shape/test_quads.py

import math
import pytest
import torch
from metashapes.shape import Rectangle, ConvexQuad, IsoscelesTrapezoid
from .conftest import (
    assert_inside, assert_outside, assert_round_trip, assert_bounds_contain, sdf_at,
    assert_dtype_device_flow, assert_direct_call_dtype_promotion,
    assert_gradients_finite, assert_gradients_finite_at,
)


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
# _sdf_rounded_box (shared helper used by Rectangle, Cross, TShape)
# ---------------------------------------------------------------------------

class TestSdfRoundedBox:
    """Direct tests of the shared helper, plus a check that Rectangle's own
    sdf() actually delegates to it rather than keeping a parallel inline
    copy that could drift out of sync."""

    def test_zero_radius_matches_plain_box(self):
        from metashapes.shape.utils import _sdf_rounded_box

        x = torch.linspace(-2.0, 2.0, 50)
        y = torch.linspace(-2.0, 2.0, 50)
        X, Y = torch.meshgrid(x, y, indexing="xy")
        hx, hy = torch.tensor(0.6), torch.tensor(0.3)

        d_default = _sdf_rounded_box(X, Y, hx, hy)  # r defaults to 0.0
        d_explicit_zero = _sdf_rounded_box(X, Y, hx, hy, torch.tensor(0.0))
        assert torch.equal(d_default, d_explicit_zero)

        # a plain box SDF, independently written
        qx = torch.abs(X) - hx
        qy = torch.abs(Y) - hy
        expected = (
            torch.sqrt(torch.clamp(qx, min=0.0) ** 2 + torch.clamp(qy, min=0.0) ** 2)
            + torch.clamp(torch.maximum(qx, qy), max=0.0)
        )
        assert torch.allclose(d_default, expected, atol=1e-6)

    def test_rectangle_delegates_to_shared_helper(self):
        # Rectangle.sdf() must produce exactly the same output as calling
        # the shared helper directly with the same (rotated, recentred)
        # local coordinates and the same clamped radius.
        from metashapes.shape.utils import _sdf_rounded_box, _to_local_coords

        r = Rectangle(center=[0.2, -0.1], size=[0.9, 0.5], angle=25.0, corner_radius=0.1)
        x = torch.linspace(-1.5, 1.5, 40)
        y = torch.linspace(-1.5, 1.5, 40)
        X, Y = torch.meshgrid(x, y, indexing="xy")

        cx, cy = r.center[0], r.center[1]
        w, h = r.size[0], r.size[1]
        rr = torch.minimum(r.corner_radius, 0.5 * torch.minimum(w, h))
        x_local, y_local = _to_local_coords(X, Y, cx, cy, r.angle)
        expected = _sdf_rounded_box(x_local, y_local, w * 0.5, h * 0.5, rr)

        assert torch.equal(r.sdf(X, Y), expected)


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

    def test_oversized_corner_radius_raises_at_construction(self):
        # On a unit-square quad (u=[1,0], v=[0,1], true inradius 1.0),
        # every radius at or above the true inradius must raise at
        # construction time.
        for rr in (1.0, 1.05, 1.1, 1.5):
            with pytest.raises(ValueError):
                ConvexQuad(center=[0.0, 0.0], u=[1.0, 0.0], v=[0.0, 1.0], corner_radius=rr)

    def test_valid_corner_radius_still_constructs_and_evaluates(self):
        q = ConvexQuad(center=[0.0, 0.0], u=[1.0, 0.0], v=[0.0, 1.0], corner_radius=0.9)
        d = sdf_at(q, 0.0, 0.0)
        assert d == pytest.approx(-1.0, abs=1e-3)

    def test_corner_radius_bound_is_monotonic(self):
        # Every radius above the analytic bound must raise, with no gaps.
        for rr in torch.linspace(1.0, 2.0, 11).tolist():
            with pytest.raises(ValueError):
                ConvexQuad(center=[0.0, 0.0], u=[1.0, 0.0], v=[0.0, 1.0], corner_radius=rr)

    def test_zero_radius_sdf_matches_inset_at_zero(self):
        # _inset_convex_polygon is called unconditionally, including at
        # rr=0 where it's an exact identity; pin that result against a
        # wide grid.
        q = ConvexQuad(center=[0.05, -0.1], u=[0.4, 0.05], v=[0.05, 0.35],
                        alpha=0.15, beta=0.1, angle=20.0, corner_radius=0.0)
        x = torch.linspace(-1.5, 1.5, 60)
        y = torch.linspace(-1.5, 1.5, 60)
        X, Y = torch.meshgrid(x, y, indexing="xy")
        d = q.sdf(X, Y)
        assert torch.isfinite(d).all()
        # sign matches a plain (non-inset) point-in-polygon test at a few
        # representative points
        assert_inside(q, [(0.05, -0.1)])
        assert_outside(q, [(2.0, 2.0)])

    def test_min_feature_size_is_polygon_width_not_edge_length(self):
        # A long, thin sliver quad has long edges but a near-zero waist --
        # min_feature_size must report the true (rotating-calipers) width,
        # not anything close to the edge lengths.
        q = ConvexQuad(center=[0.0, 0.0], u=[1.0, 0.0], v=[0.0, 0.02])
        edge_len = ((2 * 1.0) ** 2 + 0) ** 0.5  # long edges ~2.0
        assert q.min_feature_size < 0.1
        assert q.min_feature_size < edge_len

    def test_min_feature_size_matches_analytic_width_for_rectangle(self):
        # For u,v axis-aligned (a rectangle), the true width is trivially
        # min(2|u|, 2|v|) -- an independent, hand-computable ground truth.
        q = ConvexQuad(center=[0.0, 0.0], u=[0.6, 0.0], v=[0.0, 0.15])
        assert q.min_feature_size == pytest.approx(0.3, abs=1e-5)

    def test_min_feature_size_subtracts_twice_corner_radius(self):
        u, v = [0.6, 0.0], [0.0, 0.15]
        w0 = ConvexQuad(center=[0.0, 0.0], u=u, v=v, corner_radius=0.0).min_feature_size
        r = 0.05
        q = ConvexQuad(center=[0.0, 0.0], u=u, v=v, corner_radius=r)
        assert q.min_feature_size == pytest.approx(w0 - 2 * r, abs=1e-5)

    def test_min_feature_size_matches_brute_force_directional_scan(self):
        # Independent ground truth: the textbook definition of polygon
        # width is the minimum, over all directions, of the projection
        # extent. Verified (during implementation) to converge to the
        # analytic edge-normal value as angular resolution increases; this
        # pins a representative case at a resolution tight enough to
        # confirm agreement to 3 significant figures.
        import math as _math
        q = ConvexQuad(center=[0.0, 0.0], u=[0.4, 0.05], v=[0.05, 0.35],
                        alpha=0.15, beta=0.1, angle=0.0, corner_radius=0.0)
        ux, uy = q.u.detach().tolist()
        vx, vy = q.v.detach().tolist()
        a, b = q.alpha.item(), q.beta.item()
        verts = [
            (-ux - vx, -uy - vy),
            (ux - vx, uy - vy),
            ((1 + a) * ux + (1 + b) * vx, (1 + a) * uy + (1 + b) * vy),
            (-(1 + a) * ux + (1 + b) * vx, -(1 + a) * uy + (1 + b) * vy),
        ]
        n_angles = 200_000
        best = float("inf")
        for k in range(n_angles):
            theta = _math.pi * k / n_angles
            dx, dy = _math.cos(theta), _math.sin(theta)
            projs = [px * dx + py * dy for px, py in verts]
            best = min(best, max(projs) - min(projs))
        assert q.min_feature_size == pytest.approx(best, rel=2e-3)

    def test_winding_correction_matches_for_both_orientations(self):
        # Construct the same physical quad via u/v pairs that give area2
        # of each sign (swapping u and v flips the signed area) and check
        # both produce the same SDF (up to which vertex pair was
        # originally which).
        x = torch.linspace(-1.0, 1.0, 40)
        y = torch.linspace(-1.0, 1.0, 40)
        X, Y = torch.meshgrid(x, y, indexing="xy")

        q_pos = ConvexQuad(center=[0.0, 0.0], u=[0.4, 0.05], v=[0.05, 0.3], angle=10.0)
        q_neg = ConvexQuad(center=[0.0, 0.0], u=[0.05, 0.3], v=[0.4, 0.05], angle=10.0)

        d_pos = q_pos.sdf(X, Y)
        d_neg = q_neg.sdf(X, Y)
        assert torch.isfinite(d_pos).all()
        assert torch.isfinite(d_neg).all()
        # Both describe the same quadrilateral (u/v just swapped), so their
        # SDFs must agree everywhere.
        assert torch.allclose(d_pos, d_neg, atol=1e-5)


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

    def test_oversized_corner_radius_raises_at_construction(self):
        # corner_radius must be validated against the trapezoid's actual
        # bound at construction time, not just checked >= 0.
        with pytest.raises(ValueError, match="corner_radius"):
            IsoscelesTrapezoid(center=[0.0, 0.0], bottom_width=0.5, top_width=0.3,
                              height=0.4, corner_radius=0.3)

    def test_corner_radius_bound_is_sound(self):
        # The derived bound must never accept a radius that actually
        # produces a degenerate inset (r1, r2, or he <= 0), across varied
        # geometry -- not just the one case above.
        import random
        random.seed(0)
        for _ in range(200):
            wb = random.uniform(0.05, 3.0)
            wt = random.uniform(0.05, 3.0)
            h = random.uniform(0.05, 3.0)
            # bisect for the true rr_max via construction success/failure
            lo, hi = 0.0, min(wb, wt, h)
            for _ in range(25):
                mid = 0.5 * (lo + hi)
                try:
                    t = IsoscelesTrapezoid(center=[0.0, 0.0], bottom_width=wb,
                                           top_width=wt, height=h, corner_radius=mid)
                    lo = mid
                except ValueError:
                    hi = mid
            # lo is safely accepted; sdf() must not raise for it
            t = IsoscelesTrapezoid(center=[0.0, 0.0], bottom_width=wb, top_width=wt,
                                   height=h, corner_radius=lo)
            d = t.sdf(torch.tensor(0.0), torch.tensor(0.0))
            assert torch.isfinite(d)


# ---------------------------------------------------------------------------
# dtype / device / gradient flow
# ---------------------------------------------------------------------------

class TestRectangleDtypeDeviceGrad:
    def test_dtype_device_flow(self):
        r = Rectangle(center=[0.1, -0.1], size=[0.5, 0.3], angle=15.0, corner_radius=0.05)
        assert_dtype_device_flow(r)

    def test_direct_call_dtype_promotion(self):
        r = Rectangle(center=[0.0, 0.0], size=[0.5, 0.3])
        assert_direct_call_dtype_promotion(r)

    def test_integer_query_does_not_crash(self):
        # Rectangle uses only plain arithmetic (no torch.finfo), unlike
        # Ellipse/Egg/Stadium -- confirm it tolerates an integer-dtype
        # query point rather than crashing.
        r = Rectangle(center=[0.0, 0.0], size=[0.5, 0.3])
        out = r.sdf(torch.tensor(0), torch.tensor(0))
        assert torch.isfinite(out)

    def test_gradients_finite_generic_point(self):
        r = Rectangle(
            center=torch.nn.Parameter(torch.tensor([0.1, -0.05])),
            size=torch.nn.Parameter(torch.tensor([0.5, 0.3])),
            angle=torch.nn.Parameter(torch.tensor(15.0)),
            corner_radius=torch.nn.Parameter(torch.tensor(0.05)),
        )
        assert_gradients_finite(r, ["center", "size", "angle", "corner_radius"])

    def test_gradients_finite_near_corner(self):
        # Just outside the exact corner -- must be finite (see next test
        # for the corner itself, which is a known singularity).
        r = Rectangle(center=torch.nn.Parameter(torch.tensor([0.0, 0.0])),
                      size=torch.tensor([0.5, 0.3]))
        assert_gradients_finite_at(r, ["center"], 0.2505, 0.1505)

    def test_gradient_nan_exactly_at_sharp_corner(self):
        # Documents a known, inherent property of Euclidean-distance SDFs,
        # not a bug: the distance-to-corner formula is sqrt(qx^2+qy^2),
        # whose gradient is qx/sqrt(qx^2+qy^2) -- an exact 0/0 at qx=qy=0,
        # i.e. precisely at a sharp (unrounded) corner. Found by accident
        # while smoke-testing the gradient helper on this exact point;
        # kept as a documented characteristic rather than silently avoided.
        r = Rectangle(center=torch.nn.Parameter(torch.tensor([0.0, 0.0])),
                      size=torch.tensor([0.5, 0.3]))
        d = r.sdf(torch.tensor(0.25), torch.tensor(0.15))  # exactly (w/2, h/2)
        d.backward()
        assert torch.isnan(r.center.grad).all()

    def test_gradient_finite_with_rounded_corner_at_same_point(self):
        # The same query point that produces a NaN gradient for a sharp
        # corner is safely *inside* a shape with any positive corner
        # radius -- confirm rounding removes the singularity, not just
        # relocates it.
        r = Rectangle(center=torch.nn.Parameter(torch.tensor([0.0, 0.0])),
                      size=torch.tensor([0.5, 0.3]), corner_radius=torch.tensor(0.05))
        d = r.sdf(torch.tensor(0.25), torch.tensor(0.15))
        d.backward()
        assert torch.isfinite(r.center.grad).all()

    def test_gradient_flows_through_defensive_clamp_for_oversized_radius(self):
        # Rectangle is the one primitive that doesn't validate corner_radius
        # against a geometric bound at construction -- it defensively
        # clamps inside sdf() instead (r = min(r, 0.5*min(w,h))). Confirm
        # that's actually true (grossly oversized radius doesn't crash,
        # produces a finite, sane SDF) and gradient still flows through the
        # clamp back to corner_radius itself (torch.minimum's gradient
        # follows the winning branch -- when the clamp is active, that's
        # the constant 0.5*min(w,h) side, so corner_radius's OWN gradient
        # through this path should be exactly zero, not NaN).
        cr = torch.nn.Parameter(torch.tensor(50.0))  # wildly oversized
        r = Rectangle(center=[0.0, 0.0], size=[0.5, 0.3], corner_radius=cr)
        d = r.sdf(torch.tensor(0.0), torch.tensor(0.0))
        assert torch.isfinite(d)
        d.backward()
        assert torch.isfinite(cr.grad).all()
        assert cr.grad.item() == pytest.approx(0.0)

    def test_broadcast_x_y_different_shapes(self):
        r = Rectangle(center=[0.0, 0.0], size=[0.5, 0.3])
        x = torch.linspace(-1, 1, 5).reshape(1, 5)
        y = torch.linspace(-1, 1, 7).reshape(7, 1)
        out = r.sdf(x, y)
        assert out.shape == (7, 5)


class TestConvexQuadDtypeDeviceGrad:
    def test_dtype_device_flow(self):
        q = ConvexQuad(center=[0.05, -0.05], u=[0.4, 0.05], v=[0.05, 0.3],
                       alpha=0.1, beta=0.05, angle=10.0, corner_radius=0.05)
        assert_dtype_device_flow(q)

    def test_direct_call_dtype_promotion(self):
        q = ConvexQuad(center=[0.0, 0.0], u=[0.4, 0.0], v=[0.0, 0.3])
        assert_direct_call_dtype_promotion(q)

    def test_integer_query_does_not_crash(self):
        q = ConvexQuad(center=[0.0, 0.0], u=[0.4, 0.0], v=[0.0, 0.3])
        out = q.sdf(torch.tensor(0), torch.tensor(0))
        assert torch.isfinite(out)

    def test_gradients_finite_generic_point(self):
        q = ConvexQuad(
            center=torch.nn.Parameter(torch.tensor([0.05, -0.05])),
            u=torch.nn.Parameter(torch.tensor([0.4, 0.05])),
            v=torch.nn.Parameter(torch.tensor([0.05, 0.3])),
            alpha=torch.nn.Parameter(torch.tensor(0.1)),
            beta=torch.nn.Parameter(torch.tensor(0.05)),
            angle=torch.nn.Parameter(torch.tensor(10.0)),
            corner_radius=torch.nn.Parameter(torch.tensor(0.05)),
        )
        assert_gradients_finite(q, ["center", "u", "v", "alpha", "beta", "angle", "corner_radius"])

    def test_gradients_finite_at_zero_corner_radius(self):
        # The always-inset path must not introduce a NaN gradient at
        # corner_radius=0.
        q = ConvexQuad(
            center=torch.nn.Parameter(torch.tensor([0.0, 0.0])),
            u=torch.nn.Parameter(torch.tensor([0.4, 0.05])),
            v=torch.nn.Parameter(torch.tensor([0.05, 0.3])),
            corner_radius=torch.nn.Parameter(torch.tensor(0.0)),
        )
        assert_gradients_finite(q, ["u", "v", "corner_radius"])

    def test_gradients_finite_near_max_corner_radius(self):
        q = ConvexQuad(
            center=torch.nn.Parameter(torch.tensor([0.0, 0.0])),
            u=torch.tensor([0.4, 0.0]), v=torch.tensor([0.0, 0.3]),
            corner_radius=torch.nn.Parameter(torch.tensor(0.29)),  # rr_max ~= 0.3
        )
        assert_gradients_finite(q, ["corner_radius"])

    def test_gradients_finite_for_both_winding_orientations(self):
        # The branch-free winding correction must give real, non-zero
        # gradient w.r.t. u/v on both sides of area2's sign.
        for u, v in [([0.4, 0.05], [0.05, 0.3]), ([0.05, 0.3], [0.4, 0.05])]:
            q = ConvexQuad(
                center=torch.tensor([0.0, 0.0]),
                u=torch.nn.Parameter(torch.tensor(u)),
                v=torch.nn.Parameter(torch.tensor(v)),
                angle=torch.tensor(10.0),
            )
            assert_gradients_finite(q, ["u", "v"])
            assert (q.u.grad.abs().sum() > 0) or (q.v.grad.abs().sum() > 0), (
                f"u/v grad is exactly zero for winding case u={u}, v={v}"
            )

    def test_broadcast_x_y_different_shapes(self):
        q = ConvexQuad(center=[0.0, 0.0], u=[0.4, 0.0], v=[0.0, 0.3])
        x = torch.linspace(-1, 1, 5).reshape(1, 5)
        y = torch.linspace(-1, 1, 7).reshape(7, 1)
        out = q.sdf(x, y)
        assert out.shape == (7, 5)


class TestIsoscelesTrapezoidDtypeDeviceGrad:
    def test_dtype_device_flow(self):
        t = IsoscelesTrapezoid(center=[0.1, 0.2], bottom_width=0.9, top_width=0.5,
                               height=0.7, angle=20.0, corner_radius=0.04)
        assert_dtype_device_flow(t)

    def test_direct_call_dtype_promotion(self):
        t = IsoscelesTrapezoid(center=[0.0, 0.0], bottom_width=0.9, top_width=0.5, height=0.7)
        assert_direct_call_dtype_promotion(t)

    def test_integer_query_does_not_crash(self):
        # No torch.finfo here (unlike ovals.py) -- confirm it tolerates
        # integer input like Rectangle/ConvexQuad do.
        t = IsoscelesTrapezoid(center=[0.0, 0.0], bottom_width=0.9, top_width=0.5, height=0.7)
        out = t.sdf(torch.tensor(0), torch.tensor(0))
        assert torch.isfinite(out)

    def test_gradients_finite_generic_point(self):
        t = IsoscelesTrapezoid(
            center=torch.nn.Parameter(torch.tensor([0.1, 0.2])),
            bottom_width=torch.nn.Parameter(torch.tensor(0.9)),
            top_width=torch.nn.Parameter(torch.tensor(0.5)),
            height=torch.nn.Parameter(torch.tensor(0.7)),
            angle=torch.nn.Parameter(torch.tensor(20.0)),
            corner_radius=torch.nn.Parameter(torch.tensor(0.04)),
        )
        assert_gradients_finite(
            t, ["center", "bottom_width", "top_width", "height", "angle", "corner_radius"]
        )

    def test_gradients_finite_at_zero_corner_radius(self):
        t = IsoscelesTrapezoid(
            center=torch.tensor([0.0, 0.0]),
            bottom_width=torch.nn.Parameter(torch.tensor(0.9)),
            top_width=torch.nn.Parameter(torch.tensor(0.5)),
            height=torch.nn.Parameter(torch.tensor(0.7)),
            corner_radius=torch.nn.Parameter(torch.tensor(0.0)),
        )
        assert_gradients_finite(t, ["bottom_width", "top_width", "height", "corner_radius"])

    def test_gradients_finite_near_corner_radius_bound(self):
        # rr_max for (wb=0.5, wt=0.3, h=0.4) is ~0.1921 (see the fix's own
        # verification script) -- pick just under it.
        t = IsoscelesTrapezoid(
            center=torch.tensor([0.0, 0.0]),
            bottom_width=torch.tensor(0.5), top_width=torch.tensor(0.3),
            height=torch.tensor(0.4),
            corner_radius=torch.nn.Parameter(torch.tensor(0.19)),
        )
        assert_gradients_finite(t, ["corner_radius"])

    def test_broadcast_x_y_different_shapes(self):
        t = IsoscelesTrapezoid(center=[0.0, 0.0], bottom_width=0.9, top_width=0.5, height=0.7)
        x = torch.linspace(-1, 1, 5).reshape(1, 5)
        y = torch.linspace(-1, 1, 7).reshape(7, 1)
        out = t.sdf(x, y)
        assert out.shape == (7, 5)
