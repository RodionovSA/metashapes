# tests/shape/test_conics.py

import math
import pytest
import torch
from metashapes.shape import Shape
from metashapes.shape.primitives.conics import Ellipse, Egg, Stadium
from .conftest import (
    assert_inside, assert_outside, assert_round_trip, assert_bounds_contain, sdf_at,
    assert_dtype_device_flow, assert_direct_call_dtype_promotion,
    assert_gradients_finite, assert_gradients_finite_at,
)


class TestEllipse:
    def test_center_is_inside(self):
        e = Ellipse(center=[0.0, 0.0], axes=[1.0, 0.5])
        assert_inside(e, [(0.0, 0.0)])

    def test_far_point_is_outside(self):
        e = Ellipse(center=[0.0, 0.0], axes=[1.0, 0.5])
        assert_outside(e, [(2.0, 0.0), (0.0, 2.0), (-1.5, -1.5)])

    def test_circle_boundary(self):
        # axes equal → circle; point on the boundary
        e = Ellipse(center=[0.0, 0.0], axes=[2.0, 2.0])
        d = sdf_at(e, 1.0, 0.0)
        assert abs(d) < 1e-4

    def test_circle_interior_value(self):
        e = Ellipse(center=[0.0, 0.0], axes=[2.0, 2.0])
        d = sdf_at(e, 0.0, 0.0)
        assert d == pytest.approx(-1.0, abs=5e-4)

    def test_ellipse_axis_boundary(self):
        # point at end of major axis
        e = Ellipse(center=[0.0, 0.0], axes=[2.0, 1.0])
        d = sdf_at(e, 1.0, 0.0)  # end of major semi-axis
        assert abs(d) < 1e-4

    def test_offset_center(self):
        e = Ellipse(center=[2.0, -1.0], axes=[0.6, 0.4])
        assert_inside(e, [(2.0, -1.0)])
        assert_outside(e, [(0.0, 0.0)])

    def test_rotated_ellipse(self):
        e = Ellipse(center=[0.0, 0.0], axes=[1.0, 0.4], angle=90.0)
        # after 90° rotation the long axis is now vertical
        assert_inside(e, [(0.0, 0.3)])   # inside along new long axis
        assert_outside(e, [(0.4, 0.0)]) # outside along new short axis

    def test_bounds(self):
        e = Ellipse(center=[1.0, 2.0], axes=[2.0, 1.0])
        (x0, y0), (x1, y1) = e.bounds()
        assert x0 == pytest.approx(0.0, abs=1e-5)
        assert x1 == pytest.approx(2.0, abs=1e-5)
        assert y0 == pytest.approx(1.5, abs=1e-5)
        assert y1 == pytest.approx(2.5, abs=1e-5)

    def test_bounds_contain_interior(self):
        e = Ellipse(center=[0.0, 0.0], axes=[1.2, 0.8], angle=35.0)
        assert_bounds_contain(e, [(0.0, 0.0)])

    def test_invalid_axes(self):
        with pytest.raises(ValueError):
            Ellipse(center=[0.0, 0.0], axes=[0.0, 1.0])
        with pytest.raises(ValueError):
            Ellipse(center=[0.0, 0.0], axes=[1.0, -0.5])

    def test_round_trip(self):
        e = Ellipse(center=[0.2, -0.3], axes=[0.8, 0.5], angle=25.0)
        assert_round_trip(e)

    def test_round_trip_circle(self):
        e = Ellipse(center=[0.0, 0.0], axes=[0.6, 0.6])
        assert_round_trip(e)

    def test_center_value_matches_semi_minor_axis(self):
        # Regression for S-03: the old `sign(py2 - ry)` proxy returned 0.0
        # at the exact center of a non-circular ellipse instead of
        # -min(semi_a, semi_b).
        e = Ellipse(center=[0.0, 0.0], axes=[2.0, 4.0])  # semi-axes 1.0, 2.0
        d = sdf_at(e, 0.0, 0.0)
        assert d == pytest.approx(-1.0, abs=5e-4)

    def test_no_interior_sign_flip(self):
        # Regression for S-03: the proxy sign test misclassified some
        # interior points near the minor-axis region (e.g. sdf(0, 1e-4)
        # used to read +1.0 for axes=[2, 4] -- outside, deep inside the
        # shape). Sweep the interior and check every sample against the
        # exact implicit inside/outside test.
        a, b = 1.0, 2.0  # semi-axes
        e = Ellipse(center=[0.0, 0.0], axes=[2 * a, 2 * b])
        xs = torch.linspace(-0.999, 0.999, 51)
        ys = torch.linspace(-1.999, 1.999, 51)
        X, Y = torch.meshgrid(xs, ys, indexing="xy")
        d = e.sdf(X, Y)
        truth_inside = (X / a) ** 2 + (Y / b) ** 2 < 1.0
        assert not torch.any(truth_inside & (d > 0)), (
            "found interior point(s) with positive (outside) SDF"
        )


class TestEgg:
    def test_center_is_inside(self):
        e = Egg(center=[0.0, 0.0], width=1.0, height=1.0)
        assert_inside(e, [(0.0, 0.0)])

    def test_far_point_is_outside(self):
        e = Egg(center=[0.0, 0.0], width=1.0, height=1.0)
        assert_outside(e, [(2.0, 0.0), (0.0, 2.0), (-1.5, -1.5)])

    def test_symmetric_matches_ellipse(self):
        # skew=0 → egg is equivalent to an ellipse with axes (width, height)
        egg = Egg(center=[0.0, 0.0], width=2.0, height=1.0, skew=0.0)
        ellipse = Ellipse(center=[0.0, 0.0], axes=[2.0, 1.0])
        xs = torch.linspace(-1.5, 1.5, 20)
        ys = torch.linspace(-1.0, 1.0, 20)
        X, Y = torch.meshgrid(xs, ys, indexing="xy")
        d_egg = egg.sdf(X, Y)
        d_ell = ellipse.sdf(X, Y)
        assert torch.allclose(d_egg, d_ell, atol=1e-4)

    def test_boundary_top_pole(self):
        # SDF ≈ 0 at the top pole (0, b_top)
        e = Egg(center=[0.0, 0.0], width=1.0, height=1.0, skew=0.4)
        b_top = 0.5 * (1.0 + 0.4)
        d = sdf_at(e, 0.0, b_top)
        assert abs(d) < 1e-4

    def test_boundary_bottom_pole(self):
        # SDF ≈ 0 at the bottom pole (0, -b_bot)
        e = Egg(center=[0.0, 0.0], width=1.0, height=1.0, skew=0.4)
        b_bot = 0.5 * (1.0 - 0.4)
        d = sdf_at(e, 0.0, -b_bot)
        assert abs(d) < 1e-4

    def test_boundary_side(self):
        # SDF ≈ 0 at the widest point (a, 0)
        e = Egg(center=[0.0, 0.0], width=2.0, height=1.0, skew=0.3)
        d = sdf_at(e, 1.0, 0.0)
        assert abs(d) < 1e-4

    def test_asymmetric_skew_top_vs_bottom(self):
        # With positive skew the top half is taller than the bottom
        e = Egg(center=[0.0, 0.0], width=1.0, height=1.0, skew=0.5)
        b_top = 0.5 * (1.0 + 0.5)
        b_bot = 0.5 * (1.0 - 0.5)
        # Point slightly inside top pole should be inside
        assert_inside(e, [(0.0, b_top * 0.9)])
        # Point beyond bottom pole should be outside
        assert_outside(e, [(0.0, -(b_bot + 0.1))])

    def test_offset_center(self):
        e = Egg(center=[1.0, -0.5], width=0.8, height=0.6, skew=0.2)
        assert_inside(e, [(1.0, -0.5)])
        assert_outside(e, [(0.0, 0.0)])

    def test_rotated_egg(self):
        e = Egg(center=[0.0, 0.0], width=1.0, height=0.5, skew=0.3, angle=90.0)
        # After 90° rotation, the long axis is vertical; (0.4, 0) should be outside (was short axis)
        assert_outside(e, [(0.4, 0.0)])
        # (0.0, 0.4) should be inside (was long axis)
        assert_inside(e, [(0.0, 0.4)])

    def test_bounds(self):
        e = Egg(center=[0.0, 0.0], width=2.0, height=1.0, skew=0.0)
        (x0, y0), (x1, y1) = e.bounds()
        assert x0 <= -1.0 + 1e-5
        assert x1 >= 1.0 - 1e-5
        assert y0 <= -0.5 + 1e-5
        assert y1 >= 0.5 - 1e-5

    def test_bounds_contain_interior(self):
        e = Egg(center=[0.5, -0.3], width=1.2, height=0.8, skew=0.3, angle=40.0)
        assert_bounds_contain(e, [(0.5, -0.3)])

    def test_invalid_width(self):
        with pytest.raises(ValueError):
            Egg(center=[0.0, 0.0], width=0.0, height=1.0)

    def test_invalid_height(self):
        with pytest.raises(ValueError):
            Egg(center=[0.0, 0.0], width=1.0, height=-0.5)

    def test_invalid_skew(self):
        with pytest.raises(ValueError):
            Egg(center=[0.0, 0.0], width=1.0, height=1.0, skew=1.0)
        with pytest.raises(ValueError):
            Egg(center=[0.0, 0.0], width=1.0, height=1.0, skew=-1.0)

    def test_round_trip(self):
        e = Egg(center=[0.1, -0.2], width=0.9, height=0.7, skew=0.25, angle=15.0)
        assert_round_trip(e)

    def test_no_sign_flip_at_seam(self):
        # Regression for S-01: a query 1e-4 above the seam used to read
        # +1.0 (outside) while the mirrored point 1e-4 below read ~-0.5
        # (inside) -- an O(1) discontinuity exactly at the reported bug.
        e = Egg(center=[0.0, 0.0], width=2.0, height=2.5, skew=0.6)
        d_above = sdf_at(e, 0.0, 1e-4)
        d_below = sdf_at(e, 0.0, -1e-4)
        assert d_above < 0.0, f"expected inside just above the seam, got {d_above}"
        assert d_below < 0.0, f"expected inside just below the seam, got {d_below}"
        assert d_above == pytest.approx(d_below, abs=1e-2)

    def test_continuity_across_seam(self):
        # The old per-point b_eff switch produced a jump of up to ~1.5 at
        # x=0; check it stays small at several x across the seam.
        e = Egg(center=[0.0, 0.0], width=2.0, height=2.5, skew=0.6)
        for xv in (0.0, 0.2, 0.4, 0.6, 0.8):
            d_above = sdf_at(e, xv, 1e-4)
            d_below = sdf_at(e, xv, -1e-4)
            assert abs(d_above - d_below) < 5e-3, (
                f"seam jump at x={xv}: {d_above} vs {d_below}"
            )

    def test_no_sign_flip_grid_scan(self):
        # Same regression as Ellipse's, but for Egg's asymmetric two-arc
        # boundary (b_top != b_bot), across the full shape including the
        # seam region where the original bug lived.
        e = Egg(center=[0.0, 0.0], width=2.0, height=2.5, skew=0.6)
        a, b_top, b_bot = 1.0, 2.0, 0.5
        xs = torch.linspace(-0.999, 0.999, 41)
        ys = torch.linspace(-1.999, 1.999, 41)
        X, Y = torch.meshgrid(xs, ys, indexing="xy")
        d = e.sdf(X, Y)
        b_eff = torch.where(Y >= 0, torch.tensor(b_top), torch.tensor(b_bot))
        truth_inside = (X / a) ** 2 + (Y / b_eff) ** 2 < 1.0
        assert not torch.any(truth_inside & (d > 0)), (
            "found interior point(s) with positive (outside) SDF"
        )


class TestStadium:
    def test_center_is_inside(self):
        s = Stadium(center=[0.0, 0.0], length=2.0, width=1.0)
        assert_inside(s, [(0.0, 0.0)])

    def test_far_point_is_outside(self):
        s = Stadium(center=[0.0, 0.0], length=2.0, width=1.0)
        assert_outside(s, [(3.0, 0.0), (0.0, 2.0), (-2.0, -1.0)])

    def test_cap_tip_boundary(self):
        # SDF ≈ 0 at right cap tip (length/2, 0)
        s = Stadium(center=[0.0, 0.0], length=2.0, width=1.0)
        d = sdf_at(s, 1.0, 0.0)
        assert abs(d) < 1e-4

    def test_side_boundary(self):
        # SDF ≈ 0 at the widest side (0, radius)
        s = Stadium(center=[0.0, 0.0], length=2.0, width=1.0)
        d = sdf_at(s, 0.0, 0.5)
        assert abs(d) < 1e-4

    def test_center_sdf_value(self):
        # Center of stadium is exactly -radius from the boundary
        s = Stadium(center=[0.0, 0.0], length=2.0, width=1.0)
        d = sdf_at(s, 0.0, 0.0)
        assert d == pytest.approx(-0.5, abs=5e-4)

    def test_degenerate_circle(self):
        # length == width → stadium is a circle; SDF matches circle formula
        s = Stadium(center=[0.0, 0.0], length=1.0, width=1.0)
        d = sdf_at(s, 0.5, 0.0)
        assert abs(d) < 1e-4

    def test_offset_center(self):
        s = Stadium(center=[1.0, -0.5], length=2.0, width=0.8)
        assert_inside(s, [(1.0, -0.5)])
        assert_outside(s, [(0.0, 0.0)])

    def test_rotated_stadium(self):
        # After 90° rotation the long axis is vertical
        s = Stadium(center=[0.0, 0.0], length=2.0, width=0.5, angle=90.0)
        assert_inside(s, [(0.0, 0.6)])    # inside along new long axis
        assert_outside(s, [(0.4, 0.0)])   # outside along new short axis

    def test_bounds(self):
        s = Stadium(center=[0.0, 0.0], length=2.0, width=1.0)
        (x0, y0), (x1, y1) = s.bounds()
        assert x0 == pytest.approx(-1.0, abs=1e-5)
        assert x1 == pytest.approx(1.0, abs=1e-5)
        assert y0 == pytest.approx(-0.5, abs=1e-5)
        assert y1 == pytest.approx(0.5, abs=1e-5)

    def test_bounds_contain_interior(self):
        s = Stadium(center=[0.5, -0.3], length=1.5, width=0.6, angle=35.0)
        assert_bounds_contain(s, [(0.5, -0.3)])

    def test_invalid_length(self):
        with pytest.raises(ValueError):
            Stadium(center=[0.0, 0.0], length=0.0, width=0.5)
        with pytest.raises(ValueError):
            Stadium(center=[0.0, 0.0], length=-1.0, width=0.5)

    def test_invalid_width(self):
        with pytest.raises(ValueError):
            Stadium(center=[0.0, 0.0], length=1.0, width=0.0)

    def test_length_less_than_width(self):
        with pytest.raises(ValueError):
            Stadium(center=[0.0, 0.0], length=0.5, width=1.0)

    def test_round_trip(self):
        s = Stadium(center=[0.2, -0.1], length=1.8, width=0.6, angle=20.0)
        assert_round_trip(s)


# ---------------------------------------------------------------------------
# dtype / device / gradient flow
# ---------------------------------------------------------------------------

class TestEllipseDtypeDeviceGrad:
    def test_dtype_device_flow(self):
        e = Ellipse(center=[0.1, -0.1], axes=[1.0, 0.5], angle=15.0)
        assert_dtype_device_flow(e)

    def test_direct_call_dtype_promotion(self):
        e = Ellipse(center=[0.0, 0.0], axes=[1.0, 0.5])
        assert_direct_call_dtype_promotion(e)

    def test_integer_query_raises_typeerror(self):
        # Ellipse (and Egg, Stadium below) use torch.finfo(x.dtype).eps,
        # which requires a floating-point dtype -- unlike Rectangle/
        # ConvexQuad/etc, which use only plain arithmetic and tolerate
        # integer query points. This inconsistency across primitives is
        # real (confirmed for all 12); pinned here as Ellipse's actual
        # current behavior, not silently worked around.
        e = Ellipse(center=[0.0, 0.0], axes=[1.0, 0.5])
        with pytest.raises(TypeError, match="finfo"):
            e.sdf(torch.tensor(0), torch.tensor(0))

    def test_gradients_finite_generic_point(self):
        # A single, deliberately chosen safe point -- NOT a dense grid.
        # See test_gradient_nan_at_cube_root_singularity below: a plain
        # 40x40 grid over this same shape hits the solver's derivative
        # singularity at ~3.5% of points, so a grid here would be flaky by
        # construction, not a meaningful "generic point" check.
        e = Ellipse(
            center=torch.nn.Parameter(torch.tensor([0.1, -0.1])),
            axes=torch.nn.Parameter(torch.tensor([1.0, 0.5])),
            angle=torch.nn.Parameter(torch.tensor(15.0)),
        )
        assert_gradients_finite_at(e, ["center", "axes", "angle"], 0.35, 0.15)

    def test_gradient_nan_at_cube_root_singularity(self):
        # KNOWN, UNFIXED ISSUE (found by this test pass; documented, not
        # fixed -- see screening_shape_lattice.md). _ellipse_closest_point's
        # branch-2 cubic solve computes u = sign(qm) * |qm|**(1/3), where
        # qm is algebraically (sqrt(d) - m*n)**2 -- always >= 0, and
        # (numerically) at or near 0 whenever sqrt(d) ~= m*n. That's a
        # real, non-rare configuration: a plain 40x40 grid of query points
        # over this exact shape hit it at 56/1600 points. The forward
        # value at qm=0 is correct (0), but d/dqm(|qm|**(1/3)) diverges as
        # |qm| -> 0, poisoning the whole gradient with NaN -- the same
        # *class* of hazard as S-07 (a discarded/near-singular branch
        # corrupting the selected gradient), but from an inherent
        # derivative singularity in the closed-form solver itself, not a
        # memory-initialization issue. A real fix needs a numerically
        # stable reformulation of this cube-root step; out of scope here.
        e = Ellipse(center=torch.tensor([0.1, -0.1]),
                   axes=torch.nn.Parameter(torch.tensor([1.0, 0.5])),
                   angle=torch.tensor(15.0))
        d = e.sdf(torch.tensor(0.12820512), torch.tensor(-0.84615386))
        d.backward()
        assert torch.isnan(e.axes.grad).any(), (
            "expected the known cube-root gradient singularity to still reproduce here; "
            "if this now passes, the solver may have been fixed -- update this test to "
            "document that instead of asserting NaN"
        )

    def test_gradients_finite_at_swap_boundary(self):
        # _ellipse_closest_point's `swap = px > py` branch selects
        # differently on either side of the px == py tie -- confirm no
        # gradient discontinuity/NaN right at it.
        e = Ellipse(center=torch.tensor([0.0, 0.0]),
                   axes=torch.nn.Parameter(torch.tensor([1.0, 0.5])))
        assert_gradients_finite_at(e, ["axes"], 0.3, 0.3)

    def test_gradients_finite_near_circular(self):
        # a ~= b exercises _ellipse_closest_point's circle_mask branch
        # (radial projection), specifically chosen to avoid the general
        # cubic-solve path, which is singular as b^2 - a^2 -> 0.
        e = Ellipse(center=torch.tensor([0.0, 0.0]),
                   axes=torch.nn.Parameter(torch.tensor([0.5, 0.5000001])))
        assert_gradients_finite_at(e, ["axes"], 0.1, 0.05)

    def test_gradients_finite_at_degenerate_center(self):
        # Querying exactly at the ellipse's own center hits
        # _ellipse_closest_point's degenerate_center fallback (the radial
        # direction (px,py)/r is undefined at the origin).
        e = Ellipse(center=torch.nn.Parameter(torch.tensor([0.0, 0.0])),
                   axes=torch.tensor([0.5, 0.5]))
        assert_gradients_finite_at(e, ["center"], 0.0, 0.0)

    def test_sign_zero_gradient_does_not_zero_out_magnitude_gradient(self):
        # torch.sign(...) (used for the inside/outside classification) has
        # zero gradient almost everywhere by construction -- confirm this
        # doesn't zero out the *magnitude* term's real gradient via product
        # rule (it shouldn't: d(sign*mag)/dp = sign * d(mag)/dp when
        # d(sign)/dp = 0), rather than trusting that reasoning unverified.
        e = Ellipse(center=torch.tensor([0.0, 0.0]),
                   axes=torch.nn.Parameter(torch.tensor([1.0, 0.5])))
        d = e.sdf(torch.tensor(0.6), torch.tensor(0.05))  # clearly outside
        d.backward()
        assert torch.isfinite(e.axes.grad).all()
        assert e.axes.grad.abs().sum() > 0, "gradient unexpectedly zeroed out"

    def test_broadcast_x_y_different_shapes(self):
        e = Ellipse(center=[0.0, 0.0], axes=[1.0, 0.5])
        x = torch.linspace(-1, 1, 5).reshape(1, 5)
        y = torch.linspace(-1, 1, 7).reshape(7, 1)
        out = e.sdf(x, y)
        assert out.shape == (7, 5)


class TestEggDtypeDeviceGrad:
    def test_dtype_device_flow(self):
        e = Egg(center=[0.1, -0.1], width=0.6, height=0.8, skew=0.2, angle=15.0)
        assert_dtype_device_flow(e)

    def test_direct_call_dtype_promotion(self):
        e = Egg(center=[0.0, 0.0], width=0.6, height=0.8, skew=0.2)
        assert_direct_call_dtype_promotion(e)

    def test_integer_query_raises_typeerror(self):
        e = Egg(center=[0.0, 0.0], width=0.6, height=0.8, skew=0.2)
        with pytest.raises(TypeError, match="finfo"):
            e.sdf(torch.tensor(0), torch.tensor(0))

    def test_gradients_finite_generic_point(self):
        # A single, deliberately chosen safe point -- see
        # TestEllipseDtypeDeviceGrad.test_gradient_nan_at_cube_root_singularity;
        # Egg shares the same underlying solver, so the same caveat about
        # grids being flaky by construction applies here.
        e = Egg(
            center=torch.nn.Parameter(torch.tensor([0.1, -0.1])),
            width=torch.nn.Parameter(torch.tensor(0.6)),
            height=torch.nn.Parameter(torch.tensor(0.8)),
            skew=torch.nn.Parameter(torch.tensor(0.2)),
            angle=torch.nn.Parameter(torch.tensor(15.0)),
        )
        assert_gradients_finite_at(e, ["center", "width", "height", "skew", "angle"], 0.15, 0.3)

    def test_gradients_finite_off_seam(self):
        # Once clearly off the exact seam (not within the same solver's
        # singular band -- see below), gradients are finite.
        e = Egg(center=torch.tensor([0.0, 0.0]), width=torch.tensor(0.6),
               height=torch.tensor(0.8), skew=torch.nn.Parameter(torch.tensor(0.6)))
        assert_gradients_finite_at(e, ["skew"], 0.2, 0.002)

    def test_gradient_nan_very_close_to_seam(self):
        # KNOWN, UNFIXED ISSUE (same root cause as Ellipse's
        # test_gradient_nan_at_cube_root_singularity -- Egg calls the same
        # shared _ellipse_closest_point solver twice per point, once per
        # arc). Found while probing near the seam (S-01's original bug
        # location): within 1e-4 of y_local=0 for this shape, the query
        # lands in the solver's qm~=0 derivative-singularity band and the
        # gradient w.r.t. skew is NaN, even though the *value* there is
        # correct (S-01 fixed value-continuity at the seam, not gradient
        # smoothness -- these are different properties). Not fixed here;
        # see screening_shape_lattice.md.
        e = Egg(center=torch.tensor([0.0, 0.0]), width=torch.tensor(0.6),
               height=torch.tensor(0.8), skew=torch.nn.Parameter(torch.tensor(0.6)))
        d = e.sdf(torch.tensor(0.1), torch.tensor(1e-4))
        d.backward()
        assert torch.isnan(e.skew.grad), (
            "expected the known near-seam gradient singularity to still reproduce here; "
            "if this now passes, the solver may have been fixed -- update this test to "
            "document that instead of asserting NaN"
        )

    def test_gradients_finite_near_skew_boundary(self):
        # skew is validated strictly inside (-1, 1) at construction; confirm
        # approaching that boundary doesn't blow up b_bot/b_top's gradient.
        e = Egg(center=torch.tensor([0.0, 0.0]), width=torch.tensor(0.6),
               height=torch.tensor(0.8), skew=torch.nn.Parameter(torch.tensor(0.99)))
        assert_gradients_finite_at(e, ["skew"], 0.1, 0.1)

    def test_broadcast_x_y_different_shapes(self):
        e = Egg(center=[0.0, 0.0], width=0.6, height=0.8, skew=0.2)
        x = torch.linspace(-1, 1, 5).reshape(1, 5)
        y = torch.linspace(-1, 1, 7).reshape(7, 1)
        out = e.sdf(x, y)
        assert out.shape == (7, 5)


class TestStadiumDtypeDeviceGrad:
    def test_dtype_device_flow(self):
        s = Stadium(center=[0.1, -0.1], length=1.5, width=0.6, angle=15.0)
        assert_dtype_device_flow(s)

    def test_direct_call_dtype_promotion(self):
        s = Stadium(center=[0.0, 0.0], length=1.5, width=0.6)
        assert_direct_call_dtype_promotion(s)

    def test_integer_query_raises_typeerror(self):
        s = Stadium(center=[0.0, 0.0], length=1.5, width=0.6)
        with pytest.raises(TypeError, match="finfo"):
            s.sdf(torch.tensor(0), torch.tensor(0))

    def test_gradients_finite_generic_point(self):
        s = Stadium(
            center=torch.nn.Parameter(torch.tensor([0.1, -0.1])),
            length=torch.nn.Parameter(torch.tensor(1.5)),
            width=torch.nn.Parameter(torch.tensor(0.6)),
            angle=torch.nn.Parameter(torch.tensor(15.0)),
        )
        assert_gradients_finite(s, ["center", "length", "width", "angle"])

    def test_gradients_finite_at_length_equals_width(self):
        # A pure circle -- the validated edge of length >= width. half_span
        # clamps to exactly 0 here; confirm that doesn't introduce a
        # gradient hazard.
        s = Stadium(center=torch.tensor([0.0, 0.0]),
                   length=torch.nn.Parameter(torch.tensor(0.6)),
                   width=torch.nn.Parameter(torch.tensor(0.6)))
        assert_gradients_finite_at(s, ["length", "width"], 0.2, 0.1)

    def test_gradients_finite_on_cap_axis(self):
        # Directly along the capsule's long axis (y_local=0): dx's clamp
        # boundary (|x_local| == half_span) is where the flat-side and
        # rounded-cap regions meet.
        s = Stadium(center=torch.tensor([0.0, 0.0]), length=torch.tensor(1.5),
                   width=torch.nn.Parameter(torch.tensor(0.6)))
        assert_gradients_finite_at(s, ["width"], 0.75, 0.0)

    def test_broadcast_x_y_different_shapes(self):
        s = Stadium(center=[0.0, 0.0], length=1.5, width=0.6)
        x = torch.linspace(-1, 1, 5).reshape(1, 5)
        y = torch.linspace(-1, 1, 7).reshape(7, 1)
        out = s.sdf(x, y)
        assert out.shape == (7, 5)
