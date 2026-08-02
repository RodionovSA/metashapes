# tests/shape/test_ovals.py

import math
import pytest
import torch
from metashapes.shape import Shape
from metashapes.shape.primitives.ovals import Ellipse, Egg, Stadium
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
        # At the exact center of a non-circular ellipse the sdf should
        # equal -min(semi_a, semi_b).
        e = Ellipse(center=[0.0, 0.0], axes=[2.0, 4.0])  # semi-axes 1.0, 2.0
        d = sdf_at(e, 0.0, 0.0)
        assert d == pytest.approx(-1.0, abs=5e-4)

    def test_no_interior_sign_flip(self):
        # Sweep the interior and check every sample's sign against the
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

    @pytest.mark.parametrize("a,b,px,py", [
        (1.0, 0.5, 0.9, 0.7),     # exterior, generic aspect
        (1.0, 0.5, 0.2, 0.1),     # interior, generic aspect
        (1.8, 0.08, 1.0, 0.05),   # extreme aspect ratio
        (0.5, 0.5000001, 0.42, 0.33),  # near-circular (circle_mask branch)
    ])
    def test_matches_brute_force_nearest_point(self, a, b, px, py):
        # Checks sdf() accuracy against a dense brute-force scan of the
        # parametric ellipse boundary (float64), independent of the
        # closed-form solve. Points are chosen with true distance well
        # clear of sqrt(eps) (~3.5e-4) -- sdf()'s own `+ eps` inside its
        # magnitude sqrt floors sub-eps true distances and would otherwise
        # swamp this comparison.
        e = Ellipse(center=[0.0, 0.0], axes=[2 * a, 2 * b])
        d = sdf_at(e, px, py)

        t = torch.linspace(0, math.pi / 2, 400_001, dtype=torch.float64)
        ex, ey = a * torch.cos(t), b * torch.sin(t)
        dist = ((ex - px) ** 2 + (ey - py) ** 2).sqrt().min().item()
        sign = 1.0 if (px / a) ** 2 + (py / b) ** 2 > 1.0 else -1.0

        assert d == pytest.approx(sign * dist, abs=1e-5)

    def test_near_circular_sdf_is_finite(self):
        # An aspect ratio this close to 1 should route to the
        # near-circular radial-projection branch and stay finite.
        e = Ellipse(center=[0.0, 0.0], axes=[1.0, 1.0000005])
        xs = torch.linspace(-0.6, 0.6, 25)
        X, Y = torch.meshgrid(xs, xs, indexing="xy")
        d = e.sdf(X, Y)
        assert torch.isfinite(d).all(), "near-circular sdf produced non-finite values"
        # matches the radius-0.5 circle closely, since axes are ~equal
        # (float32 grid precision near axis boundaries needs a slightly
        # looser tolerance than the interior)
        expected = torch.sqrt(X ** 2 + Y ** 2) - 0.5
        assert torch.allclose(d, expected, atol=5e-4)

    def test_sdf_continuous_across_circle_mask_threshold(self):
        # The sdf shouldn't jump as an ellipse's aspect ratio crosses the
        # circle_mask threshold, i.e. as evaluation switches between
        # radial projection and the general cubic solve.
        a = 0.5
        ratios = torch.linspace(1.0 + 1e-6, 1.0 + 1e-3, 25)
        px, py = 0.35, 0.2
        vals = []
        for r in ratios:
            e = Ellipse(center=[0.0, 0.0], axes=[2 * a, 2 * a * r.item()])
            vals.append(sdf_at(e, px, py))
        vals = torch.tensor(vals)
        assert torch.isfinite(vals).all()
        max_jump = (vals[1:] - vals[:-1]).abs().max().item()
        assert max_jump < 1e-3, f"sdf jumped by {max_jump} across circle_mask threshold"


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
        # Points just above and below the seam should both read inside,
        # with closely matching sdf values -- no sign discontinuity there.
        e = Egg(center=[0.0, 0.0], width=2.0, height=2.5, skew=0.6)
        d_above = sdf_at(e, 0.0, 1e-4)
        d_below = sdf_at(e, 0.0, -1e-4)
        assert d_above < 0.0, f"expected inside just above the seam, got {d_above}"
        assert d_below < 0.0, f"expected inside just below the seam, got {d_below}"
        assert d_above == pytest.approx(d_below, abs=1e-2)

    def test_continuity_across_seam(self):
        # sdf should stay close on either side of the seam at several x
        # positions, not just at x=0.
        e = Egg(center=[0.0, 0.0], width=2.0, height=2.5, skew=0.6)
        for xv in (0.0, 0.2, 0.4, 0.6, 0.8):
            d_above = sdf_at(e, xv, 1e-4)
            d_below = sdf_at(e, xv, -1e-4)
            assert abs(d_above - d_below) < 5e-3, (
                f"seam jump at x={xv}: {d_above} vs {d_below}"
            )

    def test_continuity_across_seam_at_tighter_scale(self):
        # Same check as test_continuity_across_seam, an order of magnitude
        # closer to the seam.
        e = Egg(center=[0.0, 0.0], width=2.0, height=2.5, skew=0.6)
        for xv in (0.0, 0.2, 0.4, 0.6, 0.8):
            d_above = sdf_at(e, xv, 1e-6)
            d_below = sdf_at(e, xv, -1e-6)
            assert abs(d_above - d_below) < 5e-3, (
                f"seam jump at x={xv}: {d_above} vs {d_below}"
            )

    def test_no_sign_flip_grid_scan(self):
        # Same check as Ellipse's, but for Egg's asymmetric two-arc
        # boundary (b_top != b_bot), across the full shape including the
        # seam region.
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
        e = Ellipse(
            center=torch.nn.Parameter(torch.tensor([0.1, -0.1])),
            axes=torch.nn.Parameter(torch.tensor([1.0, 0.5])),
            angle=torch.nn.Parameter(torch.tensor(15.0)),
        )
        assert_gradients_finite_at(e, ["center", "axes", "angle"], 0.35, 0.15)

    def test_gradients_finite_dense_grid(self):
        # A dense grid exercises _ellipse_closest_point's branches across
        # many (m, n) configurations, including several points near each
        # branch's own degenerate inputs.
        e = Ellipse(
            center=torch.nn.Parameter(torch.tensor([0.1, -0.1])),
            axes=torch.nn.Parameter(torch.tensor([1.0, 0.5])),
            angle=torch.nn.Parameter(torch.tensor(15.0)),
        )
        assert_gradients_finite(e, ["center", "axes", "angle"])

    def test_gradients_finite_extreme_aspect_dense_grid(self):
        # Same grid regression, but with a much larger aspect ratio, to
        # exercise the general solve over a wider (m, n) range.
        e = Ellipse(
            center=torch.tensor([0.0, 0.0]),
            axes=torch.nn.Parameter(torch.tensor([1.8, 0.08])),
            angle=torch.tensor(37.0),
        )
        assert_gradients_finite(e, ["axes"])

    def test_gradients_finite_at_cube_root_degeneracy(self):
        # This point lands _ellipse_closest_point's branch-2 solve at
        # qm ~= 0, where a naive cube root has an infinite derivative;
        # confirm the gradient stays finite there.
        e = Ellipse(center=torch.tensor([0.1, -0.1]),
                   axes=torch.nn.Parameter(torch.tensor([1.0, 0.5])),
                   angle=torch.tensor(15.0))
        d = e.sdf(torch.tensor(0.12820512), torch.tensor(-0.84615386))
        d.backward()
        assert torch.isfinite(e.axes.grad).all(), f"got {e.axes.grad}"

    def test_gradients_finite_on_axis(self):
        # Any on-axis query (px == 0 or py == 0) drives m*n == 0 in
        # branch 1; confirm the gradient stays finite there, inside and
        # outside the ellipse on both axes.
        e = Ellipse(center=torch.tensor([0.0, 0.0]),
                   axes=torch.nn.Parameter(torch.tensor([1.0, 0.5])))
        # on the x-axis, inside and outside
        assert_gradients_finite_at(e, ["axes"], 0.3, 0.0)
        assert_gradients_finite_at(e, ["axes"], 0.9, 0.0)
        # on the y-axis, inside and outside
        assert_gradients_finite_at(e, ["axes"], 0.0, 0.1)
        assert_gradients_finite_at(e, ["axes"], 0.0, 0.6)

    def test_gradients_finite_at_evolute_cusp(self):
        # (0, (b^2 - a^2)/b) is the evolute cusp of an axis-aligned
        # ellipse -- the point where branch 2's P = sqrt(d) + m*n vanishes,
        # hitting the solve's own zero-guards.
        a, b = 1.0, 0.5
        cusp_y = (b * b - a * a) / b
        e = Ellipse(center=torch.tensor([0.0, 0.0]),
                   axes=torch.nn.Parameter(torch.tensor([2 * a, 2 * b])))
        assert_gradients_finite_at(e, ["axes"], 0.0, abs(cusp_y))

    def test_gradients_finite_at_branch_boundary(self):
        # d == 0 is the boundary between branch 1 (three real roots) and
        # branch 2 (one real root), where sqrt(d) has an infinite
        # derivative. Point located by bisecting py at fixed px=0.3 until
        # d crosses zero.
        e = Ellipse(center=torch.tensor([0.0, 0.0]),
                   axes=torch.nn.Parameter(torch.tensor([1.0, 0.5])))
        assert_gradients_finite_at(e, ["axes"], 0.3, 0.03854308373662893)

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
        e = Egg(
            center=torch.nn.Parameter(torch.tensor([0.1, -0.1])),
            width=torch.nn.Parameter(torch.tensor(0.6)),
            height=torch.nn.Parameter(torch.tensor(0.8)),
            skew=torch.nn.Parameter(torch.tensor(0.2)),
            angle=torch.nn.Parameter(torch.tensor(15.0)),
        )
        assert_gradients_finite_at(e, ["center", "width", "height", "skew", "angle"], 0.15, 0.3)

    def test_gradients_finite_dense_grid(self):
        # Egg calls the same shared _ellipse_closest_point solver twice per
        # point (once per arc); sweep it across a dense grid.
        e = Egg(
            center=torch.nn.Parameter(torch.tensor([0.1, -0.1])),
            width=torch.nn.Parameter(torch.tensor(0.6)),
            height=torch.nn.Parameter(torch.tensor(0.8)),
            skew=torch.nn.Parameter(torch.tensor(0.2)),
            angle=torch.nn.Parameter(torch.tensor(15.0)),
        )
        assert_gradients_finite(e, ["center", "width", "height", "skew", "angle"])

    def test_gradients_finite_off_seam(self):
        # Once clearly off the exact seam, gradients are finite.
        e = Egg(center=torch.tensor([0.0, 0.0]), width=torch.tensor(0.6),
               height=torch.tensor(0.8), skew=torch.nn.Parameter(torch.tensor(0.6)))
        assert_gradients_finite_at(e, ["skew"], 0.2, 0.002)

    def test_gradients_finite_very_close_to_seam(self):
        # Within 1e-4 of y_local=0, both arcs' queries are on-axis
        # (m*n ~= 0); confirm the gradient stays finite there and closer
        # in, even though the sdf *value* is already correct at the seam.
        e = Egg(center=torch.tensor([0.0, 0.0]), width=torch.tensor(0.6),
               height=torch.tensor(0.8), skew=torch.nn.Parameter(torch.tensor(0.6)))
        for y in (1e-4, -1e-4, 1e-6, -1e-6):
            e.skew.grad = None
            d = e.sdf(torch.tensor(0.1), torch.tensor(y))
            d.backward()
            assert torch.isfinite(e.skew.grad), (
                f"the near-seam gradient singularity at y={y} should be fixed; "
                f"got {e.skew.grad}"
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
