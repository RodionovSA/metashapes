# tests/shape/test_polygons.py

import math
import pytest
import torch
from metashapes.shape.primitives.polygons import RegularPolygon, Triangle, Star
from .conftest import (
    assert_inside, assert_outside, assert_round_trip, assert_bounds_contain, sdf_at,
    assert_dtype_device_flow, assert_direct_call_dtype_promotion,
    assert_gradients_finite, assert_gradients_finite_at,
)


class TestRegularPolygon:
    # --- triangle ---------------------------------------------------------

    def test_triangle_center_inside(self):
        p = RegularPolygon(center=[0.0, 0.0], n=3, side_length=1.0)
        assert_inside(p, [(0.0, 0.0)])

    def test_triangle_far_outside(self):
        p = RegularPolygon(center=[0.0, 0.0], n=3, side_length=1.0)
        assert_outside(p, [(2.0, 0.0), (0.0, -2.0)])

    # --- square (n=4) acts as a diamond by default ------------------------

    def test_square_center_inside(self):
        p = RegularPolygon(center=[0.0, 0.0], n=4, side_length=1.0)
        assert_inside(p, [(0.0, 0.0)])

    def test_square_corner_outside(self):
        # circumradius of a square with side 1 is sqrt(2)/2 ≈ 0.707
        p = RegularPolygon(center=[0.0, 0.0], n=4, side_length=1.0)
        assert_outside(p, [(0.8, 0.8)])

    # --- hexagon ----------------------------------------------------------

    def test_hexagon_center_inside(self):
        p = RegularPolygon(center=[0.0, 0.0], n=6, side_length=0.5)
        assert_inside(p, [(0.0, 0.0)])

    def test_hexagon_far_outside(self):
        p = RegularPolygon(center=[0.0, 0.0], n=6, side_length=0.5)
        assert_outside(p, [(1.0, 0.0), (0.0, 1.0)])

    # --- rotation ---------------------------------------------------------

    def test_rotated(self):
        p = RegularPolygon(center=[0.0, 0.0], n=6, side_length=0.6, angle=30.0)
        assert_inside(p, [(0.0, 0.0)])

    # --- corner radius ----------------------------------------------------

    def test_corner_radius(self):
        p = RegularPolygon(center=[0.0, 0.0], n=6, side_length=0.6, corner_radius=0.05)
        assert_inside(p, [(0.0, 0.0)])

    def test_corner_radius_too_large(self):
        # apothem of hexagon with side 0.4: rho = 0.4 / (2*tan(π/6)) ≈ 0.346
        with pytest.raises(ValueError):
            RegularPolygon(center=[0.0, 0.0], n=6, side_length=0.4, corner_radius=0.5)

    # --- offset center ----------------------------------------------------

    def test_offset_center(self):
        p = RegularPolygon(center=[1.0, -1.0], n=5, side_length=0.4)
        assert_inside(p, [(1.0, -1.0)])
        assert_outside(p, [(0.0, 0.0)])

    # --- validation -------------------------------------------------------

    def test_n_too_small(self):
        with pytest.raises(ValueError):
            RegularPolygon(center=[0.0, 0.0], n=2, side_length=1.0)

    def test_negative_side_length(self):
        with pytest.raises(ValueError):
            RegularPolygon(center=[0.0, 0.0], n=4, side_length=-1.0)

    # --- bounds -----------------------------------------------------------

    def test_bounds_contain_interior(self):
        p = RegularPolygon(center=[0.0, 0.0], n=5, side_length=0.6, angle=18.0)
        assert_bounds_contain(p, [(0.0, 0.0)])

    # --- serialization ----------------------------------------------------

    def test_round_trip_triangle(self):
        p = RegularPolygon(center=[0.1, 0.2], n=3, side_length=0.7, angle=15.0)
        assert_round_trip(p)

    def test_round_trip_hexagon(self):
        p = RegularPolygon(center=[0.0, 0.0], n=6, side_length=0.5, corner_radius=0.04)
        assert_round_trip(p)

    def test_round_trip_preserves_n(self):
        from metashapes.shape import Shape
        p = RegularPolygon(center=[0.0, 0.0], n=5, side_length=0.5)
        restored = Shape.from_parametric(p.to_parametric())
        assert restored.n == 5


class TestTriangle:
    def test_center_inside(self):
        t = Triangle(center=[0.0, 0.0], base=2.0, alpha=60.0, beta=60.0)
        assert_inside(t, [(0.0, 0.0)])

    def test_far_outside(self):
        t = Triangle(center=[0.0, 0.0], base=2.0, alpha=60.0, beta=60.0)
        assert_outside(t, [(5.0, 0.0), (0.0, -5.0), (-5.0, 0.0)])

    def test_base_right_vertex_boundary(self):
        # Equilateral, base=2: right base vertex B is at (1, -cy_apex/3)
        t = Triangle(center=[0.0, 0.0], base=2.0, alpha=60.0, beta=60.0)
        cy_apex = 2.0 * math.sin(math.radians(60.0)) ** 2 / math.sin(math.radians(120.0))
        Bx = 1.0 - 0.0          # base/2 - 0 (gcx=0 for equilateral)
        By = -cy_apex / 3.0
        d = sdf_at(t, Bx, By)
        assert abs(d) < 1e-4

    def test_isosceles_left_right_symmetry(self):
        # alpha == beta → triangle symmetric about y-axis, SDF(x, y) == SDF(-x, y)
        t = Triangle(center=[0.0, 0.0], base=2.0, alpha=50.0, beta=50.0)
        xs = torch.linspace(-0.5, 0.5, 10)
        ys = torch.tensor([0.0]).expand(10)
        X, Y = torch.meshgrid(xs, ys, indexing="xy")
        sdf_pos = t.sdf(X, Y)
        sdf_neg = t.sdf(-X, Y)
        assert torch.allclose(sdf_pos, sdf_neg, atol=1e-5)

    def test_right_triangle(self):
        # alpha=90, beta=45 → right triangle; center inside, far points outside
        t = Triangle(center=[0.0, 0.0], base=1.0, alpha=90.0, beta=45.0)
        assert_inside(t, [(0.0, 0.0)])
        assert_outside(t, [(3.0, 0.0), (0.0, -3.0)])

    def test_scalene_inside_outside(self):
        t = Triangle(center=[0.0, 0.0], base=1.5, alpha=40.0, beta=70.0)
        assert_inside(t, [(0.0, 0.0)])
        assert_outside(t, [(2.0, 0.0)])

    def test_offset_center(self):
        t = Triangle(center=[1.0, -1.0], base=1.0, alpha=60.0, beta=60.0)
        assert_inside(t, [(1.0, -1.0)])
        assert_outside(t, [(0.0, 0.0)])

    def test_rotation(self):
        # Rotation preserves centroid inside; same triangle rotated 90°
        t = Triangle(center=[0.0, 0.0], base=2.0, alpha=60.0, beta=60.0, angle=90.0)
        assert_inside(t, [(0.0, 0.0)])
        # A point clearly beyond the rotated apex should be outside
        assert_outside(t, [(0.0, -2.0)])

    def test_corner_radius_center_inside(self):
        t = Triangle(center=[0.0, 0.0], base=2.0, alpha=60.0, beta=60.0, corner_radius=0.1)
        assert_inside(t, [(0.0, 0.0)])

    def test_corner_radius_too_large(self):
        # inradius of equilateral with base=1: r = base / (2*sqrt(3)) ≈ 0.289
        with pytest.raises(ValueError):
            Triangle(center=[0.0, 0.0], base=1.0, alpha=60.0, beta=60.0, corner_radius=0.5)

    def test_zero_radius_sdf_finite_and_correct(self):
        # The corner-inset block is called unconditionally, including at
        # rr=0 where it's an exact identity; pin that result against a
        # wide grid.
        t = Triangle(center=[0.05, -0.1], base=1.3, alpha=50.0, beta=65.0,
                     angle=15.0, corner_radius=0.0)
        x = torch.linspace(-1.5, 1.5, 60)
        y = torch.linspace(-1.5, 1.5, 60)
        X, Y = torch.meshgrid(x, y, indexing="xy")
        d = t.sdf(X, Y)
        assert torch.isfinite(d).all()
        assert_inside(t, [(0.05, -0.1)])
        assert_outside(t, [(3.0, 3.0)])

    def test_bounds_contain_centroid(self):
        t = Triangle(center=[0.5, -0.3], base=1.5, alpha=45.0, beta=70.0, angle=30.0)
        assert_bounds_contain(t, [(0.5, -0.3)])

    def test_invalid_base(self):
        with pytest.raises(ValueError):
            Triangle(center=[0.0, 0.0], base=0.0, alpha=60.0, beta=60.0)
        with pytest.raises(ValueError):
            Triangle(center=[0.0, 0.0], base=-1.0, alpha=60.0, beta=60.0)

    def test_invalid_angles_zero(self):
        with pytest.raises(ValueError):
            Triangle(center=[0.0, 0.0], base=1.0, alpha=0.0, beta=60.0)
        with pytest.raises(ValueError):
            Triangle(center=[0.0, 0.0], base=1.0, alpha=60.0, beta=0.0)

    def test_invalid_angles_sum(self):
        with pytest.raises(ValueError):
            Triangle(center=[0.0, 0.0], base=1.0, alpha=90.0, beta=90.0)

    def test_round_trip(self):
        t = Triangle(center=[0.1, -0.2], base=1.2, alpha=55.0, beta=75.0, angle=20.0)
        assert_round_trip(t)

    def test_round_trip_with_corner_radius(self):
        t = Triangle(center=[0.0, 0.0], base=1.5, alpha=60.0, beta=60.0, corner_radius=0.05)
        assert_round_trip(t)


class TestStar:
    # --- basic inside / outside ---

    def test_center_inside(self):
        s = Star(center=[0.0, 0.0], n=5, outer_radius=0.5, inner_radius=0.2)
        assert_inside(s, [(0.0, 0.0)])

    def test_far_outside(self):
        s = Star(center=[0.0, 0.0], n=5, outer_radius=0.5, inner_radius=0.2)
        assert_outside(s, [(1.0, 0.0), (0.0, -1.0), (-1.0, 1.0)])

    def test_tip_boundary(self):
        # First tip of an unrotated star is at (0, outer_radius)
        s = Star(center=[0.0, 0.0], n=5, outer_radius=0.5, inner_radius=0.2)
        d = sdf_at(s, 0.0, 0.5)
        assert abs(d) < 1e-4

    def test_valley_boundary(self):
        # First valley is at (inner_radius*sin(π/n), inner_radius*cos(π/n))
        n = 5
        r = 0.2
        an = math.pi / n
        vx = r * math.sin(an)
        vy = r * math.cos(an)
        s = Star(center=[0.0, 0.0], n=n, outer_radius=0.5, inner_radius=r)
        d = sdf_at(s, vx, vy)
        assert abs(d) < 1e-4

    # --- classic 5-pointed star ---

    def test_five_pointed_arm_inside(self):
        # Point along tip axis (y > 0) but within outer_radius
        s = Star(center=[0.0, 0.0], n=5, outer_radius=0.5, inner_radius=0.2)
        assert_inside(s, [(0.0, 0.3)])

    def test_five_pointed_gap_outside(self):
        # Between arms: midway between two tips in polar space, at inner_radius+epsilon → outside
        n = 5
        an = math.pi / n
        r_inner = 0.2
        # The re-entrant notch: at angle π/n from tip, just beyond inner_radius
        gx = (r_inner + 0.05) * math.sin(an)
        gy = (r_inner + 0.05) * math.cos(an)
        s = Star(center=[0.0, 0.0], n=n, outer_radius=0.5, inner_radius=r_inner)
        assert_outside(s, [(gx, gy)])

    # --- n variation ---

    def test_three_pointed(self):
        s = Star(center=[0.0, 0.0], n=3, outer_radius=0.5, inner_radius=0.15)
        assert_inside(s, [(0.0, 0.0)])
        assert_outside(s, [(0.0, 0.6)])

    def test_eight_pointed(self):
        s = Star(center=[0.0, 0.0], n=8, outer_radius=0.4, inner_radius=0.2)
        assert_inside(s, [(0.0, 0.0)])
        assert_outside(s, [(0.0, 0.5)])

    # --- outer corner radius ---

    def test_outer_corner_radius_center_inside(self):
        s = Star(center=[0.0, 0.0], n=5, outer_radius=0.5, inner_radius=0.2,
                 outer_corner_radius=0.05)
        assert_inside(s, [(0.0, 0.0)])

    def test_outer_corner_radius_too_large(self):
        with pytest.raises(ValueError):
            Star(center=[0.0, 0.0], n=5, outer_radius=0.5, inner_radius=0.2,
                 outer_corner_radius=0.35)  # >= 0.5 - 0.2 = 0.3

    # --- inner corner radius ---

    def test_inner_corner_radius_center_inside(self):
        s = Star(center=[0.0, 0.0], n=5, outer_radius=0.5, inner_radius=0.2,
                 inner_corner_radius=0.03)
        assert_inside(s, [(0.0, 0.0)])

    def test_inner_corner_radius_too_large(self):
        n = 5
        R = 0.5
        r = 0.2
        # New bound: icr_max = L * tan(beta), where
        #   L = sqrt(R^2 + r^2 - 2*R*r*cos(pi/n))
        #   sin(beta) = R * sin(pi/n) / L
        # For these parameters icr_max ≈ 0.514, so 0.6 exceeds it.
        with pytest.raises(ValueError):
            Star(center=[0.0, 0.0], n=n, outer_radius=R, inner_radius=r,
                inner_corner_radius=0.6)

    # --- both corner radii together ---

    def test_both_corner_radii(self):
        s = Star(center=[0.0, 0.0], n=5, outer_radius=0.5, inner_radius=0.2,
                 outer_corner_radius=0.04, inner_corner_radius=0.03)
        assert_inside(s, [(0.0, 0.0)])

    # --- rotation ---

    def test_rotation(self):
        # Rotating by π/n maps a tip to where a valley was
        s = Star(center=[0.0, 0.0], n=5, outer_radius=0.5, inner_radius=0.2)
        # Tip at (0, 0.5) before rotation; after rotating by 36° the star tip is elsewhere
        s_rot = Star(center=[0.0, 0.0], n=5, outer_radius=0.5, inner_radius=0.2,
                     angle=180.0 / 5)
        assert_inside(s_rot, [(0.0, 0.0)])

    # --- offset center ---

    def test_offset_center(self):
        s = Star(center=[1.0, -1.0], n=5, outer_radius=0.3, inner_radius=0.12)
        assert_inside(s, [(1.0, -1.0)])
        assert_outside(s, [(0.0, 0.0)])

    # --- validation ---

    def test_n_too_small(self):
        with pytest.raises(ValueError):
            Star(center=[0.0, 0.0], n=2, outer_radius=0.5, inner_radius=0.2)

    def test_inner_radius_zero(self):
        with pytest.raises(ValueError):
            Star(center=[0.0, 0.0], n=5, outer_radius=0.5, inner_radius=0.0)

    def test_inner_radius_exceeds_outer(self):
        with pytest.raises(ValueError):
            Star(center=[0.0, 0.0], n=5, outer_radius=0.5, inner_radius=0.6)

    # --- bounds ---

    def test_bounds_contain_center(self):
        s = Star(center=[0.1, -0.2], n=5, outer_radius=0.4, inner_radius=0.15, angle=12.0)
        assert_bounds_contain(s, [(0.1, -0.2)])

    # --- serialization ---

    def test_round_trip(self):
        s = Star(center=[0.1, 0.2], n=5, outer_radius=0.5, inner_radius=0.2, angle=10.0)
        assert_round_trip(s)

    def test_round_trip_preserves_n(self):
        from metashapes.shape import Shape
        s = Star(center=[0.0, 0.0], n=6, outer_radius=0.4, inner_radius=0.2)
        restored = Shape.from_parametric(s.to_parametric())
        assert restored.n == 6

    def test_round_trip_with_corner_radii(self):
        s = Star(center=[0.0, 0.0], n=5, outer_radius=0.5, inner_radius=0.2,
                 outer_corner_radius=0.04, inner_corner_radius=0.03)
        assert_round_trip(s)

    def test_min_feature_size_is_twice_outer_corner_radius(self):
        # The tip is where a spike actually gets thin: it's an exact
        # circular arc of outer_corner_radius by construction of the SDF,
        # so its own width (through the arc's center) is exactly
        # 2*outer_corner_radius -- not 2*inner_radius (the diameter across
        # valleys, unrelated to the spike's own thinness).
        s = Star(center=[0.0, 0.0], n=5, outer_radius=1.0, inner_radius=0.4,
                 outer_corner_radius=0.08)
        assert s.min_feature_size == pytest.approx(0.16, abs=1e-6)

    def test_min_feature_size_zero_for_sharp_tip(self):
        # An unrounded tip is mathematically a sharp point -- zero is the
        # correct, useful answer (an infinitely sharp feature should fail
        # any nonzero min_feature_size manufacturability check), not
        # 2*inner_radius (which was always large and never caught this).
        s = Star(center=[0.0, 0.0], n=5, outer_radius=1.0, inner_radius=0.4)
        assert s.min_feature_size == pytest.approx(0.0, abs=1e-9)

    def test_min_feature_size_independent_of_inner_radius(self):
        # The old formula (2*inner_radius) varied with inner_radius even
        # though it has no bearing on how thin the tip is; the new one
        # must not, for a fixed outer_corner_radius.
        s1 = Star(center=[0.0, 0.0], n=5, outer_radius=1.0, inner_radius=0.2,
                  outer_corner_radius=0.05)
        s2 = Star(center=[0.0, 0.0], n=5, outer_radius=1.0, inner_radius=0.6,
                  outer_corner_radius=0.05)
        assert s1.min_feature_size == pytest.approx(s2.min_feature_size, abs=1e-9)
        assert s1.min_feature_size == pytest.approx(0.10, abs=1e-6)

    def test_min_feature_size_updates_with_current_parameter_value(self):
        # min_feature_size must be computed fresh from the current
        # parameter value (matters if outer_corner_radius is an
        # nn.Parameter being optimized), not cached at construction.
        ocr = torch.nn.Parameter(torch.tensor(0.03))
        s = Star(center=[0.0, 0.0], n=5, outer_radius=1.0, inner_radius=0.4,
                 outer_corner_radius=ocr)
        assert s.min_feature_size == pytest.approx(0.06, abs=1e-6)
        with torch.no_grad():
            ocr.copy_(torch.tensor(0.07))
        assert s.min_feature_size == pytest.approx(0.14, abs=1e-6)


# ---------------------------------------------------------------------------
# dtype / device / gradient flow
# ---------------------------------------------------------------------------

class TestRegularPolygonDtypeDeviceGrad:
    def test_dtype_device_flow(self):
        p = RegularPolygon(center=[0.1, -0.1], n=5, side_length=0.5, angle=10.0,
                           corner_radius=0.05)
        assert_dtype_device_flow(p)

    def test_direct_call_dtype_promotion(self):
        p = RegularPolygon(center=[0.0, 0.0], n=5, side_length=0.5)
        assert_direct_call_dtype_promotion(p)

    def test_integer_query_does_not_crash(self):
        p = RegularPolygon(center=[0.0, 0.0], n=5, side_length=0.5)
        out = p.sdf(torch.tensor(0), torch.tensor(0))
        assert torch.isfinite(out)

    def test_gradients_finite_generic_point(self):
        p = RegularPolygon(
            center=torch.nn.Parameter(torch.tensor([0.1, -0.1])),
            n=5,
            side_length=torch.nn.Parameter(torch.tensor(0.5)),
            angle=torch.nn.Parameter(torch.tensor(10.0)),
            corner_radius=torch.nn.Parameter(torch.tensor(0.05)),
        )
        assert_gradients_finite(p, ["center", "side_length", "angle", "corner_radius"])

    def test_gradients_finite_at_zero_and_near_bound_corner_radius(self):
        for rr in (0.0, 0.34):  # rho (rr_max) ~= 0.3441 for n=5, side_length=0.5
            p = RegularPolygon(center=torch.tensor([0.0, 0.0]), n=5,
                               side_length=torch.tensor(0.5),
                               corner_radius=torch.nn.Parameter(torch.tensor(rr)))
            assert_gradients_finite(p, ["corner_radius"])

    def test_n_extremes(self):
        # Minimum valid n and a large n, for correctness and no perf/NaN
        # surprise (not part of screening, but cheap to pin).
        for n in (3, 64):
            p = RegularPolygon(center=[0.0, 0.0], n=n, side_length=0.3)
            d = p.sdf(torch.tensor(0.0), torch.tensor(0.0))
            assert d.item() < 0
            xs = torch.linspace(-1, 1, 20)
            X, Y = torch.meshgrid(xs, xs, indexing="xy")
            assert torch.isfinite(p.sdf(X, Y)).all()

    def test_corner_radius_drift_past_bound_is_clamped_by_project(self):
        # Parameters aren't re-validated at construction time against a
        # later in-place update, but sdf()/bounds()/min_feature_size all
        # call _project() first, which clamps corner_radius back under the
        # apothem instead of letting it produce inverted geometry.
        rr = torch.nn.Parameter(torch.tensor(0.1))
        p = RegularPolygon(center=[0.0, 0.0], n=5, side_length=torch.tensor(0.5),
                           corner_radius=rr)
        with torch.no_grad():
            rr.copy_(torch.tensor(0.5))  # past rho ~= 0.3441
        d = p.sdf(torch.tensor(0.0), torch.tensor(0.0))
        assert torch.isfinite(d)
        rho = 0.5 / (2.0 * math.tan(math.pi / 5))
        assert rr.item() < rho
        assert d.item() < 0  # center is still comfortably inside

    def test_gradients_finite_exactly_at_vertex(self):
        # min_d2 is exactly 0.0 at a vertex of the (r=0) polygon -- the
        # same sqrt(0) singularity as Rectangle's sharp-corner case (see
        # TestRectangleDtypeDeviceGrad). RegularPolygonSDF routes through
        # _PolygonSDF's clamp_min-before-sqrt, so this must stay finite.
        n, s = 4, 1.0
        p = RegularPolygon(
            center=torch.tensor([0.0, 0.0]), n=n,
            side_length=torch.nn.Parameter(torch.tensor(s)),
        )
        R = s / (2.0 * math.sin(math.pi / n))
        vx, vy = 0.0, R  # first vertex, at phi0 = pi/2
        d = p.sdf(torch.tensor(vx), torch.tensor(vy))
        assert d.item() == pytest.approx(0.0, abs=1e-6)
        d.backward()
        assert torch.isfinite(p.side_length.grad).all()

    def test_broadcast_x_y_different_shapes(self):
        p = RegularPolygon(center=[0.0, 0.0], n=5, side_length=0.5)
        x = torch.linspace(-1, 1, 5).reshape(1, 5)
        y = torch.linspace(-1, 1, 7).reshape(7, 1)
        out = p.sdf(x, y)
        assert out.shape == (7, 5)


class TestRegularPolygonProject:
    def test_clamps_side_length_and_corner_radius(self):
        p = RegularPolygon(center=[0.0, 0.0], n=5, side_length=torch.tensor(0.5),
                           corner_radius=torch.tensor(0.05))
        with torch.no_grad():
            p.side_length.fill_(-1.0)
            p.corner_radius.fill_(10.0)

        p._project()

        # float32 storage of 1e-6 rounds down very slightly on this
        # hardware, so compare with a tolerance rather than >=.
        assert p.side_length.item() == pytest.approx(p._MIN_SIZE, abs=1e-9)
        rho = p.side_length.item() / (2.0 * math.tan(math.pi / 5))
        assert p.corner_radius.item() < rho
        out = p.sdf(torch.tensor(0.0), torch.tensor(0.0))
        assert torch.isfinite(out)

    def test_sdf_bounds_and_min_feature_size_self_project(self):
        p = RegularPolygon(center=[0.0, 0.0], n=5, side_length=torch.tensor(0.5),
                           corner_radius=torch.tensor(0.05))
        with torch.no_grad():
            p.corner_radius.fill_(10.0)

        out = p.sdf(torch.tensor(0.0), torch.tensor(0.0))
        assert torch.isfinite(out)
        p.bounds()  # must not raise
        assert p.min_feature_size > 0


class TestTriangleDtypeDeviceGrad:
    def test_dtype_device_flow(self):
        t = Triangle(center=[0.1, -0.1], base=0.8, alpha=50.0, beta=65.0, angle=10.0,
                    corner_radius=0.05)
        assert_dtype_device_flow(t)

    def test_direct_call_dtype_promotion(self):
        t = Triangle(center=[0.0, 0.0], base=0.8, alpha=60.0, beta=60.0)
        assert_direct_call_dtype_promotion(t)

    def test_integer_query_does_not_crash(self):
        t = Triangle(center=[0.0, 0.0], base=0.8, alpha=60.0, beta=60.0)
        out = t.sdf(torch.tensor(0), torch.tensor(0))
        assert torch.isfinite(out)

    def test_gradients_finite_generic_point(self):
        t = Triangle(
            center=torch.nn.Parameter(torch.tensor([0.1, -0.1])),
            base=torch.nn.Parameter(torch.tensor(0.8)),
            alpha=torch.nn.Parameter(torch.tensor(50.0)),
            beta=torch.nn.Parameter(torch.tensor(65.0)),
            angle=torch.nn.Parameter(torch.tensor(10.0)),
            corner_radius=torch.nn.Parameter(torch.tensor(0.05)),
        )
        assert_gradients_finite(t, ["center", "base", "alpha", "beta", "angle", "corner_radius"])

    def test_gradients_finite_at_zero_and_near_inradius_corner_radius(self):
        for rr in (0.0, 0.23):  # inradius ~= 0.2309 for base=0.8, alpha=beta=60
            t = Triangle(center=torch.tensor([0.0, 0.0]), base=torch.tensor(0.8),
                        alpha=torch.tensor(60.0), beta=torch.tensor(60.0),
                        corner_radius=torch.nn.Parameter(torch.tensor(rr)))
            assert_gradients_finite(t, ["corner_radius"])

    def test_gradients_finite_right_angle(self):
        t = Triangle(
            center=torch.tensor([0.0, 0.0]),
            base=torch.nn.Parameter(torch.tensor(1.0)),
            alpha=torch.nn.Parameter(torch.tensor(90.0)),
            beta=torch.nn.Parameter(torch.tensor(45.0)),
        )
        assert_gradients_finite_at(t, ["base", "alpha", "beta"], 0.05, 0.05)

    def test_gradients_finite_exactly_on_vertical_leg(self):
        # This right-angle triangle's leg happens to be exactly vertical
        # (at x=-1/3 for base=1.0), which used to land exactly on a
        # linspace(-1,1,40) grid column in the generic-point gradient
        # test above -- i.e. an on-edge point was already being hit
        # incidentally. TriangleSDF now clamps the sqrt argument (not the
        # result), so query it deliberately instead of relying on
        # coincidence.
        t = Triangle(
            center=torch.tensor([0.0, 0.0]),
            base=torch.nn.Parameter(torch.tensor(1.0)),
            alpha=torch.nn.Parameter(torch.tensor(90.0)),
            beta=torch.nn.Parameter(torch.tensor(45.0)),
        )
        (Ax, Ay), (Bx, By), (Cx, Cy) = t._vertices()
        x_on_edge = Ax.item()
        y_on_edge = 0.5 * (Ay.item() + Cy.item())  # midpoint of the A-C leg
        assert_gradients_finite_at(t, ["base", "alpha", "beta"], x_on_edge, y_on_edge)

    def test_gradients_finite_near_degenerate_angle_sum(self):
        # alpha + beta -> 180 (validated strictly less than 180 at
        # construction) is a near-degenerate, very "flat" triangle.
        t = Triangle(
            center=torch.tensor([0.0, 0.0]),
            base=torch.nn.Parameter(torch.tensor(1.0)),
            alpha=torch.nn.Parameter(torch.tensor(89.0)),
            beta=torch.nn.Parameter(torch.tensor(89.0)),
        )
        assert_gradients_finite(t, ["base", "alpha", "beta"])

    def test_broadcast_x_y_different_shapes(self):
        t = Triangle(center=[0.0, 0.0], base=0.8, alpha=60.0, beta=60.0)
        x = torch.linspace(-1, 1, 5).reshape(1, 5)
        y = torch.linspace(-1, 1, 7).reshape(7, 1)
        out = t.sdf(x, y)
        assert out.shape == (7, 5)


class TestTriangleProject:
    def test_clamps_base_corner_radius_and_angle_sum(self):
        t = Triangle(center=[0.0, 0.0], base=torch.tensor(0.8),
                    alpha=torch.tensor(60.0), beta=torch.tensor(60.0),
                    corner_radius=torch.tensor(0.05))
        with torch.no_grad():
            t.base.fill_(-1.0)
            t.alpha.fill_(-10.0)
            t.beta.fill_(-10.0)
            t.corner_radius.fill_(10.0)

        t._project()

        # float32 storage of 1e-6 rounds down very slightly on this
        # hardware, so compare with a tolerance rather than >=.
        assert t.base.item() == pytest.approx(t._MIN_SIZE, abs=1e-9)
        assert t.alpha.item() >= t._MIN_ANGLE_DEG
        assert t.beta.item() >= t._MIN_ANGLE_DEG
        assert t.alpha.item() + t.beta.item() < 180.0
        assert t.corner_radius.item() < t._inradius().item()
        out = t.sdf(torch.tensor(0.0), torch.tensor(0.0))
        assert torch.isfinite(out)

    def test_clamps_coupled_angle_sum_preserving_ratio(self):
        # alpha, beta drift to a sum >= 180 -- _project scales both down
        # together rather than picking one arbitrarily, so their ratio
        # survives.
        t = Triangle(center=[0.0, 0.0], base=torch.tensor(0.8),
                    alpha=torch.tensor(60.0), beta=torch.tensor(30.0))
        ratio_before = t.alpha.item() / t.beta.item()
        with torch.no_grad():
            t.alpha.fill_(140.0)
            t.beta.fill_(70.0)  # sum = 210 > 180, ratio unchanged (2:1)

        t._project()

        assert t.alpha.item() + t.beta.item() < 180.0
        ratio_after = t.alpha.item() / t.beta.item()
        assert ratio_after == pytest.approx(ratio_before, rel=1e-4)

    def test_sdf_bounds_and_min_feature_size_self_project(self):
        t = Triangle(center=[0.0, 0.0], base=torch.tensor(0.8),
                    alpha=torch.tensor(60.0), beta=torch.tensor(60.0))
        with torch.no_grad():
            t.alpha.fill_(179.0)
            t.beta.fill_(179.0)

        out = t.sdf(torch.tensor(0.0), torch.tensor(0.0))
        assert torch.isfinite(out)
        t.bounds()  # must not raise
        assert t.min_feature_size > 0


class TestStarDtypeDeviceGrad:
    def test_dtype_device_flow(self):
        s = Star(center=[0.1, -0.1], n=5, outer_radius=0.5, inner_radius=0.2,
                 angle=10.0, outer_corner_radius=0.02, inner_corner_radius=0.02)
        assert_dtype_device_flow(s)

    def test_direct_call_dtype_promotion(self):
        s = Star(center=[0.0, 0.0], n=5, outer_radius=0.5, inner_radius=0.2)
        assert_direct_call_dtype_promotion(s)

    def test_integer_query_does_not_crash(self):
        s = Star(center=[0.0, 0.0], n=5, outer_radius=0.5, inner_radius=0.2)
        out = s.sdf(torch.tensor(0), torch.tensor(0))
        assert torch.isfinite(out)

    def test_gradients_finite_generic_point(self):
        s = Star(
            center=torch.nn.Parameter(torch.tensor([0.1, -0.1])), n=5,
            outer_radius=torch.nn.Parameter(torch.tensor(0.5)),
            inner_radius=torch.nn.Parameter(torch.tensor(0.2)),
            angle=torch.nn.Parameter(torch.tensor(10.0)),
            outer_corner_radius=torch.nn.Parameter(torch.tensor(0.02)),
            inner_corner_radius=torch.nn.Parameter(torch.tensor(0.02)),
        )
        assert_gradients_finite(
            s, ["center", "outer_radius", "inner_radius", "angle",
                "outer_corner_radius", "inner_corner_radius"]
        )

    def test_gradients_finite_exactly_at_sector_seam(self):
        # bn (from atan2 + remainder folding) is a genuinely discontinuous
        # reparameterization at each sector seam by construction. Confirm
        # this doesn't produce a NaN gradient right at one (existing tests
        # already cover the SDF *value* staying continuous there).
        n = 5
        an = math.pi / n
        r = 0.3
        x = r * math.sin(an)
        y = r * math.cos(an)
        s = Star(center=torch.tensor([0.0, 0.0]), n=n,
                 outer_radius=torch.nn.Parameter(torch.tensor(0.5)),
                 inner_radius=torch.nn.Parameter(torch.tensor(0.2)))
        assert_gradients_finite_at(s, ["outer_radius", "inner_radius"], x, y)

    def test_gradients_finite_at_zero_and_near_bound_corner_radii(self):
        # ocr_max ~= 0.1245, icr_max ~= 0.5145 for n=5, R=0.5, r=0.2 --
        # but Star also enforces a *joint* bound (ocr/tan_alpha +
        # icr/tan_beta < L), so both individually near their own max
        # simultaneously is itself invalid; tested one at a time instead.
        for ocr, icr in [(0.0, 0.0), (0.12, 0.0), (0.0, 0.51)]:
            s = Star(center=torch.tensor([0.0, 0.0]), n=5,
                     outer_radius=torch.tensor(0.5), inner_radius=torch.tensor(0.2),
                     outer_corner_radius=torch.nn.Parameter(torch.tensor(ocr)),
                     inner_corner_radius=torch.nn.Parameter(torch.tensor(icr)))
            assert_gradients_finite(s, ["outer_corner_radius", "inner_corner_radius"])

    def test_n_extremes(self):
        for n in (3, 64):
            s = Star(center=[0.0, 0.0], n=n, outer_radius=0.5, inner_radius=0.2)
            d = s.sdf(torch.tensor(0.0), torch.tensor(0.0))
            assert d.item() < 0
            xs = torch.linspace(-1, 1, 20)
            X, Y = torch.meshgrid(xs, xs, indexing="xy")
            assert torch.isfinite(s.sdf(X, Y)).all()

    def test_gradients_finite_exactly_at_origin(self):
        # (0, 0) -- the star's own center -- is the single most likely
        # point anyone samples. r_point = sqrt(x^2+y^2) is exactly 0 there;
        # StarSDF clamps that sqrt's argument (not its result), so this
        # must stay finite.
        s = Star(
            center=torch.tensor([0.0, 0.0]), n=5,
            outer_radius=torch.nn.Parameter(torch.tensor(0.5)),
            inner_radius=torch.nn.Parameter(torch.tensor(0.2)),
        )
        assert_gradients_finite_at(s, ["outer_radius", "inner_radius"], 0.0, 0.0)

    def test_gradients_finite_exactly_at_tip(self):
        # The outer tip (first tip at +y, unrounded) is another exact
        # zero-distance point for the same class of singularity.
        s = Star(
            center=torch.tensor([0.0, 0.0]), n=5,
            outer_radius=torch.nn.Parameter(torch.tensor(0.5)),
            inner_radius=torch.nn.Parameter(torch.tensor(0.2)),
        )
        assert_gradients_finite_at(s, ["outer_radius", "inner_radius"], 0.0, 0.5)

    def test_broadcast_x_y_different_shapes(self):
        s = Star(center=[0.0, 0.0], n=5, outer_radius=0.5, inner_radius=0.2)
        x = torch.linspace(-1, 1, 5).reshape(1, 5)
        y = torch.linspace(-1, 1, 7).reshape(7, 1)
        out = s.sdf(x, y)
        assert out.shape == (7, 5)


class TestStarProject:
    def test_clamps_all_params_back_into_valid_range(self):
        s = Star(center=[0.0, 0.0], n=5, outer_radius=torch.tensor(0.5),
                 inner_radius=torch.tensor(0.2), outer_corner_radius=torch.tensor(0.02),
                 inner_corner_radius=torch.tensor(0.02))
        with torch.no_grad():
            s.outer_radius.fill_(-1.0)
            s.inner_radius.fill_(5.0)
            s.outer_corner_radius.fill_(5.0)
            s.inner_corner_radius.fill_(5.0)

        s._project()

        # float32 storage of 1e-6 rounds down very slightly on this
        # hardware, so compare with a tolerance rather than >=.
        assert s.outer_radius.item() == pytest.approx(s._MIN_SIZE, abs=1e-9)
        assert s.inner_radius.item() < s.outer_radius.item()
        an = math.pi / 5
        R, r = s.outer_radius.item(), s.inner_radius.item()
        L = math.sqrt(R * R + r * r - 2.0 * R * r * math.cos(an))
        sin_alpha = (r * math.sin(an)) / L
        tan_alpha = sin_alpha / math.sqrt(max(0.0, 1.0 - sin_alpha * sin_alpha))
        assert s.outer_corner_radius.item() < L * tan_alpha
        out = s.sdf(torch.tensor(0.0), torch.tensor(0.0))
        assert torch.isfinite(out)

    def test_sdf_bounds_and_min_feature_size_self_project(self):
        s = Star(center=[0.0, 0.0], n=5, outer_radius=torch.tensor(0.5),
                 inner_radius=torch.tensor(0.2))
        with torch.no_grad():
            s.inner_radius.fill_(5.0)  # now > outer_radius

        out = s.sdf(torch.tensor(0.0), torch.tensor(0.0))
        assert torch.isfinite(out)
        s.bounds()  # must not raise
        assert s.min_feature_size >= 0.0
