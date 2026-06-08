# tests/shape/test_periodic.py

import pytest
import torch
from metashapes.shape.primitives.periodic import Stripe
from metashapes.shape.primitives.quads import Rectangle
from metashapes.shape.transforms import Rotate
from metashapes.shape import Shape
from .conftest import assert_inside, assert_outside, assert_round_trip, sdf_at


class TestStripe:
    # --- x-axis stripe (perpendicular = y) --------------------------------

    def test_x_stripe_center_inside(self):
        s = Stripe(offset=0.0, width=0.4, axis='x')
        assert_inside(s, [(0.0, 0.0), (100.0, 0.0), (-100.0, 0.0)])

    def test_x_stripe_outside(self):
        s = Stripe(offset=0.0, width=0.4, axis='x')
        assert_outside(s, [(0.0, 0.5), (0.0, -0.5)])

    def test_x_stripe_boundary(self):
        s = Stripe(offset=0.0, width=0.4, axis='x')
        d = sdf_at(s, 0.0, 0.2)
        assert abs(d) < 1e-5

    def test_x_stripe_interior_value(self):
        s = Stripe(offset=0.0, width=0.4, axis='x')
        d = sdf_at(s, 0.0, 0.0)
        assert d == pytest.approx(-0.2, abs=1e-5)

    def test_x_stripe_offset(self):
        s = Stripe(offset=1.0, width=0.4, axis='x')
        assert_inside(s, [(0.0, 1.0), (50.0, 1.0)])
        assert_outside(s, [(0.0, 0.0)])

    # --- y-axis stripe (perpendicular = x) --------------------------------

    def test_y_stripe_center_inside(self):
        s = Stripe(offset=0.0, width=0.3, axis='y')
        assert_inside(s, [(0.0, 0.0), (0.0, 100.0)])

    def test_y_stripe_outside(self):
        s = Stripe(offset=0.0, width=0.3, axis='y')
        assert_outside(s, [(0.5, 0.0), (-0.5, 0.0)])

    def test_y_stripe_boundary(self):
        s = Stripe(offset=0.0, width=0.3, axis='y')
        d = sdf_at(s, 0.15, 0.0)
        assert abs(d) < 1e-5

    def test_y_stripe_offset(self):
        s = Stripe(offset=-0.5, width=0.2, axis='y')
        assert_inside(s, [(-0.5, 0.0)])
        assert_outside(s, [(0.0, 0.0)])

    # --- bounds -----------------------------------------------------------

    def test_x_stripe_bounds_infinite_x(self):
        s = Stripe(offset=0.0, width=0.4, axis='x')
        (x0, y0), (x1, y1) = s.bounds()
        assert x0 == float('-inf')
        assert x1 == float('inf')
        assert y0 == pytest.approx(-0.2, abs=1e-5)
        assert y1 == pytest.approx( 0.2, abs=1e-5)

    def test_y_stripe_bounds_infinite_y(self):
        s = Stripe(offset=0.0, width=0.3, axis='y')
        (x0, y0), (x1, y1) = s.bounds()
        assert x0 == pytest.approx(-0.15, abs=1e-5)
        assert x1 == pytest.approx( 0.15, abs=1e-5)
        assert y0 == float('-inf')
        assert y1 == float('inf')

    # --- validation -------------------------------------------------------

    def test_invalid_width(self):
        with pytest.raises(ValueError):
            Stripe(offset=0.0, width=0.0, axis='x')

    def test_invalid_axis(self):
        with pytest.raises(ValueError):
            Stripe(offset=0.0, width=0.3, axis='z')

    def test_rotate_raises(self):
        s = Stripe(offset=0.0, width=0.3, axis='x')
        with pytest.raises(NotImplementedError):
            s.rotate(45.0)

    def test_union_with_stripe_rotate_raises(self):
        stripe = Stripe(offset=0.0, width=0.2, axis='x')
        rect = Rectangle(center=(0, 0), size=(0.4, 0.4), angle=0.0)
        union = stripe | rect
        with pytest.raises(NotImplementedError, match="infinite"):
            union.rotate(45.0)

    def test_rotate_composition_via_rotate_class_raises(self):
        stripe = Stripe(offset=0.1, width=0.2, axis='x')
        rect = Rectangle(center=(0, 0), size=(0.3, 0.3), angle=0.0)
        shape = stripe | rect
        with pytest.raises(NotImplementedError, match="infinite"):
            Rotate(shape, angle=30.0)

    # --- serialization ----------------------------------------------------

    def test_round_trip_x_axis(self):
        s = Stripe(offset=0.2, width=0.3, axis='x')
        assert_round_trip(s, grid_range=(-2.0, 2.0))

    def test_round_trip_y_axis(self):
        s = Stripe(offset=-0.1, width=0.2, axis='y')
        assert_round_trip(s, grid_range=(-2.0, 2.0))

    def test_round_trip_preserves_axis(self):
        for axis in ('x', 'y'):
            s = Stripe(offset=0.0, width=0.3, axis=axis)
            restored = Shape.from_parametric(s.to_parametric())
            assert restored.axis == axis, f"axis lost after round-trip (expected {axis!r})"
