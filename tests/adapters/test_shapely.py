# tests/adapters/test_shapely.py

import math
import pytest
import torch

from metashapes.adapters.shapely.dispatch import shape_to_shapely
from metashapes.shape.primitives.quads import Rectangle, ConvexQuad, IsoscelesTrapezoid
from metashapes.shape.primitives.conics import Ellipse, Egg, Stadium
from metashapes.shape.primitives.polygons import RegularPolygon, Triangle, Star
from metashapes.shape.primitives.junctions import Cross, TShape
from metashapes.shape.primitives.periodic import Stripe
from metashapes.shape.boolean import Union, Intersection, Difference
from metashapes.shape.transforms import Translate, Rotate, Scale


def _centroid(geom):
    return geom.centroid.x, geom.centroid.y


# ---------------------------------------------------------------------------
# Primitive shapes
# ---------------------------------------------------------------------------

class TestPrimitivesToShapely:
    def test_rectangle(self):
        shape = Rectangle(center=[0.0, 0.0], size=[2.0, 1.0])
        geom = shape_to_shapely(shape)
        assert not geom.is_empty
        assert geom.area == pytest.approx(2.0, abs=1e-4)

    def test_rectangle_with_corner_radius(self):
        shape = Rectangle(center=[0.0, 0.0], size=[2.0, 2.0], corner_radius=0.2)
        geom = shape_to_shapely(shape)
        assert not geom.is_empty
        assert geom.area > 0.0
        assert geom.area < 4.0

    def test_ellipse(self):
        # axes=[2.0, 1.0] means full diameters; semi-axes are 1.0 and 0.5
        shape = Ellipse(center=[0.0, 0.0], axes=torch.tensor([2.0, 1.0]))
        geom = shape_to_shapely(shape)
        assert not geom.is_empty
        assert geom.area == pytest.approx(math.pi * 1.0 * 0.5, abs=0.05)

    def test_ellipse_centroid(self):
        shape = Ellipse(center=[1.0, -2.0], axes=torch.tensor([2.0, 1.0]))
        geom = shape_to_shapely(shape)
        cx, cy = _centroid(geom)
        assert cx == pytest.approx(1.0, abs=0.01)
        assert cy == pytest.approx(-2.0, abs=0.01)

    def test_egg(self):
        # area ≈ π * a * (b_top + b_bot) / 2 = π * a * height / 2
        shape = Egg(center=[0.0, 0.0], width=2.0, height=1.0, skew=0.4)
        geom = shape_to_shapely(shape)
        assert not geom.is_empty
        a = 1.0
        expected_area = math.pi * a * 0.5  # π * a * height/2
        assert geom.area == pytest.approx(expected_area, rel=0.02)

    def test_egg_centroid(self):
        shape = Egg(center=[1.0, -2.0], width=1.0, height=0.8, skew=0.3)
        geom = shape_to_shapely(shape)
        cx, cy = _centroid(geom)
        assert cx == pytest.approx(1.0, abs=0.02)
        # centroid y is above center when skew > 0 (top half is larger)
        assert cy > -2.0

    def test_stadium(self):
        # length=2.0, width=1.0 → radius=0.5, span=1.0
        # area = π * r² + 2 * r * span = π * 0.25 + 1.0
        import math
        shape = Stadium(center=[0.0, 0.0], length=2.0, width=1.0)
        geom = shape_to_shapely(shape)
        assert not geom.is_empty
        expected_area = math.pi * 0.5 ** 2 + 2 * 0.5 * 1.0
        assert geom.area == pytest.approx(expected_area, abs=0.02)

    def test_stadium_centroid(self):
        shape = Stadium(center=[1.0, -2.0], length=2.0, width=0.8)
        geom = shape_to_shapely(shape)
        cx, cy = _centroid(geom)
        assert cx == pytest.approx(1.0, abs=0.01)
        assert cy == pytest.approx(-2.0, abs=0.01)

    def test_stadium_degenerate_circle(self):
        # length == width → circle of radius 0.5
        import math
        shape = Stadium(center=[0.0, 0.0], length=1.0, width=1.0)
        geom = shape_to_shapely(shape)
        assert geom.area == pytest.approx(math.pi * 0.5 ** 2, abs=0.02)

    def test_convex_quad(self):
        # u=[0.5,0], v=[0,0.5] with alpha=beta=0 → 1×1 square at origin
        shape = ConvexQuad(
            center=torch.zeros(2),
            u=torch.tensor([0.5, 0.0]),
            v=torch.tensor([0.0, 0.5]),
            alpha=torch.tensor(0.0),
            beta=torch.tensor(0.0),
        )
        geom = shape_to_shapely(shape)
        assert not geom.is_empty
        assert geom.is_valid
        assert geom.area == pytest.approx(1.0, abs=1e-4)

    def test_isosceles_trapezoid(self):
        shape = IsoscelesTrapezoid(
            center=torch.zeros(2),
            bottom_width=torch.tensor(2.0),
            top_width=torch.tensor(1.0),
            height=torch.tensor(1.0),
        )
        geom = shape_to_shapely(shape)
        assert not geom.is_empty
        # area = 0.5 * (bottom + top) * height = 0.5 * 3 * 1 = 1.5
        assert geom.area == pytest.approx(1.5, abs=1e-4)

    def test_triangle_equilateral_area(self):
        # Equilateral: area = base² * sqrt(3) / 4
        shape = Triangle(center=[0.0, 0.0], base=2.0, alpha=60.0, beta=60.0)
        geom = shape_to_shapely(shape)
        assert not geom.is_empty
        assert geom.area == pytest.approx(math.sqrt(3), abs=0.01)

    def test_triangle_centroid(self):
        shape = Triangle(center=[1.0, -2.0], base=1.5, alpha=60.0, beta=60.0)
        geom = shape_to_shapely(shape)
        cx, cy = _centroid(geom)
        assert cx == pytest.approx(1.0, abs=0.01)
        assert cy == pytest.approx(-2.0, abs=0.01)

    def test_triangle_corner_radius_reduces_area(self):
        shape_plain = Triangle(center=[0.0, 0.0], base=2.0, alpha=60.0, beta=60.0)
        shape_rounded = Triangle(center=[0.0, 0.0], base=2.0, alpha=60.0, beta=60.0, corner_radius=0.05)
        geom_plain = shape_to_shapely(shape_plain)
        geom_rounded = shape_to_shapely(shape_rounded)
        assert geom_rounded.area < geom_plain.area

    def test_star_area_in_bounds(self):
        shape = Star(center=[0.0, 0.0], n=5, outer_radius=0.5, inner_radius=0.2)
        geom = shape_to_shapely(shape)
        assert not geom.is_empty
        import math as _math
        assert geom.area > 0
        assert geom.area < _math.pi * 0.5 ** 2  # strictly inside circumscribed circle

    def test_star_centroid(self):
        shape = Star(center=[1.0, -2.0], n=5, outer_radius=0.5, inner_radius=0.2)
        geom = shape_to_shapely(shape)
        cx, cy = _centroid(geom)
        assert cx == pytest.approx(1.0, abs=0.01)
        assert cy == pytest.approx(-2.0, abs=0.01)

    def test_star_outer_corner_radius_reduces_area(self):
        plain = Star(center=[0.0, 0.0], n=5, outer_radius=0.5, inner_radius=0.2)
        rounded = Star(center=[0.0, 0.0], n=5, outer_radius=0.5, inner_radius=0.2,
                       outer_corner_radius=0.05)
        assert shape_to_shapely(rounded).area < shape_to_shapely(plain).area

    def test_regular_polygon_square(self):
        shape = RegularPolygon(center=torch.zeros(2), n=4, side_length=torch.tensor(1.0))
        geom = shape_to_shapely(shape)
        assert not geom.is_empty
        assert geom.area == pytest.approx(1.0, abs=0.02)

    def test_regular_polygon_hexagon(self):
        shape = RegularPolygon(center=torch.zeros(2), n=6, side_length=torch.tensor(1.0))
        geom = shape_to_shapely(shape)
        assert not geom.is_empty
        # regular hexagon area = 3√3/2 * s² ≈ 2.598
        assert geom.area == pytest.approx(3 * math.sqrt(3) / 2, abs=0.05)

    def test_cross(self):
        shape = Cross(
            center=torch.zeros(2),
            length=torch.tensor(2.0),
            width=torch.tensor(0.5),
        )
        geom = shape_to_shapely(shape)
        assert not geom.is_empty
        assert geom.area > 0.0

    def test_tshape(self):
        shape = TShape(
            center=torch.zeros(2),
            length=torch.tensor(2.0),
            width=torch.tensor(0.5),
        )
        geom = shape_to_shapely(shape)
        assert not geom.is_empty
        assert geom.area > 0.0

    def test_stripe_x_height(self):
        # axis='x': stripe spans x, bounded in y with the given width
        shape = Stripe(offset=torch.tensor(0.0), width=torch.tensor(1.0), axis='x')
        geom = shape_to_shapely(shape)
        assert not geom.is_empty
        bounds = geom.bounds  # (minx, miny, maxx, maxy)
        height = bounds[3] - bounds[1]
        assert height == pytest.approx(1.0, abs=1e-4)

    def test_stripe_y_width(self):
        # axis='y': stripe spans y, bounded in x with the given width
        shape = Stripe(offset=torch.tensor(0.0), width=torch.tensor(0.8), axis='y')
        geom = shape_to_shapely(shape)
        assert not geom.is_empty
        bounds = geom.bounds  # (minx, miny, maxx, maxy)
        width = bounds[2] - bounds[0]
        assert width == pytest.approx(0.8, abs=1e-4)

    def test_unsupported_shape_raises(self):
        from metashapes.shape.base import Shape

        class _Unknown(Shape):
            def sdf(self, x, y):
                return x
            def bounds(self):
                return (0.0, 0.0), (1.0, 1.0)

        with pytest.raises(TypeError, match="Unsupported shape type"):
            shape_to_shapely(_Unknown())


# ---------------------------------------------------------------------------
# Boolean operations
# ---------------------------------------------------------------------------

class TestBooleansToShapely:
    def _non_overlapping_pair(self):
        r1 = Rectangle(center=[-2.0, 0.0], size=[1.0, 1.0])
        r2 = Rectangle(center=[2.0, 0.0], size=[1.0, 1.0])
        return r1, r2

    def _overlapping_pair(self):
        r1 = Rectangle(center=[0.0, 0.0], size=[2.0, 2.0])
        r2 = Rectangle(center=[1.0, 0.0], size=[2.0, 2.0])
        return r1, r2

    def test_union_non_overlapping_area(self):
        r1, r2 = self._non_overlapping_pair()
        geom = shape_to_shapely(Union(r1, r2))
        assert geom.area == pytest.approx(2.0, abs=1e-4)

    def test_union_non_overlapping_contains_both_centers(self):
        r1, r2 = self._non_overlapping_pair()
        geom = shape_to_shapely(Union(r1, r2))
        from shapely.geometry import Point
        assert geom.contains(Point(-2.0, 0.0))
        assert geom.contains(Point(2.0, 0.0))

    def test_union_overlapping_area(self):
        r1, r2 = self._overlapping_pair()
        geom = shape_to_shapely(Union(r1, r2))
        # r1 area=4, r2 area=4, overlap x∈[0,1]×y∈[-1,1] = 2 → union = 6
        assert geom.area == pytest.approx(6.0, abs=1e-3)

    def test_intersection_overlapping_area(self):
        r1, r2 = self._overlapping_pair()
        geom = shape_to_shapely(Intersection(r1, r2))
        # overlap strip x∈[0,1], y∈[-1,1] → area = 2.0
        assert geom.area == pytest.approx(2.0, abs=1e-3)

    def test_intersection_non_overlapping_is_empty(self):
        r1, r2 = self._non_overlapping_pair()
        geom = shape_to_shapely(Intersection(r1, r2))
        assert geom.is_empty

    def test_difference_area(self):
        r1, r2 = self._overlapping_pair()
        geom = shape_to_shapely(Difference(r1, r2))
        # r1 (4) minus overlap (2) = 2.0
        assert geom.area == pytest.approx(2.0, abs=1e-3)

    def test_difference_excludes_r2_region(self):
        r1, r2 = self._overlapping_pair()
        geom = shape_to_shapely(Difference(r1, r2))
        from shapely.geometry import Point
        # center of r2 at (1, 0) should not be in r1 - r2
        assert not geom.contains(Point(1.5, 0.0))
        # left half of r1 should still be in result
        assert geom.contains(Point(-0.5, 0.0))


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class TestTransformsToShapely:
    def test_translate_moves_centroid(self):
        rect = Rectangle(center=[0.0, 0.0], size=[1.0, 1.0])
        geom = shape_to_shapely(Translate(rect, dx=3.0, dy=1.0))
        cx, cy = _centroid(geom)
        assert cx == pytest.approx(3.0, abs=1e-4)
        assert cy == pytest.approx(1.0, abs=1e-4)

    def test_translate_preserves_area(self):
        rect = Rectangle(center=[0.0, 0.0], size=[2.0, 1.5])
        orig = shape_to_shapely(rect)
        translated = shape_to_shapely(Translate(rect, dx=5.0, dy=-3.0))
        assert translated.area == pytest.approx(orig.area, abs=1e-4)

    def test_rotate_preserves_area(self):
        rect = Rectangle(center=[0.0, 0.0], size=[2.0, 1.0])
        orig = shape_to_shapely(rect)
        rotated = shape_to_shapely(Rotate(rect, angle=45.0))
        assert rotated.area == pytest.approx(orig.area, abs=1e-3)

    def test_rotate_moves_centroid(self):
        # Rectangle centered at (1, 0); rotating 90° around origin moves centroid to (0, 1)
        rect = Rectangle(center=[1.0, 0.0], size=[0.5, 0.5])
        geom = shape_to_shapely(Rotate(rect, angle=90.0))
        cx, cy = _centroid(geom)
        assert cx == pytest.approx(0.0, abs=0.01)
        assert cy == pytest.approx(1.0, abs=0.01)

    def test_scale_area(self):
        rect = Rectangle(center=[0.0, 0.0], size=[1.0, 1.0])
        orig = shape_to_shapely(rect)
        scaled = shape_to_shapely(Scale(rect, s=2.0))
        # area scales by s² = 4
        assert scaled.area == pytest.approx(orig.area * 4.0, abs=1e-3)

    def test_scale_centroid_unchanged_at_origin(self):
        rect = Rectangle(center=[0.0, 0.0], size=[1.0, 1.0])
        scaled = shape_to_shapely(Scale(rect, s=3.0))
        cx, cy = _centroid(scaled)
        assert cx == pytest.approx(0.0, abs=1e-4)
        assert cy == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Compound / composed shapes
# ---------------------------------------------------------------------------

class TestCompoundShapesToShapely:
    def test_translated_union(self):
        rect1 = Rectangle(center=[0.0, 0.0], size=[1.0, 1.0])
        rect2 = Rectangle(center=[0.0, 0.0], size=[1.0, 1.0])
        shape = Union(Translate(rect1, dx=0.5, dy=0.0), rect2)
        geom = shape_to_shapely(shape)
        assert not geom.is_empty
        assert geom.area > 0.0

    def test_intersection_of_ellipses(self):
        e1 = Ellipse(center=[0.0, 0.0], axes=torch.tensor([2.0, 2.0]))
        e2 = Ellipse(center=[0.5, 0.0], axes=torch.tensor([2.0, 2.0]))
        geom = shape_to_shapely(Intersection(e1, e2))
        assert not geom.is_empty
        assert geom.area > 0.0

    def test_difference_rect_minus_ellipse(self):
        rect = Rectangle(center=[0.0, 0.0], size=[3.0, 3.0])
        ellipse = Ellipse(center=[0.0, 0.0], axes=torch.tensor([2.0, 2.0]))
        rect_geom = shape_to_shapely(rect)
        geom = shape_to_shapely(Difference(rect, ellipse))
        assert not geom.is_empty
        assert geom.area < rect_geom.area

    def test_scale_then_translate(self):
        rect = Rectangle(center=[0.0, 0.0], size=[1.0, 1.0])
        shape = Translate(Scale(rect, s=2.0), dx=5.0, dy=0.0)
        geom = shape_to_shapely(shape)
        cx, cy = _centroid(geom)
        assert cx == pytest.approx(5.0, abs=1e-3)
        assert geom.area == pytest.approx(4.0, abs=1e-3)

    def test_nested_booleans(self):
        r1 = Rectangle(center=[0.0, 0.0], size=[2.0, 2.0])
        r2 = Rectangle(center=[0.5, 0.0], size=[2.0, 2.0])
        r3 = Rectangle(center=[0.0, 0.0], size=[0.5, 0.5])
        shape = Difference(Union(r1, r2), r3)
        geom = shape_to_shapely(shape)
        assert not geom.is_empty
        assert geom.area > 0.0
