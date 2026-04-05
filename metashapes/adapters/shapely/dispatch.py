# metashapes/adapters/shapely/dispatch.py
# This module defines the main dispatch function for converting Metashapes Shape objects into Shapely geometries.

from __future__ import annotations

from shapely.geometry.base import BaseGeometry

import metashapes.shape.primitives as prim
from metashapes.shape.base import Shape
from metashapes.shape.boolean import Union, Intersection, Difference
from metashapes.shape.transforms import Translate, Rotate, Scale

from .quads import (
    convex_quad_to_shapely,
    rectangle_to_shapely,
    isosceles_trapezoid_to_shapely,
)
from .polygons import regular_polygon_to_shapely
from .conics import ellipse_to_shapely
from .junctions import cross_to_shapely, tshape_to_shapely
from .booleans import (
    union_to_shapely,
    intersection_to_shapely,
    difference_to_shapely,
)
from .transforms import (
    translate_to_shapely,
    rotate_to_shapely,
    scale_to_shapely,
)

def shape_to_shapely(shape: Shape) -> BaseGeometry:
    if isinstance(shape, prim.ConvexQuad):
        return convex_quad_to_shapely(shape)

    if isinstance(shape, prim.Rectangle):
        return rectangle_to_shapely(shape)

    if isinstance(shape, prim.Ellipse):
        return ellipse_to_shapely(shape)

    if isinstance(shape, prim.RegularPolygon):
        return regular_polygon_to_shapely(shape)

    if isinstance(shape, prim.IsoscelesTrapezoid):
        return isosceles_trapezoid_to_shapely(shape)

    if isinstance(shape, prim.Cross):
        return cross_to_shapely(shape)

    if isinstance(shape, prim.TShape):
        return tshape_to_shapely(shape)

    if isinstance(shape, Union):
        return union_to_shapely(shape, shape_to_shapely)

    if isinstance(shape, Intersection):
        return intersection_to_shapely(shape, shape_to_shapely)

    if isinstance(shape, Difference):
        return difference_to_shapely(shape, shape_to_shapely)

    if isinstance(shape, Translate):
        return translate_to_shapely(shape, shape_to_shapely)

    if isinstance(shape, Rotate):
        return rotate_to_shapely(shape, shape_to_shapely)

    if isinstance(shape, Scale):
        return scale_to_shapely(shape, shape_to_shapely)

    raise TypeError(f"Unsupported shape type: {type(shape).__name__}")