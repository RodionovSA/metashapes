# metashapes/adapters/shapely/booleans.py
# This module implements boolean operations (union, intersection, difference) for Shapely geometries.

from __future__ import annotations

from shapely.geometry.base import BaseGeometry

from metashapes.shape.boolean import Union, Intersection, Difference

def union_to_shapely(shape: Union, shape_to_shapely_fn) -> BaseGeometry:
    return shape_to_shapely_fn(shape.left).union(shape_to_shapely_fn(shape.right))

def intersection_to_shapely(shape: Intersection, shape_to_shapely_fn) -> BaseGeometry:
    return shape_to_shapely_fn(shape.left).intersection(shape_to_shapely_fn(shape.right))

def difference_to_shapely(shape: Difference, shape_to_shapely_fn) -> BaseGeometry:
    return shape_to_shapely_fn(shape.left).difference(shape_to_shapely_fn(shape.right))