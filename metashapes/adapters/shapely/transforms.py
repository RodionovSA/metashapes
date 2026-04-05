# metashapes/adapters/shapely/transforms.py
# This module provides functions to convert transformation ops and Shapely geometries. 

from __future__ import annotations

from shapely.affinity import rotate as shp_rotate
from shapely.affinity import scale as shp_scale
from shapely.affinity import translate as shp_translate
from shapely.geometry.base import BaseGeometry

from metashapes.shape.transforms import Translate, Rotate, Scale

def translate_to_shapely(shape: Translate, shape_to_shapely_fn) -> BaseGeometry:
    geom = shape_to_shapely_fn(shape.shape)
    return shp_translate(geom, xoff=shape.dx, yoff=shape.dy)

def rotate_to_shapely(shape: Rotate, shape_to_shapely_fn) -> BaseGeometry:
    geom = shape_to_shapely_fn(shape.shape)
    return shp_rotate(geom, shape.angle, origin=shape.origin)

def scale_to_shapely(shape: Scale, shape_to_shapely_fn) -> BaseGeometry:
    geom = shape_to_shapely_fn(shape.shape)
    return shp_scale(geom, xfact=shape.s, yfact=shape.s, origin=shape.origin)