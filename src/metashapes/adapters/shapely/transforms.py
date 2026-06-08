# metashapes/adapters/shapely/transforms.py
# This module provides functions to convert transformation ops and Shapely geometries.

from __future__ import annotations

from shapely.affinity import rotate as shp_rotate
from shapely.affinity import scale as shp_scale
from shapely.affinity import translate as shp_translate
from shapely.geometry.base import BaseGeometry

from metashapes.shape.transforms import Translate, Rotate, Scale
from .helpers import as_float, as_list

def translate_to_shapely(shape: Translate, shape_to_shapely_fn) -> BaseGeometry:
    geom = shape_to_shapely_fn(shape.shape)
    return shp_translate(geom, xoff=as_float(shape.dx), yoff=as_float(shape.dy))

def rotate_to_shapely(shape: Rotate, shape_to_shapely_fn) -> BaseGeometry:
    geom = shape_to_shapely_fn(shape.shape)
    ox, oy = as_list(shape.origin)
    return shp_rotate(geom, as_float(shape.angle), origin=(ox, oy))

def scale_to_shapely(shape: Scale, shape_to_shapely_fn) -> BaseGeometry:
    geom = shape_to_shapely_fn(shape.shape)
    s = as_float(shape.s)
    ox, oy = as_list(shape.origin)
    return shp_scale(geom, xfact=s, yfact=s, origin=(ox, oy))
