# metashapes/adapters/shapely/conics.py
# This file contains the adapter function to convert Conic shapes to Shapely geometries.

from __future__ import annotations

from shapely.affinity import rotate as shp_rotate
from shapely.affinity import scale as shp_scale
from shapely.affinity import translate as shp_translate
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

import metashapes.shape.primitives as prim

def ellipse_to_shapely(shape: prim.Ellipse) -> BaseGeometry:
    cx, cy = shape.center
    a, b = shape.axes

    geom = Point(0.0, 0.0).buffer(1.0, quad_segs=64 // 4)
    geom = shp_scale(geom, xfact=a*0.5, yfact=b*0.5, origin=(0.0, 0.0))
    geom = shp_rotate(geom, shape.angle, origin=(0.0, 0.0))
    geom = shp_translate(geom, xoff=cx, yoff=cy)
    return geom
