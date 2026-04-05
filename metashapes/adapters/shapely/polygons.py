# metashapes/adapters/shapely/polygons.py
# This file contains functions to convert Metashapes polygons to Shapely geometries.

from __future__ import annotations

import numpy as np
from shapely.affinity import rotate as shp_rotate
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

import metashapes.shape.primitives as prim
from .helpers import round_corners

def regular_polygon_to_shapely(shape: prim.RegularPolygon) -> BaseGeometry:
    cx, cy = shape.center
    n = shape.n
    s = shape.side_length

    R = s / (2.0 * np.sin(np.pi / n))
    phi = np.pi / 2.0

    points = [
        (
            cx + R * np.cos(2.0 * np.pi * k / n + phi),
            cy + R * np.sin(2.0 * np.pi * k / n + phi),
        )
        for k in range(n)
    ]

    geom = Polygon(points)
    if shape.angle != 0:
        geom = shp_rotate(geom, shape.angle, origin=shape.center)
        
    if shape.corner_radius == 0:
        return geom
    return round_corners(geom, radius=shape.corner_radius, mode="inner")
