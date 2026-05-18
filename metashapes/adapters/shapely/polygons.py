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
    cx, cy = shape.center.tolist()
    n = shape.n
    s = shape.side_length.item()
    angle = shape.angle.item()
    rr = shape.corner_radius.item()

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
    if angle != 0:
        geom = shp_rotate(geom, angle, origin=(cx, cy))

    if rr == 0:
        return geom
    return round_corners(geom, radius=rr, mode="inner")
