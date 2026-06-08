# metashapes/adapters/shapely/polygons.py
# This file contains functions to convert Metashapes polygons to Shapely geometries.

from __future__ import annotations

import math
import numpy as np
from shapely.affinity import rotate as shp_rotate
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

import src.metashapes.shape.primitives as prim
from .helpers import as_float, as_list, round_corners


def triangle_to_shapely(shape: prim.Triangle) -> BaseGeometry:
    cx, cy = as_list(shape.center)
    angle = as_float(shape.angle)
    rr = as_float(shape.corner_radius)

    (Ax, Ay), (Bx, By), (Cx, Cy) = shape._vertices()

    theta = np.deg2rad(angle)
    c_t, s_t = np.cos(theta), np.sin(theta)

    def _to_world(vx, vy):
        vx, vy = as_float(vx), as_float(vy)
        return (cx + vx * c_t - vy * s_t, cy + vx * s_t + vy * c_t)

    pts = [_to_world(Ax, Ay), _to_world(Bx, By), _to_world(Cx, Cy)]
    geom = Polygon(pts)
    if rr > 0:
        return round_corners(geom, radius=rr, mode="inner")
    return geom


def star_to_shapely(shape: prim.Star) -> BaseGeometry:
    cx, cy = as_list(shape.center)
    R = as_float(shape.outer_radius)
    r = as_float(shape.inner_radius)
    angle_deg = as_float(shape.angle)
    ocr = as_float(shape.outer_corner_radius)
    icr = as_float(shape.inner_corner_radius)
    n = shape.n
    an = math.pi / n

    pts = []
    for k in range(n):
        tip_ang = math.radians(angle_deg) + math.pi / 2 + 2 * math.pi * k / n
        pts.append((cx + R * math.cos(tip_ang), cy + R * math.sin(tip_ang)))
        val_ang = tip_ang + an
        pts.append((cx + r * math.cos(val_ang), cy + r * math.sin(val_ang)))

    geom = Polygon(pts)
    if ocr > 0:
        geom = round_corners(geom, radius=ocr, mode="inner")   # opening: rounds convex tips
    if icr > 0:
        geom = round_corners(geom, radius=icr, mode="outer")   # closing: rounds concave valleys
    return geom


def regular_polygon_to_shapely(shape: prim.RegularPolygon) -> BaseGeometry:
    cx, cy = as_list(shape.center)
    n = shape.n
    s = as_float(shape.side_length)
    angle = as_float(shape.angle)
    rr = as_float(shape.corner_radius)

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
