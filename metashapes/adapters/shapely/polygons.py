# metashapes/adapters/shapely/polygons.py
# This file contains functions to convert Metashapes polygons to Shapely geometries.

from __future__ import annotations

import math
import numpy as np
from shapely.affinity import rotate as shp_rotate
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

import metashapes.shape.primitives as prim
from .helpers import round_corners


def triangle_to_shapely(shape: prim.Triangle) -> BaseGeometry:
    cx, cy = shape.center.tolist()
    angle = shape.angle.item()
    rr = shape.corner_radius.item()

    (Ax, Ay), (Bx, By), (Cx, Cy) = shape._vertices()

    theta = np.deg2rad(angle)
    c_t, s_t = np.cos(theta), np.sin(theta)

    def _to_world(vx, vy):
        vx, vy = vx.item(), vy.item()
        return (cx + vx * c_t - vy * s_t, cy + vx * s_t + vy * c_t)

    pts = [_to_world(Ax, Ay), _to_world(Bx, By), _to_world(Cx, Cy)]
    geom = Polygon(pts)
    if rr > 0:
        return round_corners(geom, radius=rr, mode="inner")
    return geom


def star_to_shapely(shape: prim.Star) -> BaseGeometry:
    cx, cy = shape.center.tolist()
    R = shape.outer_radius.item()
    r = shape.inner_radius.item()
    angle_deg = shape.angle.item()
    ocr = shape.outer_corner_radius.item()
    icr = shape.inner_corner_radius.item()
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
