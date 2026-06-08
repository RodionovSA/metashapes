# metashapes/adapters/shapely/quads.py
# This file contains the adapter function to convert Quad shapes to Shapely geometries.

from __future__ import annotations

import numpy as np
from shapely.affinity import rotate as shp_rotate
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient

import src.metashapes.shape.primitives as prim
from .helpers import as_float, as_list, round_corners

def convex_quad_to_shapely(shape: prim.ConvexQuad) -> BaseGeometry:
    cx, cy = as_list(shape.center)
    ux0, uy0 = as_list(shape.u)
    vx0, vy0 = as_list(shape.v)
    alpha = as_float(shape.alpha)
    beta = as_float(shape.beta)
    angle = as_float(shape.angle)
    rr = as_float(shape.corner_radius)

    if rr < 0:
        raise ValueError("corner_radius must be non-negative")

    theta = np.deg2rad(angle)
    ct = np.cos(theta)
    st = np.sin(theta)

    ux = ct * ux0 - st * uy0
    uy = st * ux0 + ct * uy0
    vx = ct * vx0 - st * vy0
    vy = st * vx0 + ct * vy0

    uv_cross = ux * vy - uy * vx
    if abs(uv_cross) <= 1e-12:
        raise ValueError("u and v must not be collinear")

    points = [
        (cx - ux - vx, cy - uy - vy),
        (cx + ux - vx, cy + uy - vy),
        (cx + (1.0 + alpha) * ux + (1.0 + beta) * vx,
         cy + (1.0 + alpha) * uy + (1.0 + beta) * vy),
        (cx - (1.0 + alpha) * ux + (1.0 + beta) * vx,
         cy - (1.0 + alpha) * uy + (1.0 + beta) * vy),
    ]

    geom = Polygon(points)

    if not geom.is_valid or geom.area <= 0:
        raise ValueError("Degenerate quadrilateral")

    geom = orient(geom, sign=1.0)

    if rr == 0:
        return geom

    return round_corners(geom, radius=rr, mode="inner")

def rectangle_to_shapely(shape: prim.Rectangle):
    cx, cy = as_list(shape.center)
    w, h = as_list(shape.size)
    r = as_float(shape.corner_radius)
    angle = as_float(shape.angle)

    if w <= 0 or h <= 0:
        raise ValueError("Rectangle size components must be positive")
    if r < 0:
        raise ValueError("corner_radius must be non-negative")

    r = min(r, 0.5 * min(w, h))

    if r == 0:
        geom = box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    else:
        geom = box(
            cx - (w / 2 - r),
            cy - (h / 2 - r),
            cx + (w / 2 - r),
            cy + (h / 2 - r),
        ).buffer(r)

    if angle != 0:
        geom = shp_rotate(geom, angle, origin=(cx, cy))

    return geom


def isosceles_trapezoid_to_shapely(shape: prim.IsoscelesTrapezoid) -> BaseGeometry:
    cx, cy = as_list(shape.center)
    wb = as_float(shape.bottom_width)
    wt = as_float(shape.top_width)
    h = as_float(shape.height)
    angle = as_float(shape.angle)
    rr = as_float(shape.corner_radius)

    r1 = 0.5 * wb
    r2 = 0.5 * wt
    he = 0.5 * h

    points = [
        (cx - r1, cy - he),
        (cx + r1, cy - he),
        (cx + r2, cy + he),
        (cx - r2, cy + he),
    ]

    geom = Polygon(points)

    if angle != 0:
        geom = shp_rotate(geom, angle, origin=(cx, cy))

    if rr == 0:
        return geom

    return round_corners(geom, radius=rr, mode="inner")
