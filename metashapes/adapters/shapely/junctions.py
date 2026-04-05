# metashapes/adapters/shapely/junctions.py
# This file contains functions for converting junction shapes to Shapely geometries.

from __future__ import annotations

from shapely.affinity import rotate as shp_rotate
from shapely.affinity import translate as shp_translate
from shapely.geometry import Point, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

import metashapes.shape.primitives as prim

def _rounded_box(hx: float, hy: float, rr: float) -> BaseGeometry:
        if rr == 0:
            return box(-hx, -hy, hx, hy)
        return box(-(hx - rr), -(hy - rr), hx - rr, hy - rr).buffer(rr, join_style=1)

def cross_to_shapely(shape: prim.Cross) -> BaseGeometry:
    cx, cy = shape.center
    length = shape.length
    width = shape.width
    ro = shape.outer_corner_radius
    ri = shape.inner_corner_radius

    if length <= 0:
        raise ValueError("length must be positive")
    if width <= 0:
        raise ValueError("width must be positive")
    if width > length:
        raise ValueError("width must be less than or equal to length")
    if ro < 0:
        raise ValueError("outer_corner_radius must be non-negative")
    if ri < 0:
        raise ValueError("inner_corner_radius must be non-negative")

    bx = 0.5 * length
    by = 0.5 * width

    if ro >= by:
        raise ValueError("outer_corner_radius is too large")
    if ri > (bx - by - ro):
        raise ValueError("inner_corner_radius is too large")

    # base cross with outer rounding
    hrect = _rounded_box(bx, by, ro)
    vrect = _rounded_box(by, bx, ro)
    geom = unary_union([hrect, vrect])

    # inner rounding: add square \ circle patch in each notch
    if ri > 0:
        patches = []

        # (+,+)
        sq = box(by, by, by + ri, by + ri)
        cc = Point(by + ri, by + ri).buffer(ri)
        patches.append(sq.difference(cc))

        # (-,+)
        sq = box(-by - ri, by, -by, by + ri)
        cc = Point(-by - ri, by + ri).buffer(ri)
        patches.append(sq.difference(cc))

        # (-,-)
        sq = box(-by - ri, -by - ri, -by, -by)
        cc = Point(-by - ri, -by - ri).buffer(ri)
        patches.append(sq.difference(cc))

        # (+,-)
        sq = box(by, -by - ri, by + ri, -by)
        cc = Point(by + ri, -by - ri).buffer(ri)
        patches.append(sq.difference(cc))

        geom = unary_union([geom, *patches])

    if shape.angle != 0:
        geom = shp_rotate(geom, shape.angle, origin=(0.0, 0.0))

    geom = shp_translate(geom, xoff=cx, yoff=cy)
    return geom

def tshape_to_shapely(shape: prim.TShape) -> BaseGeometry:
    cx, cy = shape.center
    length = shape.length
    width = shape.width
    ro = shape.outer_corner_radius
    ri = shape.inner_corner_radius

    if length <= 0:
        raise ValueError("length must be positive")
    if width <= 0:
        raise ValueError("width must be positive")
    if width > length:
        raise ValueError("width must be less than or equal to length")
    if ro < 0:
        raise ValueError("outer_corner_radius must be non-negative")
    if ri < 0:
        raise ValueError("inner_corner_radius must be non-negative")

    bx = 0.5 * length
    by = 0.5 * width

    if ro >= by:
        raise ValueError("outer_corner_radius is too large")
    if ri > (bx - by - ro):
        raise ValueError("inner_corner_radius is too large")

    top = _rounded_box(bx, by, ro)
    top = shp_translate(top, xoff=0.0, yoff=bx - by)

    stem = _rounded_box(by, bx, ro)

    # keep only upper strip of the top box
    y_cut = bx - 2.0 * by
    top = top.intersection(box(-bx, y_cut, bx, bx))

    geom = unary_union([top, stem])

    if ri > 0:
        patches = [
            box(by, y_cut - ri, by + ri, y_cut).difference(
                Point(by + ri, y_cut - ri).buffer(ri)
            ),
            box(-by - ri, y_cut - ri, -by, y_cut).difference(
                Point(-by - ri, y_cut - ri).buffer(ri)
            ),
        ]
        geom = unary_union([geom, *patches])

    if shape.angle != 0:
        geom = shp_rotate(geom, shape.angle, origin=(0.0, 0.0))

    geom = shp_translate(geom, xoff=cx, yoff=cy)
    return geom