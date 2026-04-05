# metashapes/adapters/shapely/helpers.py
# Helper functions for Shapely adapter.

from __future__ import annotations
from typing import Literal

from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry


RoundMode = Literal["inner", "outer", "both"]

def round_corners(
    geom: BaseGeometry,
    radius: float,
    *,
    mode: RoundMode = "both",
    quad_segs: int = 32,
) -> BaseGeometry:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if quad_segs < 1:
        raise ValueError("quad_segs must be at least 1")
    if radius == 0:
        return geom

    if mode in {"outer", "both"}:
        geom = geom.buffer(radius, quad_segs=quad_segs)
        geom = geom.buffer(-radius, quad_segs=quad_segs)

    if mode in {"inner", "both"}:
        geom = geom.buffer(-radius, quad_segs=quad_segs)
        geom = geom.buffer(radius, quad_segs=quad_segs)

    return geom

def remove_holes(geom):
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    if geom.geom_type == "MultiPolygon":
        largest = max(geom.geoms, key=lambda g: g.area)
        return Polygon(largest.exterior)
    return geom