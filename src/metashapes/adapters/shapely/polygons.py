# metashapes/adapters/shapely/polygons.py
# This file contains functions to convert Metashapes polygons to Shapely geometries.

from __future__ import annotations

import math
import numpy as np
from shapely.affinity import rotate as shp_rotate
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

import metashapes.shape.primitives as prim
from .helpers import arc_quad_segs, as_float, as_list, round_corners


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
    """Star outline built directly: straight edges plus exact fillet arcs.

    Building it with buffers (opening then closing) is ill-conditioned when
    the tip and valley roundings nearly meet on an edge: the closing then
    intersects two almost-parallel offset curves and amplifies the
    opening's chord error. Here every vertex lies exactly on the outline.

    Tips are convex corners rounded by ``outer_corner_radius``. Valleys are
    concave corners rounded by ``inner_corner_radius``, unless
    ``r >= R cos(pi/n)``: then they are convex and, matching the SDF, are
    rounded by ``outer_corner_radius`` instead.
    """
    cx, cy = as_list(shape.center)
    R = as_float(shape.outer_radius)
    r = as_float(shape.inner_radius)
    angle_deg = as_float(shape.angle)
    ocr = as_float(shape.outer_corner_radius)
    icr = as_float(shape.inner_corner_radius)
    n = shape.n
    an = math.pi / n

    corners = []
    for k in range(n):
        tip_ang = math.radians(angle_deg) + math.pi / 2 + 2 * math.pi * k / n
        corners.append((R * math.cos(tip_ang), R * math.sin(tip_ang)))
        val_ang = tip_ang + an
        corners.append((r * math.cos(val_ang), r * math.sin(val_ang)))
    valley_convex = r >= R * math.cos(an)
    radii = [ocr, ocr if valley_convex else icr] * n

    pts = _rounded_polygon(corners, radii)
    return Polygon([(cx + x, cy + y) for x, y in pts])


def _rounded_polygon(corners, radii) -> list[tuple[float, float]]:
    """Outline of a simple CCW polygon with each corner filleted by its own
    radius (convex corners inward, reflex corners outward). Fillets must
    not overlap along an edge (the shapes' own parameter bounds ensure it).
    """
    m = len(corners)
    out: list[tuple[float, float]] = []
    for i in range(m):
        px, py = corners[i - 1]
        qx, qy = corners[i]
        nx, ny = corners[(i + 1) % m]
        rho = radii[i]
        if rho <= 0:
            out.append((qx, qy))
            continue
        # unit directions from the corner towards its neighbours
        ax, ay = px - qx, py - qy
        bx, by = nx - qx, ny - qy
        la, lb = math.hypot(ax, ay), math.hypot(bx, by)
        ax, ay, bx, by = ax / la, ay / la, bx / lb, by / lb
        half = 0.5 * math.acos(max(-1.0, min(1.0, ax * bx + ay * by)))  # half the corner angle
        t = rho / math.tan(half)                      # corner -> tangent point
        d = rho / math.sin(half)                      # corner -> arc centre
        mx, my = ax + bx, ay + by                     # bisector, pointing into the corner's angle
        lm = math.hypot(mx, my)
        mx, my = mx / lm, my / lm
        # convex (CCW turn left) -> centre inside, along the bisector;
        # reflex -> the bisector points outside, centre is there too
        ox, oy = qx + d * mx, qy + d * my
        t1 = (qx + t * ax, qy + t * ay)
        t2 = (qx + t * bx, qy + t * by)
        a1 = math.atan2(t1[1] - oy, t1[0] - ox)
        a2 = math.atan2(t2[1] - oy, t2[0] - ox)
        sweep = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi   # |sweep| = pi - corner angle
        quad = arc_quad_segs(rho, 32, fillet=False)
        k = max(1, math.ceil(abs(sweep) / (math.pi / 2 / quad)))
        for j in range(k + 1):
            a = a1 + sweep * j / k
            out.append((ox + rho * math.cos(a), oy + rho * math.sin(a)))
    return out


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
