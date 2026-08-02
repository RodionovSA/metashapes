# sdflib/quads.py
# SDF functions for quadrilaterals

import torch
from typing import Callable

from .polygons import _inset_convex, _PolygonSDF

def RectangleSDF(hx, hy, r) -> Callable:
    """ 
    SDF for a rectangle. 
    
    Parameters: 
        hx: Half of the rectangle size along x-axis in local coordinates
        hy: Half of the rectangle size along y-axis in local coordinates
        r: Corners radius
    """
    def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            qx = torch.abs(x) - (hx - r)
            qy = torch.abs(y) - (hy - r)
            s = torch.clamp_min(qx, 0.0)**2 + torch.clamp_min(qy, 0.0)**2
            outside = torch.sqrt(s.clamp_min(torch.finfo(s.dtype).tiny))
            inside = torch.clamp_max(torch.maximum(qx, qy), 0.0)
            return outside + inside - r
    return f

def IsoscelesTrapezoidSDF(hb, ht, hy, r) -> Callable:
    """
    SDF for an isosceles trapezoid, symmetric about the y-axis.

    The bottom base (width 2*hb) sits at y = -hy, the top base (width 2*ht)
    at y = +hy.

    Parameters:
        hb: Half of the bottom base width in local coordinates
        ht: Half of the top base width in local coordinates
        hy: Half of the trapezoid height in local coordinates
        r:  Corners radius

    Precondition: r must be small enough to keep the inset trapezoid
    non-degenerate, i.e. r < min(hb/(q - slope), ht/(q + slope), hy). Larger
    values silently return the distance to a collapsed shape.
    """
    # Inset every edge by r, then round by r, so the outer support lines stay
    # exactly where the caller put them (same convention as RectangleSDF).
    # The bases just move inward by r; the two slanted sides move along their
    # own normals, which is what the q -/+ slope terms account for.
    slope = (ht - hb) / (2.0 * hy)       # dx/dy of the right-hand side
    q = (1.0 + slope * slope) ** 0.5     # 1/cos of that side's tilt
    b = hb - r * (q - slope)             # inset half-width at the bottom
    t = ht - r * (q + slope)             # inset half-width at the top
    e = hy - r                           # inset half-height

    def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        px = torch.abs(x)                # mirror: solve the right half only

        # ca: offset to the nearest point on the near horizontal base,
        # clipped to that base's width (so it also covers the two corners).
        lim = torch.where(y < 0.0, b, t)
        cax = px - torch.minimum(px, lim)
        cay = torch.abs(y) - e

        # cb: offset to the nearest point on the slanted side, by projecting
        # onto the segment (t, e) -> (b, -e) and clamping to its extent.
        ex = t - b
        ey = 2.0 * e
        u = torch.clamp(((t - px) * ex + (e - y) * ey) / (ex * ex + ey * ey), 0.0, 1.0)
        cbx = px - t + ex * u
        cby = y - e + ey * u

        # Nearer of the two is the true distance; inside iff left of the
        # slanted side and between the two bases.
        d2 = torch.minimum(cax * cax + cay * cay, cbx * cbx + cby * cby)
        d = torch.sqrt(d2.clamp_min(torch.finfo(d2.dtype).tiny))
        return torch.where((cbx < 0.0) & (cay < 0.0), -d, d) - r
    return f

def ConvexQuadSDF(u, v, alpha, beta, r) -> Callable:
    """
    SDF for a convex quadrilateral.

    Corners sit at +-u +-v, with the far corner pushed out by alpha along u
    and beta along v, so the shape ranges from a parallelogram (alpha = beta
    = 0) to a general convex quad.

    Parameters:
        u:     first frame vector (half-diagonal direction).
        v:     second frame vector (half-diagonal direction); must not be collinear with u.
        alpha: stretch of the far corner along u (default 0).
        beta:  stretch of the far corner along v (default 0).
        r: Corners radius

    Precondition: u and v must not be collinear, and r must be small enough
    to keep the inset quad non-degenerate -- for every edge, r < edge_len /
    (1/tan(a/2) + 1/tan(b/2)) with a, b its two corner angles. Larger values
    silently return the distance to a collapsed shape.
    """
    ux, uy = u[0], u[1]
    vx, vy = v[0], v[1]
    au = 1.0 + alpha
    bv = 1.0 + beta

    corners = [
        (-ux - vx,           -uy - vy),
        ( ux - vx,            uy - vy),
        ( au * ux + bv * vx,  au * uy + bv * vy),
        (-au * ux + bv * vx, -au * uy + bv * vy),
    ]

    # Inset every edge by r, then round by r, so the outer support lines stay
    # exactly where the caller put them (same convention as RectangleSDF).
    verts = _inset_convex(corners, r)
    return _PolygonSDF(verts, r)