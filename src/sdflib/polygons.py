# sdflib/polygons.py
# SDF functions for general polygons

import math
import torch
from typing import Callable


def _orientation(verts):
    """+1 if verts winds CCW, -1 if CW."""
    n = len(verts)
    area2 = 0.0
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        area2 = area2 + x1 * y2 - x2 * y1
    return area2 / abs(area2)


def _inset_convex(verts, r):
    """Move every edge of a convex polygon inward by r; each inset corner is
    where the two adjacent offset edge-lines cross, so r=0 reproduces verts
    exactly."""
    n = len(verts)
    orient = _orientation(verts)

    offset = []
    for i in range(n):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        elen = torch.sqrt((ex * ex + ey * ey).clamp_min(torch.finfo(ex.dtype).tiny))
        nx, ny = -orient * ey / elen, orient * ex / elen   # inward unit normal
        offset.append((ax + r * nx, ay + r * ny, ex, ey))

    out = []
    for i in range(n):
        px, py, dx, dy = offset[i - 1]
        qx, qy, gx, gy = offset[i]
        t = ((qx - px) * gy - (qy - py) * gx) / (dx * gy - dy * gx)
        out.append((px + t * dx, py + t * dy))
    return out


def _PolygonSDF(verts, r) -> Callable:
    """SDF of the convex polygon `verts`, already inset/rounded by r."""
    n = len(verts)
    orient = _orientation(verts)

    def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d2 = None
        inside = torch.ones_like(x, dtype=torch.bool)

        for i in range(n):
            ax, ay = verts[i]
            bx, by = verts[(i + 1) % n]
            ex, ey = bx - ax, by - ay
            wx, wy = x - ax, y - ay

            t = torch.clamp((wx * ex + wy * ey) / (ex * ex + ey * ey), 0.0, 1.0)
            sx = wx - t * ex
            sy = wy - t * ey
            e2 = sx * sx + sy * sy
            d2 = e2 if d2 is None else torch.minimum(d2, e2)

            inside = inside & (orient * (ex * wy - ey * wx) >= 0)

        d = torch.sqrt(d2.clamp_min(torch.finfo(d2.dtype).tiny))
        return torch.where(inside, -d, d) - r
    return f


def RegularPolygonSDF(n: int, R, r) -> Callable:
    """SDF for a regular n-gon centered at the origin, first vertex at +y.

    R is the circumradius.
    """
    an = math.pi / n
    rho_in = R * math.cos(an) - r
    R_in = rho_in / math.cos(an)

    phi0 = math.pi / 2.0
    verts = []
    for k in range(n):
        th = 2.0 * math.pi * k / n + phi0
        verts.append((R_in * math.cos(th), R_in * math.sin(th)))
    return _PolygonSDF(verts, r)


def TriangleSDF(A, B, C, r) -> Callable:
    """SDF for the triangle with vertices A, B, C (CCW), rounded by r."""
    verts = _inset_convex([A, B, C], r)
    return _PolygonSDF(verts, r)


def StarSDF(n: int, R, r, r_out, r_in) -> Callable:
    """
    SDF for a regular n-pointed star centered at the origin, first tip at +y.

    R, r: outer/inner radii. r_out, r_in: tip/valley rounding radii.

    Precondition: 0 < r < R, r_out < L*tan(alpha), r_in < L*tan(beta), and
    r_out/tan(alpha) + r_in/tan(beta) < L, where L is the tip-valley edge
    length and alpha/beta its two corner angles. Larger values silently
    return the distance to a collapsed/self-overlapping shape.
    """
    an = math.pi / n  # half-sector angle at the origin

    def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Fold into canonical half-sector with outer tip A on the sector's
        # bisector, valley vertex B at angle an. Reflect across the
        # bisector so p_y >= 0.
        r2 = x * x + y * y
        r_point = torch.sqrt(r2.clamp_min(torch.finfo(r2.dtype).tiny))
        theta = torch.atan2(x, y)  # first tip at +y in local frame
        bn = torch.remainder(theta + an, 2.0 * an) - an
        p_x = r_point * torch.cos(bn)
        p_y = r_point * torch.abs(torch.sin(bn))

        Bx = r * math.cos(an)
        By = r * math.sin(an)
        L2 = R * R + r * r - 2.0 * R * r * math.cos(an)
        L = torch.sqrt(L2.clamp_min(torch.finfo(L2.dtype).tiny))

        # ---- Outer tip rounding ----
        denom_outer = (r * math.sin(an)).clamp_min(1e-9)
        shift_A = r_out * L / denom_outer
        A_eff_x = R - shift_A
        A_eff_y = torch.zeros_like(shift_A)

        # B_eff: B shifted outward along the bisector through B by
        # r_out/sin(beta). sin_beta, bis_x/bis_y and OB are the
        # half-angle-at-the-valley geometry shared by both the outer-tip
        # and inner-valley rounding below.
        sin_beta = (R * math.sin(an) / L).clamp(min=1e-7, max=1.0 - 1e-7)
        bis_x = math.cos(an)
        bis_y = math.sin(an)
        OB2 = Bx * Bx + By * By
        OB = torch.sqrt(OB2.clamp_min(torch.finfo(OB2.dtype).tiny))
        OB_eff = OB - r_out / sin_beta
        B_eff_x = OB_eff * bis_x
        B_eff_y = OB_eff * bis_y

        # Signed distance to segment A_eff -> B_eff in the half-sector.
        ex = B_eff_x - A_eff_x
        ey = B_eff_y - A_eff_y
        wx = p_x - A_eff_x
        wy = p_y - A_eff_y
        ee = (ex * ex + ey * ey).clamp_min(1e-18)
        t = torch.clamp((wx * ex + wy * ey) / ee, 0.0, 1.0)
        nearest_x = A_eff_x + t * ex
        nearest_y = A_eff_y + t * ey
        dist2 = (p_x - nearest_x) ** 2 + (p_y - nearest_y) ** 2
        dist_seg = torch.sqrt(dist2.clamp_min(torch.finfo(x.dtype).tiny))
        cross_z = ex * wy - ey * wx
        d_sector = torch.where(cross_z >= 0, -dist_seg, dist_seg) - r_out

        # ---- Inner valley rounding ----
        edge_dx = Bx - A_eff_x   # NOTE: edge direction from A_eff to B
        edge_dy = By - A_eff_y   # (matches ex, ey from d_sector computation)
        edge_len2 = edge_dx * edge_dx + edge_dy * edge_dy
        edge_len = torch.sqrt(edge_len2.clamp_min(torch.finfo(x.dtype).tiny))
        edge_ux = edge_dx / edge_len
        edge_uy = edge_dy / edge_len

        cos_beta = torch.sqrt((1.0 - sin_beta * sin_beta).clamp_min(torch.finfo(x.dtype).tiny))
        tan_beta = sin_beta / cos_beta

        # Clamp r_in so tangent points stay on the edges.
        icr_max = edge_len * tan_beta
        icr_eff = torch.minimum(r_in, icr_max)

        # Disk center on outward bisector, past B. (sin_beta, bis_x/bis_y,
        # OB reused from the outer-tip rounding above -- same geometry.)
        OC = OB + icr_eff / sin_beta
        C_x = OC * bis_x
        C_y = OC * bis_y

        d_disk2 = (p_x - C_x) ** 2 + (p_y - C_y) ** 2
        d_disk = torch.sqrt(d_disk2.clamp_min(torch.finfo(x.dtype).tiny)) - icr_eff

        # Half-kite constraint 1: valley-exterior side of edge A_eff -> B.
        # cross_z > 0 is polygon interior; we want cross_z < 0 (valley
        # side) to be inside the half-kite, so the signed distance is
        # -cross_z / edge_len.
        cross_edge = edge_dx * wy - edge_dy * wx
        d_half1 = cross_edge / edge_len

        # Half-kite constraint 2: B side of the perpendicular through T1.
        # T1 is on the edge at distance (edge_len - icr_eff/tan(beta)) from
        # A_eff. Projection of (p - A_eff) onto edge direction: proj = w . edge_unit.
        # B side is where proj > T1_proj, i.e. (T1_proj - proj) < 0 is inside.
        proj = wx * edge_ux + wy * edge_uy
        T1_proj = edge_len - icr_eff / tan_beta
        d_half2 = T1_proj - proj

        # Half-kite (in folded coords, the part of the kite with p_y >= 0).
        d_half_kite = torch.maximum(d_half1, d_half2)

        # Patch: half-kite minus disk.
        d_patch = torch.maximum(d_half_kite, -d_disk)

        # Union with sharp star.
        return torch.minimum(d_sector, d_patch)
    return f
