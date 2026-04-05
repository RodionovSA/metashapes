# metashapes/shape/primitives/quads.py
# This module defines symbolic quadrilateral shapes

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar
import torch
import numpy as np

from metashapes.shape.base import Shape, to_plain_data, to_plain_scalar
from metashapes.shape.registry import register_shape
from metashapes.shape.utils import _to_local_coords

__all__ = [
    "Rectangle",
    "ConvexQuad",
    "IsoscelesTrapezoid",
]

@register_shape("Rectangle")
@dataclass(slots=True)
class Rectangle(Shape):
    """
    Symbolic rectangle.

    Parameters:
        center: (cx, cy)
        size: (width, height)
        angle: counter-clockwise rotation angle in degrees
        corner_radius: radius for soft corners (ignored if soft_corners=False)
    """
    center: tuple[Any, Any]
    size: tuple[Any, Any]
    angle: Any = 0.0
    corner_radius: Any = 0.0

    def __post_init__(self) -> None:
        if len(self.center) != 2:
            raise ValueError("center must have length 2")
        if len(self.size) != 2:
            raise ValueError("size must have length 2")
        
    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cx = torch.as_tensor(self.center[0], dtype=x.dtype, device=x.device)
        cy = torch.as_tensor(self.center[1], dtype=y.dtype, device=y.device)
        w = torch.as_tensor(self.size[0], dtype=x.dtype, device=x.device)
        h = torch.as_tensor(self.size[1], dtype=y.dtype, device=y.device)
        a = torch.as_tensor(self.angle, dtype=x.dtype, device=x.device)
        r = torch.as_tensor(self.corner_radius, dtype=x.dtype, device=x.device)

        if torch.any(w <= 0) or torch.any(h <= 0):
            raise ValueError("Rectangle size components must be positive")
        if torch.any(r < 0):
            raise ValueError("corner_radius must be non-negative")

        r = torch.minimum(r, 0.5 * torch.minimum(w, h))

        # shift to local center and rotate
        x_local, y_local = _to_local_coords(x, y, cx, cy, a)

        hx = w * 0.5 - r
        hy = h * 0.5 - r

        qx = torch.abs(x_local) - hx
        qy = torch.abs(y_local) - hy

        outside = torch.sqrt(torch.clamp(qx, min=0)**2 + torch.clamp(qy, min=0)**2)
        inside = torch.minimum(torch.maximum(qx, qy), torch.zeros((), dtype=x.dtype, device=x.device))

        return outside + inside - r
    
    @property
    def min_feature_size(self) -> float:
        return min(to_plain_scalar(v) for v in self.size)

@register_shape("ConvexQuad")
@dataclass(slots=True)
class ConvexQuad(Shape):
    center: tuple[Any, Any]
    u: tuple[Any, Any]
    v: tuple[Any, Any]
    alpha: Any = 0.0
    beta: Any = 0.0
    angle: Any = 0.0
    corner_radius: Any = 0.0

    _tuple_fields = ("center", "u", "v")
    _scalar_fields = Shape._scalar_fields + ("alpha", "beta")

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cx = torch.as_tensor(self.center[0], dtype=x.dtype, device=x.device)
        cy = torch.as_tensor(self.center[1], dtype=y.dtype, device=y.device)

        u0x = torch.as_tensor(self.u[0], dtype=x.dtype, device=x.device)
        u0y = torch.as_tensor(self.u[1], dtype=y.dtype, device=y.device)
        v0x = torch.as_tensor(self.v[0], dtype=x.dtype, device=x.device)
        v0y = torch.as_tensor(self.v[1], dtype=y.dtype, device=y.device)

        angle = torch.as_tensor(self.angle, dtype=x.dtype, device=x.device)
        alpha = torch.as_tensor(self.alpha, dtype=x.dtype, device=x.device)
        beta = torch.as_tensor(self.beta, dtype=x.dtype, device=x.device)
        rr = torch.as_tensor(self.corner_radius, dtype=x.dtype, device=x.device)

        if torch.any(rr < 0):
            raise ValueError("corner_radius must be non-negative")

        theta = angle * (torch.pi / 180.0)
        ct = torch.cos(theta)
        st = torch.sin(theta)

        # rotate local frame vectors by global angle
        ux = ct * u0x - st * u0y
        uy = st * u0x + ct * u0y
        vx = ct * v0x - st * v0y
        vy = st * v0x + ct * v0y

        uv_cross = ux * vy - uy * vx
        if abs(to_plain_scalar(uv_cross)) <= 1e-12:
            raise ValueError("u and v must not be collinear")

        v0 = (cx - ux - vx, cy - uy - vy)
        v1 = (cx + ux - vx, cy + uy - vy)
        v2 = (cx + (1.0 + alpha) * ux + (1.0 + beta) * vx,
            cy + (1.0 + alpha) * uy + (1.0 + beta) * vy)
        v3 = (cx - (1.0 + alpha) * ux + (1.0 + beta) * vx,
            cy - (1.0 + alpha) * uy + (1.0 + beta) * vy)

        verts = [v0, v1, v2, v3]

        def _signed_area2(poly):
            s = torch.zeros((), dtype=x.dtype, device=x.device)
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                s = s + x1 * y2 - x2 * y1
            return s

        area2 = _signed_area2(verts)
        if abs(to_plain_scalar(area2)) <= 1e-12:
            raise ValueError("Degenerate quadrilateral")

        if to_plain_scalar(area2) < 0:
            verts = [verts[0], verts[3], verts[2], verts[1]]

        def _line_intersection(px, py, rx, ry, qx, qy, sx, sy):
            det = rx * sy - ry * sx
            if abs(to_plain_scalar(det)) <= 1e-12:
                raise ValueError("Degenerate inset polygon")
            t = ((qx - px) * sy - (qy - py) * sx) / det
            return px + t * rx, py + t * ry

        def _inset_convex_polygon(poly, rad):
            n = len(poly)
            lines = []

            for i in range(n):
                ax, ay = poly[i]
                bx, by = poly[(i + 1) % n]

                ex = bx - ax
                ey = by - ay
                elen = torch.sqrt(ex * ex + ey * ey)

                if abs(to_plain_scalar(elen)) <= 1e-12:
                    raise ValueError("Degenerate polygon edge")

                nx = -ey / elen
                ny = ex / elen
                lines.append((ax + rad * nx, ay + rad * ny, ex, ey))

            out = []
            for i in range(n):
                p1x, p1y, r1x, r1y = lines[i - 1]
                p2x, p2y, r2x, r2y = lines[i]
                ix, iy = _line_intersection(p1x, p1y, r1x, r1y, p2x, p2y, r2x, r2y)
                out.append((ix, iy))

            area2_in = _signed_area2(out)
            if abs(to_plain_scalar(area2_in)) <= 1e-12 or to_plain_scalar(area2_in) <= 0:
                raise ValueError("corner_radius is too large")

            return out

        if to_plain_scalar(rr) > 0:
            verts = _inset_convex_polygon(verts, rr)

        min_d2 = None
        inside = torch.ones_like(x, dtype=torch.bool)

        n = len(verts)
        for i in range(n):
            ax, ay = verts[i]
            bx, by = verts[(i + 1) % n]

            ex = bx - ax
            ey = by - ay
            wx = x - ax
            wy = y - ay

            ee = ex * ex + ey * ey
            t = torch.clamp((wx * ex + wy * ey) / ee, 0.0, 1.0)

            px = ax + t * ex
            py = ay + t * ey

            d2 = (x - px) ** 2 + (y - py) ** 2
            min_d2 = d2 if min_d2 is None else torch.minimum(min_d2, d2)

            cross = ex * (y - ay) - ey * (x - ax)
            inside = inside & (cross >= 0)

        d = torch.sqrt(torch.clamp(min_d2, min=0.0))
        d = torch.where(inside, -d, d)

        return d - rr
    
    @property
    def min_feature_size(self) -> float:
        cx, cy = (to_plain_scalar(v) for v in self.center)
        ux, uy = (to_plain_scalar(v) for v in self.u)
        vx, vy = (to_plain_scalar(v) for v in self.v)
        a = to_plain_scalar(self.alpha)
        b = to_plain_scalar(self.beta)

        verts = [
            (cx - ux - vx, cy - uy - vy),
            (cx + ux - vx, cy + uy - vy),
            (cx + (1.0 + a) * ux + (1.0 + b) * vx, cy + (1.0 + a) * uy + (1.0 + b) * vy),
            (cx - (1.0 + a) * ux + (1.0 + b) * vx, cy - (1.0 + a) * uy + (1.0 + b) * vy),
        ]

        def seg_len(p, q):
            return ((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2) ** 0.5

        return min(seg_len(verts[i], verts[(i + 1) % 4]) for i in range(4))

@register_shape("IsoscelesTrapezoid")
@dataclass(slots=True)
class IsoscelesTrapezoid(Shape):
    """
    Symbolic isosceles trapezoid.

    Parameters:
        center: (cx, cy)
        bottom_width: full width of the bottom base
        top_width: full width of the top base
        height: full trapezoid height
        angle: counter-clockwise rotation angle in degrees
        corner_radius: rounding radius
    """
    center: tuple[Any, Any]
    bottom_width: Any
    top_width: Any
    height: Any
    angle: Any = 0.0
    corner_radius: Any = 0.0
    
    _scalar_fields: ClassVar[tuple[str, ...]] = Shape._scalar_fields + ('bottom_width', 'top_width', 'height')

    def __post_init__(self) -> None:
        if len(self.center) != 2:
            raise ValueError("center must have length 2")

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cx = torch.as_tensor(self.center[0], dtype=x.dtype, device=x.device)
        cy = torch.as_tensor(self.center[1], dtype=y.dtype, device=y.device)
        wb = torch.as_tensor(self.bottom_width, dtype=x.dtype, device=x.device)
        wt = torch.as_tensor(self.top_width, dtype=x.dtype, device=x.device)
        h = torch.as_tensor(self.height, dtype=x.dtype, device=x.device)
        rr = torch.as_tensor(self.corner_radius, dtype=x.dtype, device=x.device)
        angle = torch.as_tensor(self.angle, dtype=x.dtype, device=x.device)

        if torch.any(wb <= 0):
            raise ValueError("bottom_width must be positive")
        if torch.any(wt <= 0):
            raise ValueError("top_width must be positive")
        if torch.any(h <= 0):
            raise ValueError("height must be positive")
        if torch.any(rr < 0):
            raise ValueError("corner_radius must be non-negative")

        r1 = 0.5 * wb
        r2 = 0.5 * wt
        he = 0.5 * h

        # inset trapezoid so outer support lines stay fixed after rounding
        slope = (r2 - r1) / (2.0 * he)
        q = torch.sqrt(1.0 + slope * slope)

        r1 = r1 - rr * (q - slope)
        r2 = r2 - rr * (q + slope)
        he = he - rr

        if torch.any(r1 <= 0) or torch.any(r2 <= 0) or torch.any(he <= 0):
            raise ValueError("corner_radius is too large")

        x_local, y_local = _to_local_coords(x, y, cx, cy, angle)

        px = torch.abs(x_local)
        py = y_local

        k1x = r2
        k1y = he
        k2x = r2 - r1
        k2y = 2.0 * he

        lim = torch.where(py < 0.0, r1, r2)
        cax = px - torch.minimum(px, lim)
        cay = torch.abs(py) - he

        dx = k1x - px
        dy = k1y - py
        k2_dot_k2 = k2x * k2x + k2y * k2y
        t = torch.clamp((dx * k2x + dy * k2y) / k2_dot_k2, 0.0, 1.0)

        cbx = px - k1x + k2x * t
        cby = py - k1y + k2y * t

        ca2 = cax * cax + cay * cay
        cb2 = cbx * cbx + cby * cby

        s = torch.where((cbx < 0.0) & (cay < 0.0), -1.0, 1.0)
        d = s * torch.sqrt(torch.clamp(torch.minimum(ca2, cb2), min=0.0))

        return d - rr

    @property
    def min_feature_size(self) -> float:
        return min(
            to_plain_scalar(self.bottom_width),
            to_plain_scalar(self.top_width),
            to_plain_scalar(self.height),
        )
        
