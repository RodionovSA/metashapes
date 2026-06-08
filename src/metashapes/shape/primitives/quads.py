# metashapes/shape/primitives/quads.py
# This module defines symbolic quadrilateral shapes

from __future__ import annotations

import math
import torch

from metashapes.shape.base import Shape
from metashapes.shape.registry import register_shape
from metashapes.shape.utils import _to_local_coords, register

__all__ = [
    "Rectangle",
    "ConvexQuad",
    "IsoscelesTrapezoid",
]

@register_shape("Rectangle")
class Rectangle(Shape):
    """
    Symbolic rectangle.

    Parameters:
        center: (cx, cy)
        size: (width, height)
        angle: counter-clockwise rotation angle in degrees
        corner_radius: radius for soft corners (ignored if soft_corners=False)
    """
    def __init__(self, 
                 center: torch.tensor, 
                 size: torch.Tensor,
                 angle: torch.Tensor = 0.0,
                 corner_radius: torch.Tensor = 0.0):
        super().__init__()
        register(self, "center", center)
        register(self, "size", size)
        register(self, "angle", angle)
        register(self, "corner_radius", corner_radius)
        
        if torch.any(self.size <= 0):
            raise ValueError("Rectangle size components must be positive")
        if torch.any(self.corner_radius < 0):
            raise ValueError("corner_radius must be non-negative")
        
    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cx, cy = self.center[0], self.center[1]
        w, h = self.size[0], self.size[1]
        a = self.angle
        r = self.corner_radius

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
    
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        cx, cy = self.center.detach().tolist()
        w, h = self.size.detach().tolist()
        angle = self.angle.detach().item()

        theta = math.radians(angle)
        c, s = abs(math.cos(theta)), abs(math.sin(theta))
        hw = 0.5 * (c * w + s * h)
        hh = 0.5 * (s * w + c * h)
        return (cx - hw, cy - hh), (cx + hw, cy + hh)

    @property
    def min_feature_size(self) -> float:
        return self.size.detach().min().item()

@register_shape("ConvexQuad")
class ConvexQuad(Shape):
    """
    Symbolic convex quadrilateral with optional rounded corners.

    The quad is built from a center point and two frame vectors `u`, `v`.
    Vertices are placed at combinations of ±u and ±v; `alpha` and `beta`
    stretch one corner along the u- and v-directions respectively, so the
    shape ranges from a parallelogram (alpha = beta = 0) to a general
    convex quad. `angle` rotates the whole frame about `center`.

    Parameters:
        center:        (cx, cy) center of the base parallelogram.
        u:             first frame vector (half-diagonal direction).
        v:             second frame vector (half-diagonal direction);
                       must not be collinear with u.
        alpha:         stretch of the far corner along u (default 0).
        beta:          stretch of the far corner along v (default 0).
        angle:         counter-clockwise rotation in degrees (default 0).
        corner_radius: radius for rounded corners; 0 = sharp corners.
                       Must be non-negative and small enough that the
                       inset polygon stays non-degenerate.
    """
    def __init__(self, 
                 center: torch.tensor, 
                 u: torch.Tensor,
                 v: torch.Tensor,
                 alpha: torch.Tensor = 0.0,
                 beta: torch.Tensor = 0.0,
                 angle: torch.Tensor = 0.0,
                 corner_radius: torch.Tensor = 0.0):
        super().__init__()
        register(self, "center", center)
        register(self, "u", u)
        register(self, "v", v)
        register(self, "alpha", alpha)
        register(self, "beta", beta)
        register(self, "angle", angle)
        register(self, "corner_radius", corner_radius)
        
        if torch.any(self.corner_radius < 0):
            raise ValueError("corner_radius must be non-negative")
        
        uv_cross = self.u[0] * self.v[1] - self.u[1] * self.v[0]
        if uv_cross.abs().item() <= 1e-12:
            raise ValueError("u and v must not be collinear")
        
    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cx, cy = self.center[0], self.center[1]
        u0x, u0y = self.u[0], self.u[1]
        v0x, v0y = self.v[0], self.v[1]
        angle = self.angle
        alpha = self.alpha
        beta  = self.beta
        rr    = self.corner_radius

        theta = angle * (torch.pi / 180.0)
        ct = torch.cos(theta)
        st = torch.sin(theta)

        # rotate local frame vectors by global angle
        ux = ct * u0x - st * u0y
        uy = st * u0x + ct * u0y
        vx = ct * v0x - st * v0y
        vy = st * v0x + ct * v0y

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
        if area2.abs().item() <= 1e-12:
            raise ValueError("Degenerate quadrilateral")

        if area2.item() < 0:
            verts = [verts[0], verts[3], verts[2], verts[1]]

        def _line_intersection(px, py, rx, ry, qx, qy, sx, sy):
            det = rx * sy - ry * sx
            if det.abs().item() <= 1e-12:
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

                if elen.abs().item() <= 1e-12:
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
            if area2_in.abs().item() <= 1e-12 or area2_in.item() <= 0:
                raise ValueError("corner_radius is too large")

            return out

        if rr.item() > 0:
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
    
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        cx, cy = self.center.detach().tolist()
        u0x, u0y = self.u.detach().tolist()
        v0x, v0y = self.v.detach().tolist()
        alpha = self.alpha.detach().item()
        beta  = self.beta.detach().item()
        angle = self.angle.detach().item()

        theta = math.radians(angle)
        ct, st = math.cos(theta), math.sin(theta)
        ux = ct * u0x - st * u0y
        uy = st * u0x + ct * u0y
        vx = ct * v0x - st * v0y
        vy = st * v0x + ct * v0y

        verts = [
            (cx - ux - vx,                       cy - uy - vy),
            (cx + ux - vx,                       cy + uy - vy),
            (cx + (1.0 + alpha) * ux + (1.0 + beta) * vx,
             cy + (1.0 + alpha) * uy + (1.0 + beta) * vy),
            (cx - (1.0 + alpha) * ux + (1.0 + beta) * vx,
             cy - (1.0 + alpha) * uy + (1.0 + beta) * vy),
        ]
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        return (min(xs), min(ys)), (max(xs), max(ys))

    @property
    def min_feature_size(self) -> float:
        cx, cy = self.center.detach().tolist()
        ux, uy = self.u.detach().tolist()
        vx, vy = self.v.detach().tolist()
        a = self.alpha.item()
        b = self.beta.item()

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
    def __init__(self,
                 center: torch.Tensor,
                 bottom_width: torch.Tensor,
                 top_width: torch.Tensor,
                 height: torch.Tensor,
                 angle: torch.Tensor = 0.0,
                 corner_radius: torch.Tensor = 0.0):
        super().__init__()
        register(self, "center", center)
        register(self, "bottom_width", bottom_width)
        register(self, "top_width", top_width)
        register(self, "height", height)
        register(self, "angle", angle)
        register(self, "corner_radius", corner_radius)

        if torch.any(self.bottom_width <= 0):
            raise ValueError("bottom_width must be positive")
        if torch.any(self.top_width <= 0):
            raise ValueError("top_width must be positive")
        if torch.any(self.height <= 0):
            raise ValueError("height must be positive")
        if torch.any(self.corner_radius < 0):
            raise ValueError("corner_radius must be non-negative")

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cx, cy = self.center[0], self.center[1]
        wb    = self.bottom_width
        wt    = self.top_width
        h     = self.height
        rr    = self.corner_radius
        angle = self.angle

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

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        cx, cy = self.center.detach().tolist()
        wb = self.bottom_width.detach().item()
        wt = self.top_width.detach().item()
        h  = self.height.detach().item()
        angle = self.angle.detach().item()

        theta = math.radians(angle)
        ct, st = math.cos(theta), math.sin(theta)
        corners = [(-wb / 2, -h / 2), (wb / 2, -h / 2),
                   (wt / 2,  h / 2), (-wt / 2,  h / 2)]
        xs = [cx + ct * lx - st * ly for lx, ly in corners]
        ys = [cy + st * lx + ct * ly for lx, ly in corners]
        return (min(xs), min(ys)), (max(xs), max(ys))

    @property
    def min_feature_size(self) -> float:
        return min(
            self.bottom_width.detach().item(),
            self.top_width.detach().item(),
            self.height.detach().item(),
        )
        
