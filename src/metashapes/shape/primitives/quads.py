# metashapes/shape/primitives/quads.py
# This module defines symbolic quadrilateral shapes

from __future__ import annotations

import math
import torch

from metashapes.shape.base import Shape
from metashapes.shape.registry import register_shape
from metashapes.shape.utils import _sdf_rounded_box, _to_local_coords, register

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

        return _sdf_rounded_box(x_local, y_local, w * 0.5, h * 0.5, r)

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

        # Corner-radius validity depends only on the quad's own geometry
        # (edge lengths and interior angles), not on rotation -- a rigid
        # rotation changes neither. So it's checked once here, against the
        # unrotated frame, rather than being (re-)discovered inside sdf()
        # on every call via a degeneracy side-effect of the inset
        # construction (see _max_corner_radius for why that guard was
        # unsound above a certain radius).
        zero = torch.zeros((), dtype=self.u.dtype, device=self.u.device)
        base_verts = ConvexQuad._quad_vertices(
            zero, zero, self.u[0], self.u[1], self.v[0], self.v[1], self.alpha, self.beta
        )
        area2 = ConvexQuad._signed_area2(base_verts)
        if area2.abs().item() <= 1e-12:
            raise ValueError("Degenerate quadrilateral")
        if area2.item() < 0:
            base_verts = [base_verts[0], base_verts[3], base_verts[2], base_verts[1]]

        if torch.any(self.corner_radius > 0):
            rr_max = self._max_corner_radius(base_verts)
            if torch.any(self.corner_radius >= rr_max):
                raise ValueError(
                    f"corner_radius must be < {rr_max:.6g} for this quad's "
                    "geometry (offsetting each edge inward by corner_radius "
                    "would make the rounded corners overlap or invert)"
                )

    @staticmethod
    def _quad_vertices(cx, cy, ux, uy, vx, vy, alpha, beta):
        """The four corners of the (possibly stretched) parallelogram
        frame, in construction order -- CCW or CW depending on the sign of
        u x v; callers that need a definite orientation check/fix it via
        `_signed_area2`. Shared by `__init__` (unrotated, for validation)
        and `sdf` (rotated, for the actual query).
        """
        v0 = (cx - ux - vx, cy - uy - vy)
        v1 = (cx + ux - vx, cy + uy - vy)
        v2 = (cx + (1.0 + alpha) * ux + (1.0 + beta) * vx,
              cy + (1.0 + alpha) * uy + (1.0 + beta) * vy)
        v3 = (cx - (1.0 + alpha) * ux + (1.0 + beta) * vx,
              cy - (1.0 + alpha) * uy + (1.0 + beta) * vy)
        return [v0, v1, v2, v3]

    @staticmethod
    def _signed_area2(poly):
        s = torch.zeros((), dtype=poly[0][0].dtype, device=poly[0][0].device)
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            s = s + x1 * y2 - x2 * y1
        return s

    @staticmethod
    def _max_corner_radius(verts) -> torch.Tensor:
        """Largest radius for which eroding `verts` (via the same
        edge-normal line-intersection construction `sdf` uses) stays a
        valid, positively-oriented polygon.

        Standard convex-polygon erosion bound: offsetting each edge inward
        by `rad` advances the intersection of adjacent offset lines by
        `rad / tan(interior_angle / 2)` along the edge from each endpoint.
        The construction stays valid for every edge only while the two
        endpoints' advances don't together exceed the edge's own length --
        `rr_max` is the tightest (minimum) such bound over all edges.
        """
        n = len(verts)
        eps = 1e-9
        half_pi = math.pi / 2 - eps
        angles = []
        for i in range(n):
            vx, vy = verts[i]
            px, py = verts[i - 1]
            nx, ny = verts[(i + 1) % n]
            d1x, d1y = px - vx, py - vy
            d2x, d2y = nx - vx, ny - vy
            l1 = torch.sqrt(d1x * d1x + d1y * d1y).clamp(min=eps)
            l2 = torch.sqrt(d2x * d2x + d2y * d2y).clamp(min=eps)
            cos_t = torch.clamp((d1x * d2x + d1y * d2y) / (l1 * l2), -1.0, 1.0)
            angles.append(torch.acos(cos_t))

        rr_max = None
        for i in range(n):
            j = (i + 1) % n
            ex = verts[j][0] - verts[i][0]
            ey = verts[j][1] - verts[i][1]
            edge_len = torch.sqrt(ex * ex + ey * ey)
            tan_a = torch.tan(torch.clamp(angles[i] / 2, min=eps, max=half_pi))
            tan_b = torch.tan(torch.clamp(angles[j] / 2, min=eps, max=half_pi))
            edge_rr_max = edge_len / (1.0 / tan_a + 1.0 / tan_b)
            rr_max = edge_rr_max if rr_max is None else torch.minimum(rr_max, edge_rr_max)
        return rr_max

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

        verts = ConvexQuad._quad_vertices(cx, cy, ux, uy, vx, vy, alpha, beta)

        # Rotation is rigid, so it can't turn a valid (non-degenerate) quad
        # degenerate or flip which winding correction __init__ already
        # determined -- but alpha/beta/u/v aren't re-validated here (they
        # were fixed at construction), so this stays a cheap live check
        # rather than trusting stale __init__-time state.
        area2 = ConvexQuad._signed_area2(verts)
        if area2.abs().item() <= 1e-12:
            # An .item()-based raise is fine here even though sdf() must
            # otherwise stay branch-free: this validates degeneracy and
            # must raise a clear error, not silently select a
            # NaN-producing branch. __init__ already guarantees this is
            # unreachable for any valid construction; this is a live
            # safety net, not a per-point SDF decision.
            raise ValueError("Degenerate quadrilateral")

        # Winding correction stays branch-free: swap v1<->v3 by masked
        # selection rather than an `if area2.item() < 0:` -- safe because
        # both orderings are already-finite affine rearrangements of the
        # same computed vertices.
        flip = area2 < 0
        v0, v1, v2, v3 = verts

        def _select(a, b):
            return torch.where(flip, b, a)

        verts = [
            v0,
            (_select(v1[0], v3[0]), _select(v1[1], v3[1])),
            v2,
            (_select(v3[0], v1[0]), _select(v3[1], v1[1])),
        ]

        def _line_intersection(px, py, rx, ry, qx, qy, sx, sy):
            det = rx * sy - ry * sx
            if det.abs().item() <= 1e-12:
                # __init__ already proves this can't trigger for a valid
                # quad (see _max_corner_radius); this raise is a defensive
                # safety net, not a data-dependent SDF branch.
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

            # No degeneracy check here: __init__ already guarantees
            # corner_radius stays under the exact bound for which this
            # construction is valid (see _max_corner_radius).
            return out

        # No `if rr.item() > 0:` guard: offsetting each edge by 0
        # reconstructs the original vertices exactly (the line-intersection
        # of two unmoved adjacent edges is the original shared vertex), so
        # the inset is already an identity at rr=0. Always insetting keeps
        # this branch-free.
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
        """Narrowest place the (possibly rounded) quad gets.

        Computed fresh from the current parameter values (not cached at
        construction) so it stays correct as u/v/alpha/beta are updated
        during optimization -- same as every other primitive's
        min_feature_size.

        This is the polygon's *width* (the rotating-calipers quantity: for
        each edge, the perpendicular distance from the farthest other
        vertex to that edge's line; the width is the minimum of those over
        all edges) -- the shortest edge length has no relationship to how
        thin the shape actually gets (a long, thin sliver quad can have
        long edges but a near-zero waist).
        Rotation-invariant, so this uses the unrotated frame directly, like
        `_max_corner_radius` in `__init__`. Rounding erodes a convex
        shape's width by exactly `2 * corner_radius` in every direction.
        """
        cx = cy = 0.0  # width is translation-invariant; skip self.center
        ux, uy = self.u.detach().tolist()
        vx, vy = self.v.detach().tolist()
        a = self.alpha.detach().item()
        b = self.beta.detach().item()

        verts = [
            (cx - ux - vx, cy - uy - vy),
            (cx + ux - vx, cy + uy - vy),
            (cx + (1.0 + a) * ux + (1.0 + b) * vx, cy + (1.0 + a) * uy + (1.0 + b) * vy),
            (cx - (1.0 + a) * ux + (1.0 + b) * vx, cy - (1.0 + a) * uy + (1.0 + b) * vy),
        ]
        n = len(verts)

        widths = []
        for i in range(n):
            ax, ay = verts[i]
            bx, by = verts[(i + 1) % n]
            ex, ey = bx - ax, by - ay
            elen = math.hypot(ex, ey)
            # __init__ already guarantees non-degenerate edges.
            nx, ny = -ey / elen, ex / elen  # unit normal (either direction --
                                             # only the |.| distance is used)
            dists = [
                abs((verts[j][0] - ax) * nx + (verts[j][1] - ay) * ny)
                for j in range(n) if j != i and j != (i + 1) % n
            ]
            widths.append(max(dists))

        width = min(widths)
        return max(0.0, width - 2.0 * self.corner_radius.detach().item())

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

        # Corner-radius validity depends only on the trapezoid's own
        # geometry (widths and height), not on rotation, so it's checked
        # once here rather than being (re-)discovered inside sdf() on every
        # call. Same bound sdf()'s own inset construction below uses,
        # solved for the radius at which each of r1, r2, he first reaches
        # zero.
        r1_0 = 0.5 * self.bottom_width.item()
        r2_0 = 0.5 * self.top_width.item()
        he_0 = 0.5 * self.height.item()
        slope0 = (r2_0 - r1_0) / (2.0 * he_0)
        q0 = math.sqrt(1.0 + slope0 * slope0)
        rr_max = min(r1_0 / (q0 - slope0), r2_0 / (q0 + slope0), he_0)
        if torch.any(self.corner_radius >= rr_max):
            raise ValueError(
                f"corner_radius must be < {rr_max:.6g} for this trapezoid's "
                "geometry (bottom_width, top_width, height) -- larger values "
                "make the inset trapezoid degenerate"
            )

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

        # No degeneracy check here: __init__ now guarantees corner_radius
        # is below the exact bound for which this inset stays valid (see
        # above), matching the pattern already used by ConvexQuad's
        # _inset_convex_polygon and keeping sdf() free of data-dependent
        # Python control flow.

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
        
