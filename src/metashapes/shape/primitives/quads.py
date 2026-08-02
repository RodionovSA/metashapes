# metashapes/shape/primitives/quads.py
# This module defines symbolic quadrilateral shapes

from __future__ import annotations

import math
import torch

from metashapes.shape.base import Shape
from metashapes.shape.registry import register_shape
from metashapes.shape.utils import _to_local_coords, register
from sdflib.quads import ConvexQuadSDF, IsoscelesTrapezoidSDF, RectangleSDF

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
                 center: torch.Tensor,
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

    @torch.no_grad()
    def _project(self) -> None:
        """Snap size/corner_radius back into their valid ranges in place."""
        self.size.clamp_(min=self._MIN_SIZE)
        self.corner_radius.clamp_(min=0.0, max=0.5 * self.size.min())

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._project()
        cx, cy = self.center[0], self.center[1]
        a = self.angle
        r = self.corner_radius
        w, h = self.size

        # shift to local center and rotate
        x_local, y_local = _to_local_coords(x, y, cx, cy, a)

        return RectangleSDF(w * 0.5, h * 0.5, r)(x_local, y_local)

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        self._project()
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
        self._project()
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
    def _floor_vector_norm(vec: torch.Tensor, min_norm: float,
                            fallback: tuple[float, float]) -> None:
        """In-place: rescale `vec` up to at least `min_norm`, preserving its
        direction. `vec` exactly zero has no direction to preserve, so it
        snaps to `fallback` instead of dividing by zero.
        """
        norm = torch.linalg.vector_norm(vec)
        if norm.item() <= 1e-12:
            vec.copy_(torch.as_tensor(fallback, dtype=vec.dtype, device=vec.device))
        elif norm.item() < min_norm:
            vec.mul_(min_norm / norm)

    @torch.no_grad()
    def _project(self) -> None:
        """Snap u/v/corner_radius back into their valid ranges in place.

        Collinear-but-nonzero u/v and an alpha/beta that collapses the quad
        to zero area are NOT corrected -- unlike a scalar bound, there's no
        unique "nearest valid frame" to snap to. Only the unambiguous case
        (a frame vector collapsed to exactly zero) is fixed; corner_radius's
        clamp is skipped for a step where the frame is otherwise degenerate,
        rather than computed from invalid geometry.
        """
        ConvexQuad._floor_vector_norm(self.u, self._MIN_SIZE, fallback=(self._MIN_SIZE, 0.0))
        ConvexQuad._floor_vector_norm(self.v, self._MIN_SIZE, fallback=(0.0, self._MIN_SIZE))

        zero = torch.zeros((), dtype=self.u.dtype, device=self.u.device)
        base_verts = ConvexQuad._quad_vertices(
            zero, zero, self.u[0], self.u[1], self.v[0], self.v[1], self.alpha, self.beta
        )
        area2 = ConvexQuad._signed_area2(base_verts)
        if area2.abs().item() <= 1e-12:
            return
        if area2.item() < 0:
            base_verts = [base_verts[0], base_verts[3], base_verts[2], base_verts[1]]

        # Strictly less than rr_max: at exactly rr_max the inset polygon
        # collapses to a single point, and ConvexQuadSDF's line-intersection
        # divides by a near-zero determinant there.
        rr_max = self._max_corner_radius(base_verts)
        self.corner_radius.clamp_(min=0.0, max=rr_max * self._MAX_RADIUS_FRACTION)

    @staticmethod
    def _quad_vertices(cx, cy, ux, uy, vx, vy, alpha, beta):
        """The four corners of the (possibly stretched) parallelogram
        frame, in construction order -- CCW or CW depending on the sign of
        u x v; callers that need a definite orientation check/fix it via
        `_signed_area2`. Shared by `__init__` (unrotated, for validation)
        and `bounds` (rotated, for the extent box); `sdf` builds its own
        corners inside `ConvexQuadSDF`.
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
        self._project()
        cx, cy = self.center[0], self.center[1]
        x_local, y_local = _to_local_coords(x, y, cx, cy, self.angle)

        return ConvexQuadSDF(
            self.u, self.v, self.alpha, self.beta, self.corner_radius
        )(x_local, y_local)

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        self._project()
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

        verts = ConvexQuad._quad_vertices(cx, cy, ux, uy, vx, vy, alpha, beta)
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
        self._project()
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

        rr_max = self._max_corner_radius(self.bottom_width, self.top_width, self.height)
        if torch.any(self.corner_radius >= rr_max):
            raise ValueError(
                f"corner_radius must be < {rr_max:.6g} for this trapezoid's "
                "geometry (bottom_width, top_width, height) -- larger values "
                "make the inset trapezoid degenerate"
            )

    @staticmethod
    def _max_corner_radius(bottom_width, top_width, height) -> torch.Tensor:
        """Largest corner_radius for which the inset trapezoid (the same
        construction IsoscelesTrapezoidSDF uses) stays non-degenerate: the
        tightest of the three bounds at which r1, r2, or he first reaches
        zero.
        """
        r1 = 0.5 * bottom_width
        r2 = 0.5 * top_width
        he = 0.5 * height
        slope = (r2 - r1) / (2.0 * he)
        q = torch.sqrt(1.0 + slope * slope)
        return torch.minimum(torch.minimum(r1 / (q - slope), r2 / (q + slope)), he)

    @torch.no_grad()
    def _project(self) -> None:
        """Snap bottom_width/top_width/height/corner_radius back into their
        valid ranges in place."""
        self.bottom_width.clamp_(min=self._MIN_SIZE)
        self.top_width.clamp_(min=self._MIN_SIZE)
        self.height.clamp_(min=self._MIN_SIZE)
        # Strictly less than rr_max: at exactly rr_max the inset trapezoid
        # collapses to a single point (b=t=e=0), and IsoscelesTrapezoidSDF's
        # edge-projection divides by ex^2+ey^2 == 0 there.
        rr_max = self._max_corner_radius(self.bottom_width, self.top_width, self.height)
        self.corner_radius.clamp_(min=0.0, max=rr_max * self._MAX_RADIUS_FRACTION)

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._project()
        cx, cy = self.center[0], self.center[1]
        wb    = self.bottom_width
        wt    = self.top_width
        h     = self.height
        rr    = self.corner_radius
        angle = self.angle

        x_local, y_local = _to_local_coords(x, y, cx, cy, angle)

        return IsoscelesTrapezoidSDF(wb * 0.5, wt * 0.5, h * 0.5, rr)(x_local, y_local)

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        self._project()
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
        self._project()
        return min(
            self.bottom_width.detach().item(),
            self.top_width.detach().item(),
            self.height.detach().item(),
        )
        
