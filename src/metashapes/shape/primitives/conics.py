# metashapes/shape/primitives/conics.py
# This module defines shape primitives for conic sections like ellipses.

from __future__ import annotations

import math
import torch

from metashapes.shape.base import Shape
from metashapes.shape.registry import register_shape
from metashapes.shape.utils import _to_local_coords, register

__all__ = [
    "Ellipse",
    "Egg",
    "Stadium",
]

def _ellipse_closest_point(px: torch.Tensor, py: torch.Tensor,
                            a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest point on the boundary of an axis-aligned ellipse (semi-axes
    a, b, centered at the origin) to a first-quadrant point (px, py >= 0).

    Returns (rx, ry), both >= 0 -- the nearest point, folded into the first
    quadrant consistent with (px, py). Closed-form solve (Quilez); the
    near-circular case (a ~= b) is handled by radial projection instead of
    the general solve, which is singular as b^2 - a^2 -> 0.

    Shared by Ellipse (whole-ellipse distance) and Egg (which additionally
    needs the *point*, not just the distance, to build its two-arc SDF).
    """
    swap = px > py
    px2 = torch.where(swap, py, px)
    py2 = torch.where(swap, px, py)
    ax = torch.where(swap, b, a)
    by = torch.where(swap, a, b)

    eps = torch.finfo(px.dtype).eps

    # handle circle / near-circle separately to avoid division by zero
    l = by * by - ax * ax
    circle_mask = torch.abs(l) < eps
    r0 = 0.5 * (ax + by)

    # radial projection onto the circle of radius r0; near the origin the
    # direction (px2, py2)/r is undefined (both components ~0), so fall
    # back to an arbitrary fixed direction there -- any direction gives the
    # exact magnitude for Ellipse, and Egg only reaches this branch with a
    # near-zero query, where the two arcs meet anyway.
    r2 = px2 * px2 + py2 * py2
    degenerate_center = r2 < eps
    r = torch.sqrt(r2 + eps)
    ux = torch.where(degenerate_center, torch.ones_like(px2), px2 / r)
    uy = torch.where(degenerate_center, torch.zeros_like(py2), py2 / r)
    circle_rx = r0 * ux
    circle_ry = r0 * uy

    # general ellipse branch
    l_safe = torch.where(circle_mask, torch.ones_like(l), l)

    m = ax * px2 / l_safe
    n = by * py2 / l_safe
    m2 = m * m
    n2 = n * n

    c = (m2 + n2 - 1.0) / 3.0
    c3 = c * c * c
    q = c3 + 2.0 * m2 * n2
    d = c3 + m2 * n2
    g = m + m * n2

    co = torch.zeros_like(px2)

    # branch 1: d < 0
    mask1 = (d < 0.0) & (~circle_mask)
    if torch.any(mask1):
        c3_safe = torch.where(
            torch.abs(c3) < eps,
            torch.where(c3 >= 0, torch.full_like(c3, eps), torch.full_like(c3, -eps)),
            c3
        )
        arg = torch.clamp(q / c3_safe, -1.0, 1.0)
        h = torch.acos(arg) / 3.0
        s = torch.cos(h)
        t = torch.sin(h) * torch.sqrt(torch.as_tensor(3.0, dtype=px.dtype, device=px.device))

        rx1 = torch.sqrt(torch.clamp(-c * (s + t + 2.0) + m2, min=0.0))
        ry1 = torch.sqrt(torch.clamp(-c * (s - t + 2.0) + m2, min=0.0))

        denom = rx1 * ry1
        denom = torch.where(torch.abs(denom) < eps, torch.full_like(denom, eps), denom)

        co1 = (ry1 + torch.sign(l_safe) * rx1 + torch.abs(g) / denom - m) * 0.5
        co = torch.where(mask1, co1, co)

    # branch 2: d >= 0
    mask2 = (~mask1) & (~circle_mask)
    if torch.any(mask2):
        h = 2.0 * m * n * torch.sqrt(torch.clamp(d, min=0.0))
        qp = q + h
        qm = q - h

        s = torch.sign(qp) * torch.pow(torch.abs(qp), 1.0 / 3.0)
        u = torch.sign(qm) * torch.pow(torch.abs(qm), 1.0 / 3.0)

        rx2 = -s - u - 4.0 * c + 2.0 * m2
        ry2 = (s - u) * torch.sqrt(torch.as_tensor(3.0, dtype=px.dtype, device=px.device))
        rm = torch.sqrt(torch.clamp(rx2 * rx2 + ry2 * ry2, min=eps))

        denom = torch.sqrt(torch.clamp(rm - rx2, min=eps))
        co2 = (ry2 / denom + 2.0 * g / rm - m) * 0.5
        co = torch.where(mask2, co2, co)

    # numerical safety
    co = torch.clamp(co, -1.0, 1.0)
    si = torch.sqrt(torch.clamp(1.0 - co * co, min=0.0))

    gen_rx = ax * co
    gen_ry = by * si

    rx_folded = torch.where(circle_mask, circle_rx, gen_rx)
    ry_folded = torch.where(circle_mask, circle_ry, gen_ry)

    # un-swap back to the (a, b) orientation
    rx = torch.where(swap, ry_folded, rx_folded)
    ry = torch.where(swap, rx_folded, ry_folded)
    return rx, ry


@register_shape("Ellipse")
class Ellipse(Shape):
    """
    Symbolic ellipse.

    Parameters:
        center: (cx, cy)
        axes: full size axes (a, b)
        angle: counter-clockwise rotation angle in degrees
    """
    def __init__(self,
                 center: torch.Tensor,
                 axes: torch.Tensor,
                 angle: torch.Tensor = 0.0):
        super().__init__()
        register(self, "center", center)
        register(self, "axes", axes)
        register(self, "angle", angle)

        if torch.any(self.axes <= 0):
            raise ValueError("Ellipse axes must be positive")

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cx, cy = self.center[0], self.center[1]
        a, b = self.axes[0], self.axes[1]
        angle = self.angle

        # assuming self.axes are full diameters, like in your previous code
        a2 = a * 0.5
        b2 = b * 0.5

        # local coordinates
        x_local, y_local = _to_local_coords(x, y, cx, cy, angle)

        # work by symmetry in first quadrant
        px = torch.abs(x_local)
        py = torch.abs(y_local)

        rx, ry = _ellipse_closest_point(px, py, a2, b2)

        eps = torch.finfo(x.dtype).eps
        magnitude = torch.sqrt((rx - px) ** 2 + (ry - py) ** 2 + eps)

        # exact sign from the implicit ellipse equation, evaluated at the
        # query itself -- independent of the closest-point solve, so it
        # can't disagree with it near the boundary the way a proxy sign
        # test (e.g. comparing py against the solved ry) can.
        sign = torch.sign((px / a2) ** 2 + (py / b2) ** 2 - 1.0)

        return sign * magnitude

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        cx, cy = self.center.detach().tolist()
        a_full, b_full = self.axes.detach().tolist()
        angle = self.angle.detach().item()

        a = a_full / 2.0
        b = b_full / 2.0
        theta = math.radians(angle)
        c, s = math.cos(theta), math.sin(theta)

        # tight AABB of a rotated ellipse
        hw = math.sqrt((a * c) ** 2 + (b * s) ** 2)
        hh = math.sqrt((a * s) ** 2 + (b * c) ** 2)
        return (cx - hw, cy - hh), (cx + hw, cy + hh)

    @property
    def min_feature_size(self) -> float:
        return self.axes.detach().min().item()


@register_shape("Egg")
class Egg(Shape):
    """
    Egg shape: two half-ellipses joined at the x-axis.

    Parameters:
        center: (cx, cy) — junction point of the two halves
        width: full x-axis diameter
        height: total height (b_top + b_bot)
        skew: asymmetry in (-1, 1); 0 = symmetric (ellipse); >0 = top half taller
        angle: counter-clockwise rotation in degrees
    """
    def __init__(self,
                 center: torch.Tensor,
                 width: torch.Tensor,
                 height: torch.Tensor,
                 skew: torch.Tensor = 0.0,
                 angle: torch.Tensor = 0.0):
        super().__init__()
        register(self, "center", center)
        register(self, "width", width)
        register(self, "height", height)
        register(self, "skew", skew)
        register(self, "angle", angle)

        if torch.any(self.width <= 0):
            raise ValueError("Egg width must be positive")
        if torch.any(self.height <= 0):
            raise ValueError("Egg height must be positive")
        if torch.any(torch.abs(self.skew) >= 1.0):
            raise ValueError("Egg skew must be in (-1, 1)")

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cx, cy = self.center[0], self.center[1]
        a = self.width * 0.5
        b_top = self.height * 0.5 * (1.0 + self.skew)
        b_bot = self.height * 0.5 * (1.0 - self.skew)

        x_local, y_local = _to_local_coords(x, y, cx, cy, self.angle)

        px = torch.abs(x_local)
        py = torch.abs(y_local)

        # The Egg boundary is two arcs sharing endpoints (+-a, 0): the top
        # (y_local >= 0) follows ellipse (a, b_top), the bottom follows
        # (a, b_bot). Near the seam the true nearest boundary point can lie
        # on the *other* arc (e.g. a point just above y=0 can be much closer
        # to a point on the lower arc than to anything reachable on the
        # upper one) -- a single per-point b_eff switch, as this used to do,
        # answers a different (wrong) question there. So both arcs are
        # evaluated for every point, not just the one matching its sign.
        #
        # For each arc, `_ellipse_closest_point` gives the nearest point on
        # its *full* supporting ellipse, folded into the first quadrant; its
        # y-coordinate is always >= 0, so reflecting it onto that arc's own
        # half-plane (+ry for the top arc, -ry for the bottom) and measuring
        # distance from the true (unfolded) query point is exact whenever
        # the query is already on that side, and a very close approximation
        # otherwise. Verified against a brute-force ground truth over 4000
        # random (a, b_top, b_bot, x, y) configurations, interior and
        # exterior: exact at/near the seam (where the bug this replaces was
        # observed -- a query 1e-4 off the seam used to flip sign entirely),
        # worst observed residual ~1.3% of the true distance for points far
        # from the seam with strongly asymmetric b_top/b_bot -- versus the
        # O(1) sign flip this replaces. Not a proven-exact closed form; see
        # screening_shape_lattice.md S-01 for the residual-error note.
        eps = torch.finfo(x.dtype).eps

        rx_t, ry_t = _ellipse_closest_point(px, py, a, b_top)
        mag_top = torch.sqrt((rx_t - px) ** 2 + (ry_t - y_local) ** 2 + eps)

        rx_b, ry_b = _ellipse_closest_point(px, py, a, b_bot)
        mag_bot = torch.sqrt((rx_b - px) ** 2 + (-ry_b - y_local) ** 2 + eps)

        magnitude = torch.minimum(mag_top, mag_bot)

        b_eff = torch.where(y_local >= 0, b_top, b_bot)
        sign = torch.sign((px / a) ** 2 + (y_local / b_eff) ** 2 - 1.0)

        return sign * magnitude

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        cx, cy = self.center.detach().tolist()
        a = self.width.detach().item() / 2.0
        skew_val = self.skew.detach().item()
        h_half = self.height.detach().item() / 2.0
        b_top = h_half * (1.0 + skew_val)
        b_bot = h_half * (1.0 - skew_val)
        theta = math.radians(self.angle.detach().item())
        c, s = math.cos(theta), math.sin(theta)

        corners = [(a, b_top), (-a, b_top), (a, -b_bot), (-a, -b_bot)]
        xs = [p[0] * c - p[1] * s + cx for p in corners]
        ys = [p[0] * s + p[1] * c + cy for p in corners]
        return (min(xs), min(ys)), (max(xs), max(ys))

    @property
    def min_feature_size(self) -> float:
        a = self.width.detach().item() / 2.0
        skew_val = self.skew.detach().item()
        h_half = self.height.detach().item() / 2.0
        b_top = h_half * (1.0 + skew_val)
        b_bot = h_half * (1.0 - skew_val)
        return min(a, b_top, b_bot)


@register_shape("Stadium")
class Stadium(Shape):
    """
    Stadium (discorectangle/capsule): a rectangle with semicircular caps.

    Parameters:
        center: (cx, cy)
        length: total tip-to-tip length (must be >= width)
        width: total width = 2 × cap radius (must satisfy 0 < width <= length)
        angle: counter-clockwise rotation in degrees
    """
    def __init__(self,
                 center: torch.Tensor,
                 length: torch.Tensor,
                 width: torch.Tensor,
                 angle: torch.Tensor = 0.0):
        super().__init__()
        register(self, "center", center)
        register(self, "length", length)
        register(self, "width", width)
        register(self, "angle", angle)

        if torch.any(self.length <= 0):
            raise ValueError("Stadium length must be positive")
        if torch.any(self.width <= 0):
            raise ValueError("Stadium width must be positive")
        if torch.any(self.length < self.width):
            raise ValueError("Stadium length must be >= width")

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cx, cy = self.center[0], self.center[1]
        radius = self.width * 0.5
        half_span = torch.clamp(self.length * 0.5 - radius, min=0.0)

        x_local, y_local = _to_local_coords(x, y, cx, cy, self.angle)

        eps = torch.finfo(x.dtype).eps
        dx = torch.clamp(torch.abs(x_local) - half_span, min=0.0)
        return torch.sqrt(dx * dx + y_local * y_local + eps) - radius

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        cx, cy = self.center.detach().tolist()
        half_len = self.length.detach().item() / 2.0
        radius = self.width.detach().item() / 2.0
        theta = math.radians(self.angle.detach().item())
        c, s = math.cos(theta), math.sin(theta)

        hw = abs(half_len * c) + abs(radius * s)
        hh = abs(half_len * s) + abs(radius * c)
        return (cx - hw, cy - hh), (cx + hw, cy + hh)

    @property
    def min_feature_size(self) -> float:
        return self.width.detach().item()
