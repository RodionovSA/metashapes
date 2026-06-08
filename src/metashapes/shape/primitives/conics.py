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

        # swap so px <= py, matching Quilez code
        swap = px > py
        px2 = torch.where(swap, py, px)
        py2 = torch.where(swap, px, py)
        ax = torch.where(swap, b2, a2)
        by = torch.where(swap, a2, b2)

        eps = torch.finfo(x.dtype).eps

        # handle circle / near-circle separately to avoid division by zero
        l = by * by - ax * ax
        circle_mask = torch.abs(l) < eps

        # default circle result for a ~= b
        r0 = 0.5 * (ax + by)
        circle_dist = torch.sqrt(px2 * px2 + py2 * py2 + eps) - r0

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

        co = torch.empty_like(px2)

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
            t = torch.sin(h) * torch.sqrt(torch.as_tensor(3.0, dtype=x.dtype, device=x.device))

            rx = torch.sqrt(torch.clamp(-c * (s + t + 2.0) + m2, min=0.0))
            ry = torch.sqrt(torch.clamp(-c * (s - t + 2.0) + m2, min=0.0))

            denom = rx * ry
            denom = torch.where(torch.abs(denom) < eps, torch.full_like(denom, eps), denom)

            co1 = (ry + torch.sign(l_safe) * rx + torch.abs(g) / denom - m) * 0.5
            co = torch.where(mask1, co1, torch.zeros_like(co))

        # branch 2: d >= 0
        mask2 = (~mask1) & (~circle_mask)
        if torch.any(mask2):
            h = 2.0 * m * n * torch.sqrt(torch.clamp(d, min=0.0))
            qp = q + h
            qm = q - h

            s = torch.sign(qp) * torch.pow(torch.abs(qp), 1.0 / 3.0)
            u = torch.sign(qm) * torch.pow(torch.abs(qm), 1.0 / 3.0)

            rx = -s - u - 4.0 * c + 2.0 * m2
            ry = (s - u) * torch.sqrt(torch.as_tensor(3.0, dtype=x.dtype, device=x.device))
            rm = torch.sqrt(torch.clamp(rx * rx + ry * ry, min=eps))

            denom = torch.sqrt(torch.clamp(rm - rx, min=eps))
            co2 = (ry / denom + 2.0 * g / rm - m) * 0.5
            co = torch.where(mask2, co2, co)

        # numerical safety
        co = torch.clamp(co, -1.0, 1.0)
        si = torch.sqrt(torch.clamp(1.0 - co * co, min=0.0))

        rx = ax * co
        ry = by * si

        dist = torch.sqrt((rx - px2) ** 2 + (ry - py2) ** 2 + eps)
        sign = torch.sign(py2 - ry)
        ellipse_dist = dist * sign

        return torch.where(circle_mask, circle_dist, ellipse_dist)

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

        if self.width <= 0:
            raise ValueError("Egg width must be positive")
        if self.height <= 0:
            raise ValueError("Egg height must be positive")
        if torch.abs(self.skew) >= 1.0:
            raise ValueError("Egg skew must be in (-1, 1)")

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cx, cy = self.center[0], self.center[1]
        a = self.width * 0.5
        b_top = self.height * 0.5 * (1.0 + self.skew)
        b_bot = self.height * 0.5 * (1.0 - self.skew)

        x_local, y_local = _to_local_coords(x, y, cx, cy, self.angle)

        # Per-point effective semi-axis in y: upper half uses b_top, lower uses b_bot.
        # By symmetry proof, for py >= 0 the Quilez closest point always has qy >= 0,
        # so reflecting the lower half up (py = |y_local|) gives the exact half-ellipse SDF.
        b_eff = torch.where(y_local >= 0,
                            b_top * torch.ones_like(y_local),
                            b_bot * torch.ones_like(y_local))

        px = torch.abs(x_local)
        py = torch.abs(y_local)

        # Quilez ellipse SDF with tensor axes (a, b_eff)
        a2 = a
        b2 = b_eff

        swap = px > py
        px2 = torch.where(swap, py, px)
        py2 = torch.where(swap, px, py)
        ax = torch.where(swap, b2, a2 * torch.ones_like(b2))
        by = torch.where(swap, a2 * torch.ones_like(b2), b2)

        eps = torch.finfo(x.dtype).eps

        l = by * by - ax * ax
        circle_mask = torch.abs(l) < eps

        r0 = 0.5 * (ax + by)
        circle_dist = torch.sqrt(px2 * px2 + py2 * py2 + eps) - r0

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

        co = torch.empty_like(px2)

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
            t = torch.sin(h) * torch.sqrt(torch.as_tensor(3.0, dtype=x.dtype, device=x.device))

            rx = torch.sqrt(torch.clamp(-c * (s + t + 2.0) + m2, min=0.0))
            ry = torch.sqrt(torch.clamp(-c * (s - t + 2.0) + m2, min=0.0))

            denom = rx * ry
            denom = torch.where(torch.abs(denom) < eps, torch.full_like(denom, eps), denom)

            co1 = (ry + torch.sign(l_safe) * rx + torch.abs(g) / denom - m) * 0.5
            co = torch.where(mask1, co1, torch.zeros_like(co))

        mask2 = (~mask1) & (~circle_mask)
        if torch.any(mask2):
            h = 2.0 * m * n * torch.sqrt(torch.clamp(d, min=0.0))
            qp = q + h
            qm = q - h

            s = torch.sign(qp) * torch.pow(torch.abs(qp), 1.0 / 3.0)
            u = torch.sign(qm) * torch.pow(torch.abs(qm), 1.0 / 3.0)

            rx = -s - u - 4.0 * c + 2.0 * m2
            ry = (s - u) * torch.sqrt(torch.as_tensor(3.0, dtype=x.dtype, device=x.device))
            rm = torch.sqrt(torch.clamp(rx * rx + ry * ry, min=eps))

            denom = torch.sqrt(torch.clamp(rm - rx, min=eps))
            co2 = (ry / denom + 2.0 * g / rm - m) * 0.5
            co = torch.where(mask2, co2, co)

        co = torch.clamp(co, -1.0, 1.0)
        si = torch.sqrt(torch.clamp(1.0 - co * co, min=0.0))

        rx = ax * co
        ry = by * si

        dist = torch.sqrt((rx - px2) ** 2 + (ry - py2) ** 2 + eps)
        sign = torch.sign(py2 - ry)
        ellipse_dist = dist * sign

        return torch.where(circle_mask, circle_dist, ellipse_dist)

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

        if self.length <= 0:
            raise ValueError("Stadium length must be positive")
        if self.width <= 0:
            raise ValueError("Stadium width must be positive")
        if self.length < self.width:
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
