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
