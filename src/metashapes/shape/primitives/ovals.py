# metashapes/shape/primitives/ovals.py
# This module defines shape primitives for oval shapes like ellipses.

from __future__ import annotations

import math
import torch

from metashapes.shape.base import Shape
from metashapes.shape.registry import register_shape
from metashapes.shape.utils import _to_local_coords, register
from sdflib.ovals import EllipseSDF, EggSDF, StadiumSDF

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
        x_local, y_local = _to_local_coords(x, y, cx, cy, self.angle)
        # axes are full diameters
        return EllipseSDF(self.axes[0] * 0.5, self.axes[1] * 0.5)(x_local, y_local)

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
        return EggSDF(a, b_top, b_bot)(x_local, y_local)

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
        # Full dimensions, consistent with Ellipse (min full axis) and
        # Stadium (full width). Skew redistributes height between the two
        # halves but doesn't change the egg's overall thickness.
        return min(self.width.detach().item(), self.height.detach().item())


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
        return StadiumSDF(half_span, radius)(x_local, y_local)

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
