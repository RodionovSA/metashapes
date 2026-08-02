# metashapes/shape/primitives/junctions.py
# This module defines shape primitives for junctions like crosses and T-shapes.

from __future__ import annotations

import math
import torch

from metashapes.shape.base import Shape
from metashapes.shape.registry import register_shape
from metashapes.shape.utils import _to_local_coords, register
from sdflib.junctions import CrossSDF, TShapeSDF

__all__ = [
    "Cross",
    "TShape",
]

@register_shape("Cross")
class Cross(Shape):
    """
    Symbolic symmetric cross.

    Parameters:
        center: (cx, cy)
        length: full tip-to-tip size of the cross
        width: full arm width
        angle: counter-clockwise rotation angle in degrees
        outer_corner_radius:
            rounding radius for the 8 outer convex corners
        inner_corner_radius:
            rounding radius for the 4 inner concave corners
    """
    def __init__(self,
                 center: torch.Tensor,
                 length: torch.Tensor,
                 width: torch.Tensor,
                 angle: torch.Tensor = 0.0,
                 outer_corner_radius: torch.Tensor = 0.0,
                 inner_corner_radius: torch.Tensor = 0.0):
        super().__init__()
        register(self, "center", center)
        register(self, "length", length)
        register(self, "width", width)
        register(self, "angle", angle)
        register(self, "outer_corner_radius", outer_corner_radius)
        register(self, "inner_corner_radius", inner_corner_radius)

        if torch.any(self.length <= 0):
            raise ValueError("length must be positive")
        if torch.any(self.width <= 0):
            raise ValueError("width must be positive")
        if torch.any(self.width > self.length):
            raise ValueError("width must be less than or equal to length")
        if torch.any(self.outer_corner_radius < 0):
            raise ValueError("outer_corner_radius must be non-negative")
        if torch.any(self.inner_corner_radius < 0):
            raise ValueError("inner_corner_radius must be non-negative")
        if torch.any(self.outer_corner_radius >= 0.5 * self.width):
            raise ValueError("outer_corner_radius is too large")
        if torch.any(self.inner_corner_radius > (0.5 * self.length - 0.5 * self.width - self.outer_corner_radius)):
            raise ValueError("inner_corner_radius is too large")

    @torch.no_grad()
    def _project(self) -> None:
        """Snap length/width/outer_corner_radius/inner_corner_radius back
        into their valid ranges in place."""
        self.length.clamp_(min=self._MIN_SIZE)
        self.width.clamp_(min=self._MIN_SIZE, max=self.length.detach())
        self.outer_corner_radius.clamp_(min=0.0, max=0.5 * self.width * self._MAX_RADIUS_FRACTION)
        self.inner_corner_radius.clamp_(
            min=0.0,
            max=(0.5 * self.length - 0.5 * self.width - self.outer_corner_radius).clamp_min(0.0),
        )

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._project()
        cx, cy = self.center[0], self.center[1]
        angle  = self.angle

        bx = 0.5 * self.length
        by = 0.5 * self.width

        x_local, y_local = _to_local_coords(x, y, cx, cy, angle)

        return CrossSDF(bx, by, self.outer_corner_radius, self.inner_corner_radius)(x_local, y_local)

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        self._project()
        cx, cy = self.center.detach().tolist()
        length = self.length.detach().item()
        angle = self.angle.detach().item()

        # unrotated AABB is length x length, centred at (cx, cy)
        theta = math.radians(angle)
        c, s = abs(math.cos(theta)), abs(math.sin(theta))
        hw = 0.5 * length * (c + s)
        hh = 0.5 * length * (s + c)
        return (cx - hw, cy - hh), (cx + hw, cy + hh)

    @property
    def min_feature_size(self) -> float:
        self._project()
        return self.width.detach().item()


@register_shape("TShape")
class TShape(Shape):
    """
    Symbolic T-shape.

    Parameters:
        center: (cx, cy)
        length: full total width of the top bar and full total height of the shape
        width: full bar/stem thickness
        angle: counter-clockwise rotation angle in degrees
        outer_corner_radius: rounding radius for convex outer corners
        inner_corner_radius: rounding radius for concave inner corners
    """
    def __init__(self,
                 center: torch.Tensor,
                 length: torch.Tensor,
                 width: torch.Tensor,
                 angle: torch.Tensor = 0.0,
                 outer_corner_radius: torch.Tensor = 0.0,
                 inner_corner_radius: torch.Tensor = 0.0):
        super().__init__()
        register(self, "center", center)
        register(self, "length", length)
        register(self, "width", width)
        register(self, "angle", angle)
        register(self, "outer_corner_radius", outer_corner_radius)
        register(self, "inner_corner_radius", inner_corner_radius)

        if torch.any(self.length <= 0):
            raise ValueError("length must be positive")
        if torch.any(self.width <= 0):
            raise ValueError("width must be positive")
        if torch.any(self.width > self.length):
            raise ValueError("width must be less than or equal to length")
        if torch.any(self.outer_corner_radius < 0):
            raise ValueError("outer_corner_radius must be non-negative")
        if torch.any(self.inner_corner_radius < 0):
            raise ValueError("inner_corner_radius must be non-negative")
        if torch.any(self.outer_corner_radius >= 0.5 * self.width):
            raise ValueError("outer_corner_radius is too large")
        if torch.any(self.inner_corner_radius > (0.5 * self.length - 0.5 * self.width - self.outer_corner_radius)):
            raise ValueError("inner_corner_radius is too large")

    @torch.no_grad()
    def _project(self) -> None:
        """Snap length/width/outer_corner_radius/inner_corner_radius back
        into their valid ranges in place."""
        self.length.clamp_(min=self._MIN_SIZE)
        self.width.clamp_(min=self._MIN_SIZE, max=self.length.detach())
        self.outer_corner_radius.clamp_(min=0.0, max=0.5 * self.width * self._MAX_RADIUS_FRACTION)
        self.inner_corner_radius.clamp_(
            min=0.0,
            max=(0.5 * self.length - 0.5 * self.width - self.outer_corner_radius).clamp_min(0.0),
        )

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._project()
        cx, cy = self.center[0], self.center[1]
        angle  = self.angle

        bx = 0.5 * self.length
        by = 0.5 * self.width

        x_local, y_local = _to_local_coords(x, y, cx, cy, angle)

        return TShapeSDF(bx, by, self.outer_corner_radius, self.inner_corner_radius)(x_local, y_local)

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        self._project()
        cx, cy = self.center.detach().tolist()
        length = self.length.detach().item()
        angle = self.angle.detach().item()

        # unrotated AABB is length x length, centred at (cx, cy)
        theta = math.radians(angle)
        c, s = abs(math.cos(theta)), abs(math.sin(theta))
        hw = 0.5 * length * (c + s)
        hh = 0.5 * length * (s + c)
        return (cx - hw, cy - hh), (cx + hw, cy + hh)

    @property
    def min_feature_size(self) -> float:
        self._project()
        return self.width.detach().item()
