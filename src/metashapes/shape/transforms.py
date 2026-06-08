# metashapes/shape/transforms.py
# This module defines symbolic shape transformations: translation, rotation, scaling.

from __future__ import annotations

import math
from typing import Any, Tuple
import torch

from .base import Shape, to_plain_data
from .registry import register_shape
from .utils import register


@register_shape("Translate")
class Translate(Shape):
    """
    Symbolic translation of a shape.
    """
    def __init__(self, 
                 shape: Shape, 
                 dx: float | torch.Tensor = 0.0,
                 dy: float | torch.Tensor = 0.0):
        super().__init__()
        self.shape = shape
        register(self, "dx", dx)
        register(self, "dy", dy)
    
    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.shape.sdf(x - self.dx, y - self.dy)

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        dx = self.dx.detach().item()
        dy = self.dy.detach().item()
        (x0, y0), (x1, y1) = self.shape.bounds()
        return (x0 + dx, y0 + dy), (x1 + dx, y1 + dy)

    def to_parametric(self) -> dict:
        return {
            "type": "Translate",
            "shape": self.shape.to_parametric(),
            "dx": to_plain_data(self.dx),
            "dy": to_plain_data(self.dy),
        }
        
    @classmethod
    def from_parametric(cls, data: dict) -> "Translate":
        return cls(
            shape=Shape.from_parametric(data["shape"]),
            dx=data.get("dx", 0.0),
            dy=data.get("dy", 0.0),
        )

@register_shape("Rotate")
class Rotate(Shape):
    """
    Symbolic rotation of a shape.
    """
    def __init__(self,
                 shape: Shape,
                 angle: float | torch.Tensor,
                 origin: Tuple[float | torch.Tensor,
                               float | torch.Tensor]  = (0.0, 0.0)):
        super().__init__()
        (x0, y0), (x1, y1) = shape.bounds()
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
            raise NotImplementedError(
                "Cannot rotate a shape with infinite spatial extent "
                "(e.g. a Stripe or any boolean composition containing one). "
                "Define infinite shapes in their final orientation before composing."
            )
        self.shape = shape
        register(self, "angle", angle)
        register(self, "origin", origin)

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        (x0, y0), (x1, y1) = self.shape.bounds()
        angle = self.angle.detach().item()
        ox, oy = self.origin.detach().tolist()

        theta = math.radians(angle)
        c, s = math.cos(theta), math.sin(theta)
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        xs = [ox + c * (x - ox) - s * (y - oy) for x, y in corners]
        ys = [oy + s * (x - ox) + c * (y - oy) for x, y in corners]
        return (min(xs), min(ys)), (max(xs), max(ys))

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xr = x - self.origin[0]
        yr = y - self.origin[1]

        angle_rad = torch.deg2rad(self.angle)
        c = torch.cos(angle_rad)
        s = torch.sin(angle_rad)

        # inverse rotation
        x_local =  c * xr + s * yr + self.origin[0]
        y_local = -s * xr + c * yr + self.origin[1]
        return self.shape.sdf(x_local, y_local)

    def to_parametric(self) -> dict:
        return {
            "type": "Rotate",
            "shape": self.shape.to_parametric(),
            "angle": to_plain_data(self.angle),
            "origin": to_plain_data(self.origin),
        }
        
    @classmethod
    def from_parametric(cls, data: dict) -> "Rotate":
        return cls(
            shape=Shape.from_parametric(data["shape"]),
            angle=data["angle"],
            origin=data.get("origin", "center"),
        )

@register_shape("Scale")
class Scale(Shape):
    """
    Symbolic scaling of a shape.
    """
    def __init__(self, 
                 shape: Shape, 
                 s: float | torch.Tensor,
                 origin: Tuple[float | torch.Tensor,
                               float | torch.Tensor]  = (0.0, 0.0)):
        super().__init__()
        self.shape = shape
        register(self, "s", s)
        register(self, "origin", origin)
        
        if torch.any(self.s <= 0):
            raise ValueError("scale factor s must be positive")
            
    
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        s = self.s.detach().item()
        ox, oy = self.origin.detach().tolist()
        (x0, y0), (x1, y1) = self.shape.bounds()
        xs = [ox + s * (x0 - ox), ox + s * (x1 - ox)]
        ys = [oy + s * (y0 - oy), oy + s * (y1 - oy)]
        return (min(xs), min(ys)), (max(xs), max(ys))

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_local = (x - self.origin[0]) / self.s + self.origin[0]
        y_local = (y - self.origin[1]) / self.s + self.origin[1]
        return self.s * self.shape.sdf(x_local, y_local)

    def to_parametric(self) -> dict:
        return {
            "type": "Scale",
            "shape": self.shape.to_parametric(),
            "s": to_plain_data(self.s),
            "origin": to_plain_data(self.origin),
        }
        
    @classmethod
    def from_parametric(cls, data: dict) -> "Scale":
        return cls(
            shape=Shape.from_parametric(data["shape"]),
            s=data["s"],
            origin=data.get("origin", "center"),
        )