# metashapes/shape/transforms.py
# This module defines symbolic shape transformations: translation, rotation, scaling.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch

from .base import Shape
from .registry import register_shape


@register_shape("Translate")
@dataclass(slots=True)
class Translate(Shape):
    """
    Symbolic translation of a shape.
    """
    shape: Shape
    dx: Any = 0.0
    dy: Any = 0.0
    
    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dx = torch.as_tensor(self.dx, dtype=x.dtype, device=x.device)
        dy = torch.as_tensor(self.dy, dtype=y.dtype, device=y.device)
        return self.shape.sdf(x - dx, y - dy)

    def to_parametric(self) -> dict:
        return {
            "type": "Translate",
            "shape": self.shape.to_parametric(),
            "dx": self.dx,
            "dy": self.dy,
        }
        
    @classmethod
    def from_parametric(cls, data: dict) -> "Translate":
        return cls(
            shape=Shape.from_parametric(data["shape"]),
            dx=data.get("dx", 0.0),
            dy=data.get("dy", 0.0),
        )

@register_shape("Rotate")
@dataclass(slots=True)
class Rotate(Shape):
    """
    Symbolic rotation of a shape.
    """
    shape: Shape
    angle: Any
    origin: tuple[Any, Any] = (0.0, 0.0)
    
    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        angle = torch.as_tensor(self.angle, dtype=x.dtype, device=x.device)
        ox = torch.as_tensor(self.origin[0], dtype=x.dtype, device=x.device)
        oy = torch.as_tensor(self.origin[1], dtype=y.dtype, device=y.device)

        xr = x - ox
        yr = y - oy

        angle_rad = torch.deg2rad(angle)
        c = torch.cos(angle_rad)
        s = torch.sin(angle_rad)

        # inverse rotation
        x_local =  c * xr + s * yr + ox
        y_local = -s * xr + c * yr + oy
        return self.shape.sdf(x_local, y_local)

    def to_parametric(self) -> dict:
        return {
            "type": "Rotate",
            "shape": self.shape.to_parametric(),
            "angle": self.angle,
            "origin": self.origin,
        }
        
    @classmethod
    def from_parametric(cls, data: dict) -> "Rotate":
        return cls(
            shape=Shape.from_parametric(data["shape"]),
            angle=data["angle"],
            origin=data.get("origin", "center"),
        )

@register_shape("Scale")
@dataclass(slots=True)
class Scale(Shape):
    """
    Symbolic scaling of a shape.
    """
    shape: Shape
    s: Any
    origin: tuple[Any, Any] = (0.0, 0.0)
    
    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ox = torch.as_tensor(self.origin[0], dtype=x.dtype, device=x.device)
        oy = torch.as_tensor(self.origin[1], dtype=y.dtype, device=y.device)
        s = torch.as_tensor(self.s, dtype=x.dtype, device=x.device)

        x_local = (x - ox) / s + ox
        y_local = (y - oy) / s + oy
        return s * self.shape.sdf(x_local, y_local)

    def to_parametric(self) -> dict:
        return {
            "type": "Scale",
            "shape": self.shape.to_parametric(),
            "s": self.s,
            "origin": self.origin,
        }
        
    @classmethod
    def from_parametric(cls, data: dict) -> "Scale":
        return cls(
            shape=Shape.from_parametric(data["shape"]),
            s=data["s"],
            origin=data.get("origin", "center"),
        )