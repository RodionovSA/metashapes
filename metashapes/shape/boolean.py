# metashapes/shape/boolean.py
# This module defines boolean operations on shapes: union, intersection, difference.

from __future__ import annotations

from dataclasses import dataclass
import torch

from .base import Shape
from .registry import register_shape


@register_shape("Union")
@dataclass(slots=True)
class Union(Shape):
    """
    Symbolic union of two shapes.
    """
    left: Shape
    right: Shape
    smooth: bool = False
    k: float | torch.Tensor = 1.0
    
    def sdf(self, x, y):
        d1 = self.left.sdf(x, y)
        d2 = self.right.sdf(x, y)
        if not self.smooth:
            return torch.minimum(d1, d2)
        return smooth_min_poly(d1, d2, self.k)
    
    def to_parametric(self) -> dict:
        return {
            "type": "Union",
            "left": self.left.to_parametric(),
            "right": self.right.to_parametric(),
        }
        
    @classmethod
    def from_parametric(cls, data: dict) -> "Union":
        return cls(
            left=Shape.from_parametric(data["left"]),
            right=Shape.from_parametric(data["right"]),
        )


@register_shape("Intersection")
@dataclass(slots=True)
class Intersection(Shape):
    """
    Symbolic intersection of two shapes.
    """
    left: Shape
    right: Shape
    smooth: bool = False
    k: float | torch.Tensor = 1.0
    
    def sdf(self, x, y):
        d1 = self.left.sdf(x, y)
        d2 = self.right.sdf(x, y)
        if not self.smooth:
            return torch.maximum(d1, d2)
        return smooth_max_poly(d1, d2, self.k)
    
    def to_parametric(self) -> dict:
        return {
            "type": "Intersection",
            "left": self.left.to_parametric(),
            "right": self.right.to_parametric(),
        }
        
    @classmethod
    def from_parametric(cls, data: dict) -> "Intersection":
        return cls(
            left=Shape.from_parametric(data["left"]),
            right=Shape.from_parametric(data["right"]),
        )


@register_shape("Difference")
@dataclass(slots=True)
class Difference(Shape):
    """
    Symbolic difference of two shapes: left - right.
    """
    left: Shape
    right: Shape
    smooth: bool = False
    k: float | torch.Tensor = 1.0
    
    def sdf(self, x, y):
        d1 = self.left.sdf(x, y)
        d2 = self.right.sdf(x, y)
        if not self.smooth:
            return torch.maximum(d1, -d2)
        return smooth_max_poly(d1, -d2, self.k)
    
    def to_parametric(self) -> dict:
        return {
            "type": "Difference",
            "left": self.left.to_parametric(),
            "right": self.right.to_parametric(),
        }
        
    @classmethod
    def from_parametric(cls, data: dict) -> "Difference":
        return cls(
            left=Shape.from_parametric(data["left"]),
            right=Shape.from_parametric(data["right"]),
        )
        
""" Helper functions for smooth boolean operations. 
These are based on polynomial smooth min/max functions."""

def smooth_min_poly(a: torch.Tensor, b: torch.Tensor, k: float | torch.Tensor) -> torch.Tensor:
    k = torch.as_tensor(k, dtype=a.dtype, device=a.device)
    h = torch.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return torch.lerp(b, a, h) - k * h * (1.0 - h)

def smooth_max_poly(a: torch.Tensor, b: torch.Tensor, k: float | torch.Tensor) -> torch.Tensor:
    return -smooth_min_poly(-a, -b, k)