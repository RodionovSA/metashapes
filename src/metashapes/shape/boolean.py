# metashapes/shape/boolean.py
# This module defines boolean operations on shapes: union, intersection, difference.

from __future__ import annotations

import torch

from .base import Shape, to_plain_data
from .registry import register_shape
from .utils import register


@register_shape("Union")
class Union(Shape):
    """
    Symbolic union of two shapes.
    """
    def __init__(self, 
                 left: Shape, 
                 right: Shape, 
                 smooth: bool = False, 
                 k: float | torch.Tensor = 1.0):
        super().__init__()
        self.left = left
        self.right = right
        self.smooth = smooth
        register(self, "k", k)
    
    def sdf(self, x, y):
        d1 = self.left.sdf(x, y)
        d2 = self.right.sdf(x, y)
        if not self.smooth:
            return torch.minimum(d1, d2)
        return smooth_min_poly(d1, d2, self.k)

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        (x0, y0), (x1, y1) = self.left.bounds()
        (x2, y2), (x3, y3) = self.right.bounds()
        return (min(x0, x2), min(y0, y2)), (max(x1, x3), max(y1, y3))

    def to_parametric(self) -> dict:
        return {
            "type": "Union",
            "left": self.left.to_parametric(),
            "right": self.right.to_parametric(),
            "smooth": self.smooth,
            "k":      to_plain_data(self.k),
        }
        
    @classmethod
    def from_parametric(cls, data: dict) -> "Union":
        return cls(
            left=Shape.from_parametric(data["left"]),
            right=Shape.from_parametric(data["right"]),
            smooth=data.get("smooth", False),
            k=data.get("k", 1.0),
        )


@register_shape("Intersection")
class Intersection(Shape):
    """
    Symbolic intersection of two shapes.
    """
    def __init__(self, 
                 left: Shape, 
                 right: Shape, 
                 smooth: bool = False, 
                 k: float | torch.Tensor = 1.0):
        super().__init__()
        self.left = left
        self.right = right
        self.smooth = smooth
        register(self, "k", k)
    
    def sdf(self, x, y):
        d1 = self.left.sdf(x, y)
        d2 = self.right.sdf(x, y)
        if not self.smooth:
            return torch.maximum(d1, d2)
        return smooth_max_poly(d1, d2, self.k)

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        (x0, y0), (x1, y1) = self.left.bounds()
        (x2, y2), (x3, y3) = self.right.bounds()
        return (max(x0, x2), max(y0, y2)), (min(x1, x3), min(y1, y3))

    def to_parametric(self) -> dict:
        return {
            "type": "Intersection",
            "left": self.left.to_parametric(),
            "right": self.right.to_parametric(),
            "smooth": self.smooth,
            "k":      to_plain_data(self.k),
        }
        
    @classmethod
    def from_parametric(cls, data: dict) -> "Intersection":
        return cls(
            left=Shape.from_parametric(data["left"]),
            right=Shape.from_parametric(data["right"]),
            smooth=data.get("smooth", False),
            k=data.get("k", 1.0),
        )


@register_shape("Difference")
class Difference(Shape):
    """
    Symbolic difference of two shapes: left - right.
    """
    def __init__(self, 
                 left: Shape, 
                 right: Shape, 
                 smooth: bool = False, 
                 k: float | torch.Tensor = 1.0):
        super().__init__()
        self.left = left
        self.right = right
        self.smooth = smooth
        register(self, "k", k)
    
    def sdf(self, x, y):
        d1 = self.left.sdf(x, y)
        d2 = self.right.sdf(x, y)
        if not self.smooth:
            return torch.maximum(d1, -d2)
        return smooth_max_poly(d1, -d2, self.k)

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return self.left.bounds()

    def to_parametric(self) -> dict:
        return {
            "type": "Difference",
            "left": self.left.to_parametric(),
            "right": self.right.to_parametric(),
            "smooth": self.smooth,
            "k":      to_plain_data(self.k),
        }
        
    @classmethod
    def from_parametric(cls, data: dict) -> "Difference":
        return cls(
            left=Shape.from_parametric(data["left"]),
            right=Shape.from_parametric(data["right"]),
            smooth=data.get("smooth", False),
            k=data.get("k", 1.0),
        )
        
""" Helper functions for smooth boolean operations. 
These are based on polynomial smooth min/max functions."""

def smooth_min_poly(a: torch.Tensor, b: torch.Tensor, k: float | torch.Tensor) -> torch.Tensor:
    k = torch.as_tensor(k, dtype=a.dtype, device=a.device)
    h = torch.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return torch.lerp(b, a, h) - k * h * (1.0 - h)

def smooth_max_poly(a: torch.Tensor, b: torch.Tensor, k: float | torch.Tensor) -> torch.Tensor:
    return -smooth_min_poly(-a, -b, k)