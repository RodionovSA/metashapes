# metashapes/shape/boolean.py
# This module defines boolean operations on shapes: union, intersection, difference.

from __future__ import annotations

import torch

from .base import EMPTY_BOUNDS, Shape, is_empty_bounds, to_plain_data
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

    # No min_feature_size override: unlike Translate/Rotate/Scale (rigid or
    # uniform-scale motions that can't change a shape's narrowest width),
    # combining two shapes can create a *new*, thinner feature at the seam
    # between them (e.g. two rectangles unioned edge-to-edge can pinch to a
    # feature narrower than either operand's own min_feature_size). There is
    # no cheap, generally-correct formula from the children's values alone,
    # so this deliberately stays None ("unknown") rather than propagating a
    # possibly-too-large min() that would understate the risk.

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

    Note: `max(d1, d2)` is the standard CSG intersection SDF, but unlike
    `Union`'s `min` (which is exact), it is only a conservative *bound* on
    the true Euclidean distance -- exact near the dominant surface, looser
    near concave corners where both operands' distances are comparable.
    The sign (and hence the hard mask) is still always correct; only the
    magnitude near concave features is approximate. This matters for
    anything reading SDF magnitude rather than sign -- gap distance,
    feature-size estimates, `UnitCell.mask(soft=True)`'s sigmoid edge width.
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
        box = (max(x0, x2), max(y0, y2)), (min(x1, x3), min(y1, y3))
        # For disjoint operands the naive componentwise max/min produces an
        # inverted box (xmin > xmax and/or ymin > ymax) rather than a
        # meaningful extent -- normalize to the canonical empty sentinel so
        # downstream consumers (ring sizing, centering, further transforms)
        # never do arithmetic on a nonsense box.
        return EMPTY_BOUNDS if is_empty_bounds(box) else box

    # See Union's min_feature_size comment: intersecting/subtracting can
    # create a feature thinner than either child's own value, so this stays
    # None ("unknown") rather than propagating a possibly-too-optimistic
    # min() from the children.

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

    Note: like `Intersection`, `max(d1, -d2)` is a conservative bound on
    the true SDF magnitude near concave corners of the cut, not exact --
    see `Intersection`'s docstring for what that means for consumers of
    SDF magnitude.
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

    # See Union's min_feature_size comment: subtracting `right` can leave a
    # sliver of `left` thinner than left.min_feature_size, so this stays
    # None ("unknown") rather than propagating the child's value unchanged.

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