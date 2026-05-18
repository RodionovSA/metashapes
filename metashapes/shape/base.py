#metashapes/shape/base.py
# This module defines the base Shape class, which represents symbolic 2D shapes.

from __future__ import annotations

import torch
import torch.nn as nn

from shapely.geometry.base import BaseGeometry
from .registry import SHAPE_REGISTRY


class Shape(nn.Module):
    """
    Base class for all symbolic 2D shapes.
    """

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Signed distance evaluated on torch tensors.
        x, y can be broadcastable tensors.
        Returns tensor of same broadcasted shape.
        """
        raise NotImplementedError
    
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Axis-aligned bounding box of the shape, in world coordinates.

        Returns ((xmin, ymin), (xmax, ymax)).

        Shapes that are infinite along a direction return -inf / +inf
        for that coordinate (e.g. a horizontal stripe is infinite in x).
        UnitCell uses this only to size the periodic-copy search, so an
        infinite extent simply means "one copy is enough in that
        direction" — the copies there are identical.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement bounds()")
    
    @property
    def min_feature_size(self) -> float:
        return None

    # Boolean operators for easy shape composition
    def __or__(self, other: "Shape") -> "Shape":
        from .boolean import Union
        return Union(self, other)

    def __and__(self, other: "Shape") -> "Shape":
        from .boolean import Intersection
        return Intersection(self, other)

    def __sub__(self, other: "Shape") -> "Shape":
        from .boolean import Difference
        return Difference(self, other)
    
    def union(self, other: "Shape", 
              *, smooth: bool = False, k: float | torch.Tensor = 1.0) -> "Shape":
        from .boolean import Union
        return Union(self, other, smooth=smooth, k=k)
    
    def intersection(self, other: "Shape", 
                     *, smooth: bool = False, k: float | torch.Tensor = 1.0) -> "Shape":
        from .boolean import Intersection
        return Intersection(self, other, smooth=smooth, k=k)
    
    def difference(self, other: "Shape", 
                   *, smooth: bool = False, k: float | torch.Tensor = 1.0) -> "Shape":
        from .boolean import Difference
        return Difference(self, other, smooth=smooth, k=k)
    
    # Transformations for shape manipulation
    def translate(self, 
                  dx: float | torch.Tensor = 0.0, 
                  dy: float | torch.Tensor = 0.0) -> "Shape":
        from .transforms import Translate
        return Translate(self, dx=dx, dy=dy)

    def rotate(
        self,
        angle: float | torch.Tensor,
        origin: tuple[float | torch.Tensor, float | torch.Tensor] = (0.0, 0.0),
    ) -> "Shape":
        from .transforms import Rotate
        return Rotate(self, angle=angle, origin=origin)

    def scale(
        self,
        s: float | torch.Tensor,
        origin: tuple[float | torch.Tensor, float | torch.Tensor] = (0.0, 0.0),
    ) -> "Shape":
        from .transforms import Scale
        return Scale(self, s=s, origin=origin)

    # Serialization
    def to_parametric(self) -> dict:
        data = {"type": type(self).__name__}
        # registered tensors (params + buffers)
        for name, t in (*self.named_parameters(recurse=False),
                    *self.named_buffers(recurse=False)):
            data[name] = to_plain_data(t)
        # child shapes
        for name, child in self.named_children():
            data[name] = child.to_parametric()
        return data
    
    @classmethod
    def from_parametric(cls, data: dict) -> "Shape":
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")

        shape_type = data.get("type")
        if shape_type is None:
            raise ValueError("Missing 'type' in shape parametric data")

        if cls is not Shape:
            raise NotImplementedError(f"{cls.__name__} must define from_parametric")

        try:
            shape_cls = SHAPE_REGISTRY[shape_type]
        except KeyError as e:
            raise ValueError(f"Unknown shape type: {shape_type}") from e

        return shape_cls.from_parametric(data)
    
    # Adapters to other representations
    def to_shapely(self) -> BaseGeometry:
        """
        Convert this Shape into a Shapely geometry.
        """
        from metashapes.adapters import shape_to_shapely
        shape_plain = Shape.from_parametric(self.to_parametric())
        return shape_to_shapely(shape_plain)
    

""" Helper functions for data conversion and serialization. """
def to_plain_data(x: torch.Tensor):
    """Tensor -> JSON/YAML-serializable Python value.
    Scalar tensor -> Python scalar; vector/matrix -> nested lists."""
    x = x.detach().cpu()
    return x.item() if x.ndim == 0 else x.tolist()