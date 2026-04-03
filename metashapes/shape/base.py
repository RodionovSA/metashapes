#metashapes/shape/base.py
# This module defines the base Shape class, which represents symbolic 2D shapes.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numbers
import numpy as np
import torch

from shapely.geometry.base import BaseGeometry
from metashapes.canvas import Canvas
from .registry import SHAPE_REGISTRY


class Shape(ABC):
    """
    Base class for all symbolic 2D shapes.

    A Shape stores a geometric recipe, not resolved polygon data.
    It can later be evaluated by different backends.
    """
    
    @abstractmethod
    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Signed distance evaluated on torch tensors.
        x, y can be broadcastable tensors.
        Returns tensor of same broadcasted shape.
        """
        raise NotImplementedError

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
    def translate(self, dx: Any = 0.0, dy: Any = 0.0) -> "Shape":
        from .transforms import Translate
        return Translate(self, dx=dx, dy=dy)

    def rotate(
        self,
        angle: Any,
        origin: tuple[Any, Any] = (0.0, 0.0),
    ) -> "Shape":
        from .transforms import Rotate
        return Rotate(self, angle=angle, origin=origin)

    def scale(
        self,
        s: Any,
        origin: tuple[Any, Any] = (0.0, 0.0),
    ) -> "Shape":
        from .transforms import Scale
        return Scale(self, s=s, origin=origin)

    # Serialization 
    @abstractmethod
    def to_parametric(self) -> dict:
        """
        Serialize the symbolic shape into a dictionary.
        """
        raise NotImplementedError
    
    @classmethod
    def from_parametric(cls, data: dict) -> "Shape":
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")

        shape_type = data.get("type")
        if shape_type is None:
            raise ValueError("Missing 'type' in shape parametric data")
        
        try:
            shape_cls = SHAPE_REGISTRY[shape_type]
        except KeyError as e:
            raise ValueError(f"Unknown shape type: {shape_type}") from e

        return shape_cls.from_parametric(data)
    
    # Adapters to other representations
    def mask(
        self,
        canvas: Canvas,
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        soft: bool = False,
        softness: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Rasterize this Shape into a torch mask on the given canvas.

        Parameters:
            canvas: The canvas to rasterize onto.
            dtype: Torch dtype of the output mask.
            device: Torch device.
            soft: If True, return a soft mask instead of a hard binary mask.
            softness: Edge smoothing scale in world units. If None, defaults to one pixel.
        
        Returns:
            Tensor of shape (H, W).
        
        """
        x, y = canvas.grid(dtype=dtype, device=device)
        d = self.sdf(x, y)

        if not soft:
            return (d <= 0).to(dtype)

        if softness is None:
            softness = min(canvas.dx, canvas.dy)

        softness = torch.as_tensor(softness, dtype=d.dtype, device=d.device)
        return torch.sigmoid(-d / softness)
    
    def mask_numpy(self, 
                   canvas: Canvas, 
                   *, 
                   dtype=np.float32, 
                   soft: bool = False, 
                   softness: float | None = None) -> np.ndarray:
        """
        Rasterize this Shape into a NumPy array of the given canvas size.

        Parameters:
            canvas: The canvas to rasterize onto.
            dtype: The desired data type of the output array.
            soft: If True, return a soft mask instead of a hard binary mask.
            softness: The scale of the softness in world units. If None, defaults to one pixel.
        """
        mask = self.mask(
            canvas,
            dtype=torch.float32,
            device="cpu",
            soft=soft,
            softness=softness,
        )
        return mask.detach().cpu().numpy().astype(dtype, copy=False)
        
    def to_shapely(self) -> BaseGeometry:
        """
        Convert this Shape into a Shapely geometry.
        """
        from metashapes.adapters import shape_to_shapely
        shape_plain = Shape.from_parametric(self.to_parametric())
        return shape_to_shapely(shape_plain)
    

""" Helper functions for data conversion and serialization. """
def to_plain_data(x: Any):
    """
    Convert values to plain Python-serializable data.

    Rules:
        - Python scalars stay unchanged
        - torch scalar tensors -> Python scalar
        - torch vectors/matrices -> nested Python lists
        - NumPy scalars -> Python scalar
        - NumPy arrays -> nested Python lists
        - tuples/lists -> recursively converted, preserving container type
    """
    if isinstance(x, numbers.Number):
        return x

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.ndim == 0:
            return x.item()
        return x.tolist()

    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        return x.tolist()

    if isinstance(x, tuple):
        return tuple(to_plain_data(v) for v in x)

    if isinstance(x, list):
        return [to_plain_data(v) for v in x]

    if isinstance(x, dict):
        return {k: to_plain_data(v) for k, v in x.items()}

    return x

def to_plain_scalar(x: Any):
    """
    Convert scalar-like values to a plain Python scalar.

    Accepts:
        - Python/NumPy scalars
        - 0D torch tensors
        - 1-element tensors / arrays / lists / tuples
    """
    x = to_plain_data(x)

    while isinstance(x, (list, tuple)):
        if len(x) != 1:
            raise ValueError(f"Expected scalar-like value, got {x}")
        x = x[0]

    return x