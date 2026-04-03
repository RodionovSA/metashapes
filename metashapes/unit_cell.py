# metashapes/unit_cell.py
# Definition of UnitCell, representing the unit cell of a periodic structure, containing Shape and Canvas information.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch
import numpy as np
from shapely.geometry.base import BaseGeometry

from .canvas import Canvas
from .shape.base import Shape
from .shape.primitives import Rectangle


@dataclass(slots=True)
class UnitCell:
    """
    Unit cell of a periodic structure, containing Shape and Canvas information.
    - `shape` defines the symbolic pattern
    - `canvas` defines the unit-cell region in x-y
    - `inverted=False`: shape = material region
    - `inverted=True`:  shape = void region inside the unit cell
    
    Note:
    - Any part of the shape extending beyond the unit-cell boundaries is clipped
    to the unit cell before periodic repetition.
    - Zero coordinates of the canvas are at the center of the unit cell.
    """
    shape: Shape
    canvas: Canvas
    inverted: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.shape, Shape):
            raise TypeError("shape must be a Shape")
        if not isinstance(self.canvas, Canvas):
            raise TypeError("canvas must be a Canvas")

    @property
    def unit_cell_shape(self) -> Shape:
        """
        Rectangle corresponding to the usable inner canvas region
        (canvas area excluding spacing).
        """
        x0, y0, x1, y1 = self.canvas.inner_bounds
        return Rectangle(
            center=((x0 + x1) / 2, (y0 + y1) / 2),
            size=(x1 - x0, y1 - y0),
        )

    @property
    def filled_region(self) -> Shape:
        """
        Symbolic material region inside the unit cell.
        """
        cell = self.unit_cell_shape
        if self.inverted:
            return cell - self.shape
        return cell & self.shape

    @property
    def void_region(self) -> Shape:
        """
        Symbolic complementary empty region inside the unit cell.
        """
        cell = self.unit_cell_shape
        if self.inverted:
            return cell & self.shape
        return cell - self.shape
    
    def _wrap_xy(self, x: Any, y: Any) -> tuple[Any, Any]:
        x0, y0, x1, y1 = self.canvas.inner_bounds
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        Lx = x1 - x0
        Ly = y1 - y0

        xp = ((x - cx + 0.5 * Lx) % Lx) - 0.5 * Lx + cx
        yp = ((y - cy + 0.5 * Ly) % Ly) - 0.5 * Ly + cy
        return xp, yp

    def sdf(self, x, y):
        xp, yp = self._wrap_xy(x, y)

        x0, y0, x1, y1 = self.canvas.inner_bounds
        Lx = x1 - x0
        Ly = y1 - y0

        ds = [
            self.filled_region.sdf(xp - i * Lx, yp - j * Ly)
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
        ]

        d = torch.stack(ds, dim=0).amin(dim=0)
        return d

    # Serialization 
    def to_parametric(self) -> dict:
        """
        Serialize the UnitCell into a dictionary of defining parameters.
        """
        return {
            "shape": self.shape.to_parametric(),
            "canvas": self.canvas.to_parametric(),
            "inverted": self.inverted,
        }

    @classmethod
    def from_parametric(cls, data: dict) -> UnitCell:
        """
        Reconstruct a UnitCell from a dictionary created by `to_parametric`.
        """
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")

        try:
            shape_data = data["shape"]
            canvas_data = data["canvas"]
            inverted = data["inverted"]
        except KeyError as e:
            raise ValueError(f"Missing required field: {e.args[0]}") from e

        shape = Shape.from_parametric(shape_data)
        canvas = Canvas.from_parametric(canvas_data)

        return cls(
            shape=shape,
            canvas=canvas,
            inverted=bool(inverted),
        )
        
    # Adapters to other representations
    def mask(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        soft: bool = False,
        softness: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Rasterize UnitCell into a torch mask.

        Parameters:
            dtype: Torch dtype of the output mask.
            device: Torch device.
            soft: If True, return a soft mask instead of a hard binary mask.
            softness: Edge smoothing scale in world units. If None, defaults to one pixel.
        
        Returns:
            Tensor of shape (H, W).
        
        """
        x, y = self.canvas.grid(dtype=dtype, device=device)
        d = self.sdf(x, y)

        if not soft:
            return (d <= 0).to(dtype)

        if softness is None:
            softness = min(self.canvas.dx, self.canvas.dy)

        softness = torch.as_tensor(softness, dtype=d.dtype, device=d.device)
        return torch.sigmoid(-d / softness)
    
    def mask_numpy(self, 
                   *, 
                   dtype=np.float32, 
                   soft: bool = False, 
                   softness: float | None = None) -> np.ndarray:
        """
        Rasterize UnitCell into a numpy array.

        Parameters:
            dtype: The desired data type of the output array.
            soft: If True, return a soft mask instead of a hard binary mask.
            softness: The scale of the softness in world units. If None, defaults to one pixel.
        """
        mask = self.mask(
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
        shape_plain = Shape.from_parametric(self.filled_region.to_parametric())
        return shape_to_shapely(shape_plain)