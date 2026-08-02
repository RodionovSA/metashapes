# metashapes/shape/primitives/stripes.py
# Shapes that span the full unit cell, infinite along one axis.

from __future__ import annotations

import torch

from metashapes.shape.base import Shape
from metashapes.shape.registry import register_shape
from metashapes.shape.utils import register
from sdflib.stripes import BarSDF

__all__ = [
    "Bar",
]


@register_shape("Bar")
class Bar(Shape):
    """
    An infinite bar spanning the full unit cell along one axis.

    The bar is unbounded along `axis` and has a finite thickness
    (`width`) in the perpendicular direction, centred at `offset`.

    Parameters:
        offset: position of the bar centre along the *perpendicular* axis
        width:  full thickness of the bar (must be positive)
        axis:   ``'x'`` — bar runs along x, bounded in y  (default)
                ``'y'`` — bar runs along y, bounded in x

    Example::

        # Horizontal bar of width 0.3 centred on y = 0
        s = Bar(offset=0.0, width=0.3, axis='x')

        # Vertical bar of width 0.2 shifted to x = 0.1
        s = Bar(offset=0.1, width=0.2, axis='y')

    """

    def __init__(self,
                 offset: torch.Tensor,
                 width: torch.Tensor,
                 axis: str = 'x'):
        super().__init__()
        if axis not in ('x', 'y'):
            raise ValueError("axis must be 'x' or 'y'")
        self.axis = axis
        register(self, "offset", offset)
        register(self, "width", width)

        if torch.any(self.width <= 0):
            raise ValueError("Bar width must be positive")
        
    @torch.no_grad()
    def _project(self) -> None:
        """Snap size/corner_radius back into their valid ranges in place."""
        self.width.clamp_(min=self._MIN_SIZE)

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._project()
        w   = self.width
        off_x = self.offset if self.axis == "y" else 0.0
        off_y = self.offset if self.axis == "x" else 0.0

        return BarSDF(w, self.axis)(x - off_x, y - off_y)

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        self._project()
        off = self.offset.detach().item()
        w = self.width.detach().item()
        inf = float('inf')
        if self.axis == 'x':
            return (-inf, off - w / 2.0), (inf, off + w / 2.0)
        return (off - w / 2.0, -inf), (off + w / 2.0, inf)

    def to_parametric(self) -> dict:
        d = super().to_parametric()
        d["axis"] = self.axis
        return d

    @property
    def min_feature_size(self) -> float:
        self._project()
        return self.width.detach().item()

    def rotate(
        self,
        angle: float | torch.Tensor,
        origin: tuple[float | torch.Tensor, float | torch.Tensor] = (0.0, 0.0),
    ) -> "Shape":
        raise NotImplementedError(
        "Bar cannot be rotated: a bar only tiles when its axis "
        "is a lattice direction. Define the bar in the desired "
        "orientation instead."
    )
