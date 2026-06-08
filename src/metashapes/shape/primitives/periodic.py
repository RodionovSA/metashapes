# metashapes/shape/primitives/periodic.py
# Shapes that span the full unit cell in one periodic direction.

from __future__ import annotations

import torch

from metashapes.shape.base import Shape
from metashapes.shape.registry import register_shape
from metashapes.shape.utils import register

__all__ = [
    "Stripe",
]


@register_shape("Stripe")
class Stripe(Shape):
    """
    An infinite stripe spanning the full unit cell along one axis.

    The stripe is unbounded along `axis` and has a finite thickness
    (`width`) in the perpendicular direction, centred at `offset`.

    Parameters:
        offset: position of the stripe centre along the *perpendicular* axis
        width:  full thickness of the stripe (must be positive)
        axis:   ``'x'`` — stripe runs along x, bounded in y  (default)
                ``'y'`` — stripe runs along y, bounded in x

    Example::

        # Horizontal stripe of width 0.3 centred on y = 0
        s = Stripe(offset=0.0, width=0.3, axis='x')

        # Vertical stripe of width 0.2 shifted to x = 0.1
        s = Stripe(offset=0.1, width=0.2, axis='y')

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
            raise ValueError("Stripe width must be positive")

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        w   = self.width
        off = self.offset

        # SDF of a 1-D band along the perpendicular coordinate t:
        #   d = |t - offset| - width/2
        t = y if self.axis == 'x' else x
        return torch.abs(t - off) - 0.5 * w

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
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
        return self.width.detach().item()
    
    def rotate(
        self,
        angle: float | torch.Tensor,
        origin: tuple[float | torch.Tensor, float | torch.Tensor] = (0.0, 0.0),
    ) -> "Shape":
        raise NotImplementedError(
        "Stripe cannot be rotated: a stripe only tiles when its axis "
        "is a lattice direction. Define the stripe in the desired "
        "orientation instead."
    )