# metashapes/shape/primitives/polygons.py
# This module defines shape primitives for general polygons

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar
import torch
import numpy as np

from metashapes.shape.base import Shape, to_plain_data, to_plain_scalar
from metashapes.shape.registry import register_shape
from metashapes.shape.utils import _to_local_coords

__all__ = [
    "RegularPolygon",
]
        
@register_shape("RegularPolygon")    
@dataclass(slots=True)
class RegularPolygon(Shape):
    """
    Symbolic regular polygon.

    Parameters:
        center: (cx, cy)
        n: Number of sides.
        side_length: Length of each side.
        angle: Counter-clockwise rotation angle in degrees.
    """
    center: tuple[Any, Any]
    n: int
    side_length: Any
    angle: Any = 0.0
    corner_radius: Any = 0.0

    def __post_init__(self) -> None:
        if len(self.center) != 2:
            raise ValueError("center must have length 2")
        if self.n < 3:
            raise ValueError("n must be at least 3")
        
    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cx = torch.as_tensor(self.center[0], dtype=x.dtype, device=x.device)
        cy = torch.as_tensor(self.center[1], dtype=y.dtype, device=y.device)
        s = torch.as_tensor(self.side_length, dtype=x.dtype, device=x.device)
        rr = torch.as_tensor(self.corner_radius, dtype=x.dtype, device=x.device)
        n = int(self.n)

        if torch.any(s <= 0):
            raise ValueError("side_length must be positive")
        if torch.any(rr < 0):
            raise ValueError("corner_radius must be non-negative")

        pi = torch.as_tensor(np.pi, dtype=x.dtype, device=x.device)
        angle = torch.as_tensor(self.angle, dtype=x.dtype, device=x.device)  # degrees if _to_local_coords expects degrees

        # original polygon radii
        n_t = torch.as_tensor(float(n), dtype=x.dtype, device=x.device)
        an = pi / n_t
        R = s / (2.0 * torch.sin(an))     # circumradius
        rho = R * torch.cos(an)           # apothem

        if torch.any(rr >= rho):
            raise ValueError("corner_radius is too large")

        # inset polygon: preserve outer support lines after rounding
        rho_in = rho - rr
        R_in = rho_in / torch.cos(an)

        # local coords
        x_local, y_local = _to_local_coords(x, y, cx, cy, angle)

        # inset polygon vertices, same convention as shapely
        phi0 = pi / 2.0
        verts = []
        for k in range(n):
            th = 2.0 * pi * k / n_t + phi0
            vx = R_in * torch.cos(th)
            vy = R_in * torch.sin(th)
            verts.append((vx, vy))

        min_d2 = None
        inside = torch.ones_like(x_local, dtype=torch.bool)

        for k in range(n):
            ax, ay = verts[k]
            bx, by = verts[(k + 1) % n]

            ex = bx - ax
            ey = by - ay
            wx = x_local - ax
            wy = y_local - ay

            ee = ex * ex + ey * ey
            t = torch.clamp((wx * ex + wy * ey) / ee, 0.0, 1.0)

            px = ax + t * ex
            py = ay + t * ey

            d2 = (x_local - px) ** 2 + (y_local - py) ** 2
            min_d2 = d2 if min_d2 is None else torch.minimum(min_d2, d2)

            cross = ex * (y_local - ay) - ey * (x_local - ax)
            inside = inside & (cross >= 0)

        d_in = torch.sqrt(torch.clamp(min_d2, min=0.0))
        d_in = torch.where(inside, -d_in, d_in)

        return d_in - rr
    
    @property
    def min_feature_size(self) -> float:
        pi = np.pi
        R = to_plain_scalar(self.side_length) / (2.0 * np.sin(pi / self.n))
        a = R * np.cos(pi / self.n)

        if self.n % 2 == 0:
            return 2.0 * a

        return a * (1.0 + np.cos(pi / self.n))
        