# metashapes/shape/primitives/polygons.py
# This module defines shape primitives for general polygons

from __future__ import annotations

import torch
import numpy as np

from metashapes.shape.base import Shape
from metashapes.shape.registry import register_shape
from metashapes.shape.utils import _to_local_coords, register

__all__ = [
    "RegularPolygon",
]

@register_shape("RegularPolygon")
class RegularPolygon(Shape):
    """
    Symbolic regular polygon.

    Parameters:
        center: (cx, cy)
        n: Number of sides.
        side_length: Length of each side.
        angle: Counter-clockwise rotation angle in degrees.
        corner_radius: rounding radius for corners.
    """
    def __init__(self,
                 center: torch.Tensor,
                 n: int,
                 side_length: torch.Tensor,
                 angle: torch.Tensor = 0.0,
                 corner_radius: torch.Tensor = 0.0):
        super().__init__()
        if n < 3:
            raise ValueError("n must be at least 3")
        self.n = n
        register(self, "center", center)
        register(self, "side_length", side_length)
        register(self, "angle", angle)
        register(self, "corner_radius", corner_radius)

        if torch.any(self.side_length <= 0):
            raise ValueError("side_length must be positive")
        if torch.any(self.corner_radius < 0):
            raise ValueError("corner_radius must be non-negative")

        an = torch.tensor(np.pi / n)
        rho = self.side_length / (2.0 * torch.tan(an))
        if torch.any(self.corner_radius >= rho):
            raise ValueError("corner_radius is too large")

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cx, cy = self.center[0], self.center[1]
        s  = self.side_length
        rr = self.corner_radius
        n  = self.n

        pi = torch.as_tensor(np.pi, dtype=x.dtype, device=x.device)
        angle = self.angle

        # original polygon radii
        n_t = torch.as_tensor(float(n), dtype=x.dtype, device=x.device)
        an = pi / n_t
        R = s / (2.0 * torch.sin(an))     # circumradius
        rho = R * torch.cos(an)           # apothem

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

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        cx, cy = self.center.detach().tolist()
        s = self.side_length.detach().item()
        angle = self.angle.detach().item()
        n = self.n

        an = np.pi / n
        R = s / (2.0 * np.sin(an))
        phi0 = np.pi / 2.0 + np.radians(angle)
        angles = [2.0 * np.pi * k / n + phi0 for k in range(n)]
        xs = [cx + R * np.cos(a) for a in angles]
        ys = [cy + R * np.sin(a) for a in angles]
        return (float(min(xs)), float(min(ys))), (float(max(xs)), float(max(ys)))

    @property
    def min_feature_size(self) -> float:
        pi = np.pi
        R = self.side_length.detach().item() / (2.0 * np.sin(pi / self.n))
        a = R * np.cos(pi / self.n)

        if self.n % 2 == 0:
            return 2.0 * a

        return a * (1.0 + np.cos(pi / self.n))
