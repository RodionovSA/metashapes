# metashapes/shape/primitives/junctions.py
# This module defines shape primitives for junctions like crosses and T-shapes.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar
import torch
import numpy as np

from metashapes.shape.base import Shape, to_plain_data, to_plain_scalar
from metashapes.shape.registry import register_shape
from metashapes.shape.utils import _to_local_coords

__all__ = [
    "Cross",
    "TShape",
]

@register_shape("Cross")
@dataclass(slots=True)
class Cross(Shape):
    """
    Symbolic symmetric cross.

    Parameters:
        center: (cx, cy)
        length: full tip-to-tip size of the cross
        width: full arm width
        angle: counter-clockwise rotation angle in degrees
        outer_corner_radius:
            rounding radius for the 8 outer convex corners
        inner_corner_radius:
            rounding radius for the 4 inner concave corners
    """
    center: tuple[Any, Any]
    length: Any
    width: Any
    angle: Any = 0.0
    outer_corner_radius: Any = 0.0
    inner_corner_radius: Any = 0.0

    def __post_init__(self) -> None:
        if len(self.center) != 2:
            raise ValueError("center must have length 2")

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cx = torch.as_tensor(self.center[0], dtype=x.dtype, device=x.device)
        cy = torch.as_tensor(self.center[1], dtype=y.dtype, device=y.device)
        length = torch.as_tensor(self.length, dtype=x.dtype, device=x.device)
        width = torch.as_tensor(self.width, dtype=x.dtype, device=x.device)
        angle = torch.as_tensor(self.angle, dtype=x.dtype, device=x.device)
        ro = torch.as_tensor(self.outer_corner_radius, dtype=x.dtype, device=x.device)
        ri = torch.as_tensor(self.inner_corner_radius, dtype=x.dtype, device=x.device)

        if torch.any(length <= 0):
            raise ValueError("length must be positive")
        if torch.any(width <= 0):
            raise ValueError("width must be positive")
        if torch.any(width > length):
            raise ValueError("width must be less than or equal to length")
        if torch.any(ro < 0):
            raise ValueError("outer_corner_radius must be non-negative")
        if torch.any(ri < 0):
            raise ValueError("inner_corner_radius must be non-negative")

        bx = 0.5 * length
        by = 0.5 * width

        if torch.any(ro >= by):
            raise ValueError("outer_corner_radius is too large")
        if torch.any(ri > (bx - by - ro)):
            raise ValueError("inner_corner_radius is too large")

        x_local, y_local = _to_local_coords(x, y, cx, cy, angle)

        def _sd_rounded_box(px: torch.Tensor,
                            py: torch.Tensor,
                            hx: torch.Tensor,
                            hy: torch.Tensor,
                            rr: torch.Tensor) -> torch.Tensor:
            qx = torch.abs(px) - (hx - rr)
            qy = torch.abs(py) - (hy - rr)

            qx_pos = torch.clamp(qx, min=0.0)
            qy_pos = torch.clamp(qy, min=0.0)

            outside = torch.sqrt(qx_pos * qx_pos + qy_pos * qy_pos)
            inside = torch.clamp(torch.maximum(qx, qy), max=0.0)

            return outside + inside - rr

        dh = _sd_rounded_box(x_local, y_local, bx, by, ro)
        dv = _sd_rounded_box(x_local, y_local, by, bx, ro)
        d_base = torch.minimum(dh, dv)

        if torch.all(ri == 0):
            return d_base

        u = torch.abs(x_local)
        v = torch.abs(y_local)

        qx = u - by
        qy = v - by

        # square [0, ri] x [0, ri]
        sx = qx - 0.5 * ri
        sy = qy - 0.5 * ri
        ax = torch.abs(sx) - 0.5 * ri
        ay = torch.abs(sy) - 0.5 * ri

        ax_pos = torch.clamp(ax, min=0.0)
        ay_pos = torch.clamp(ay, min=0.0)
        d_square = torch.sqrt(ax_pos * ax_pos + ay_pos * ay_pos) + torch.clamp(torch.maximum(ax, ay), max=0.0)

        # circle centered at (ri, ri)
        d_circle = torch.sqrt((qx - ri) ** 2 + (qy - ri) ** 2) - ri

        # correct patch: square \ circle
        d_patch = torch.maximum(d_square, -d_circle)

        # add patch to the base cross
        return torch.minimum(d_base, d_patch)

    @property
    def min_feature_size(self) -> float:
        return to_plain_scalar(self.width)

@register_shape("TShape")
@dataclass(slots=True)
class TShape(Shape):
    """
    Symbolic T-shape.

    Parameters:
        center: (cx, cy)
        length: full total width of the top bar and full total height of the shape
        width: full bar/stem thickness
        angle: counter-clockwise rotation angle in degrees
        outer_corner_radius: rounding radius for convex outer corners
        inner_corner_radius: rounding radius for concave inner corners
    """
    center: tuple[Any, Any]
    length: Any
    width: Any
    angle: Any = 0.0
    outer_corner_radius: Any = 0.0
    inner_corner_radius: Any = 0.0

    def __post_init__(self) -> None:
        if len(self.center) != 2:
            raise ValueError("center must have length 2")

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cx = torch.as_tensor(self.center[0], dtype=x.dtype, device=x.device)
        cy = torch.as_tensor(self.center[1], dtype=y.dtype, device=y.device)
        length = torch.as_tensor(self.length, dtype=x.dtype, device=x.device)
        width = torch.as_tensor(self.width, dtype=x.dtype, device=x.device)
        angle = torch.as_tensor(self.angle, dtype=x.dtype, device=x.device)
        ro = torch.as_tensor(self.outer_corner_radius, dtype=x.dtype, device=x.device)
        ri = torch.as_tensor(self.inner_corner_radius, dtype=x.dtype, device=x.device)

        if torch.any(length <= 0):
            raise ValueError("length must be positive")
        if torch.any(width <= 0):
            raise ValueError("width must be positive")
        if torch.any(width > length):
            raise ValueError("width must be less than or equal to length")
        if torch.any(ro < 0):
            raise ValueError("outer_corner_radius must be non-negative")
        if torch.any(ri < 0):
            raise ValueError("inner_corner_radius must be non-negative")

        bx = 0.5 * length
        by = 0.5 * width

        if torch.any(ro >= by):
            raise ValueError("outer_corner_radius is too large")
        if torch.any(ri > (bx - by - ro)):
            raise ValueError("inner_corner_radius is too large")

        x_local, y_local = _to_local_coords(x, y, cx, cy, angle)

        def _sd_box(px: torch.Tensor, py: torch.Tensor, hx: torch.Tensor, hy: torch.Tensor) -> torch.Tensor:
            qx = torch.abs(px) - hx
            qy = torch.abs(py) - hy
            ox = torch.clamp(qx, min=0.0)
            oy = torch.clamp(qy, min=0.0)
            return torch.sqrt(ox * ox + oy * oy) + torch.clamp(torch.maximum(qx, qy), max=0.0)

        def _sd_rounded_box(px: torch.Tensor, py: torch.Tensor, hx: torch.Tensor, hy: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
            return _sd_box(px, py, hx - rr, hy - rr) - rr

        def _patch_sdf(qx: torch.Tensor, qy: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
            d_square = _sd_box(qx - 0.5 * r, qy - 0.5 * r, 0.5 * r, 0.5 * r)
            d_circle = torch.sqrt((qx - r) ** 2 + (qy - r) ** 2) - r
            return torch.maximum(d_square, -d_circle)  # square \ circle

        # base T = top bar ∪ stem
        # top bar center at y = bx - by
        # stem center at y = 0
        d_top = _sd_rounded_box(x_local, y_local - (bx - by), bx, by, ro)
        d_stem = _sd_rounded_box(x_local, y_local, by, bx, ro)

        # cut away bottom part of horizontal bar so it becomes T, not cross
        # keep only y >= bx - 2*by
        y_cut = bx - 2.0 * by
        d_top_half = torch.maximum(d_top, -(y_local - y_cut))

        d_base = torch.minimum(d_top_half, d_stem)

        if torch.all(ri == 0):
            return d_base

        # add two concave patches under the top bar
        qx = torch.abs(x_local) - by
        qy = (bx - 2.0 * by) - y_local
        d_patch = _patch_sdf(qx, qy, ri)

        return torch.minimum(d_base, d_patch)

    @property
    def min_feature_size(self) -> float:
        return to_plain_scalar(self.width)
        
