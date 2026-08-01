# metashapes/shape/primitives/junctions.py
# This module defines shape primitives for junctions like crosses and T-shapes.

from __future__ import annotations

import math
import torch

from metashapes.shape.base import Shape
from metashapes.shape.registry import register_shape
from metashapes.shape.utils import _sdf_rounded_box, _to_local_coords, register

__all__ = [
    "Cross",
    "TShape",
]

@register_shape("Cross")
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
    def __init__(self,
                 center: torch.Tensor,
                 length: torch.Tensor,
                 width: torch.Tensor,
                 angle: torch.Tensor = 0.0,
                 outer_corner_radius: torch.Tensor = 0.0,
                 inner_corner_radius: torch.Tensor = 0.0):
        super().__init__()
        register(self, "center", center)
        register(self, "length", length)
        register(self, "width", width)
        register(self, "angle", angle)
        register(self, "outer_corner_radius", outer_corner_radius)
        register(self, "inner_corner_radius", inner_corner_radius)

        if torch.any(self.length <= 0):
            raise ValueError("length must be positive")
        if torch.any(self.width <= 0):
            raise ValueError("width must be positive")
        # width == length (a square cross/T with no arm protrusion beyond
        # the crossbar) is intentionally allowed -- it's a valid degenerate
        # case, just not a very cross/T-like one. This is deliberately
        # looser than the outer_corner_radius check below, which does
        # forbid the equivalent equality (see screening_shape_lattice.md
        # S-21).
        if torch.any(self.width > self.length):
            raise ValueError("width must be less than or equal to length")
        if torch.any(self.outer_corner_radius < 0):
            raise ValueError("outer_corner_radius must be non-negative")
        if torch.any(self.inner_corner_radius < 0):
            raise ValueError("inner_corner_radius must be non-negative")
        if torch.any(self.outer_corner_radius >= 0.5 * self.width):
            raise ValueError("outer_corner_radius is too large")
        if torch.any(self.inner_corner_radius > (0.5 * self.length - 0.5 * self.width - self.outer_corner_radius)):
            raise ValueError("inner_corner_radius is too large")

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cx, cy = self.center[0], self.center[1]
        length = self.length
        width  = self.width
        angle  = self.angle
        ro     = self.outer_corner_radius
        ri     = self.inner_corner_radius

        bx = 0.5 * length
        by = 0.5 * width

        x_local, y_local = _to_local_coords(x, y, cx, cy, angle)

        dh = _sdf_rounded_box(x_local, y_local, bx, by, ro)
        dv = _sdf_rounded_box(x_local, y_local, by, bx, ro)
        d_base = torch.minimum(dh, dv)

        # No `if ri == 0: return d_base` shortcut: the patch computed below
        # reduces to an exact d_patch >= d_base everywhere when ri=0 (the
        # square/circle both collapse to the same single-point distance),
        # so torch.minimum(d_base, d_patch) is already a no-op in that case
        # -- verified against the old branch by dense-grid comparison
        # (max diff 0.0 across randomized parameters). Always computing it
        # keeps this branch-free (S-10: no .item()/tensor-truthiness inside
        # the differentiable forward path).
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

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        cx, cy = self.center.detach().tolist()
        length = self.length.detach().item()
        angle = self.angle.detach().item()

        # unrotated AABB is length x length, centred at (cx, cy)
        theta = math.radians(angle)
        c, s = abs(math.cos(theta)), abs(math.sin(theta))
        hw = 0.5 * length * (c + s)
        hh = 0.5 * length * (s + c)
        return (cx - hw, cy - hh), (cx + hw, cy + hh)

    @property
    def min_feature_size(self) -> float:
        return self.width.detach().item()


@register_shape("TShape")
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
    def __init__(self,
                 center: torch.Tensor,
                 length: torch.Tensor,
                 width: torch.Tensor,
                 angle: torch.Tensor = 0.0,
                 outer_corner_radius: torch.Tensor = 0.0,
                 inner_corner_radius: torch.Tensor = 0.0):
        super().__init__()
        register(self, "center", center)
        register(self, "length", length)
        register(self, "width", width)
        register(self, "angle", angle)
        register(self, "outer_corner_radius", outer_corner_radius)
        register(self, "inner_corner_radius", inner_corner_radius)

        if torch.any(self.length <= 0):
            raise ValueError("length must be positive")
        if torch.any(self.width <= 0):
            raise ValueError("width must be positive")
        # width == length (a square cross/T with no arm protrusion beyond
        # the crossbar) is intentionally allowed -- it's a valid degenerate
        # case, just not a very cross/T-like one. This is deliberately
        # looser than the outer_corner_radius check below, which does
        # forbid the equivalent equality (see screening_shape_lattice.md
        # S-21).
        if torch.any(self.width > self.length):
            raise ValueError("width must be less than or equal to length")
        if torch.any(self.outer_corner_radius < 0):
            raise ValueError("outer_corner_radius must be non-negative")
        if torch.any(self.inner_corner_radius < 0):
            raise ValueError("inner_corner_radius must be non-negative")
        if torch.any(self.outer_corner_radius >= 0.5 * self.width):
            raise ValueError("outer_corner_radius is too large")
        if torch.any(self.inner_corner_radius > (0.5 * self.length - 0.5 * self.width - self.outer_corner_radius)):
            raise ValueError("inner_corner_radius is too large")

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cx, cy = self.center[0], self.center[1]
        length = self.length
        width  = self.width
        angle  = self.angle
        ro     = self.outer_corner_radius
        ri     = self.inner_corner_radius

        bx = 0.5 * length
        by = 0.5 * width

        x_local, y_local = _to_local_coords(x, y, cx, cy, angle)

        def _patch_sdf(qx: torch.Tensor, qy: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
            d_square = _sdf_rounded_box(qx - 0.5 * r, qy - 0.5 * r, 0.5 * r, 0.5 * r)
            d_circle = torch.sqrt((qx - r) ** 2 + (qy - r) ** 2) - r
            return torch.maximum(d_square, -d_circle)  # square \ circle

        # base T = top bar ∪ stem
        # top bar center at y = bx - by
        # stem center at y = 0
        d_top = _sdf_rounded_box(x_local, y_local - (bx - by), bx, by, ro)
        d_stem = _sdf_rounded_box(x_local, y_local, by, bx, ro)

        # cut away bottom part of horizontal bar so it becomes T, not cross
        # keep only y >= bx - 2*by
        # Note: max(d, -cut) is a conservative bound on the true SDF near
        # the cut's concave corners, not exact -- same caveat as
        # Intersection/Difference's docstring in boolean.py.
        y_cut = bx - 2.0 * by
        d_top_half = torch.maximum(d_top, -(y_local - y_cut))

        d_base = torch.minimum(d_top_half, d_stem)

        # No `if ri == 0: return d_base` shortcut -- same reasoning as
        # Cross.sdf(): the patch is a provable no-op at ri=0 (verified by
        # dense-grid comparison), so always computing it keeps this
        # branch-free (S-10).
        # add two concave patches under the top bar
        qx = torch.abs(x_local) - by
        qy = (bx - 2.0 * by) - y_local
        d_patch = _patch_sdf(qx, qy, ri)

        return torch.minimum(d_base, d_patch)

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        cx, cy = self.center.detach().tolist()
        length = self.length.detach().item()
        angle = self.angle.detach().item()

        # unrotated AABB is length x length, centred at (cx, cy)
        theta = math.radians(angle)
        c, s = abs(math.cos(theta)), abs(math.sin(theta))
        hw = 0.5 * length * (c + s)
        hh = 0.5 * length * (s + c)
        return (cx - hw, cy - hh), (cx + hw, cy + hh)

    @property
    def min_feature_size(self) -> float:
        return self.width.detach().item()
