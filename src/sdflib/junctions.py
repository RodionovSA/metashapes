# sdflib/junctions.py
# SDF functions for junctions

import torch
from typing import Callable

from .ovals import CircleSDF
from .quads import RectangleSDF


def _FilletSDF(r) -> Callable:
    """
    SDF of the concave-corner fillet: the square [0, r] x [0, r] minus the
    disk of radius r centered at (r, r), evaluated in corner-local
    coordinates (qx, qy) that are already offset to that corner.

    Routed through RectangleSDF/CircleSDF rather than inlined sqrt so both
    sub-SDFs keep the finfo(dtype).tiny clamp under their sqrt -- that's
    what keeps this NaN-free at r = 0 and on the square's own edges.
    """
    box = RectangleSDF(0.5 * r, 0.5 * r, 0.0)
    disk = CircleSDF(r)

    def f(qx: torch.Tensor, qy: torch.Tensor) -> torch.Tensor:
        return torch.maximum(box(qx - 0.5 * r, qy - 0.5 * r), -disk(qx - r, qy - r))
    return f


def CrossSDF(hl, hw, r_out, r_in) -> Callable:
    """
    SDF for a symmetric cross, centered at the origin.

    Parameters:
        hl:    half of the tip-to-tip size of the cross
        hw:    half of the arm width
        r_out: rounding radius for the 8 outer convex corners
        r_in:  rounding radius for the 4 inner concave corners

    Precondition: hw <= hl, r_out < hw, r_in <= hl - hw - r_out. Larger
    values silently return the distance to a collapsed/self-overlapping
    shape.
    """
    fillet = _FilletSDF(r_in)

    def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Build cross from two rectangular arms
        dh = RectangleSDF(hl, hw, r_out)(x, y)
        dv = RectangleSDF(hw, hl, r_out)(x, y)
        d_base = torch.minimum(dh, dv)

        # No `if r_in == 0: return d_base` shortcut: at r_in = 0 the patch
        # degenerates to |d| = d >= d_base everywhere, so torch.minimum
        # with d_base is already a no-op in that case. Always computing it
        # keeps this branch-free (no .item()/tensor-truthiness inside the
        # differentiable forward path).
        qx = torch.abs(x) - hw
        qy = torch.abs(y) - hw
        d_patch = fillet(qx, qy)

        return torch.minimum(d_base, d_patch)
    return f


def TShapeSDF(hl, hw, r_out, r_in) -> Callable:
    """
    SDF for a T-shape, centered at the origin.

    Parameters:
        hl:    half of the total width of the top bar and half of the
               total height of the shape
        hw:    half of the bar/stem thickness
        r_out: rounding radius for the convex outer corners
        r_in:  rounding radius for the concave inner corners

    Precondition: hw <= hl, r_out < hw, r_in <= hl - hw - r_out. Larger
    values silently return the distance to a collapsed/self-overlapping
    shape.
    """
    fillet = _FilletSDF(r_in)

    def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # base T = top bar union stem
        # top bar center at y = hl - hw
        # stem center at y = 0
        d_top = RectangleSDF(hl, hw, r_out)(x, y - (hl - hw))
        d_stem = RectangleSDF(hw, hl, r_out)(x, y)

        # cut away bottom part of horizontal bar so it becomes T, not cross
        # keep only y >= hl - 2*hw
        # Note: max(d, -cut) is a conservative bound on the true SDF near
        # the cut's concave corners, not exact -- same caveat as
        # Intersection/Difference's docstring in boolean.py.
        y_cut = hl - 2.0 * hw
        d_top_half = torch.maximum(d_top, -(y - y_cut))

        d_base = torch.minimum(d_top_half, d_stem)

        # No `if r_in == 0: return d_base` shortcut -- same reasoning as
        # CrossSDF: the patch is a provable no-op at r_in = 0, so always
        # computing it keeps this branch-free.
        # add two concave patches under the top bar
        qx = torch.abs(x) - hw
        qy = y_cut - y
        d_patch = fillet(qx, qy)

        return torch.minimum(d_base, d_patch)
    return f
