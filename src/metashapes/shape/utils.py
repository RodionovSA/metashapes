# metashapes/shape/utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F

""" Helper functions """
def _to_local_coords(
    x: torch.Tensor,
    y: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    angle_deg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xr = x - cx
    yr = y - cy

    angle = torch.deg2rad(angle_deg)
    c = torch.cos(angle)
    s = torch.sin(angle)

    x_local =  c * xr + s * yr
    y_local = -s * xr + c * yr
    return x_local, y_local

def _sdf_rounded_box(
    x_local: torch.Tensor,
    y_local: torch.Tensor,
    hx: torch.Tensor,
    hy: torch.Tensor,
    r: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    """SDF of an axis-aligned box of half-extents (hx, hy) with corners
    rounded by radius r, evaluated at already-recentered local coordinates.

    r=0 (the default) reduces exactly to a plain sharp-cornered box -- no
    separate "unrounded" formula is needed. Shared by Rectangle, Cross, and
    TShape.
    """
    qx = torch.abs(x_local) - (hx - r)
    qy = torch.abs(y_local) - (hy - r)

    outside = torch.sqrt(torch.clamp(qx, min=0.0) ** 2 + torch.clamp(qy, min=0.0) ** 2)
    inside = torch.clamp(torch.maximum(qx, qy), max=0.0)

    return outside + inside - r

def register(module, name, value):
    """Register `value` on `module` under `name`.

    If `value` is an nn.Parameter it becomes an optimizable parameter;
    otherwise it is stored as a (non-gradient) buffer that still moves
    with .to() and is saved in state_dict().
    """
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, dtype=torch.get_default_dtype())

    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError("Only float32 and float64 dtype variables can be registered.")
    
    if isinstance(value, nn.Parameter):
        setattr(module, name, value)
    else:
        module.register_buffer(name, value)
        
def positive(parameter):
    """ Make parameter stricly positive"""
    return F.softplus(parameter)

def bounded(parameter, lo, hi):
    """ Make parameter stricly bounded """          
    return lo + (hi - lo) * torch.sigmoid(parameter)

