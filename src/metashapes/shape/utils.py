# metashapes/shape/utils.py

import torch
import torch.nn as nn


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
    TShape (S-09: this was three independently-maintained inline copies of
    the same Quilez rounded-box trick).
    """
    qx = torch.abs(x_local) - (hx - r)
    qy = torch.abs(y_local) - (hy - r)

    outside = torch.sqrt(torch.clamp(qx, min=0.0) ** 2 + torch.clamp(qy, min=0.0) ** 2)
    inside = torch.clamp(torch.maximum(qx, qy), max=0.0)

    return outside + inside - r

def register(module, name, value, dtype=torch.float32):
    """Register `value` on `module` under `name`.

    If `value` is an nn.Parameter it becomes an optimizable parameter;
    otherwise it is stored as a (non-gradient) buffer that still moves
    with .to() and is saved in state_dict().
    """
    if isinstance(value, nn.Parameter):
        if value.dtype != dtype:
            value = nn.Parameter(value.detach().to(dtype),
                                  requires_grad=value.requires_grad)
        setattr(module, name, value)
    else:
        module.register_buffer(name, torch.as_tensor(value, dtype=dtype))