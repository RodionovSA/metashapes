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