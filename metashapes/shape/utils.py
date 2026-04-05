# metashapes/shape/utils.py

import torch


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