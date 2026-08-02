# sdflib/stripes.py
# SDF functions for stripes

import torch
from typing import Callable

def BarSDF(w, axis: str) -> Callable:
    def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # SDF of a 1-D band along the perpendicular coordinate t:
        #   d = |t - offset| - width/2
        t = y if axis == 'x' else x
        return torch.abs(t) - 0.5 * w
    return f

