# sdflib/ovals.py
# SDF functions for ovals

import torch
from typing import Callable

def CircleSDF(r) -> Callable:
    def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        q = x*x + y*y
        p = torch.sqrt(q.clamp_min(torch.finfo(q.dtype).tiny)) # safe grads through the origin
        return p - r
    return f