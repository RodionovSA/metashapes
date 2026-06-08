# metashapes/adapters/shapely/helpers.py
# Helper functions for Shapely adapter.

from __future__ import annotations
from typing import Literal

import torch

from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry


def as_float(t) -> float:
    """Safely extract a scalar tensor (any device/grad state) to Python float.

    Calls ``.detach().cpu()`` before ``.item()`` so the function is safe
    for CUDA tensors and for ``nn.Parameter`` values inside autograd graphs.
    Plain Python numbers are passed through ``float()`` unchanged.
    """
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().item()
    return float(t)


def as_list(t) -> list:
    """Safely extract a tensor (any device/grad state) to a Python list.

    Calls ``.detach().cpu()`` before ``.tolist()`` so the function is safe
    for CUDA tensors and for ``nn.Parameter`` values inside autograd graphs.
    Plain sequences are passed through ``list()`` unchanged.
    """
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().tolist()
    return list(t)


RoundMode = Literal["inner", "outer", "both"]

def round_corners(
    geom: BaseGeometry,
    radius: float,
    *,
    mode: RoundMode = "both",
    quad_segs: int = 32,
) -> BaseGeometry:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if quad_segs < 1:
        raise ValueError("quad_segs must be at least 1")
    if radius == 0:
        return geom

    if mode in {"outer", "both"}:
        geom = geom.buffer(radius, quad_segs=quad_segs)
        geom = geom.buffer(-radius, quad_segs=quad_segs)

    if mode in {"inner", "both"}:
        geom = geom.buffer(-radius, quad_segs=quad_segs)
        geom = geom.buffer(radius, quad_segs=quad_segs)

    return geom

def remove_holes(geom):
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    if geom.geom_type == "MultiPolygon":
        largest = max(geom.geoms, key=lambda g: g.area)
        return Polygon(largest.exterior)
    return geom