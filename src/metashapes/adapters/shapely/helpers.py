# metashapes/adapters/shapely/helpers.py
# Helper functions for Shapely adapter.

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
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


# --- curve resolution --------------------------------------------------
# Max distance between a polygon edge and the true curve, set for the
# duration of one shape_to_shapely(..., tolerance=...) call. None keeps
# each converter's historical fixed segment count.
_TOLERANCE: ContextVar[float | None] = ContextVar("metashapes_shapely_tolerance", default=None)
_MAX_QUAD_SEGS = 4096


def current_tolerance() -> float | None:
    return _TOLERANCE.get()


@contextmanager
def tolerance_scope(tolerance: float | None):
    """Set the curve tolerance for converters called inside the block."""
    if tolerance is not None and not tolerance > 0:
        raise ValueError("tolerance must be positive")
    token = _TOLERANCE.set(tolerance)
    try:
        yield
    finally:
        _TOLERANCE.reset(token)


def arc_quad_segs(radius: float, default: int, *, fillet: bool = True) -> int:
    """Segments per quarter circle for an arc of ``radius``.

    Without a tolerance returns ``default``. With one, picks the fewest
    segments whose sagitta ``r * (1 - cos(step / 2))`` is <= tolerance.
    For an ellipse drawn as a scaled circle, pass its largest semi-axis:
    the affine map scales the sagitta by at most that factor.

    ``fillet=True`` (corner arcs made by ``buffer``): GEOS splits a fillet
    of angle A into ``int(A / quantum + 0.5)`` segments, so a partial arc
    can get a step of up to 1.5 quanta; the quantum is shrunk to match.
    Full circles (``Point.buffer``, parametric ellipses) pass ``False``.
    """
    tol = _TOLERANCE.get()
    if tol is None:
        return default
    if radius <= tol:
        return 1
    step = 2.0 * math.acos(1.0 - tol / radius)
    if fillet:
        step /= 1.5
    return int(min(_MAX_QUAD_SEGS, max(1, math.ceil((math.pi / 2.0) / step))))


RoundMode = Literal["inner", "outer", "both"]

def round_corners(
    geom: BaseGeometry,
    radius: float,
    *,
    mode: RoundMode = "both",
    quad_segs: int | None = None,
) -> BaseGeometry:
    """Round convex ("inner": opening) and/or concave ("outer": closing)
    corners by ``radius``. ``quad_segs`` defaults to 32, or follows the
    active tolerance (see :func:`tolerance_scope`)."""
    if quad_segs is None:
        quad_segs = arc_quad_segs(radius, 32)
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