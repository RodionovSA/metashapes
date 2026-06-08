# metashapes/adapters/shapely/raster.py
# Convert Shapely geometries to raster masks.

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry


def shapely_to_numpy(
    geom: BaseGeometry,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    """Rasterize a Shapely geometry onto a coordinate grid.

    Parameters
    ----------
    geom : BaseGeometry
        Shapely geometry to rasterize.
    xs, ys : np.ndarray
        World-coordinate arrays of shape (H, W). Any grid layout is supported.

    Returns
    -------
    np.ndarray, shape (H, W), dtype bool
        True where the grid point is inside or on the boundary of geom.
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    if geom.is_empty:
        return np.zeros(xs.shape, dtype=bool)
    mask = shapely.contains_xy(geom, xs, ys) | shapely.intersects_xy(geom.boundary, xs, ys)
    return np.asarray(mask, dtype=bool)
