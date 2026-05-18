# metashapes/adapters/shapely/raster.py
# Convert Shapely geometries to raster masks on a Canvas grid.

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

from metashapes.lattice.canvas import Canvas

def shapely_to_numpy(geom: 'BaseGeometry', canvas: 'Canvas') -> np.ndarray:
    if geom.is_empty:
        return np.zeros((canvas.H, canvas.W), dtype=bool)

    xs, ys = canvas.grid()
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    mask = shapely.contains_xy(geom, xs, ys) | shapely.intersects_xy(geom.boundary, xs, ys)
    return np.asarray(mask, dtype=bool)
