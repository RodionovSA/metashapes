# metashapes/adapters/shapely/periodic.py
# Shapely adapters for periodic primitive shapes.

from __future__ import annotations

from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

import src.metashapes.shape.primitives as prim
from .helpers import as_float

# Large enough that distance calculations within any realistic unit cell
# are exact, while keeping Shapely operations numerically stable.
_LARGE = 1e6


def stripe_to_shapely(shape: prim.Stripe) -> BaseGeometry:
    """
    Convert a Stripe to a Shapely box.

    The stripe is infinite along its axis, so we approximate it with a
    very wide (axis='x') or very tall (axis='y') rectangle.  All
    within-cell distance computations are exact under this approximation.
    """
    off = as_float(shape.offset)
    half = as_float(shape.width) / 2.0

    if shape.axis == 'x':
        return box(-_LARGE, off - half, _LARGE, off + half)
    else:  # axis == 'y'
        return box(off - half, -_LARGE, off + half, _LARGE)
