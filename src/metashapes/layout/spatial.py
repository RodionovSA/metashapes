# metashapes/layout/spatial.py
# GeometryHash: incremental Shapely-side spatial index for generation.

from __future__ import annotations

import math
from collections import defaultdict

from shapely.geometry.base import BaseGeometry

__all__ = ["GeometryHash"]


class GeometryHash:
    """Uniform-grid index of Shapely geometries that supports insertion.

    Shapely's ``STRtree`` is static, which suits analysis but not random
    sequential addition, where every accepted atom must be visible to the
    next candidate's gap query. This keeps a ``cell -> [ids]`` dict
    instead: inserting registers a geometry in every cell its bbox touches,
    and a query visits only the cells under the (padded) query box.

    Parameters
    ----------
    cell_size : float, optional
        Grid cell edge length. If omitted it is set on the first insert to
        the larger side of that geometry's bbox, which suits populations of
        similar-size meta-atoms.
    """

    def __init__(self, cell_size: float | None = None):
        if cell_size is not None and not cell_size > 0:
            raise ValueError("cell_size must be positive")
        self.cell_size = cell_size
        self.geoms: list[BaseGeometry] = []
        self._cells: dict[tuple[int, int], list[int]] = defaultdict(list)

    def __len__(self) -> int:
        return len(self.geoms)

    def _range(self, x0, y0, x1, y1):
        cs = self.cell_size
        return (math.floor(x0 / cs), math.floor(y0 / cs),
                math.floor(x1 / cs), math.floor(y1 / cs))

    def insert(self, geom: BaseGeometry) -> int:
        x0, y0, x1, y1 = geom.bounds
        if self.cell_size is None:
            side = max(x1 - x0, y1 - y0)
            self.cell_size = side if side > 0 else 1.0
        gid = len(self.geoms)
        self.geoms.append(geom)
        i0, j0, i1, j1 = self._range(x0, y0, x1, y1)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                self._cells[(i, j)].append(gid)
        return gid

    def query(self, bounds: tuple[float, float, float, float], pad: float = 0.0) -> list[int]:
        """Ids of geometries whose bbox may intersect ``bounds`` padded by ``pad``."""
        if not self.geoms:
            return []
        x0, y0, x1, y1 = bounds
        i0, j0, i1, j1 = self._range(x0 - pad, y0 - pad, x1 + pad, y1 + pad)
        out: set[int] = set()
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                ids = self._cells.get((i, j))
                if ids:
                    out.update(ids)
        return sorted(out)

    def conflicts(self, geom: BaseGeometry, min_gap: float) -> bool:
        """True if ``geom`` is closer than ``min_gap`` to any stored geometry.

        ``min_gap=0`` allows touching but not overlapping (positive-area
        intersection).
        """
        for gid in self.query(geom.bounds, pad=min_gap):
            other = self.geoms[gid]
            if min_gap > 0:
                if geom.distance(other) < min_gap:
                    return True
            elif geom.intersects(other) and not geom.touches(other):
                return True
        return False
