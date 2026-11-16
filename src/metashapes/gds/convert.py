# metashapes/gds/convert.py
# Units and Shapely -> gdstk polygon conversion.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry.base import BaseGeometry

__all__ = ["Units", "require_gdstk", "polygons_from_shapely"]


def require_gdstk():
    """Import gdstk, with an actionable message if it is missing."""
    try:
        import gdstk
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError(
            "GDS export needs the optional dependency 'gdstk' (pip install gdstk)"
        ) from e
    return gdstk


@dataclass(frozen=True)
class Units:
    """How model coordinates map onto the GDS file.

    Parameters
    ----------
    coord_unit : float
        Metres per model unit (``1e-6``: the model is drawn in µm,
        ``1e-9``: in nm).
    unit : float
        GDS user unit in metres (``1e-6`` = µm, the usual choice).
    precision : float
        GDS database unit (grid) in metres.
    """
    coord_unit: float = 1e-6
    unit: float = 1e-6
    precision: float = 1e-9

    def __post_init__(self):
        for f in ("coord_unit", "unit", "precision"):
            if not getattr(self, f) > 0:
                raise ValueError(f"{f} must be positive")
        if self.precision > self.unit:
            raise ValueError("precision must not exceed unit")

    @property
    def scale(self) -> float:
        """Model units -> GDS user units."""
        return self.coord_unit / self.unit

    @property
    def grid_model(self) -> float:
        """Database grid in model units."""
        return self.precision / self.coord_unit

    @property
    def grid_user(self) -> float:
        """Database grid in GDS user units."""
        return self.precision / self.unit

    def default_tolerance(self) -> float:
        """1 nm in model units, but never finer than the database grid."""
        return max(1e-9, self.precision) / self.coord_unit


def _iter_polygons(geom: BaseGeometry):
    if geom.is_empty:
        return
    t = geom.geom_type
    if t == "Polygon":
        yield geom
    elif t in ("MultiPolygon", "GeometryCollection"):
        for g in geom.geoms:
            yield from _iter_polygons(g)
    # points / lines have no area: nothing to export


def polygons_from_shapely(geom: BaseGeometry, units: Units, *, layer: int = 0,
                          datatype: int = 0) -> list:
    """Shapely geometry (model units) -> ``gdstk.Polygon`` list (user units).

    GDS polygons can't have holes; a polygon with holes is turned into a
    single polygon with a cut line through to each hole by
    ``gdstk.boolean(..., "not")``. Only polygons that have holes pay for
    the boolean. Pieces smaller than one database cell are dropped.
    """
    gdstk = require_gdstk()
    s = units.scale
    min_area = units.grid_model ** 2
    out = []
    for p in _iter_polygons(geom):
        if p.area <= min_area:
            continue
        outer = np.asarray(p.exterior.coords)[:-1] * s
        if not p.interiors:
            out.append(gdstk.Polygon(outer, layer=layer, datatype=datatype))
            continue
        holes = [gdstk.Polygon(np.asarray(r.coords)[:-1] * s) for r in p.interiors]
        out.extend(gdstk.boolean(gdstk.Polygon(outer), holes, "not",
                                 precision=units.grid_user, layer=layer, datatype=datatype))
    return out
