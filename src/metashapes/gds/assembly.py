# metashapes/gds/assembly.py
# GDSAssembly: place Layouts and UnitCell arrays on one chip, check, write.

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.affinity import translate as shp_translate
from shapely.geometry import MultiPoint, Polygon, box
from shapely.strtree import STRtree

from metashapes.adapters.shapely import shape_to_shapely

from .cells import (_finite, cell_parallelogram, layout_to_cell, unit_cell_geometry,
                    unit_cell_to_cell)
from .convert import Units, require_gdstk

__all__ = ["Placement", "GDSAssembly", "save_gds"]

_NAME_RE = re.compile(r"[^A-Za-z0-9_?$]")


def _layer(layer) -> tuple[int, int]:
    if isinstance(layer, (tuple, list)):
        return int(layer[0]), int(layer[1])
    return int(layer), 0


def _kind(item) -> str:
    from metashapes.lattice.unit_cell import UnitCell
    from metashapes.layout.layout import Layout
    if isinstance(item, Layout):
        return "layout"
    if isinstance(item, UnitCell):
        return "array"
    raise TypeError(f"can place a Layout or a UnitCell, not {type(item).__name__}")


@dataclass
class Placement:
    """One placed item. ``offset`` is where the item's own (0, 0) lands
    (for arrays: lattice point (0, 0) of the first cell), in model units."""
    name: str
    item: object
    kind: str                                  # "layout" | "array"
    offset: tuple[float, float]
    layer: tuple[int, int] = (0, 0)
    repeats: tuple[int, int] | None = None
    text: str | None = None
    text_anchor: str | tuple[float, float] = "center"
    _footprint: Polygon | None = field(default=None, repr=False)


class GDSAssembly:
    """A chip: Layouts and periodic UnitCell arrays placed on one top cell.

    All lengths passed in (positions, pitches, ``tolerance``,
    ``min_spacing``, ``text_size``) are in **model units**;
    ``coord_unit`` says how long one model unit is (``1e-9`` if the
    design is drawn in nm). The file is written with GDS user unit
    ``unit`` (default 1 µm) and database grid ``precision`` (default 1 nm).

    Parameters
    ----------
    name : str
        Top cell name.
    unit, precision, coord_unit : float
        See :class:`~metashapes.gds.convert.Units`.
    tolerance : float, optional
        Largest distance between polygon edges and the true curved outline,
        in model units. Default: 1 nm.
    text_layer : (int, int)
        Layer/texttype for the optional placement texts; never the design layer.
    text_as_polygons : bool
        Write texts as polygon glyphs (``gdstk.text``) instead of labels.
    text_size : float, optional
        Glyph height for ``text_as_polygons`` (model units). Default: 5 %
        of the smaller footprint side.
    max_points : int
        Vertex limit per polygon in the written file (GDSII allows 8190).
    workers : int, optional
        Processes used to convert layouts whose atoms are all different
        (see :func:`~metashapes.gds.layout_to_cell`). Scripts using it need
        an ``if __name__ == "__main__":`` guard.
    """

    def __init__(self, name: str = "TOP", *, unit: float = 1e-6, precision: float = 1e-9,
                 coord_unit: float = 1e-6, tolerance: float | None = None,
                 text_layer=(100, 0), text_as_polygons: bool = False,
                 text_size: float | None = None, max_points: int = 8190,
                 workers: int | None = None):
        self.name = self._sanitize(name)
        self.units = Units(coord_unit=coord_unit, unit=unit, precision=precision)
        self.tolerance = self.units.default_tolerance() if tolerance is None else float(tolerance)
        self.text_layer = _layer(text_layer)
        self.text_as_polygons = bool(text_as_polygons)
        self.text_size = text_size
        self.max_points = int(max_points)
        self.workers = workers
        self.placements: list[Placement] = []

    # --- placing -----------------------------------------------------------
    @staticmethod
    def _sanitize(name: str) -> str:
        return _NAME_RE.sub("_", str(name)) or "_"

    def _add(self, p: Placement) -> Placement:
        p.name = self._sanitize(p.name)
        if p.name == self.name or any(q.name == p.name for q in self.placements):
            raise ValueError(f"duplicate placement name {p.name!r}")
        self.placements.append(p)
        return p

    def _auto_name(self, kind: str) -> str:
        return f"{kind}_{len(self.placements)}"

    def place(self, layout, at=(0.0, 0.0), *, layer=0, name: str | None = None,
              text: str | None = None, text_anchor="center") -> Placement:
        """Place a ``Layout``: its own (0, 0) goes to ``at``."""
        if _kind(layout) != "layout":
            raise TypeError("place() takes a Layout; use place_array() for a UnitCell")
        return self._add(Placement(name or self._auto_name("layout"), layout, "layout",
                                   (float(at[0]), float(at[1])), _layer(layer),
                                   text=text, text_anchor=text_anchor))

    def place_array(self, cell, at=(0.0, 0.0), *, repeats, anchor: str = "center", layer=0,
                    name: str | None = None, text: str | None = None,
                    text_anchor="center") -> Placement:
        """Place an ``n1 x n2`` array of a ``UnitCell`` (whole atoms, no clipping).

        ``anchor="center"``: the centre of the ``n1·a1 x n2·a2`` parallelogram
        goes to ``at``. ``anchor="origin"``: lattice point (0, 0) of the first
        cell goes to ``at``.
        """
        if _kind(cell) != "array":
            raise TypeError("place_array() takes a UnitCell; use place() for a Layout")
        n1, n2 = int(repeats[0]), int(repeats[1])
        if n1 < 1 or n2 < 1:
            raise ValueError("repeats must be >= 1")
        x, y = float(at[0]), float(at[1])
        if anchor == "center":
            a1 = cell.lattice.a1.detach().cpu().numpy().astype(float)
            a2 = cell.lattice.a2.detach().cpu().numpy().astype(float)
            cx, cy = 0.5 * (n1 * a1 + n2 * a2)
            x, y = x - cx, y - cy
        elif anchor != "origin":
            raise ValueError("anchor must be 'center' or 'origin'")
        return self._add(Placement(name or self._auto_name("array"), cell, "array", (x, y),
                                   _layer(layer), repeats=(n1, n2), text=text,
                                   text_anchor=text_anchor))

    def place_grid(self, items, origin=(0.0, 0.0), *, pitch, cols: int, repeats=None,
                   layer=0, name_prefix: str = "item", texts=None) -> list[Placement]:
        """Place many items on a grid, each centred in its slot.

        ``items``: a list of Layouts / UnitCells, a ``LayoutBatchResult`` or a
        ``GenerationBatchResult``. Slot ``i`` is column ``i % cols``, row
        ``i // cols``; columns go right (+x) and rows go down (-y) from
        ``origin`` (the centre of the first slot). UnitCells need ``repeats``.
        """
        items = list(getattr(items, "layouts", None) or getattr(items, "unit_cells", None)
                     or items)
        if texts is not None and len(texts) != len(items):
            raise ValueError("texts must have one entry per item")
        px, py = float(pitch[0]), float(pitch[1])
        out = []
        for i, item in enumerate(items):
            col, row = i % cols, i // cols
            sx, sy = origin[0] + col * px, origin[1] - row * py
            name = f"{name_prefix}_{i}"
            text = texts[i] if texts is not None else None
            if _kind(item) == "array":
                if repeats is None:
                    raise ValueError("UnitCells in place_grid need repeats=(n1, n2)")
                p = self.place_array(item, (0.0, 0.0), repeats=repeats, anchor="origin",
                                     layer=layer, name=name, text=text)
            else:
                p = self.place(item, (0.0, 0.0), layer=layer, name=name, text=text)
            (x0, y0, x1, y1) = self.footprint(p).bounds
            p.offset = (sx - 0.5 * (x0 + x1), sy - 0.5 * (y0 + y1))
            p._footprint = None
            out.append(p)
        return out

    # --- footprints ----------------------------------------------------------
    def footprint(self, p: Placement) -> Polygon:
        """Region the placement occupies, in model units (cached)."""
        if p._footprint is None:
            p._footprint = shp_translate(self._local_footprint(p), *p.offset)
        return p._footprint

    def _local_footprint(self, p: Placement) -> Polygon:
        if p.kind == "layout":
            lay = p.item
            if lay.clip:
                return shape_to_shapely(lay.domain, tolerance=self.tolerance)
            pts = []
            for a in lay.shapes:
                (x0, y0), (x1, y1) = a.bounds()
                pts += [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            return MultiPoint(pts).convex_hull if pts else Polygon()
        cell = p.item
        n1, n2 = p.repeats
        a1 = cell.lattice.a1.detach().cpu().numpy().astype(float)
        a2 = cell.lattice.a2.detach().cpu().numpy().astype(float)
        corners = [np.zeros(2), n1 * a1, n1 * a1 + n2 * a2, n2 * a2]
        pts = [tuple(c) for c in corners]
        bounds = cell.scene.bounds()
        if _finite(bounds):        # whole atoms may stick out past the parallelogram
            (x0, y0), (x1, y1) = bounds
            for o in (np.zeros(2), (n1 - 1) * a1, (n2 - 1) * a2, (n1 - 1) * a1 + (n2 - 1) * a2):
                pts += [(x0 + o[0], y0 + o[1]), (x1 + o[0], y0 + o[1]),
                        (x1 + o[0], y1 + o[1]), (x0 + o[0], y1 + o[1])]
        return MultiPoint(pts).convex_hull

    def footprints(self) -> dict[str, Polygon]:
        return {p.name: self.footprint(p) for p in self.placements}

    def extent(self) -> tuple[float, float, float, float]:
        """(x0, y0, x1, y1) of all footprints, model units."""
        if not self.placements:
            return (0.0, 0.0, 0.0, 0.0)
        return shapely.union_all(list(self.footprints().values())).bounds

    # --- checks -----------------------------------------------------------------
    def check(self, *, min_spacing: float = 0.0, geometry: bool = False) -> list[str]:
        """Placement problems as readable strings (empty list = OK).

        - ``overlap``: two footprints share area (touching is allowed).
        - ``spacing``: two footprints are closer than ``min_spacing``.

        Array footprints are conservative hulls. With ``geometry=True`` a
        flagged pair is re-checked against the actual exported polygons
        near the conflict and dropped if those are fine.
        """
        ps = self.placements
        if len(ps) < 2:
            return []
        fps = [self.footprint(p) for p in ps]
        tree = STRtree(fps)
        pad = max(0.0, float(min_spacing))
        src, dst = tree.query(fps, predicate="dwithin", distance=pad) if pad > 0 \
            else tree.query(fps, predicate="intersects")
        eps = 1e-12 * max(1.0, max(f.area for f in fps))
        out = []
        for i, j in sorted({(min(a, b), max(a, b)) for a, b in zip(src, dst) if a != b}):
            area = fps[i].intersection(fps[j]).area
            if area > eps:
                if geometry and not self._geom_conflict(ps[i], ps[j], 0.0):
                    continue
                out.append(f"overlap: {ps[i].name!r} and {ps[j].name!r} footprints "
                           f"overlap (area {area:.4g})")
            elif pad > 0:
                d = fps[i].distance(fps[j])
                if d < pad:
                    if geometry and not self._geom_conflict(ps[i], ps[j], pad):
                        continue
                    out.append(f"spacing: {ps[i].name!r} and {ps[j].name!r} are "
                               f"{d:.4g} apart (< {pad:g})")
        return out

    def _geom_conflict(self, p: Placement, q: Placement, spacing: float) -> bool:
        region = self.footprint(p).buffer(spacing).intersection(self.footprint(q).buffer(spacing))
        if region.is_empty:
            return False
        region = region.buffer(spacing)
        gp = self._geometry_near(p, region)
        gq = self._geometry_near(q, region)
        if gp.is_empty or gq.is_empty:
            return False
        if spacing > 0:
            return gp.distance(gq) < spacing
        return gp.intersection(gq).area > 0

    def _geometry_near(self, p: Placement, region):
        """Exported geometry of ``p`` (model units, placed) within ``region``."""
        local = shp_translate(region, -p.offset[0], -p.offset[1])
        x0, y0, x1, y1 = local.bounds
        if p.kind == "layout":
            lay = p.item
            parts = []
            for a in lay.shapes:
                (ax0, ay0), (ax1, ay1) = a.bounds()
                if ax1 < x0 or ax0 > x1 or ay1 < y0 or ay0 > y1:
                    continue
                parts.append(shape_to_shapely(a, tolerance=self.tolerance))
            g = shapely.union_all(parts) if parts else Polygon()
            if lay.clip:
                g = g.intersection(shape_to_shapely(lay.domain, tolerance=self.tolerance))
        else:
            g = self._array_geometry(p, (x0, y0, x1, y1))
        return shp_translate(g.intersection(local), *p.offset)

    def _array_geometry(self, p: Placement, window):
        """Union of the array copies whose bounding box meets ``window``."""
        cell = p.item
        n1, n2 = p.repeats
        base = unit_cell_geometry(cell, tolerance=self.tolerance)
        if base.is_empty:
            return base
        bx0, by0, bx1, by1 = base.bounds
        a1 = cell.lattice.a1.detach().cpu().numpy().astype(float)
        a2 = cell.lattice.a2.detach().cpu().numpy().astype(float)
        inv = np.linalg.inv(np.stack([a1, a2], axis=1))
        x0, y0, x1, y1 = window
        # lattice indices whose copy bbox can meet the window
        corners = np.array([[x0 - bx1, y0 - by1], [x1 - bx0, y0 - by1],
                            [x1 - bx0, y1 - by0], [x0 - bx1, y1 - by0]])
        ij = corners @ inv.T
        i0, j0 = np.floor(ij.min(axis=0)).astype(int)
        i1, j1 = np.ceil(ij.max(axis=0)).astype(int)
        i0, i1 = max(i0, 0), min(i1, n1 - 1)
        j0, j1 = max(j0, 0), min(j1, n2 - 1)
        copies = [shp_translate(base, *(i * a1 + j * a2))
                  for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]
        return shapely.union_all(copies) if copies else Polygon()

    # --- output ------------------------------------------------------------
    def to_library(self):
        """Build a ``gdstk.Library``: the top cell, one cell per placed object
        (shared when the same object is placed twice on the same layer), and
        a ``Repetition`` reference per array."""
        gdstk = require_gdstk()
        u = self.units
        s = u.scale
        lib = gdstk.Library(unit=u.unit, precision=u.precision)
        top = gdstk.Cell(self.name)
        cache: dict = {}
        for p in self.placements:
            key = (id(p.item), p.layer, p.kind)
            if key not in cache:
                if p.kind == "layout":
                    cache[key] = layout_to_cell(p.item, p.name, u, layer=p.layer[0],
                                                datatype=p.layer[1], tolerance=self.tolerance,
                                                workers=self.workers)
                else:
                    cache[key] = unit_cell_to_cell(p.item, p.name, u, layer=p.layer[0],
                                                   datatype=p.layer[1],
                                                   tolerance=self.tolerance)
            sub = cache[key]
            origin = (p.offset[0] * s, p.offset[1] * s)
            if p.kind == "layout":
                top.add(gdstk.Reference(sub, origin=origin))
            else:
                a1 = p.item.lattice.a1.detach().cpu().numpy().astype(float) * s
                a2 = p.item.lattice.a2.detach().cpu().numpy().astype(float) * s
                top.add(*_array_references(gdstk, sub, origin, a1, a2, *p.repeats))
            if p.text:
                self._add_text(top, p)
        lib.add(top, *top.dependencies(True))
        return lib

    def _text_origin(self, p: Placement) -> tuple[float, float]:
        if isinstance(p.text_anchor, (tuple, list)):
            return float(p.text_anchor[0]), float(p.text_anchor[1])
        x0, y0, x1, y1 = self.footprint(p).bounds
        cx = 0.5 * (x0 + x1)
        return {"center": (cx, 0.5 * (y0 + y1)), "top": (cx, y1),
                "bottom": (cx, y0)}[p.text_anchor]

    def _add_text(self, top, p: Placement):
        gdstk = require_gdstk()
        s = self.units.scale
        x, y = self._text_origin(p)
        layer, texttype = self.text_layer
        if not self.text_as_polygons:
            top.add(gdstk.Label(p.text, (x * s, y * s), layer=layer, texttype=texttype))
            return
        x0, y0, x1, y1 = self.footprint(p).bounds
        size = self.text_size or 0.05 * min(x1 - x0, y1 - y0)
        top.add(*gdstk.text(p.text, size * s, (x * s, y * s), layer=layer, datatype=texttype))

    def write_gds(self, path, *, min_spacing: float | None = 0.0, allow_overlap: bool = False,
                  check_geometry: bool = False):
        """Check placements, then write the file.

        Raises ``ValueError`` listing the problems unless they are allowed:
        overlaps with ``allow_overlap=True``; spacing is checked only when
        ``min_spacing`` is set (> 0).
        """
        if min_spacing is not None:
            issues = self.check(min_spacing=min_spacing, geometry=check_geometry)
            if allow_overlap:
                issues = [m for m in issues if not m.startswith("overlap")]
            if issues:
                raise ValueError("GDS placement check failed:\n  " + "\n  ".join(issues))
        lib = self.to_library()
        lib.write_gds(str(path), max_points=self.max_points)
        return lib


def _axis(v, tol: float = 1e-12) -> str | None:
    """'x' / 'y' if ``v`` lies along that axis, else None."""
    if abs(v[1]) <= tol * abs(v[0]):
        return "x"
    if abs(v[0]) <= tol * abs(v[1]):
        return "y"
    return None


def _rect_rep(gdstk, n: int, v, axis: str):
    """1D repetition of ``n`` along an axis-aligned ``v`` (written as AREF)."""
    if axis == "x":
        return gdstk.Repetition(columns=n, rows=1, spacing=(float(v[0]), 0.0))
    return gdstk.Repetition(columns=1, rows=n, spacing=(0.0, float(v[1])))


def _array_references(gdstk, cell, origin, a1, a2, n1: int, n2: int) -> list:
    """References placing ``cell`` at ``origin + i·a1 + j·a2``.

    gdstk writes a GDS array reference (AREF) only for an axis-aligned,
    rectangular repetition and expands anything else into one reference per
    copy. So:
    - a1, a2 along x and y: one AREF;
    - only one of them axis-aligned: one AREF per row along it
      (n2 or n1 references instead of n1·n2);
    - neither: a regular repetition, which gdstk expands (n1·n2 references).
    """
    ox, oy = origin
    ax1, ax2 = _axis(a1), _axis(a2)
    if ax1 and ax2 and ax1 != ax2:
        ref = gdstk.Reference(cell, origin=origin)
        if ax1 == "x":
            ref.repetition = gdstk.Repetition(columns=n1, rows=n2,
                                              spacing=(float(a1[0]), float(a2[1])))
        else:
            ref.repetition = gdstk.Repetition(columns=n2, rows=n1,
                                              spacing=(float(a2[0]), float(a1[1])))
        return [ref]
    if ax1:
        refs = []
        for j in range(n2):
            ref = gdstk.Reference(cell, origin=(ox + j * a2[0], oy + j * a2[1]))
            ref.repetition = _rect_rep(gdstk, n1, a1, ax1)
            refs.append(ref)
        return refs
    if ax2:
        refs = []
        for i in range(n1):
            ref = gdstk.Reference(cell, origin=(ox + i * a1[0], oy + i * a1[1]))
            ref.repetition = _rect_rep(gdstk, n2, a2, ax2)
            refs.append(ref)
        return refs
    ref = gdstk.Reference(cell, origin=origin)
    ref.repetition = gdstk.Repetition(n1, n2, v1=tuple(a1), v2=tuple(a2))
    return [ref]


def save_gds(path, item, *, repeats=None, layer=0, **assembly_kwargs):
    """Write one Layout (or one UnitCell array, with ``repeats``) to ``path``.

    Shorthand for a one-placement :class:`GDSAssembly` at the origin
    (arrays anchored at their lattice origin).
    """
    chip = GDSAssembly(**assembly_kwargs)
    if _kind(item) == "array":
        if repeats is None:
            raise ValueError("a UnitCell needs repeats=(n1, n2)")
        chip.place_array(item, repeats=repeats, anchor="origin", layer=layer, name="array")
    else:
        chip.place(item, layer=layer, name="layout")
    return chip.write_gds(path)
