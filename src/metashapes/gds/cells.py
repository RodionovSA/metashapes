# metashapes/gds/cells.py
# Shapes, UnitCells and Layouts -> gdstk polygons / cells.

from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np
import shapely
from shapely.affinity import translate as shp_translate
from shapely.geometry import Polygon

from metashapes.adapters.shapely import shape_to_shapely, tolerance_scope
from metashapes.shape.base import Shape
from metashapes.shape.transforms import Translate

from .convert import Units, polygons_from_shapely, require_gdstk

__all__ = ["shape_to_gdstk", "unit_cell_geometry", "unit_cell_to_cell", "layout_to_cell"]


def _tol(tolerance, units: Units) -> float:
    return units.default_tolerance() if tolerance is None else float(tolerance)


def shape_to_gdstk(shape: Shape, units: Units | None = None, *, layer: int = 0,
                   datatype: int = 0, tolerance: float | None = None) -> list:
    """One shape -> ``gdstk.Polygon`` list.

    ``tolerance`` (model units) bounds the distance between polygon edges
    and the true outline; default: 1 nm (see :meth:`Units.default_tolerance`).
    """
    units = units or Units()
    geom = shape_to_shapely(shape, tolerance=_tol(tolerance, units))
    return polygons_from_shapely(geom, units, layer=layer, datatype=datatype)


# --- unit cells -------------------------------------------------------------
def _finite(bounds) -> bool:
    (x0, y0), (x1, y1) = bounds
    return all(math.isfinite(v) for v in (x0, y0, x1, y1))


def cell_parallelogram(lattice) -> Polygon:
    a1 = lattice.a1.detach().cpu().tolist()
    a2 = lattice.a2.detach().cpu().tolist()
    return Polygon([(0.0, 0.0), (a1[0], a1[1]),
                    (a1[0] + a2[0], a1[1] + a2[1]), (a2[0], a2[1])])


def unit_cell_geometry(cell, *, tolerance: float):
    """Geometry of one period, as placed in an array.

    A finite scene is exported whole and unclipped, so an array of it is
    exactly periodic inside and has whole atoms on its outer edge. A scene
    with infinite extent (e.g. a ``Bar`` grating) can't be exported whole;
    it is wrapped into the cell parallelogram (``UnitCell.to_shapely``), so
    neighbouring copies abut seamlessly.
    """
    if _finite(cell.scene.bounds()):
        return shape_to_shapely(cell.scene, tolerance=tolerance)
    with tolerance_scope(tolerance):
        return cell.to_shapely()


def unit_cell_to_cell(cell, name: str, units: Units | None = None, *, layer: int = 0,
                      datatype: int = 0, tolerance: float | None = None):
    """One period of a ``UnitCell`` -> ``gdstk.Cell`` (see :func:`unit_cell_geometry`).

    Arrays reference this cell with a ``gdstk.Repetition`` along a1/a2.
    """
    gdstk = require_gdstk()
    units = units or Units()
    out = gdstk.Cell(name)
    geom = unit_cell_geometry(cell, tolerance=_tol(tolerance, units))
    out.add(*polygons_from_shapely(geom, units, layer=layer, datatype=datatype))
    return out


# --- layouts ----------------------------------------------------------------
def _split_translate(atom: Shape):
    """(base shape, (dx, dy)) for a top-level Translate, else (atom, (0, 0))."""
    if isinstance(atom, Translate):
        return atom.shape, (float(atom.dx.detach().cpu()), float(atom.dy.detach().cpu()))
    return atom, (0.0, 0.0)


# Parameters without a length unit. An error d in one of them moves the
# outline by ~d * size, so they are snapped to grid / size, not to grid.
_DIMENSIONLESS = frozenset({"angle", "s"})


def _max_abs(v) -> float:
    if isinstance(v, (int, float)):
        return abs(float(v))
    if isinstance(v, (list, tuple)):
        return max((_max_abs(x) for x in v), default=0.0)
    return 0.0


def _quantize(obj, grid: float, fine: float | None):
    """Parametric dict with lengths snapped to multiples of ``grid`` and
    dimensionless values (``_DIMENSIONLESS``) to multiples of ``fine``
    (``None``: kept exact)."""
    if isinstance(obj, dict):
        g = grid
        if obj.get("type") == "Scale":          # child lengths are magnified by |s|
            g = grid / max(_max_abs(obj.get("s")), 1e-12)
        out = {}
        for k, v in obj.items():
            if k in _DIMENSIONLESS:
                out[k] = _snap(v, fine)
            else:
                out[k] = _quantize(v, g if k == "shape" else grid, fine)
        return out
    if isinstance(obj, (list, tuple)):
        return [_quantize(v, grid, fine) for v in obj]
    return _snap(obj, grid)


def _snap(v, step: float | None):
    if isinstance(v, (list, tuple)):
        return [_snap(x, step) for x in v]
    if not isinstance(v, float):
        return v
    return repr(v) if step is None else round(v / step)


def _dedupe_key(data: dict, grid: float, bounds=None) -> str:
    """Grouping key: two atoms with equal keys differ by less than ~``grid``.

    ``bounds`` (the base's bounds) sets the size used for dimensionless
    parameters; unknown or infinite bounds keep them exact.
    """
    fine = None
    if bounds is not None and _finite(bounds):
        size = max(_max_abs(list(bounds[0])), _max_abs(list(bounds[1])), grid)
        fine = grid / size
    return json.dumps(_quantize(data, grid, fine), sort_keys=True)


def layout_to_cell(layout, name: str, units: Units | None = None, *, layer: int = 0,
                   datatype: int = 0, tolerance: float | None = None,
                   dedupe: bool = True, workers: int | None = None):
    """``Layout`` -> ``gdstk.Cell`` (with sub-cells for repeated atoms).

    - With ``layout.clip`` (default), atoms are clipped to the domain. Only
      atoms whose bounding box isn't inside the domain are converted and
      tested; the few that actually cross it are intersected one by one
      and written as plain polygons. Atoms entirely outside are dropped.
    - ``dedupe``: atoms are grouped by their parameters with the outer
      ``Translate`` removed (snapped to the database grid). A group used
      two or more times becomes one sub-cell named ``{name}__A{k}``,
      placed by references; atoms used once are plain polygons.
    - Overlapping atoms are exported as they are (no union).
    - ``workers``: convert the atoms used once (the expensive part when all
      atoms differ) in that many processes. Worth it from ~10⁴ such atoms;
      ``None``/``1`` converts in this process. Workers start from a
      ``forkserver``, so a *script* using this needs the usual
      ``if __name__ == "__main__":`` guard (notebooks don't).
    """
    gdstk = require_gdstk()
    units = units or Units()
    tol = _tol(tolerance, units)
    s = units.scale
    atoms = layout.shapes
    top = gdstk.Cell(name)
    if not atoms:
        return top

    # Per distinct base object (atoms built from one library shape share it):
    # its bounds, parameters and dedupe key, computed once. bounds() runs
    # first because it clamps drifted parameters, which the key then sees.
    # A Translate atom's bounds are its base's bounds shifted by (dx, dy).
    memo: dict[int, tuple] = {}
    split = []
    for a in atoms:
        base, off = _split_translate(a)
        m = memo.get(id(base))
        if m is None:
            bounds = base.bounds()
            data = base.to_parametric() if (dedupe or workers) else None
            key = _dedupe_key(data, units.grid_model, bounds) if dedupe else None
            m = memo[id(base)] = (key, bounds, data)
        split.append((base, off, m))

    whole = list(range(len(atoms)))        # exported unclipped
    clipped_geoms = []
    if layout.clip:
        domain = shape_to_shapely(layout.domain, tolerance=tol)
        shapely.prepare(domain)
        b = np.array([[x0 + dx, y0 + dy, x1 + dx, y1 + dy]
                      for _, (dx, dy), (_, ((x0, y0), (x1, y1)), _) in split])
        inside = shapely.contains(domain, shapely.box(b[:, 0], b[:, 1], b[:, 2], b[:, 3]))
        whole = [i for i in whole if inside[i]]
        for i in np.nonzero(~inside)[0]:
            g = shape_to_shapely(atoms[i], tolerance=tol)
            if domain.contains(g):
                whole.append(int(i))
            elif domain.intersects(g):
                clipped_geoms.append(g.intersection(domain))

    groups: dict[str, list] = defaultdict(list)
    for i in whole:
        base, off, (key, _, data) = split[i]
        groups[key if dedupe else str(i)].append((base, off, data))

    singles = [m[0] for m in groups.values() if len(m) == 1]   # (base, off, data)
    for g in _convert_many(singles, tol, workers):
        top.add(*polygons_from_shapely(g, units, layer=layer, datatype=datatype))

    k = 0
    for members in groups.values():
        if len(members) == 1:
            continue
        geom = shape_to_shapely(members[0][0], tolerance=tol)
        sub = gdstk.Cell(f"{name}__A{k}")
        k += 1
        sub.add(*polygons_from_shapely(geom, units, layer=layer, datatype=datatype))
        for _, (dx, dy), _ in members:
            top.add(gdstk.Reference(sub, origin=(dx * s, dy * s)))

    for g in clipped_geoms:
        top.add(*polygons_from_shapely(g, units, layer=layer, datatype=datatype))
    return top


# --- parallel conversion ------------------------------------------------------
_MIN_PER_WORKER = 2000


def _convert_chunk(args) -> list[bytes]:
    """Worker: rebuild shapes from parameters, convert, place, return WKB."""
    items, tol = args
    out = []
    for data, dx, dy in items:
        g = shape_to_shapely(Shape.from_parametric(data), tolerance=tol)
        out.append(shapely.to_wkb(shp_translate(g, dx, dy)))
    return out


def _mp_context():
    """Process start method for the workers.

    Plain ``fork`` from a process running torch threads can deadlock the
    child. A ``forkserver`` that preloads this module forks the workers
    from a single-threaded server that already has metashapes imported
    (fast start, safe). Falls back to ``spawn`` where forkserver is missing.
    """
    import multiprocessing as mp
    if "forkserver" in mp.get_all_start_methods():
        ctx = mp.get_context("forkserver")
        ctx.set_forkserver_preload(["metashapes.gds.cells"])
        return ctx
    return mp.get_context("spawn")


def _convert_many(members, tol: float, workers: int | None):
    """Placed Shapely geometry for each (base, (dx, dy), parameters),
    optionally in parallel."""
    if not workers or workers <= 1 or len(members) < 2 * _MIN_PER_WORKER:
        return [shp_translate(shape_to_shapely(b, tolerance=tol), dx, dy)
                for b, (dx, dy), _ in members]
    from concurrent.futures import ProcessPoolExecutor

    items = [(data, dx, dy) for _, (dx, dy), data in members]
    n_chunks = min(len(items) // _MIN_PER_WORKER, 4 * workers)
    size = -(-len(items) // n_chunks)
    chunks = [(items[i:i + size], tol) for i in range(0, len(items), size)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=_mp_context()) as pool:
        return [shapely.from_wkb(w) for part in pool.map(_convert_chunk, chunks) for w in part]
