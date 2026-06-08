# metashapes/analysis.py
# UnitCellAnalyzer — compute metrics and validate UnitCells.

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from shapely.affinity import translate
from shapely.geometry.base import BaseGeometry

from src.metashapes.lattice.basis import Lattice
from src.metashapes.lattice.unit_cell import UnitCell
from src.metashapes.shape.base import Shape
from src.metashapes.shape.boolean import Union, Intersection, Difference
from src.metashapes.shape.transforms import Translate, Rotate, Scale

__all__ = ["CellMetrics", "UnitCellAnalyzer"]

_FILL_RESOLUTION = 64  # grid size for fill-fraction rasterization


# ---------------------------------------------------------------------------
# Shape-tree traversal
# ---------------------------------------------------------------------------

def _leaf_shapes(shape: Shape) -> Iterator[Shape]:
    """
    Yield leaf (primitive) shapes from a composed shape tree.

    Union nodes are decomposed into their components.
    Transform wrappers (Translate, Rotate, Scale) are passed through so the
    leaf geometry remains correctly positioned.
    Intersection / Difference are treated as atomic — their internal shapes
    are not separated because the result is not simply a union of parts.
    """
    if isinstance(shape, Union):
        yield from _leaf_shapes(shape.left)
        yield from _leaf_shapes(shape.right)
    elif isinstance(shape, (Translate, Rotate, Scale)):
        # Wrap each leaf in the same transform so positions are preserved
        for leaf in _leaf_shapes(shape.shape):
            if isinstance(shape, Translate):
                yield leaf.translate(shape.dx, shape.dy)
            elif isinstance(shape, Rotate):
                yield leaf.rotate(shape.angle, shape.origin)
            else:
                yield leaf.scale(shape.s, shape.origin)
    else:
        # Primitive or Intersection/Difference — treat as atomic
        yield shape


# ---------------------------------------------------------------------------
# CellMetrics
# ---------------------------------------------------------------------------

@dataclass
class CellMetrics:
    """
    Per-cell measurements produced by :class:`UnitCellAnalyzer`.

    Attributes
    ----------
    fill_fraction : float
        Fraction of canvas pixels inside material (hard mask mean).
    num_shapes : int
        Number of leaf shapes in the cell's shape tree.
    shape_areas : list[float]
        Shapely area of each leaf shape (world units²).
    shape_bbox_sizes : list[tuple[float, float]]
        Axis-aligned bounding-box (width, height) of each leaf shape.
        For rotated shapes this is larger than the intrinsic shape dimensions.
    min_feature_size : float or None
        Smallest ``min_feature_size`` across all leaf shapes.
        ``None`` if no shape reports a finite feature size.
    min_gap : float or None
        Minimum distance between any two shapes (and their periodic images).
        For a single shape, the distance to its own nearest periodic copy.
        ``None`` if it could not be computed.
    """
    fill_fraction: float
    num_shapes: int
    shape_areas: list[float] = field(default_factory=list)
    shape_bbox_sizes: list[tuple[float, float]] = field(default_factory=list)
    min_feature_size: float | None = None
    min_gap: float | None = None


# ---------------------------------------------------------------------------
# UnitCellAnalyzer
# ---------------------------------------------------------------------------

class UnitCellAnalyzer:
    """
    Computes metrics for :class:`~metashapes.unit_cell.UnitCell` objects and
    validates them against configurable constraints.

    All constraint parameters are optional; unset ones are not checked.

    Parameters
    ----------
    min_gap : float, optional
        Minimum allowed distance between shapes (and periodic images).
    min_fill : float, optional
        Minimum allowed fill fraction (0–1).
    max_fill : float, optional
        Maximum allowed fill fraction (0–1).
    min_shape_size : float, optional
        Minimum allowed bounding-box dimension for each shape.
    min_feature_size : float, optional
        Minimum allowed feature size for each shape.
    min_num_shapes : int, optional
        Minimum number of leaf shapes.
    max_num_shapes : int, optional
        Maximum number of leaf shapes.
    """

    def __init__(
        self,
        *,
        min_gap: float | None = None,
        min_fill: float | None = None,
        max_fill: float | None = None,
        min_shape_size: float | None = None,
        min_feature_size: float | None = None,
        min_num_shapes: int | None = None,
        max_num_shapes: int | None = None,
    ) -> None:
        self.min_gap = min_gap
        self.min_fill = min_fill
        self.max_fill = max_fill
        self.min_shape_size = min_shape_size
        self.min_feature_size = min_feature_size
        self.min_num_shapes = min_num_shapes
        self.max_num_shapes = max_num_shapes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def metrics(self, cell: UnitCell) -> CellMetrics:
        """Compute all metrics for a single cell."""
        leaves = list(_leaf_shapes(cell.scene))
        geoms = [_to_geom(s) for s in leaves]

        fill_fraction = float(
            cell.mask(_FILL_RESOLUTION, _FILL_RESOLUTION).mean().item()
        )

        shape_areas = [float(g.area) for g in geoms]
        shape_bbox_sizes = [_bbox_size(g) for g in geoms]

        feature_sizes = [s.min_feature_size for s in leaves]
        finite_fs = [f for f in feature_sizes if f is not None]
        min_fs = min(finite_fs) if finite_fs else None

        min_gap = _compute_min_gap(leaves, geoms, cell.lattice)

        return CellMetrics(
            fill_fraction=fill_fraction,
            num_shapes=len(leaves),
            shape_areas=shape_areas,
            shape_bbox_sizes=shape_bbox_sizes,
            min_feature_size=min_fs,
            min_gap=min_gap,
        )

    def check(self, cell: UnitCell) -> list[str]:
        """
        Return a list of constraint violation descriptions.
        An empty list means all constraints pass.
        """
        m = self.metrics(cell)
        failures: list[str] = []

        if self.min_fill is not None and m.fill_fraction < self.min_fill:
            failures.append(
                f"fill_fraction {m.fill_fraction:.3f} < min_fill {self.min_fill}"
            )
        if self.max_fill is not None and m.fill_fraction > self.max_fill:
            failures.append(
                f"fill_fraction {m.fill_fraction:.3f} > max_fill {self.max_fill}"
            )
        if self.min_num_shapes is not None and m.num_shapes < self.min_num_shapes:
            failures.append(
                f"num_shapes {m.num_shapes} < min_num_shapes {self.min_num_shapes}"
            )
        if self.max_num_shapes is not None and m.num_shapes > self.max_num_shapes:
            failures.append(
                f"num_shapes {m.num_shapes} > max_num_shapes {self.max_num_shapes}"
            )
        if self.min_feature_size is not None and m.min_feature_size is not None:
            if m.min_feature_size < self.min_feature_size:
                failures.append(
                    f"min_feature_size {m.min_feature_size:.4f} < "
                    f"required {self.min_feature_size}"
                )
        if self.min_shape_size is not None:
            for i, (w, h) in enumerate(m.shape_bbox_sizes):
                if w < self.min_shape_size or h < self.min_shape_size:
                    failures.append(
                        f"shape {i} size ({w:.4f}, {h:.4f}) < "
                        f"min_shape_size {self.min_shape_size}"
                    )
        if self.min_gap is not None and m.min_gap is not None:
            if m.min_gap <= self.min_gap:
                failures.append(
                    f"min_gap {m.min_gap:.4f} <= required {self.min_gap}"
                )

        return failures

    def validate(self, cell: UnitCell) -> str | None:
        """
        Generator-compatible interface.

        Returns the first constraint violation string, or ``None`` if all
        constraints pass.  Can be used directly as a
        :class:`~metashapes.generators.validator.UnitCellValidator`.
        """
        failures = self.check(cell)
        return failures[0] if failures else None

    def analyze(self, cells: list[UnitCell]) -> list[CellMetrics]:
        """Compute metrics for every cell in a batch."""
        return [self.metrics(c) for c in cells]

    def find_duplicates(
        self,
        cells: list[UnitCell],
        *,
        resolution: int = 32,
        atol: float = 1e-3,
    ) -> list[list[int]]:
        """
        Find groups of cells that have identical SDFs (within tolerance).

        Each cell's SDF is evaluated on a ``resolution × resolution`` grid
        and the resulting fingerprints are compared pairwise using the
        max absolute difference (Chebyshev distance).

        Parameters
        ----------
        cells : list[UnitCell]
        resolution : int
            Grid resolution for SDF fingerprinting (default 32).
            Higher values give fewer false positives at the cost of speed.
        atol : float
            Absolute tolerance in world units (default 1e-3).
            Two cells are considered duplicates when
            ``max|sdf_a - sdf_b| <= atol`` everywhere on the grid.

        Returns
        -------
        list[list[int]]
            Each inner list contains the indices (into ``cells``) of a
            duplicate group.  Only groups of size ≥ 2 are returned.
            Indices within each group are sorted.
        """
        n = len(cells)
        if n == 0:
            return []

        # --- compute fingerprints ---
        fps = np.empty((n, resolution * resolution), dtype=np.float32)
        for idx, cell in enumerate(cells):
            fps[idx] = (
                cell.rasterize(resolution, resolution)
                .detach().cpu().numpy().ravel()
            )

        # --- union-find ---
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pi] = pj

        # pairwise comparison: for each i, diff against all j > i at once
        for i in range(n):
            if i + 1 >= n:
                break
            diffs = np.abs(fps[i + 1:] - fps[i])   # (n-i-1, D)
            matches = diffs.max(axis=1) <= atol      # (n-i-1,)
            for rel, matched in enumerate(matches):
                if matched:
                    union(i, i + 1 + rel)

        # --- collect groups ---
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)

        return [sorted(g) for g in groups.values() if len(g) >= 2]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _to_geom(shape: Shape) -> BaseGeometry:
    from src.metashapes.adapters.shapely import shape_to_shapely, remove_holes
    return remove_holes(shape_to_shapely(shape))


def _bbox_size(geom: BaseGeometry) -> tuple[float, float]:
    minx, miny, maxx, maxy = geom.bounds
    return (maxx - minx, maxy - miny)


def _skip_self_shift(shape: Shape, ox: float, oy: float) -> bool:
    """True if this periodic shift should be skipped for self-gap measurement.

    A shape with infinite extent along an axis reaches the cell boundary in
    that direction, so shifts with a component along that axis are skipped —
    the periodic copies overlap and the zero-distance is not a real gap.
    """
    (x0, y0), (x1, y1) = shape.bounds()
    eps = 1e-10
    inf_x = not math.isfinite(x0) or not math.isfinite(x1)
    inf_y = not math.isfinite(y0) or not math.isfinite(y1)
    if inf_x and abs(ox) > eps:
        return True
    if inf_y and abs(oy) > eps:
        return True
    return False


def _compute_min_gap(
    shapes: list[Shape],
    geoms: list[BaseGeometry],
    lattice: Lattice,
) -> float | None:
    """
    Minimum distance between any two shapes (or a shape and its own periodic
    images), considering the 8 nearest periodic copies.

    Shifts where a shape with infinite extent along a lattice direction would
    trivially overlap its own copy are excluded, so a Stripe's gap is measured
    only in the perpendicular direction.
    """
    if not geoms:
        return None

    # (i, j) lattice index pairs for the 8 nearest periodic images
    neighbor_indices = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),           (0,  1),
        ( 1, -1), ( 1, 0), ( 1, 1),
    ]

    min_d = float("inf")

    for i, (si, gi) in enumerate(zip(shapes, geoms)):
        # shape vs its own periodic images
        for ix, iy in neighbor_indices:
            off = lattice.offset(ix, iy)
            ox, oy = off[0].item(), off[1].item()
            if _skip_self_shift(si, ox, oy):
                continue
            shifted = translate(gi, xoff=ox, yoff=oy)
            min_d = min(min_d, gi.distance(shifted))

        # shape vs every other shape and their periodic images
        for gj in geoms[i + 1:]:
            for ix, iy in neighbor_indices + [(0, 0)]:
                off = lattice.offset(ix, iy)
                ox, oy = off[0].item(), off[1].item()
                shifted = translate(gj, xoff=ox, yoff=oy)
                min_d = min(min_d, gi.distance(shifted))

    return min_d if min_d < float("inf") else None
