# metashapes/layout/analysis.py
# LayoutAnalyzer: non-periodic metrics and constraint checks for Layouts.

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.strtree import STRtree

from metashapes.analysis import _bbox_size, _leaf_shapes, _to_geom  # noqa: F401  (shared helpers)

from .generator import LayoutValidator
from .layout import Layout

__all__ = ["LayoutMetrics", "LayoutAnalyzer"]

# Relative shrink of the erosion radius in the sliver test, so a piece of
# exactly the threshold width isn't flagged by floating-point noise.
_EROSION_SLACK = 1e-6


@dataclass
class LayoutMetrics:
    """Measurements produced by :class:`LayoutAnalyzer`.

    Attributes
    ----------
    n_shapes : int
        Number of atoms.
    fill_fraction : float
        ``area(scene ∩ domain) / area(domain)``.
    min_gap : float or None
        Smallest distance between two atoms (0 if any touch or overlap);
        ``None`` with fewer than two atoms. Non-periodic.
    min_feature_size : float or None
        Smallest atom ``min_feature_size``, ignoring atoms that don't
        report one; ``None`` if none do.
    min_edge_clearance : float or None
        Smallest distance from a fully contained atom to the domain
        boundary; 0 if any atom is not contained. ``None`` without atoms.
    n_outside : int
        Atoms not fully inside the domain (shrunk by the analyzer's
        ``edge_margin``).
    n_clipped_features : int
        Atoms cut by the domain boundary whose remaining piece is narrower
        than the feature threshold everywhere (a sliver that the clip turns
        into a sub-resolution feature).
    """
    n_shapes: int
    fill_fraction: float
    min_gap: float | None
    min_feature_size: float | None
    min_edge_clearance: float | None
    n_outside: int
    n_clipped_features: int


class LayoutAnalyzer(LayoutValidator):
    """Metrics and checks for :class:`~metashapes.layout.Layout` objects.

    Unset constraints are not checked. The analyzer is a
    ``LayoutValidator``, so it can be passed straight to
    ``LayoutGenerator(validator=...)``.

    Parameters
    ----------
    min_gap : float, optional
        Minimum allowed atom-to-atom distance (``0`` allows touching).
    min_feature_size : float, optional
        Minimum allowed feature size, for atoms and for clipped remainders.
    edge_margin : float
        Required clearance between atoms and the domain boundary.
    """

    def __init__(self, *, min_gap: float | None = None,
                 min_feature_size: float | None = None, edge_margin: float = 0.0):
        self.min_gap = min_gap
        self.min_feature_size = min_feature_size
        self.edge_margin = float(edge_margin)

    # ------------------------------------------------------------------
    def metrics(self, layout: Layout) -> LayoutMetrics:
        layout.project()
        atoms = layout.shapes
        geoms = np.array([_to_geom(a) for a in atoms], dtype=object)
        domain = layout._domain_geom()
        shapely.prepare(domain)
        n = len(geoms)

        if n:
            fill = shapely.union_all(geoms).intersection(domain).area / domain.area
        else:
            fill = 0.0

        sizes = [a.min_feature_size for a in atoms]
        known = [s for s in sizes if s is not None]
        mfs = min(known) if known else None

        contained = shapely.contains(domain, geoms) if n else np.zeros(0, bool)
        inner = domain.buffer(-self.edge_margin) if self.edge_margin else domain
        inside_margin = shapely.contains(inner, geoms) if n else np.zeros(0, bool)

        if not n:
            clearance = None
        elif not contained.all():
            clearance = 0.0
        else:
            clearance = float(shapely.distance(domain.boundary, geoms).min())

        clipped = 0
        cut = np.nonzero(~contained & shapely.intersects(domain, geoms))[0] if n else []
        for i in cut:
            thr = self.min_feature_size if self.min_feature_size is not None else sizes[i]
            if thr is None or thr <= 0:
                continue
            piece = geoms[i].intersection(domain)
            if piece.buffer(-0.5 * thr * (1 - _EROSION_SLACK)).is_empty:
                clipped += 1

        return LayoutMetrics(
            n_shapes=n,
            fill_fraction=float(fill),
            min_gap=self._min_gap(geoms),
            min_feature_size=mfs,
            min_edge_clearance=clearance,
            n_outside=int((~inside_margin).sum()) if n else 0,
            n_clipped_features=clipped,
        )

    @staticmethod
    def _min_gap(geoms: np.ndarray) -> float | None:
        """Nearest-neighbour atom distance via STRtree."""
        n = len(geoms)
        if n < 2:
            return None
        tree = STRtree(geoms)
        # any intersecting (touching / overlapping) pair -> 0, including
        # identical geometries that query_nearest(exclusive=True) would skip
        src, dst = tree.query(geoms, predicate="intersects")
        if np.any(src != dst):
            return 0.0
        _, dist = tree.query_nearest(geoms, exclusive=True, return_distance=True, all_matches=False)
        return float(dist.min()) if len(dist) else None

    def outside(self, layout: Layout) -> list[int]:
        """Indices of atoms not inside the domain shrunk by ``edge_margin``."""
        return layout.violations(margin=self.edge_margin)

    def check(self, layout: Layout) -> list[str]:
        """Constraint violations as readable strings (empty list = OK)."""
        m = self.metrics(layout)
        out: list[str] = []
        if m.n_outside:
            what = "domain" if not self.edge_margin else f"domain (edge_margin {self.edge_margin})"
            out.append(f"outside_domain: {m.n_outside} atom(s) not fully inside the {what}")
        if m.n_clipped_features:
            out.append(f"clipped_feature: {m.n_clipped_features} atom(s) clipped to a "
                       "sliver narrower than the feature size")
        if self.min_feature_size is not None and m.min_feature_size is not None \
                and m.min_feature_size < self.min_feature_size:
            out.append(f"min_feature_size {m.min_feature_size:.4g} < "
                       f"required {self.min_feature_size}")
        if self.min_gap is not None and m.min_gap is not None:
            bad = m.min_gap < self.min_gap if self.min_gap > 0 else self._overlaps(layout)
            if bad:
                out.append(f"min_gap {m.min_gap:.4g} < required {self.min_gap}"
                           if self.min_gap > 0 else "overlap: atoms overlap (min_gap=0)")
        return out

    @staticmethod
    def _overlaps(layout: Layout) -> bool:
        layout.project()
        geoms = np.array([_to_geom(a) for a in layout.shapes], dtype=object)
        if len(geoms) < 2:
            return False
        tree = STRtree(geoms)
        src, dst = tree.query(geoms, predicate="overlaps")
        if np.any(src != dst):
            return True
        src, dst = tree.query(geoms, predicate="contains")
        return bool(np.any(src != dst))

    def validate(self, layout: Layout) -> str | None:
        failures = self.check(layout)
        return failures[0] if failures else None

    def analyze(self, layouts: list[Layout]) -> list[LayoutMetrics]:
        return [self.metrics(lay) for lay in layouts]
