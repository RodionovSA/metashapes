# metashapes/layout/generator.py
# LayoutGenerator: sites (where) x sources (what) x constraints.

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import shapely

from metashapes.adapters.shapely import remove_holes, shape_to_shapely
from metashapes.generators.report import GenerationBatchResult, GenerationReport
from metashapes.shape.base import Shape, is_empty_bounds
from metashapes.shape.shapeset import ShapeSet
from metashapes.shape.transforms import Translate

from .config import LayoutConfig
from .layout import Layout
from .sites import Sites
from .sources import AtomSource
from .spatial import GeometryHash

__all__ = ["LayoutGenerator", "LayoutBatchResult", "LayoutValidator"]


class LayoutValidator(ABC):
    """Whole-layout acceptance test run after a layout is built."""

    @abstractmethod
    def validate(self, layout: Layout) -> str | None:
        """None if the layout is acceptable, otherwise a short reason."""
        raise NotImplementedError


@dataclass(slots=True)
class LayoutBatchResult:
    layouts: list[Layout]
    report: GenerationReport

    @classmethod
    def merge(cls, *results: "LayoutBatchResult") -> "LayoutBatchResult":
        """Concatenate layouts and merge reports (same rules as
        :meth:`GenerationBatchResult.merge`)."""
        if not results:
            raise ValueError("merge() requires at least one result")
        if len(results) == 1:
            return results[0]
        report = GenerationBatchResult.merge(
            *(GenerationBatchResult(unit_cells=[], report=r.report) for r in results)
        ).report
        layouts = [lay for r in results for lay in r.layouts]
        return cls(layouts=layouts, report=report)


def _place(atom: Shape, x: float, y: float) -> Shape:
    """Translate an origin-centred atom to (x, y), folding a top-level
    Translate into one node instead of nesting two."""
    if isinstance(atom, Translate):
        return Translate(atom.shape, atom.dx.item() + x, atom.dy.item() + y)
    return Translate(atom, x, y)


class _Rejected(Exception):
    pass


class LayoutGenerator:
    """Builds layouts by offering atoms from ``source`` at ``sites``.

    For every candidate site the source produces an atom, which is
    translated to the site and checked against the constraints in
    ``config``: finite extent, ``min_shape_size``, ``min_feature_size``,
    containment in the domain shrunk by ``edge_margin`` (always enforced, so
    generated layouts never need clipping), and ``min_gap`` against the
    atoms placed so far (via an incremental ``GeometryHash``). Accepted
    atoms are collected into a ``ShapeSet``.

    One ``random.Random(config.seed)`` drives sites and sources in a fixed
    order, so the same seed reproduces the same batch.

    Parameters
    ----------
    config : LayoutConfig
    sites : Sites
        Where candidates go (``LatticeSites``, ``RandomSites``, ...).
    source : AtomSource
        What goes there (``SamplerSource``, ``ParametricSource``, ...).
    validator : LayoutValidator, optional
        Whole-layout check (e.g. a ``LayoutAnalyzer``). A rejected layout
        is regenerated up to ``max_tries_per_layout`` times.
    """

    def __init__(self, config: LayoutConfig, sites: Sites, source: AtomSource,
                 validator: LayoutValidator | None = None):
        self.config = config
        self.sites = sites
        self.source = source
        self.validator = validator
        self._rng = random.Random(config.seed)

    # --- public API ---------------------------------------------------------
    def generate(self, domain: Shape) -> LayoutBatchResult:
        cfg = self.config
        layouts: list[Layout] = []
        stats: list[dict] = []
        failure_reasons: dict[str, int] = {}
        site_rejections: dict[str, int] = {}
        attempts = 0
        for _ in range(cfg.target_count):
            for _ in range(cfg.max_tries_per_layout):
                attempts += 1
                layout, reason, st = self._attempt(domain)
                for k, v in st["rejections"].items():
                    site_rejections[k] = site_rejections.get(k, 0) + v
                if reason is None:
                    layouts.append(layout)
                    stats.append(st)
                    break
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

        ranges: dict[str, tuple[float, float]] = {}
        if stats:
            n = [s["n_shapes"] for s in stats]
            f = [s["fill_fraction"] for s in stats]
            ranges = {"num_shapes": (min(n), max(n)), "fill_fraction": (min(f), max(f))}
        report = GenerationReport(
            requested=cfg.target_count,
            generated=len(layouts),
            failed=cfg.target_count - len(layouts),
            total_attempts=attempts,
            failure_reasons=failure_reasons,
            parameter_ranges=ranges,
            metadata={
                "generator": type(self).__name__,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "sites": type(self.sites).__name__,
                "source": type(self.source).__name__,
                "domain": domain.to_parametric(),
                "config": asdict(cfg),
                "site_rejections": site_rejections,
                "saturated": sum(bool(s["saturated"]) for s in stats),
                "layouts": stats,
            },
        )
        return LayoutBatchResult(layouts=layouts, report=report)

    def generate_one(self, domain: Shape) -> Layout:
        """One layout, retried up to ``max_tries_per_layout`` times.

        Raises ``RuntimeError`` listing the reasons if every attempt fails.
        """
        reasons = []
        for _ in range(self.config.max_tries_per_layout):
            layout, reason, _ = self._attempt(domain)
            if reason is None:
                return layout
            reasons.append(reason)
        raise RuntimeError(f"layout generation failed: {reasons}")

    # --- one layout ---------------------------------------------------------
    def _attempt(self, domain: Shape) -> tuple[Layout | None, str | None, dict]:
        cfg = self.config
        domain_geom = shape_to_shapely(domain)
        domain_area = domain_geom.area
        if not domain_area > 0:
            raise ValueError("domain has zero area")
        inner = domain_geom.buffer(-cfg.edge_margin) if cfg.edge_margin else domain_geom
        shapely.prepare(inner)
        ix0, iy0, ix1, iy1 = inner.bounds if not inner.is_empty else (0.0, 0.0, -1.0, -1.0)

        index = GeometryHash()
        placed: list[Shape] = []
        area = 0.0
        rejections: dict[str, int] = {}
        sites_tried = 0
        consecutive = 0
        saturated = False
        exhaustible = self.sites.exhaustible
        tries = 1 if (self.source.deterministic or not exhaustible) else cfg.max_tries_per_site

        def done() -> bool:
            if cfg.target_num_shapes is not None and len(placed) >= cfg.target_num_shapes:
                return True
            if cfg.target_fill_fraction is not None and area / domain_area >= cfg.target_fill_fraction:
                return True
            return False

        if not done():
            for x, y in self.sites.candidates(domain, self._rng):
                sites_tried += 1
                accepted = False
                for _ in range(tries):
                    try:
                        atom, geom, atom_area = self._candidate(
                            x, y, inner, (ix0, iy0, ix1, iy1), index)
                    except _Rejected as r:
                        key = str(r)
                        rejections[key] = rejections.get(key, 0) + 1
                        continue
                    placed.append(atom)
                    index.insert(geom)
                    area += atom_area
                    accepted = True
                    break
                consecutive = 0 if accepted else consecutive + 1
                if done():
                    break
                if not exhaustible and consecutive >= cfg.max_consecutive_failures:
                    saturated = True
                    break

        if cfg.min_gap is None and placed:
            # atoms may overlap: measure the real union instead of the sum
            fill = shapely.union_all([shape_to_shapely(a) for a in placed]).area / domain_area
        else:
            fill = area / domain_area
        stats = {
            "n_shapes": len(placed),
            "fill_fraction": float(fill),
            "saturated": saturated,
            "sites_tried": sites_tried,
            "rejections": rejections,
        }
        if not placed:
            return None, "empty", stats
        layout = Layout(domain, ShapeSet(placed))
        if self.validator is not None:
            reason = self.validator.validate(layout)
            if reason is not None:
                return None, reason, stats
        return layout, None, stats

    def _candidate(self, x, y, inner, inner_box, index: GeometryHash):
        cfg = self.config
        try:
            atom = self.source.atom(x, y, self._rng)
        except (RuntimeError, ValueError) as e:
            raise _Rejected(str(e) or type(e).__name__) from e
        if atom is None:
            raise _Rejected("no_source")
        b = atom.bounds()
        if is_empty_bounds(b):
            raise _Rejected("empty_atom")
        if not all(math.isfinite(v) for pt in b for v in pt):
            raise _Rejected("infinite_extent")

        placed = _place(atom, x, y)
        (bx0, by0), (bx1, by1) = placed.bounds()
        # cheap bbox pre-check before any Shapely work
        if bx0 < inner_box[0] or by0 < inner_box[1] or bx1 > inner_box[2] or by1 > inner_box[3]:
            raise _Rejected("outside_domain")
        if cfg.min_feature_size is not None:
            fs = placed.min_feature_size
            if fs is not None and fs < cfg.min_feature_size:
                raise _Rejected("feature_too_small")
        raw = shape_to_shapely(placed)
        geom = remove_holes(raw)          # gap/containment ignore holes
        if cfg.min_shape_size is not None:
            gx0, gy0, gx1, gy1 = geom.bounds
            if gx1 - gx0 < cfg.min_shape_size or gy1 - gy0 < cfg.min_shape_size:
                raise _Rejected("shape_too_small")
        if not inner.contains(geom):
            raise _Rejected("outside_domain")
        if cfg.min_gap is not None and index.conflicts(geom, cfg.min_gap):
            raise _Rejected("gap_violation")
        return placed, geom, raw.area
