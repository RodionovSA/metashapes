# metashapes/generators/base.py
# This module defines the base class for unit cell generators.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import datetime
import random

from metashapes.generators.config import GeneratorConfig
from metashapes.generators.report import GenerationBatchResult, GenerationReport
from metashapes.generators.validator import UnitCellValidator
from metashapes.lattice.unit_cell import UnitCell
from metashapes.lattice.canvas import Canvas


class UnitCellGenerator(ABC):
    """
    Base API for periodic unit-cell generators.

    Generation pipeline for each candidate:
    1. sample shapes
    2. build candidate UnitCell
    3. clip/intersect with the unit-cell rectangle defined by canvas
    4. validate final cell
    5. accept or reject
    """

    def __init__(self, config: GeneratorConfig, validator: UnitCellValidator | None = None) -> None:
        self.config = config
        self.validator = validator
        self._rng = random.Random(config.seed)
        
    def generate(self, canvas: "Canvas") -> GenerationBatchResult:
        unit_cells: list[UnitCell] = []

        requested = self.config.target_count
        total_attempts = 0
        failure_reasons: dict[str, int] = {}
    
        for _ in range(requested):
            success = False

            for _ in range(self.config.max_tries_per_cell):
                total_attempts += 1
                cell_canvas = self._sample_canvas(canvas)

                try:
                    cell = self._generate_one(cell_canvas)
                except RuntimeError as e:
                    reason = str(e)
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                    continue

                reason = self._validate(cell)

                if reason is None:
                    unit_cells.append(cell)
                    success = True
                    break

                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

            # if not success: just mark implicitly as failed and move on

        report = GenerationReport(
            requested=requested,
            generated=len(unit_cells),
            failed=requested - len(unit_cells),
            total_attempts=total_attempts,
            failure_reasons=failure_reasons,
            parameter_ranges=self._parameter_ranges(unit_cells),
            metadata=self._build_metadata(canvas),
        )

        return GenerationBatchResult(unit_cells=unit_cells, report=report)

    def _sample_canvas(self, canvas: "Canvas") -> "Canvas":
        """
        Return a (possibly resized) canvas for a single cell.

        Uses ``fixed_canvas_Lx`` / ``canvas_Lx_range`` and the Ly
        equivalents from config.  If neither is set the original canvas
        is returned unchanged.
        """
        cfg = self.config
        needs_resize = (
            cfg.fixed_canvas_Lx is not None
            or cfg.fixed_canvas_Ly is not None
            or cfg.canvas_Lx_range is not None
            or cfg.canvas_Ly_range is not None
        )
        if not needs_resize:
            return canvas

        if cfg.fixed_canvas_Lx is not None:
            Lx = cfg.fixed_canvas_Lx
        elif cfg.canvas_Lx_range is not None:
            lo, hi = cfg.canvas_Lx_range
            Lx = lo + self._rng.random() * (hi - lo)
        else:
            Lx = canvas.Lx

        if cfg.fixed_canvas_Ly is not None:
            Ly = cfg.fixed_canvas_Ly
        elif cfg.canvas_Ly_range is not None:
            lo, hi = cfg.canvas_Ly_range
            Ly = lo + self._rng.random() * (hi - lo)
        else:
            Ly = canvas.Ly

        from dataclasses import replace
        return replace(canvas, Lx=Lx, Ly=Ly)

    @abstractmethod
    def _generate_one(self, canvas: "Canvas") -> UnitCell:
        """
        Generate one raw periodic unit cell candidate before final validation.
        """
        raise NotImplementedError

    def _validate(self, cell: UnitCell) -> str | None:
        if self.validator is None:
            return None
        return self.validator.validate(cell)

    def _build_metadata(self, canvas: Canvas) -> dict:
        """
        Build metadata dict attached to every GenerationReport.
        Subclasses can call super() and extend the result.
        """
        return {
            "generator": type(self).__name__,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "canvas": canvas.to_parametric(),
            "config": asdict(self.config),
        }

    def _parameter_ranges(self, unit_cells: list[UnitCell]) -> dict:
        """
        Summarise key parameter ranges across generated cells.
        Subclasses can override to add shape-specific ranges.
        """
        if not unit_cells:
            return {}
        fills = [float(c.mask().mean().item()) for c in unit_cells]
        return {
            "fill_fraction": (min(fills), max(fills)),
        }