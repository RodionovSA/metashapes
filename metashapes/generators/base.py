# metashapes/generators/base.py
# This module defines the base class for unit cell generators.

from __future__ import annotations

from abc import ABC, abstractmethod
import random

from metashapes.generators.config import GeneratorConfig
from metashapes.generators.report import GenerationBatchResult, GenerationReport
from metashapes.generators.validator import UnitCellValidator
from metashapes.unit_cell import UnitCell
from metashapes.canvas import Canvas


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

                try:
                    cell = self._generate_one(canvas)
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
        )

        return GenerationBatchResult(unit_cells=unit_cells, report=report)

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

    def _count_shapes(self, cell: UnitCell) -> int:
        return 0

    def _parameter_ranges(self, unit_cells: list[UnitCell]) -> dict[str, tuple[float, float]]:
        """
        Optional batch summary hook. Override later.
        """
        return {}