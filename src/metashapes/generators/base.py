# metashapes/generators/base.py
# This module defines the base class for unit cell generators.

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import datetime
import random

from src.metashapes.generators.config import GeneratorConfig
from src.metashapes.generators.report import GenerationBatchResult, GenerationReport
from src.metashapes.generators.validator import UnitCellValidator
from src.metashapes.lattice.basis import Lattice
from src.metashapes.lattice.unit_cell import UnitCell


class UnitCellGenerator(ABC):
    """
    Base API for periodic unit-cell generators.

    Generation pipeline for each candidate:
    1. sample shapes
    2. build candidate UnitCell
    3. validate
    4. accept or reject
    """

    def __init__(self, config: GeneratorConfig, validator: UnitCellValidator | None = None) -> None:
        self.config = config
        self.validator = validator
        self._rng = random.Random(config.seed)

    def generate(self, lattice: Lattice) -> GenerationBatchResult:
        unit_cells: list[UnitCell] = []

        requested = self.config.target_count
        total_attempts = 0
        failure_reasons: dict[str, int] = {}

        for _ in range(requested):
            success = False

            for _ in range(self.config.max_tries_per_cell):
                total_attempts += 1
                cell_lattice = self._sample_lattice(lattice)

                try:
                    cell = self._generate_one(cell_lattice)
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

            # if not success: implicitly counted as failed

        report = GenerationReport(
            requested=requested,
            generated=len(unit_cells),
            failed=requested - len(unit_cells),
            total_attempts=total_attempts,
            failure_reasons=failure_reasons,
            parameter_ranges=self._parameter_ranges(unit_cells),
            metadata=self._build_metadata(lattice),
        )

        return GenerationBatchResult(unit_cells=unit_cells, report=report)

    def _sample_lattice(self, lattice: Lattice) -> Lattice:
        """Return a (possibly rescaled) lattice for a single cell.

        Uniform scaling (``fixed_lattice_L`` / ``lattice_L_range``) scales
        both vectors by the same factor, preserving the lattice angles and
        relative vector lengths.  Use this for hexagonal or square lattices.

        Per-axis scaling (``fixed_lattice_Lx/Ly`` / ``lattice_Lx/Ly_range``)
        scales each vector's norm independently.  Intended for rectangular
        lattices; using it on an oblique lattice distorts the angles.

        Uniform fields take priority if both are set.
        """
        cfg = self.config
        needs_resize = any([
            cfg.fixed_lattice_L, cfg.lattice_L_range,
            cfg.fixed_lattice_Lx, cfg.fixed_lattice_Ly,
            cfg.lattice_Lx_range, cfg.lattice_Ly_range,
        ])
        if not needs_resize:
            return lattice

        # --- uniform scaling (angle-preserving) ---
        if cfg.fixed_lattice_L is not None or cfg.lattice_L_range is not None:
            L_orig = float(lattice.a1.norm())
            if cfg.fixed_lattice_L is not None:
                L = cfg.fixed_lattice_L
            else:
                lo, hi = cfg.lattice_L_range
                L = lo + self._rng.random() * (hi - lo)
            scale = L / L_orig
            return Lattice(lattice.a1 * scale, lattice.a2 * scale)

        # --- independent per-vector scaling (rectangular lattices) ---
        Lx_orig = float(lattice.a1.norm())
        Ly_orig = float(lattice.a2.norm())

        if cfg.fixed_lattice_Lx is not None:
            Lx = cfg.fixed_lattice_Lx
        elif cfg.lattice_Lx_range is not None:
            lo, hi = cfg.lattice_Lx_range
            Lx = lo + self._rng.random() * (hi - lo)
        else:
            Lx = Lx_orig

        if cfg.fixed_lattice_Ly is not None:
            Ly = cfg.fixed_lattice_Ly
        elif cfg.lattice_Ly_range is not None:
            lo, hi = cfg.lattice_Ly_range
            Ly = lo + self._rng.random() * (hi - lo)
        else:
            Ly = Ly_orig

        return Lattice(lattice.a1 * (Lx / Lx_orig), lattice.a2 * (Ly / Ly_orig))

    @abstractmethod
    def _generate_one(self, lattice: Lattice) -> UnitCell:
        """Generate one raw periodic unit cell candidate before final validation."""
        raise NotImplementedError

    def _validate(self, cell: UnitCell) -> str | None:
        if self.validator is None:
            return None
        return self.validator.validate(cell)

    def _build_metadata(self, lattice: Lattice) -> dict:
        """Build metadata dict attached to every GenerationReport."""
        return {
            "generator": type(self).__name__,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "lattice": {"a1": lattice.a1.tolist(), "a2": lattice.a2.tolist()},
            "config": asdict(self.config),
        }

    def _parameter_ranges(self, unit_cells: list[UnitCell]) -> dict:
        """Summarise key parameter ranges across generated cells."""
        if not unit_cells:
            return {}
        fills = [
            c.to_shapely().area / float(c.lattice.cell_area.item())
            for c in unit_cells
        ]
        return {"fill_fraction": (min(fills), max(fills))}
