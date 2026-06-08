# metashapes/generators/report.py
# This module defines the GenerationReport and GenerationBatchResult dataclasses, 
# which encapsulate the results and statistics of the unit cell generation process. 
# These classes are used to track the number of requested, generated, and failed unit cells, 
# as well as details about accepted shapes and failure reasons.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from metashapes.lattice.unit_cell import UnitCell


@dataclass(slots=True)
class GenerationReport:
    requested: int
    generated: int
    failed: int
    total_attempts: int

    failure_reasons: dict[str, int] = field(default_factory=dict)

    parameter_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationBatchResult:
    unit_cells: list["UnitCell"]
    report: GenerationReport

    @classmethod
    def merge(cls, *results: "GenerationBatchResult") -> "GenerationBatchResult":
        """
        Merge multiple GenerationBatchResult objects into one.

        Unit cells are concatenated in order. Report counters are summed.
        ``failure_reasons`` counts are added per key. ``parameter_ranges``
        takes the widest interval across all results for each key.
        ``metadata`` from each run is collected under a ``"runs"`` list.

        Parameters
        ----------
        *results : GenerationBatchResult
            Two or more results to merge. A single result is returned as-is.

        Returns
        -------
        GenerationBatchResult
        """
        if not results:
            raise ValueError("merge() requires at least one result")
        if len(results) == 1:
            return results[0]

        cells: list[UnitCell] = []
        requested = generated = failed = total_attempts = 0
        failure_reasons: dict[str, int] = {}
        parameter_ranges: dict[str, tuple[float, float]] = {}
        run_metadata: list[dict[str, Any]] = []

        for r in results:
            cells.extend(r.unit_cells)
            rep = r.report
            requested += rep.requested
            generated += rep.generated
            failed += rep.failed
            total_attempts += rep.total_attempts

            for key, count in rep.failure_reasons.items():
                failure_reasons[key] = failure_reasons.get(key, 0) + count

            for key, (lo, hi) in rep.parameter_ranges.items():
                if key in parameter_ranges:
                    old_lo, old_hi = parameter_ranges[key]
                    parameter_ranges[key] = (min(old_lo, lo), max(old_hi, hi))
                else:
                    parameter_ranges[key] = (lo, hi)

            if rep.metadata:
                run_metadata.append(rep.metadata)

        report = GenerationReport(
            requested=requested,
            generated=generated,
            failed=failed,
            total_attempts=total_attempts,
            failure_reasons=failure_reasons,
            parameter_ranges=parameter_ranges,
            metadata={"runs": run_metadata} if run_metadata else {},
        )
        return cls(unit_cells=cells, report=report)