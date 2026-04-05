# metashapes/generators/report.py
# This module defines the GenerationReport and GenerationBatchResult dataclasses, 
# which encapsulate the results and statistics of the unit cell generation process. 
# These classes are used to track the number of requested, generated, and failed unit cells, 
# as well as details about accepted shapes and failure reasons.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from metashapes.unit_cell import UnitCell


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