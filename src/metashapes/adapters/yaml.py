# metashapes/adapters/yaml.py
# YAML serialization for UnitCells and GenerationBatchResult.

from __future__ import annotations

from pathlib import Path

import yaml

from metashapes.lattice.unit_cell import UnitCell


class _Dumper(yaml.Dumper):
    """YAML Dumper that puts 'type' first, then sorts remaining keys."""

    def represent_mapping(self, tag, mapping, flow_style=None):
        if isinstance(mapping, dict) and "type" in mapping:
            pairs = [("type", mapping["type"])]
            pairs += sorted((k, v) for k, v in mapping.items() if k != "type")
            return super().represent_mapping(tag, pairs, flow_style)
        return super().represent_mapping(tag, mapping, flow_style)
from metashapes.generators.report import GenerationBatchResult, GenerationReport

__all__ = [
    "save_unit_cells",
    "load_unit_cells",
    "save_batch_result",
    "load_batch_result",
    "save_layouts",
    "load_layouts",
    "save_layout_batch",
    "load_layout_batch",
]

_VERSION = "1"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_unit_cells(
    path: str | Path,
    cells: UnitCell | list | tuple,
) -> None:
    """
    Save one or more unit cells to a YAML file.

    Parameters
    ----------
    path : str or Path
    cells : UnitCell or a list/tuple of them
    """
    if isinstance(cells, UnitCell):
        cells = [cells]
    data = {
        "version": _VERSION,
        "unit_cells": [c.to_parametric() for c in cells],
    }
    Path(path).write_text(yaml.dump(_to_safe(data), Dumper=_Dumper, default_flow_style=False, allow_unicode=True))


def load_unit_cells(path: str | Path) -> list[UnitCell]:
    """
    Load unit cells from a YAML file saved by :func:`save_unit_cells`
    or :func:`save_batch_result`.

    Returns
    -------
    list of UnitCell 
    """
    raw = yaml.safe_load(Path(path).read_text())
    _check_version(raw)
    return [_load_cell(d) for d in raw["unit_cells"]]


def save_batch_result(
    path: str | Path,
    result: GenerationBatchResult,
) -> None:
    """
    Save a :class:`~metashapes.generators.report.GenerationBatchResult`
    (unit cells + generation report) to a YAML file.

    Parameters
    ----------
    path : str or Path
    result : GenerationBatchResult
    """
    r = result.report
    data = {
        "version": _VERSION,
        "report": {
            "requested": r.requested,
            "generated": r.generated,
            "failed": r.failed,
            "total_attempts": r.total_attempts,
            "failure_reasons": dict(r.failure_reasons),
            "parameter_ranges": {
                k: list(v) for k, v in r.parameter_ranges.items()
            },
            "metadata": r.metadata,
        },
        "unit_cells": [c.to_parametric() for c in result.unit_cells],
    }
    Path(path).write_text(yaml.dump(_to_safe(data), Dumper=_Dumper, default_flow_style=False, allow_unicode=True))


def load_batch_result(path: str | Path) -> GenerationBatchResult:
    """
    Load a :class:`~metashapes.generators.report.GenerationBatchResult`
    from a YAML file saved by :func:`save_batch_result`.

    Returns
    -------
    GenerationBatchResult
    """
    raw = yaml.safe_load(Path(path).read_text())
    _check_version(raw)

    r = raw["report"]
    report = GenerationReport(
        requested=r["requested"],
        generated=r["generated"],
        failed=r["failed"],
        total_attempts=r["total_attempts"],
        failure_reasons=r.get("failure_reasons") or {},
        parameter_ranges={
            k: tuple(v) for k, v in (r.get("parameter_ranges") or {}).items()
        },
        metadata=r.get("metadata") or {},
    )
    cells = [_load_cell(d) for d in raw.get("unit_cells") or []]
    return GenerationBatchResult(unit_cells=cells, report=report)


# ---------------------------------------------------------------------------
# Layouts (non-periodic). For 10^5+ atoms prefer the columnar .npz format in
# metashapes.adapters.npz; YAML stays convenient for small layouts.
# ---------------------------------------------------------------------------

def save_layouts(path: str | Path, layouts) -> None:
    """Save one or more :class:`~metashapes.layout.Layout` objects to YAML."""
    from metashapes.layout import Layout
    if isinstance(layouts, Layout):
        layouts = [layouts]
    data = {
        "version": _VERSION,
        "layouts": [lay.to_parametric() for lay in layouts],
    }
    _write(path, data)


def load_layouts(path: str | Path) -> list:
    """Load layouts saved by :func:`save_layouts` or :func:`save_layout_batch`."""
    from metashapes.layout import Layout
    raw = yaml.safe_load(Path(path).read_text())
    _check_version(raw)
    return [Layout.from_parametric(d) for d in raw.get("layouts") or []]


def save_layout_batch(path: str | Path, result) -> None:
    """Save a :class:`~metashapes.layout.LayoutBatchResult` (layouts + report)."""
    data = {
        "version": _VERSION,
        "report": _report_to_dict(result.report),
        "layouts": [lay.to_parametric() for lay in result.layouts],
    }
    _write(path, data)


def load_layout_batch(path: str | Path):
    """Load a :class:`~metashapes.layout.LayoutBatchResult` saved by
    :func:`save_layout_batch`."""
    from metashapes.layout import Layout, LayoutBatchResult
    raw = yaml.safe_load(Path(path).read_text())
    _check_version(raw)
    report = _report_from_dict(raw["report"])
    layouts = [Layout.from_parametric(d) for d in raw.get("layouts") or []]
    return LayoutBatchResult(layouts=layouts, report=report)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(path: str | Path, data: dict) -> None:
    Path(path).write_text(yaml.dump(_to_safe(data), Dumper=_Dumper,
                                    default_flow_style=False, allow_unicode=True))


def _report_to_dict(r: GenerationReport) -> dict:
    return {
        "requested": r.requested,
        "generated": r.generated,
        "failed": r.failed,
        "total_attempts": r.total_attempts,
        "failure_reasons": dict(r.failure_reasons),
        "parameter_ranges": {k: list(v) for k, v in r.parameter_ranges.items()},
        "metadata": r.metadata,
    }


def _report_from_dict(r: dict) -> GenerationReport:
    return GenerationReport(
        requested=r["requested"],
        generated=r["generated"],
        failed=r["failed"],
        total_attempts=r["total_attempts"],
        failure_reasons=r.get("failure_reasons") or {},
        parameter_ranges={
            k: tuple(v) for k, v in (r.get("parameter_ranges") or {}).items()
        },
        metadata=r.get("metadata") or {},
    )


def _to_safe(obj):
    """Recursively convert tuples → lists so yaml.dump produces clean YAML."""
    if isinstance(obj, dict):
        return {k: _to_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_safe(v) for v in obj]
    return obj


def _load_cell(data: dict) -> UnitCell:
    return UnitCell.from_parametric(data)


def _check_version(raw: dict) -> None:
    v = str(raw.get("version", _VERSION))
    if v != _VERSION:
        raise ValueError(
            f"Unsupported YAML version {v!r} (expected {_VERSION!r}). "
            "Re-save the file with the current version of metashapes."
        )
