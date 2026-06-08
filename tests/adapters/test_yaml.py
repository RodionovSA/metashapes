# tests/adapters/test_yaml.py

import pytest
import torch

from metashapes.lattice.basis import Lattice
from metashapes.lattice.unit_cell import UnitCell
from metashapes.adapters.yaml import (
    save_unit_cells,
    load_unit_cells,
    save_batch_result,
    load_batch_result,
)
from metashapes.generators.report import GenerationBatchResult, GenerationReport
from metashapes.shape.primitives.quads import Rectangle
from metashapes.shape.primitives.conics import Ellipse
from metashapes.shape.boolean import Union
from metashapes.shape.transforms import Translate


def _rect_cell(px=2.0, py=2.0):
    lattice = Lattice.rectangular(px, py)
    shape = Rectangle(center=[0.0, 0.0], size=[0.6, 0.4])
    return UnitCell(lattice, shape)


def _sdf_grid(cell):
    xs = torch.linspace(-1.0, 1.0, 16)
    ys = torch.linspace(-1.0, 1.0, 16)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    return cell.sdf(X, Y)


# ---------------------------------------------------------------------------
# save / load unit cells
# ---------------------------------------------------------------------------

class TestSaveLoadUnitCells:
    def test_round_trip_single_cell(self, tmp_path):
        cell = _rect_cell()
        path = tmp_path / "cells.yaml"
        save_unit_cells(path, cell)
        loaded = load_unit_cells(path)
        assert len(loaded) == 1
        assert torch.allclose(_sdf_grid(cell), _sdf_grid(loaded[0]), atol=1e-5)

    def test_round_trip_multiple_cells(self, tmp_path):
        cells = [_rect_cell(2.0, 2.0), _rect_cell(3.0, 3.0), _rect_cell(1.5, 2.5)]
        path = tmp_path / "cells.yaml"
        save_unit_cells(path, cells)
        loaded = load_unit_cells(path)
        assert len(loaded) == 3
        for orig, rest in zip(cells, loaded):
            assert torch.allclose(_sdf_grid(orig), _sdf_grid(rest), atol=1e-5)

    def test_load_always_returns_list(self, tmp_path):
        cell = _rect_cell()
        path = tmp_path / "cell.yaml"
        save_unit_cells(path, cell)
        result = load_unit_cells(path)
        assert isinstance(result, list)

    def test_round_trip_preserves_lattice_vectors(self, tmp_path):
        cell = _rect_cell(3.0, 1.5)
        path = tmp_path / "cell.yaml"
        save_unit_cells(path, cell)
        restored = load_unit_cells(path)[0]
        assert torch.allclose(cell.lattice.a1, restored.lattice.a1)
        assert torch.allclose(cell.lattice.a2, restored.lattice.a2)

    def test_round_trip_hexagonal_lattice(self, tmp_path):
        lattice = Lattice.hexagonal(2.0, orientation="pointy")
        shape = Ellipse(center=[0.0, 0.0], axes=torch.tensor([0.8, 0.6]))
        cell = UnitCell(lattice, shape)
        path = tmp_path / "hex.yaml"
        save_unit_cells(path, cell)
        restored = load_unit_cells(path)[0]
        assert torch.allclose(_sdf_grid(cell), _sdf_grid(restored), atol=1e-5)

    def test_round_trip_complex_shape(self, tmp_path):
        rect = Rectangle(center=[0.0, 0.0], size=[0.5, 0.3])
        ellipse = Ellipse(center=[0.0, 0.0], axes=torch.tensor([0.6, 0.4]))
        shape = Union(Translate(rect, dx=0.3, dy=0.0), ellipse)
        lattice = Lattice.rectangular(2.0, 2.0)
        cell = UnitCell(lattice, shape)
        path = tmp_path / "complex.yaml"
        save_unit_cells(path, cell)
        restored = load_unit_cells(path)[0]
        assert torch.allclose(_sdf_grid(cell), _sdf_grid(restored), atol=1e-5)

    def test_version_mismatch_raises(self, tmp_path):
        path = tmp_path / "bad_version.yaml"
        path.write_text("version: '99'\nunit_cells: []\n")
        with pytest.raises(ValueError, match="Unsupported YAML version"):
            load_unit_cells(path)

    def test_accepts_string_path(self, tmp_path):
        cell = _rect_cell()
        path = str(tmp_path / "string_path.yaml")
        save_unit_cells(path, cell)
        loaded = load_unit_cells(path)
        assert len(loaded) == 1


# ---------------------------------------------------------------------------
# save / load batch result
# ---------------------------------------------------------------------------

def _make_batch_result(n=2):
    cells = [_rect_cell(2.0, 2.0) for _ in range(n)]
    report = GenerationReport(
        requested=10,
        generated=n,
        failed=10 - n,
        total_attempts=12,
        failure_reasons={"gap_too_small": 2},
        parameter_ranges={"side_length": (0.2, 0.8)},
        metadata={"seed": 42},
    )
    return GenerationBatchResult(unit_cells=cells, report=report)


class TestSaveBatchResult:
    def test_round_trip_batch_result(self, tmp_path):
        result = _make_batch_result(n=2)
        path = tmp_path / "batch.yaml"
        save_batch_result(path, result)
        loaded = load_batch_result(path)

        assert len(loaded.unit_cells) == 2
        for orig, rest in zip(result.unit_cells, loaded.unit_cells):
            assert torch.allclose(_sdf_grid(orig), _sdf_grid(rest), atol=1e-5)

    def test_round_trip_preserves_report_counters(self, tmp_path):
        result = _make_batch_result(n=2)
        path = tmp_path / "batch.yaml"
        save_batch_result(path, result)
        loaded = load_batch_result(path)

        r = loaded.report
        assert r.requested == 10
        assert r.generated == 2
        assert r.failed == 8
        assert r.total_attempts == 12

    def test_batch_result_failure_reasons(self, tmp_path):
        result = _make_batch_result()
        path = tmp_path / "batch.yaml"
        save_batch_result(path, result)
        loaded = load_batch_result(path)
        assert loaded.report.failure_reasons == {"gap_too_small": 2}

    def test_batch_result_parameter_ranges(self, tmp_path):
        result = _make_batch_result()
        path = tmp_path / "batch.yaml"
        save_batch_result(path, result)
        loaded = load_batch_result(path)
        lo, hi = loaded.report.parameter_ranges["side_length"]
        assert lo == pytest.approx(0.2)
        assert hi == pytest.approx(0.8)

    def test_batch_result_metadata(self, tmp_path):
        result = _make_batch_result()
        path = tmp_path / "batch.yaml"
        save_batch_result(path, result)
        loaded = load_batch_result(path)
        assert loaded.report.metadata["seed"] == 42

    def test_load_unit_cells_reads_batch_file(self, tmp_path):
        result = _make_batch_result(n=3)
        path = tmp_path / "batch.yaml"
        save_batch_result(path, result)
        # load_unit_cells should also work on batch files
        cells = load_unit_cells(path)
        assert len(cells) == 3
