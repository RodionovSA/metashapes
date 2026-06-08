# tests/generators/test_random_generator.py

import pytest
import torch

from src.metashapes.lattice.basis import Lattice
from src.metashapes.lattice.unit_cell import UnitCell
from src.metashapes.generators.random import RandomUnitCellGenerator, RandomGeneratorConfig
from src.metashapes.generators.report import GenerationBatchResult
from src.metashapes.generators.validator import UnitCellValidator, DefaultUnitCellValidator
from src.metashapes.shape.boolean import Union
from src.metashapes.shape.primitives.quads import Rectangle
from src.metashapes.adapters.shapely.dispatch import shape_to_shapely
from src.metashapes.adapters.yaml import (
    save_batch_result,
    load_batch_result,
    save_unit_cells,
    load_unit_cells,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _gen(target_count=1, seed=42, allowed_shapes=("Rectangle",),
         fixed_num_shapes=1, lattice=None, **kwargs):
    config = RandomGeneratorConfig(
        target_count=target_count,
        seed=seed,
        allowed_shapes=allowed_shapes,
        fixed_num_shapes=fixed_num_shapes,
        **kwargs,
    )
    if lattice is None:
        lattice = Lattice.rectangular(3.0, 3.0)
    return RandomUnitCellGenerator(config).generate(lattice)


def _sdf_grid(cell):
    xs = torch.linspace(0.1, 2.9, 16)
    ys = torch.linspace(0.1, 2.9, 16)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    return cell.sdf(X, Y)


class _AlwaysFailValidator(UnitCellValidator):
    def validate(self, cell):
        return "always_fails"


# ---------------------------------------------------------------------------
# basic generation
# ---------------------------------------------------------------------------

class TestBasicGeneration:
    def test_generates_requested_count(self):
        result = _gen(target_count=5)
        assert result.report.generated == 5
        assert len(result.unit_cells) == 5

    def test_result_cells_are_unit_cells(self):
        result = _gen(target_count=3)
        for cell in result.unit_cells:
            assert isinstance(cell, UnitCell)

    def test_deterministic_with_seed(self):
        r1 = _gen(target_count=3, seed=7)
        r2 = _gen(target_count=3, seed=7)
        assert len(r1.unit_cells) == len(r2.unit_cells)
        for c1, c2 in zip(r1.unit_cells, r2.unit_cells):
            assert torch.allclose(_sdf_grid(c1), _sdf_grid(c2))

    def test_different_seeds_produce_different_results(self):
        r1 = _gen(target_count=3, seed=1)
        r2 = _gen(target_count=3, seed=2)
        # At least one cell pair should differ
        any_diff = any(
            not torch.allclose(_sdf_grid(c1), _sdf_grid(c2))
            for c1, c2 in zip(r1.unit_cells, r2.unit_cells)
        )
        assert any_diff

    def test_no_validator_generates_all(self):
        config = RandomGeneratorConfig(
            target_count=3, seed=42,
            allowed_shapes=("Rectangle",), fixed_num_shapes=1,
        )
        result = RandomUnitCellGenerator(config, validator=None).generate(
            Lattice.rectangular(3.0, 3.0)
        )
        assert result.report.generated == 3

    def test_generated_cells_have_non_empty_shapely(self):
        result = _gen(target_count=2)
        for cell in result.unit_cells:
            geom = cell.to_shapely()
            assert not geom.is_empty
            assert geom.area > 0.0


# ---------------------------------------------------------------------------
# shape types
# ---------------------------------------------------------------------------

class TestShapeTypes:
    @pytest.mark.parametrize("shape_name", [
        "Rectangle",
        "Ellipse",
        "Egg",
        "Stadium",
        "Triangle",
        "Star",
        "RegularPolygon",
        "ConvexQuad",
        "IsoscelesTrapezoid",
        "Cross",
        "TShape",
        "Stripe",
    ])
    def test_each_shape_type_generates(self, shape_name):
        result = _gen(
            target_count=1,
            allowed_shapes=(shape_name,),
            fixed_num_shapes=1,
            seed=42,
        )
        assert result.report.generated == 1
        geom = result.unit_cells[0].to_shapely()
        assert not geom.is_empty

    def test_all_shapes_allowed_when_empty_tuple(self):
        result = _gen(target_count=1, allowed_shapes=(), fixed_num_shapes=1)
        assert result.report.generated == 1

    def test_shape_weights_exclude_zero_weight(self):
        # Ellipse has weight 0 → only Rectangle should appear
        result = _gen(
            target_count=5,
            allowed_shapes=("Rectangle", "Ellipse"),
            shape_weights={"Rectangle": 1.0, "Ellipse": 0.0},
            fixed_num_shapes=1,
            seed=42,
        )
        assert result.report.generated == 5
        for cell in result.unit_cells:
            assert isinstance(cell.scene, Rectangle)


# ---------------------------------------------------------------------------
# shape count
# ---------------------------------------------------------------------------

class TestShapeCount:
    def test_fixed_one_shape_is_not_union(self):
        result = _gen(target_count=1, fixed_num_shapes=1)
        assert not isinstance(result.unit_cells[0].scene, Union)

    def test_fixed_two_shapes_is_union(self):
        result = _gen(target_count=1, fixed_num_shapes=2)
        assert isinstance(result.unit_cells[0].scene, Union)

    def test_fixed_three_shapes_generates(self):
        result = _gen(target_count=1, fixed_num_shapes=3)
        assert result.report.generated == 1
        assert isinstance(result.unit_cells[0].scene, Union)

    def test_shape_count_range_generates(self):
        config = RandomGeneratorConfig(
            target_count=3, seed=42,
            allowed_shapes=("Rectangle",),
            min_num_shapes=2, max_num_shapes=3,
        )
        result = RandomUnitCellGenerator(config).generate(Lattice.rectangular(3.0, 3.0))
        assert result.report.generated == 3


# ---------------------------------------------------------------------------
# constraints
# ---------------------------------------------------------------------------

class TestConstraints:
    def test_min_gap_positive_generates(self):
        result = _gen(target_count=2, fixed_num_shapes=1, min_gap=0.05, seed=1)
        assert result.report.generated == 2

    def test_min_shape_size_respected(self):
        threshold = 0.4
        result = _gen(target_count=3, min_shape_size=threshold, seed=42)
        for cell in result.unit_cells:
            geom = shape_to_shapely(cell.scene)
            minx, miny, maxx, maxy = geom.bounds
            assert (maxx - minx) >= threshold - 1e-4
            assert (maxy - miny) >= threshold - 1e-4

    def test_min_shape_size_too_large_fails_all(self):
        # min_shape_size larger than the entire 3×3 lattice → no cell can succeed
        result = _gen(target_count=3, min_shape_size=100.0, seed=42)
        assert result.report.generated == 0
        assert len(result.unit_cells) == 0


# ---------------------------------------------------------------------------
# lattice variants
# ---------------------------------------------------------------------------

class TestLatticeVariants:
    def test_rectangular_lattice(self):
        result = _gen(target_count=1, lattice=Lattice.rectangular(2.0, 2.0))
        assert result.report.generated == 1

    def test_hexagonal_lattice(self):
        result = _gen(target_count=1, lattice=Lattice.hexagonal(2.0))
        assert result.report.generated == 1
        geom = result.unit_cells[0].to_shapely()
        assert not geom.is_empty

    def test_fixed_lattice_lx_overrides_lattice(self):
        config = RandomGeneratorConfig(
            target_count=1, seed=42,
            allowed_shapes=("Rectangle",), fixed_num_shapes=1,
            fixed_lattice_Lx=4.0, fixed_lattice_Ly=4.0,
        )
        result = RandomUnitCellGenerator(config).generate(Lattice.rectangular(1.0, 1.0))
        assert result.report.generated == 1
        cell = result.unit_cells[0]
        assert float(cell.lattice.a1.norm()) == pytest.approx(4.0, abs=0.01)
        assert float(cell.lattice.a2.norm()) == pytest.approx(4.0, abs=0.01)

    def test_lattice_lx_range_produces_varied_sizes(self):
        config = RandomGeneratorConfig(
            target_count=5, seed=42,
            allowed_shapes=("Rectangle",), fixed_num_shapes=1,
            lattice_Lx_range=(2.0, 4.0), lattice_Ly_range=(2.0, 4.0),
        )
        result = RandomUnitCellGenerator(config).generate(Lattice.rectangular(1.0, 1.0))
        assert result.report.generated == 5
        for cell in result.unit_cells:
            lx = float(cell.lattice.a1.norm())
            ly = float(cell.lattice.a2.norm())
            assert 2.0 - 1e-6 <= lx <= 4.0 + 1e-6
            assert 2.0 - 1e-6 <= ly <= 4.0 + 1e-6

    def test_hexagonal_lattice_generates_cells(self):
        config = RandomGeneratorConfig(
            target_count=5, seed=42,
            allowed_shapes=("Rectangle",), fixed_num_shapes=1,
        )
        result = RandomUnitCellGenerator(config).generate(Lattice.hexagonal(1.0))
        assert result.report.generated == 5
        for cell in result.unit_cells:
            assert cell.lattice.a1.norm().item() == pytest.approx(
                cell.lattice.a2.norm().item(), rel=1e-5
            )

    def test_fixed_lattice_L_uniform_scaling(self):
        config = RandomGeneratorConfig(
            target_count=3, seed=0,
            allowed_shapes=("Rectangle",), fixed_num_shapes=1,
            fixed_lattice_L=2.0,
        )
        result = RandomUnitCellGenerator(config).generate(Lattice.hexagonal(1.0))
        assert result.report.generated == 3
        for cell in result.unit_cells:
            assert cell.lattice.a1.norm().item() == pytest.approx(2.0, abs=1e-5)
            assert cell.lattice.a2.norm().item() == pytest.approx(2.0, abs=1e-5)

    def test_lattice_L_range_uniform_scaling(self):
        config = RandomGeneratorConfig(
            target_count=5, seed=7,
            allowed_shapes=("Rectangle",), fixed_num_shapes=1,
            lattice_L_range=(1.0, 3.0),
        )
        result = RandomUnitCellGenerator(config).generate(Lattice.hexagonal(1.0))
        assert result.report.generated == 5
        for cell in result.unit_cells:
            n1 = cell.lattice.a1.norm().item()
            n2 = cell.lattice.a2.norm().item()
            assert n1 == pytest.approx(n2, rel=1e-5)
            assert 1.0 - 1e-5 <= n1 <= 3.0 + 1e-5

    def test_lattice_L_takes_priority_over_lx_ly(self):
        # lattice_L_range takes priority; Lx_range is ignored
        config = RandomGeneratorConfig(
            target_count=3, seed=1,
            allowed_shapes=("Rectangle",), fixed_num_shapes=1,
            fixed_lattice_L=2.0,
            lattice_Lx_range=(5.0, 6.0),  # should be ignored
        )
        result = RandomUnitCellGenerator(config).generate(Lattice.hexagonal(1.0))
        for cell in result.unit_cells:
            assert cell.lattice.a1.norm().item() == pytest.approx(2.0, abs=1e-5)


# ---------------------------------------------------------------------------
# report correctness
# ---------------------------------------------------------------------------

class TestReport:
    def test_requested_equals_target_count(self):
        result = _gen(target_count=4)
        assert result.report.requested == 4

    def test_generated_plus_failed_equals_requested(self):
        result = _gen(target_count=5)
        r = result.report
        assert r.generated + r.failed == r.requested

    def test_total_attempts_at_least_generated(self):
        result = _gen(target_count=3)
        assert result.report.total_attempts >= result.report.generated

    def test_fill_fraction_in_range(self):
        result = _gen(target_count=3)
        lo, hi = result.report.parameter_ranges["fill_fraction"]
        assert 0.0 < lo
        assert hi < 1.0
        assert lo <= hi

    def test_report_metadata_keys(self):
        result = _gen(target_count=1)
        meta = result.report.metadata
        for key in ("generator", "generated_at", "lattice", "config"):
            assert key in meta

    def test_report_metadata_generator_name(self):
        result = _gen(target_count=1)
        assert result.report.metadata["generator"] == "RandomUnitCellGenerator"

    def test_report_metadata_lattice_vectors(self):
        result = _gen(target_count=1, lattice=Lattice.rectangular(3.0, 3.0))
        lmeta = result.report.metadata["lattice"]
        assert "a1" in lmeta and "a2" in lmeta

    def test_failure_reasons_from_custom_validator(self):
        config = RandomGeneratorConfig(
            target_count=3, seed=42,
            allowed_shapes=("Rectangle",), fixed_num_shapes=1,
        )
        result = RandomUnitCellGenerator(config, validator=_AlwaysFailValidator()).generate(
            Lattice.rectangular(3.0, 3.0)
        )
        assert result.report.generated == 0
        assert "always_fails" in result.report.failure_reasons


# ---------------------------------------------------------------------------
# validator
# ---------------------------------------------------------------------------

class TestValidator:
    def test_custom_validator_rejects_all(self):
        config = RandomGeneratorConfig(
            target_count=3, seed=42,
            allowed_shapes=("Rectangle",), fixed_num_shapes=1,
        )
        result = RandomUnitCellGenerator(config, validator=_AlwaysFailValidator()).generate(
            Lattice.rectangular(3.0, 3.0)
        )
        assert result.report.generated == 0
        assert result.unit_cells == []
        assert result.report.failure_reasons.get("always_fails", 0) > 0

    def test_default_validator_accepts_normal_cells(self):
        config = RandomGeneratorConfig(
            target_count=3, seed=42,
            allowed_shapes=("Rectangle",), fixed_num_shapes=1,
        )
        result = RandomUnitCellGenerator(config, validator=DefaultUnitCellValidator()).generate(
            Lattice.rectangular(3.0, 3.0)
        )
        assert result.report.generated == 3


# ---------------------------------------------------------------------------
# merge batch results
# ---------------------------------------------------------------------------

class TestMergeBatchResults:
    def test_merge_two_results_cell_count(self):
        lattice = Lattice.rectangular(3.0, 3.0)
        config = RandomGeneratorConfig(
            target_count=2, seed=42,
            allowed_shapes=("Rectangle",), fixed_num_shapes=1,
        )
        gen = RandomUnitCellGenerator(config)
        r1 = gen.generate(lattice)
        r2 = gen.generate(lattice)
        merged = GenerationBatchResult.merge(r1, r2)
        assert len(merged.unit_cells) == 4

    def test_merge_sums_generated_counter(self):
        lattice = Lattice.rectangular(3.0, 3.0)
        config = RandomGeneratorConfig(
            target_count=2, seed=42,
            allowed_shapes=("Rectangle",), fixed_num_shapes=1,
        )
        gen = RandomUnitCellGenerator(config)
        r1 = gen.generate(lattice)
        r2 = gen.generate(lattice)
        merged = GenerationBatchResult.merge(r1, r2)
        assert merged.report.generated == r1.report.generated + r2.report.generated
        assert merged.report.requested == r1.report.requested + r2.report.requested


# ---------------------------------------------------------------------------
# YAML saving / loading
# ---------------------------------------------------------------------------

class TestYAMLSaving:
    def test_save_load_batch_result(self, tmp_path):
        result = _gen(target_count=3)
        path = tmp_path / "gen.yaml"
        save_batch_result(path, result)
        loaded = load_batch_result(path)

        assert len(loaded.unit_cells) == len(result.unit_cells)
        for orig, rest in zip(result.unit_cells, loaded.unit_cells):
            assert torch.allclose(_sdf_grid(orig), _sdf_grid(rest), atol=1e-5)

    def test_save_load_cells_only(self, tmp_path):
        result = _gen(target_count=3)
        path = tmp_path / "cells.yaml"
        save_unit_cells(path, result.unit_cells)
        loaded = load_unit_cells(path)
        assert len(loaded) == 3

    def test_loaded_report_counters_preserved(self, tmp_path):
        result = _gen(target_count=3)
        path = tmp_path / "gen.yaml"
        save_batch_result(path, result)
        loaded = load_batch_result(path)
        r = loaded.report
        assert r.requested == result.report.requested
        assert r.generated == result.report.generated
        assert r.failed == result.report.failed
        assert r.total_attempts == result.report.total_attempts

    def test_loaded_report_metadata_preserved(self, tmp_path):
        result = _gen(target_count=1)
        path = tmp_path / "gen.yaml"
        save_batch_result(path, result)
        loaded = load_batch_result(path)
        meta = loaded.report.metadata
        assert meta["generator"] == "RandomUnitCellGenerator"
        assert "a1" in meta["lattice"]

    def test_complex_shapes_survive_yaml_round_trip(self, tmp_path):
        result = _gen(target_count=2, fixed_num_shapes=2, seed=10)
        path = tmp_path / "complex.yaml"
        save_batch_result(path, result)
        loaded = load_batch_result(path)
        for orig, rest in zip(result.unit_cells, loaded.unit_cells):
            assert torch.allclose(_sdf_grid(orig), _sdf_grid(rest), atol=1e-5)

    def test_mixed_shape_types_survive_yaml_round_trip(self, tmp_path):
        result = _gen(target_count=2, allowed_shapes=("Rectangle", "Ellipse"),
                      fixed_num_shapes=1, seed=5)
        path = tmp_path / "mixed.yaml"
        save_batch_result(path, result)
        loaded = load_batch_result(path)
        assert len(loaded.unit_cells) == len(result.unit_cells)
        for orig, rest in zip(result.unit_cells, loaded.unit_cells):
            assert torch.allclose(_sdf_grid(orig), _sdf_grid(rest), atol=1e-5)
