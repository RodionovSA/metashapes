# tests/test_analysis.py

import math
import pytest
import torch

from metashapes.lattice.basis import Lattice
from metashapes.lattice.unit_cell import UnitCell
from metashapes.shape.primitives.quads import Rectangle
from metashapes.shape.primitives.conics import Ellipse
from metashapes.shape.primitives.periodic import Stripe
from metashapes.shape.boolean import Union, Intersection, Difference
from metashapes.shape.transforms import Translate, Rotate, Scale
from metashapes.analysis import (
    CellMetrics,
    UnitCellAnalyzer,
    _leaf_shapes,
    _compute_min_gap,
    _skip_self_shift,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _rect(cx=0.0, cy=0.0, w=0.4, h=0.4):
    return Rectangle(
        center=torch.tensor([cx, cy]),
        size=torch.tensor([w, h]),
    )


def _cell(shape, px=1.0, py=1.0):
    return UnitCell(Lattice.rectangular(px, py), shape)


# ---------------------------------------------------------------------------
# _leaf_shapes
# ---------------------------------------------------------------------------

class TestLeafShapes:
    def test_primitive_yields_itself(self):
        r = _rect()
        leaves = list(_leaf_shapes(r))
        assert len(leaves) == 1
        assert leaves[0] is r

    def test_union_yields_both(self):
        a, b = _rect(cx=-0.3), _rect(cx=0.3)
        u = a | b
        leaves = list(_leaf_shapes(u))
        assert len(leaves) == 2

    def test_deep_union(self):
        shapes = [_rect(cx=i * 0.2) for i in range(4)]
        u = shapes[0] | shapes[1] | shapes[2] | shapes[3]
        assert len(list(_leaf_shapes(u))) == 4

    def test_intersection_is_atomic(self):
        a, b = _rect(), _rect(w=0.2, h=0.6)
        inter = a & b
        leaves = list(_leaf_shapes(inter))
        assert len(leaves) == 1
        assert isinstance(leaves[0], Intersection)

    def test_difference_is_atomic(self):
        a, b = _rect(), _rect(w=0.1, h=0.1)
        diff = a - b
        leaves = list(_leaf_shapes(diff))
        assert len(leaves) == 1
        assert isinstance(leaves[0], Difference)

    def test_translate_wraps_leaf(self):
        r = _rect()
        t = r.translate(0.2, 0.1)
        leaves = list(_leaf_shapes(t))
        assert len(leaves) == 1
        # Leaf is a new Translate wrapping the original rect
        assert isinstance(leaves[0], Translate)

    def test_rotate_wraps_leaf(self):
        r = _rect()
        rot = r.rotate(30.0)
        leaves = list(_leaf_shapes(rot))
        assert len(leaves) == 1
        assert isinstance(leaves[0], Rotate)

    def test_scale_wraps_leaf(self):
        r = _rect()
        s = r.scale(2.0)
        leaves = list(_leaf_shapes(s))
        assert len(leaves) == 1
        assert isinstance(leaves[0], Scale)

    def test_translate_union_splits(self):
        a, b = _rect(cx=-0.3), _rect(cx=0.3)
        u = (a | b).translate(0.1, 0.0)
        leaves = list(_leaf_shapes(u))
        assert len(leaves) == 2
        assert all(isinstance(l, Translate) for l in leaves)


# ---------------------------------------------------------------------------
# CellMetrics
# ---------------------------------------------------------------------------

class TestCellMetrics:
    def test_default_fields(self):
        m = CellMetrics(fill_fraction=0.25, num_shapes=1)
        assert m.shape_areas == []
        assert m.shape_bbox_sizes == []
        assert m.min_feature_size is None
        assert m.min_gap is None

    def test_fields_set(self):
        m = CellMetrics(
            fill_fraction=0.5,
            num_shapes=2,
            shape_areas=[0.1, 0.2],
            shape_bbox_sizes=[(0.3, 0.4), (0.5, 0.6)],
            min_feature_size=0.1,
            min_gap=0.05,
        )
        assert m.fill_fraction == pytest.approx(0.5)
        assert m.num_shapes == 2
        assert m.min_gap == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# UnitCellAnalyzer.metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_num_shapes_single(self):
        cell = _cell(_rect())
        m = UnitCellAnalyzer().metrics(cell)
        assert m.num_shapes == 1

    def test_num_shapes_union(self):
        a, b = _rect(cx=-0.2), _rect(cx=0.2)
        cell = _cell(a | b)
        m = UnitCellAnalyzer().metrics(cell)
        assert m.num_shapes == 2

    def test_fill_fraction_range(self):
        cell = _cell(_rect(w=0.4, h=0.4), px=1.0, py=1.0)
        m = UnitCellAnalyzer().metrics(cell)
        assert 0.0 < m.fill_fraction < 1.0

    def test_fill_fraction_small_shape_low(self):
        # 0.1×0.1 shape in 2.0×2.0 cell → fill ~ 0.0025 / 4.0 = 0.0025
        cell = _cell(_rect(w=0.1, h=0.1), px=2.0, py=2.0)
        m = UnitCellAnalyzer().metrics(cell)
        assert m.fill_fraction < 0.1

    def test_fill_fraction_large_shape_high(self):
        # Shape nearly fills the unit cell
        cell = _cell(_rect(w=0.9, h=0.9), px=1.0, py=1.0)
        m = UnitCellAnalyzer().metrics(cell)
        assert m.fill_fraction > 0.5

    def test_shape_areas_positive(self):
        cell = _cell(_rect(w=0.3, h=0.4))
        m = UnitCellAnalyzer().metrics(cell)
        assert len(m.shape_areas) == 1
        assert m.shape_areas[0] > 0.0

    def test_shape_bbox_sizes(self):
        cell = _cell(_rect(w=0.3, h=0.4))
        m = UnitCellAnalyzer().metrics(cell)
        w, h = m.shape_bbox_sizes[0]
        assert w == pytest.approx(0.3, abs=0.02)
        assert h == pytest.approx(0.4, abs=0.02)

    def test_min_feature_size_rectangle(self):
        # Rectangle min_feature_size = min(w, h)
        cell = _cell(_rect(w=0.2, h=0.5))
        m = UnitCellAnalyzer().metrics(cell)
        assert m.min_feature_size == pytest.approx(0.2, abs=1e-4)

    def test_min_feature_size_none_for_unknown(self):
        # Intersection/Difference shapes return None from base Shape
        a, b = _rect(w=0.6, h=0.3), _rect(w=0.3, h=0.6)
        cell = _cell(a & b)
        m = UnitCellAnalyzer().metrics(cell)
        assert m.min_feature_size is None

    def test_min_gap_two_shapes(self):
        # Two rects at x=-0.3 and x=+0.3, each with width 0.2
        # right edge of left rect: -0.3 + 0.1 = -0.2
        # left edge of right rect:  0.3 - 0.1 = +0.2
        # gap = 0.4
        a = _rect(cx=-0.3, w=0.2, h=0.2)
        b = _rect(cx=0.3, w=0.2, h=0.2)
        cell = _cell(a | b, px=2.0, py=2.0)
        m = UnitCellAnalyzer().metrics(cell)
        assert m.min_gap is not None
        assert m.min_gap > 0.0

    def test_min_gap_single_shape(self):
        # Single shape — gap measured to periodic image
        cell = _cell(_rect(w=0.2, h=0.2), px=1.0, py=1.0)
        m = UnitCellAnalyzer().metrics(cell)
        assert m.min_gap is not None
        assert m.min_gap > 0.0

    def test_stripe_min_gap_not_zero(self):
        # Stripe infinite in x — self-periodic images in x direction should be skipped
        stripe = Stripe(offset=0.0, width=0.3, axis='x')
        cell = _cell(stripe, px=1.0, py=1.0)
        m = UnitCellAnalyzer().metrics(cell)
        # min_gap should be the gap to y-periodic image, not 0
        assert m.min_gap is not None
        assert m.min_gap > 0.0


# ---------------------------------------------------------------------------
# UnitCellAnalyzer.check
# ---------------------------------------------------------------------------

class TestCheck:
    def test_no_violations_empty(self):
        cell = _cell(_rect())
        failures = UnitCellAnalyzer().check(cell)
        assert failures == []

    def test_min_fill_violation(self):
        cell = _cell(_rect(w=0.1, h=0.1), px=2.0, py=2.0)
        failures = UnitCellAnalyzer(min_fill=0.9).check(cell)
        assert any("fill_fraction" in f and "min_fill" in f for f in failures)

    def test_max_fill_violation(self):
        cell = _cell(_rect(w=0.9, h=0.9))
        failures = UnitCellAnalyzer(max_fill=0.1).check(cell)
        assert any("fill_fraction" in f and "max_fill" in f for f in failures)

    def test_fill_passes(self):
        cell = _cell(_rect(w=0.5, h=0.5))
        failures = UnitCellAnalyzer(min_fill=0.1, max_fill=0.9).check(cell)
        assert not any("fill_fraction" in f for f in failures)

    def test_min_num_shapes_violation(self):
        cell = _cell(_rect())
        failures = UnitCellAnalyzer(min_num_shapes=2).check(cell)
        assert any("num_shapes" in f and "min_num_shapes" in f for f in failures)

    def test_max_num_shapes_violation(self):
        a, b = _rect(cx=-0.3), _rect(cx=0.3)
        cell = _cell(a | b)
        failures = UnitCellAnalyzer(max_num_shapes=1).check(cell)
        assert any("num_shapes" in f and "max_num_shapes" in f for f in failures)

    def test_min_shape_size_violation(self):
        cell = _cell(_rect(w=0.1, h=0.5))
        failures = UnitCellAnalyzer(min_shape_size=0.4).check(cell)
        assert any("min_shape_size" in f for f in failures)

    def test_min_feature_size_violation(self):
        cell = _cell(_rect(w=0.05, h=0.5))
        failures = UnitCellAnalyzer(min_feature_size=0.1).check(cell)
        assert any("min_feature_size" in f for f in failures)

    def test_min_gap_violation(self):
        # Two rects very close together
        a = _rect(cx=-0.1, w=0.15, h=0.2)
        b = _rect(cx=0.1, w=0.15, h=0.2)
        cell = _cell(a | b, px=1.0, py=1.0)
        m_gap = UnitCellAnalyzer().metrics(cell).min_gap
        if m_gap is not None:
            # Set min_gap just above the actual gap
            failures = UnitCellAnalyzer(min_gap=m_gap + 0.1).check(cell)
            assert any("min_gap" in f for f in failures)

    def test_multiple_violations(self):
        cell = _cell(_rect(w=0.1, h=0.1), px=2.0, py=2.0)
        failures = UnitCellAnalyzer(min_fill=0.9, min_num_shapes=5).check(cell)
        assert len(failures) >= 2


# ---------------------------------------------------------------------------
# UnitCellAnalyzer.validate
# ---------------------------------------------------------------------------

class TestValidate:
    def test_valid_returns_none(self):
        cell = _cell(_rect())
        result = UnitCellAnalyzer().validate(cell)
        assert result is None

    def test_invalid_returns_string(self):
        cell = _cell(_rect(w=0.1, h=0.1), px=2.0, py=2.0)
        result = UnitCellAnalyzer(min_fill=0.9).validate(cell)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_first_failure(self):
        cell = _cell(_rect(w=0.1, h=0.1), px=2.0, py=2.0)
        # Both fill and num_shapes constraints violated
        failures = UnitCellAnalyzer(min_fill=0.9, min_num_shapes=3).check(cell)
        first = UnitCellAnalyzer(min_fill=0.9, min_num_shapes=3).validate(cell)
        assert first == failures[0]


# ---------------------------------------------------------------------------
# UnitCellAnalyzer.analyze
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_returns_list_of_metrics(self):
        cells = [_cell(_rect(w=0.3 + i * 0.1)) for i in range(3)]
        results = UnitCellAnalyzer().analyze(cells)
        assert len(results) == 3
        assert all(isinstance(r, CellMetrics) for r in results)

    def test_empty_list(self):
        assert UnitCellAnalyzer().analyze([]) == []


# ---------------------------------------------------------------------------
# UnitCellAnalyzer.find_duplicates
# ---------------------------------------------------------------------------

class TestFindDuplicates:
    def test_empty(self):
        assert UnitCellAnalyzer().find_duplicates([]) == []

    def test_identical_cells(self):
        r = _rect()
        cells = [_cell(r), _cell(r)]
        groups = UnitCellAnalyzer().find_duplicates(cells)
        assert len(groups) == 1
        assert groups[0] == [0, 1]

    def test_different_cells(self):
        cells = [_cell(_rect(w=0.2)), _cell(_rect(w=0.6))]
        groups = UnitCellAnalyzer().find_duplicates(cells)
        assert groups == []

    def test_three_identical(self):
        r = _rect()
        cells = [_cell(r)] * 3
        groups = UnitCellAnalyzer().find_duplicates(cells)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1, 2}

    def test_mixed(self):
        r1, r2 = _rect(w=0.2), _rect(w=0.6)
        cells = [_cell(r1), _cell(r2), _cell(r1)]
        groups = UnitCellAnalyzer().find_duplicates(cells)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 2}


# ---------------------------------------------------------------------------
# _skip_self_shift and _compute_min_gap
# ---------------------------------------------------------------------------

class TestSkipSelfShift:
    def test_finite_shape_not_skipped(self):
        r = _rect()
        assert not _skip_self_shift(r, 1.0, 0.0)
        assert not _skip_self_shift(r, 0.0, 1.0)

    def test_x_infinite_stripe_skips_x_shift(self):
        stripe = Stripe(offset=0.0, width=0.3, axis='x')
        # Stripe with axis='x' is infinite in x
        assert _skip_self_shift(stripe, 1.0, 0.0)   # x-component → skip
        assert not _skip_self_shift(stripe, 0.0, 1.0)  # y-only → don't skip

    def test_y_infinite_stripe_skips_y_shift(self):
        stripe = Stripe(offset=0.0, width=0.3, axis='y')
        assert not _skip_self_shift(stripe, 1.0, 0.0)  # x-only → don't skip
        assert _skip_self_shift(stripe, 0.0, 1.0)   # y-component → skip


class TestComputeMinGap:
    def test_no_shapes_returns_none(self):
        lattice = Lattice.rectangular(1.0, 1.0)
        result = _compute_min_gap([], [], lattice)
        assert result is None

    def test_single_shape_positive_gap(self):
        # 0.2×0.2 shape in 1.0×1.0 cell → self-periodic gap = 0.8 (in x: 1.0 - 0.2)
        r = _rect(w=0.2, h=0.2)
        from metashapes.adapters.shapely import shape_to_shapely, remove_holes
        geom = remove_holes(shape_to_shapely(r))
        lattice = Lattice.rectangular(1.0, 1.0)
        gap = _compute_min_gap([r], [geom], lattice)
        assert gap is not None
        assert gap > 0.0

    def test_two_shapes_gap_matches_geometry(self):
        # Two rects: left edge at -0.3+0.1=-0.2, right edge at 0.3-0.1=0.2 → gap=0.4
        a = _rect(cx=-0.3, w=0.2, h=0.2)
        b = _rect(cx=0.3, w=0.2, h=0.2)
        from metashapes.adapters.shapely import shape_to_shapely, remove_holes
        ga = remove_holes(shape_to_shapely(a))
        gb = remove_holes(shape_to_shapely(b))
        lattice = Lattice.rectangular(2.0, 2.0)
        gap = _compute_min_gap([a, b], [ga, gb], lattice)
        assert gap == pytest.approx(0.4, abs=0.01)

    def test_stripe_gap_is_perpendicular_only(self):
        # Stripe width=0.3 centered at y=0, py=1.0 → gap to y-periodic image = 1.0 - 0.3 = 0.7
        stripe = Stripe(offset=0.0, width=0.3, axis='x')
        from metashapes.adapters.shapely import shape_to_shapely, remove_holes
        geom = remove_holes(shape_to_shapely(stripe))
        lattice = Lattice.rectangular(1.0, 1.0)
        gap = _compute_min_gap([stripe], [geom], lattice)
        assert gap is not None
        assert gap == pytest.approx(0.7, abs=0.02)
