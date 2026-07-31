# tests/shape/test_transforms.py
# Tests for Translate, Rotate, Scale: min_feature_size propagation (S-04),
# from_parametric origin defaults (S-05), and empty-bounds handling (S-06).

import math

import pytest
import torch

from metashapes.shape import Shape, Translate, Rotate, Scale, Intersection
from metashapes.shape.base import is_empty_bounds
from metashapes.shape.primitives.quads import Rectangle


def _rect(size=(0.3, 0.8), center=(0.0, 0.0)):
    return Rectangle(center=torch.tensor(center), size=torch.tensor(size))


def _disjoint_intersection():
    return Intersection(
        Rectangle(center=torch.tensor([0.0, 0.0]), size=torch.tensor([1.0, 1.0])),
        Rectangle(center=torch.tensor([5.0, 5.0]), size=torch.tensor([1.0, 1.0])),
    )


# ---------------------------------------------------------------------------
# S-04: min_feature_size propagation through rigid/uniform-scale transforms
# ---------------------------------------------------------------------------

class TestMinFeatureSizePropagation:
    def test_translate_forwards_child_value_unchanged(self):
        rect = _rect(size=(0.3, 0.8))
        assert Translate(rect, dx=0.5, dy=-0.2).min_feature_size == pytest.approx(0.3)

    def test_rotate_forwards_child_value_unchanged(self):
        rect = _rect(size=(0.3, 0.8))
        assert Rotate(rect, angle=30.0).min_feature_size == pytest.approx(0.3)

    def test_scale_multiplies_child_value(self):
        rect = _rect(size=(0.3, 0.8))
        assert Scale(rect, s=2.0).min_feature_size == pytest.approx(0.6)
        assert Scale(rect, s=0.5).min_feature_size == pytest.approx(0.15)

    def test_none_child_stays_none_through_translate(self):
        # Union has no min_feature_size (see below); wrapping it must not
        # invent a value.
        child = _rect() | _rect(center=(2.0, 2.0))
        assert child.min_feature_size is None
        assert Translate(child, dx=1.0).min_feature_size is None

    def test_none_child_stays_none_through_rotate_and_scale(self):
        child = _rect() | _rect(center=(2.0, 2.0))
        assert Rotate(child, angle=10.0).min_feature_size is None
        assert Scale(child, s=2.0).min_feature_size is None

    def test_chained_transforms_compose(self):
        rect = _rect(size=(0.3, 0.8))
        chained = Scale(Rotate(Translate(rect, dx=1.0), angle=45.0), s=3.0)
        assert chained.min_feature_size == pytest.approx(0.9)


class TestMinFeatureSizeBooleansStayUnknown:
    """Union/Intersection/Difference deliberately return None: combining
    shapes can create a feature thinner than either operand's own value, so
    there is no safe way to propagate a number (see user decision recorded
    in screening_shape_lattice.md, S-04)."""

    def test_union_of_two_finite_children_is_none(self):
        a = _rect(size=(0.3, 0.8))
        b = _rect(size=(0.5, 0.5), center=(2.0, 2.0))
        assert (a | b).min_feature_size is None

    def test_intersection_and_difference_are_none(self):
        a = _rect(size=(0.3, 0.8))
        b = _rect(size=(0.5, 0.5), center=(0.1, 0.1))
        assert (a & b).min_feature_size is None
        assert (a - b).min_feature_size is None


# ---------------------------------------------------------------------------
# S-05: Rotate/Scale.from_parametric default origin
# ---------------------------------------------------------------------------

class TestFromParametricOriginDefault:
    def test_rotate_missing_origin_defaults_to_zero(self):
        data = {"type": "Rotate", "shape": _rect().to_parametric(), "angle": 30.0}
        r = Rotate.from_parametric(data)
        assert r.origin.tolist() == pytest.approx([0.0, 0.0])

    def test_scale_missing_origin_defaults_to_zero(self):
        data = {"type": "Scale", "shape": _rect().to_parametric(), "s": 2.0}
        s = Scale.from_parametric(data)
        assert s.origin.tolist() == pytest.approx([0.0, 0.0])

    def test_rotate_missing_origin_matches_default_constructor(self):
        rect = _rect()
        data = {"type": "Rotate", "shape": rect.to_parametric(), "angle": 30.0}
        via_parametric = Rotate.from_parametric(data)
        via_ctor = Rotate(rect, angle=30.0)
        x = torch.linspace(-1.0, 1.0, 16)
        y = torch.linspace(-1.0, 1.0, 16)
        X, Y = torch.meshgrid(x, y, indexing="xy")
        assert torch.allclose(via_parametric.sdf(X, Y), via_ctor.sdf(X, Y))

    def test_generic_from_parametric_dispatch_also_defaults_origin(self):
        # Shape.from_parametric(...) is the path actually used when
        # reconstructing nested shapes (e.g. inside Translate/Union).
        data = {"type": "Rotate", "shape": _rect().to_parametric(), "angle": 30.0}
        r = Shape.from_parametric(data)
        assert r.origin.tolist() == pytest.approx([0.0, 0.0])


# ---------------------------------------------------------------------------
# S-06: empty bounds propagate correctly through transforms
# ---------------------------------------------------------------------------

class TestEmptyBoundsThroughTransforms:
    def test_translate_of_empty_stays_empty(self):
        empty = _disjoint_intersection()
        assert is_empty_bounds(Translate(empty, dx=1.0, dy=1.0).bounds())

    def test_rotate_of_empty_constructs_and_stays_empty(self):
        # Previously this raised NotImplementedError: the finite-bounds
        # guard in Rotate.__init__ treated the inverted/empty box as
        # "infinite" because inf/-inf both fail math.isfinite.
        empty = _disjoint_intersection()
        rotated = Rotate(empty, angle=30.0)
        assert is_empty_bounds(rotated.bounds())

    def test_scale_of_empty_stays_empty_no_nan(self):
        empty = _disjoint_intersection()
        scaled = Scale(empty, s=2.0)
        (x0, y0), (x1, y1) = scaled.bounds()
        assert not any(math.isnan(v) for v in (x0, y0, x1, y1))
        assert is_empty_bounds(scaled.bounds())

    def test_union_with_empty_child_equals_other_child_bounds(self):
        empty = _disjoint_intersection()
        rect = _rect(size=(0.4, 0.6), center=(1.0, 1.0))
        combined = empty | rect
        assert combined.bounds() == rect.bounds()
