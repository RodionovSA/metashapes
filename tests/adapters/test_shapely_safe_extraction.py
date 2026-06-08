# tests/adapters/test_shapely_safe_extraction.py
"""
Verify that all Shapely adapters handle nn.Parameter inputs correctly.

The risk: adapters previously used .item()/.tolist()/float() directly on
tensors without .detach().cpu() first, which breaks on CUDA tensors and can
cause unexpected behaviour with gradient-tracked Parameters.

These tests use nn.Parameter (CPU, grad-tracked) as the minimal reproducible
case that exercises the safe-extraction path without requiring a GPU.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from metashapes.adapters.shapely import shape_to_shapely
from metashapes.lattice.basis import Lattice
from metashapes.lattice.unit_cell import UnitCell
from metashapes.shape.primitives.quads import Rectangle, IsoscelesTrapezoid
from metashapes.shape.primitives.conics import Ellipse, Stadium
from metashapes.shape.primitives.polygons import RegularPolygon
from metashapes.shape.primitives.junctions import Cross
from metashapes.shape.primitives.periodic import Stripe
from metashapes.shape.transforms import Translate, Rotate, Scale


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def p(val) -> nn.Parameter:
    """Shorthand: make a scalar nn.Parameter."""
    return nn.Parameter(torch.tensor(float(val)))

def pv(*vals) -> nn.Parameter:
    """Shorthand: make a vector nn.Parameter."""
    return nn.Parameter(torch.tensor([float(v) for v in vals]))


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class TestRectangleSafeExtraction:
    def test_all_params_learnable(self):
        shape = Rectangle(
            center=pv(0.3, -0.2),
            size=pv(0.8, 0.5),
            angle=p(15.0),
            corner_radius=p(0.05),
        )
        geom = shape_to_shapely(shape)
        assert geom.area > 0

    def test_zero_angle(self):
        shape = Rectangle(center=pv(0.0, 0.0), size=pv(1.0, 0.5), angle=p(0.0))
        geom = shape_to_shapely(shape)
        assert geom.area > 0

    def test_with_corner_radius(self):
        shape = Rectangle(center=pv(0.0, 0.0), size=pv(1.0, 1.0), corner_radius=p(0.1))
        geom = shape_to_shapely(shape)
        assert geom.area > 0


class TestEllipseSafeExtraction:
    def test_all_params_learnable(self):
        shape = Ellipse(
            center=pv(0.1, 0.2),
            axes=pv(0.6, 0.4),
            angle=p(30.0),
        )
        geom = shape_to_shapely(shape)
        assert geom.area > 0

    def test_circle_case(self):
        shape = Ellipse(center=pv(0.0, 0.0), axes=pv(0.5, 0.5))
        geom = shape_to_shapely(shape)
        assert geom.area > 0


class TestStadiumSafeExtraction:
    def test_all_params_learnable(self):
        shape = Stadium(
            center=pv(0.0, 0.0),
            width=pv(0.3),
            length=pv(0.8),
            angle=p(0.0),
        )
        geom = shape_to_shapely(shape)
        assert geom.area > 0


class TestRegularPolygonSafeExtraction:
    def test_all_params_learnable(self):
        shape = RegularPolygon(
            center=pv(0.1, -0.1),
            n=6,
            side_length=p(0.3),
            angle=p(10.0),
            corner_radius=p(0.02),
        )
        geom = shape_to_shapely(shape)
        assert geom.area > 0


class TestIsoscelesTrapezoidSafeExtraction:
    def test_all_params_learnable(self):
        shape = IsoscelesTrapezoid(
            center=pv(0.0, 0.0),
            bottom_width=p(0.8),
            top_width=p(0.5),
            height=p(0.6),
            angle=p(0.0),
            corner_radius=p(0.0),
        )
        geom = shape_to_shapely(shape)
        assert geom.area > 0


class TestCrossSafeExtraction:
    def test_all_params_learnable(self):
        shape = Cross(
            center=pv(0.0, 0.0),
            length=p(0.8),
            width=p(0.3),
            outer_corner_radius=p(0.0),
            inner_corner_radius=p(0.0),
            angle=p(0.0),
        )
        geom = shape_to_shapely(shape)
        assert geom.area > 0


class TestStripeSafeExtraction:
    def test_learnable_offset_and_width(self):
        shape = Stripe(offset=p(0.1), width=p(0.3), axis="x")
        geom = shape_to_shapely(shape)
        assert geom.area > 0


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class TestTransformSafeExtraction:
    def test_translate_with_param_offsets(self):
        base = Rectangle(center=pv(0.0, 0.0), size=pv(0.5, 0.4))
        shape = Translate(base, dx=p(0.2), dy=p(-0.1))
        geom = shape_to_shapely(shape)
        assert geom.area > 0

    def test_rotate_with_param_angle(self):
        base = Rectangle(center=pv(0.0, 0.0), size=pv(0.6, 0.3))
        shape = Rotate(base, angle=p(45.0))
        geom = shape_to_shapely(shape)
        assert geom.area > 0

    def test_scale_with_param_factor(self):
        base = Rectangle(center=pv(0.0, 0.0), size=pv(0.5, 0.5))
        shape = Scale(base, s=p(2.0))
        geom = shape_to_shapely(shape)
        assert geom.area > 0


# ---------------------------------------------------------------------------
# UnitCell integration
# ---------------------------------------------------------------------------

class TestUnitCellSafeExtraction:
    def test_to_shapely_with_learnable_params(self):
        shape = Rectangle(center=pv(0.5, 0.5), size=pv(0.4, 0.3))
        lattice = Lattice.rectangular(2.0, 2.0)
        cell = UnitCell(lattice, shape)
        geom = cell.to_shapely()
        assert geom.area > 0

    def test_center_scene_centroid_with_learnable_params(self):
        """center_scene(method='centroid') must work even if params have grad."""
        shape = Rectangle(center=pv(0.0, 0.0), size=pv(0.4, 0.3))
        lattice = Lattice.rectangular(2.0, 2.0)
        cell = UnitCell(lattice, shape)
        centered = cell.center_scene(method="centroid")
        geom = shape_to_shapely(centered.scene)
        assert geom.area > 0

    def test_gradients_still_flow_after_shapely_round_trip(self):
        """
        Verifies that calling shape_to_shapely (which does .detach().cpu())
        does NOT destroy grad on the original parameter — detach() creates a
        view with no grad, but leaves the original tensor's grad_fn intact.
        """
        center = pv(0.1, 0.2)
        shape = Rectangle(center=center, size=pv(0.5, 0.4))
        lattice = Lattice.rectangular(2.0, 2.0)
        cell = UnitCell(lattice, shape)

        # Shapely conversion
        _ = cell.to_shapely()

        # grad still flows through the SDF
        mask = cell.mask(32, 32, soft=True)
        mask.sum().backward()
        assert center.grad is not None
        assert not torch.all(center.grad == 0)
