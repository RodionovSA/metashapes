# tests/lattice/conftest.py

import pytest
import torch
import torch.nn as nn

from src.metashapes.lattice.basis import Lattice
from src.metashapes.lattice.unit_cell import UnitCell
from src.metashapes.shape.primitives.polygons import RegularPolygon


@pytest.fixture
def rect_lattice():
    return Lattice.rectangular(2.0, 2.0)


@pytest.fixture
def hex_lattice():
    return Lattice.hexagonal(1.0, orientation="pointy")


@pytest.fixture
def square_in_rect(rect_lattice):
    shape = RegularPolygon(
        center=torch.zeros(2), n=4, side_length=torch.tensor(0.5)
    )
    return UnitCell(rect_lattice, shape)


def make_learnable_polygon(side_length_val=0.5, center_val=(0.0, 0.0)):
    """Return (UnitCell, side_length param, center param) with nn.Parameters."""
    side_length = nn.Parameter(torch.tensor(float(side_length_val)))
    center = nn.Parameter(torch.tensor([float(v) for v in center_val]))
    shape = RegularPolygon(center=center, n=4, side_length=side_length)
    lattice = Lattice.rectangular(2.0, 2.0)
    cell = UnitCell(lattice, shape)
    return cell, side_length, center
