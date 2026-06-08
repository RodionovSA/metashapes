# metashapes/lattice/grid.py

import torch
from .basis import Lattice


def fractional_grid(nx: int, ny: int,
                    dtype=torch.float32, device=None):
    """Sample points covering exactly one unit cell, in fractional
    coordinates [0,1) x [0,1).

    endpoint-excluded so the grid tiles seamlessly without a duplicate
    seam row/column. Returns F1, F2 each of shape [ny, nx].
    """
    f1 = torch.arange(nx, dtype=dtype, device=device) / nx
    f2 = torch.arange(ny, dtype=dtype, device=device) / ny
    F1, F2 = torch.meshgrid(f1, f2, indexing="xy")
    return F1, F2


def cartesian_grid(lattice: Lattice, nx: int, ny: int,
                    dtype=torch.float32, device=None):
    """One unit cell sampled in Cartesian coordinates.

    Builds the fractional grid then maps it through the lattice basis,
    so an oblique lattice gives a sheared grid automatically.
    Returns X, Y each of shape [ny, nx].
    """
    F1, F2 = fractional_grid(nx, ny, dtype=dtype, device=device)
    return lattice.to_cartesian(F1, F2)