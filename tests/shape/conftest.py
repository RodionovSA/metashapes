# tests/shape/conftest.py
# Shared helpers for shape tests.

import torch
import pytest
from metashapes.shape import Shape


def sdf_at(shape, x, y):
    return shape.sdf(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    ).item()


def assert_inside(shape, points, tol=1e-4):
    """All points should have SDF < tol (inside or on boundary)."""
    for x, y in points:
        d = sdf_at(shape, x, y)
        assert d < tol, f"{type(shape).__name__}: expected inside at ({x}, {y}), got SDF={d:.6f}"


def assert_outside(shape, points, tol=1e-4):
    """All points should have SDF > -tol (outside or on boundary)."""
    for x, y in points:
        d = sdf_at(shape, x, y)
        assert d > -tol, f"{type(shape).__name__}: expected outside at ({x}, {y}), got SDF={d:.6f}"


def assert_round_trip(shape, grid_range=(-1.0, 1.0), grid_n=24, atol=1e-5):
    """Serialise → deserialise and verify SDF equality on a small grid."""
    data = shape.to_parametric()
    restored = Shape.from_parametric(data)
    assert type(restored).__name__ == type(shape).__name__

    lo, hi = grid_range
    xs = torch.linspace(lo, hi, grid_n)
    ys = torch.linspace(lo, hi, grid_n)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")

    d_orig = shape.sdf(X, Y)
    d_rest = restored.sdf(X, Y)
    assert torch.allclose(d_orig, d_rest, atol=atol), (
        f"{type(shape).__name__} SDF mismatch after round-trip: "
        f"max diff = {(d_orig - d_rest).abs().max().item():.2e}"
    )


def assert_bounds_contain(shape, points):
    """All points should lie inside (or on) the reported bounding box."""
    (x0, y0), (x1, y1) = shape.bounds()
    for x, y in points:
        assert x0 - 1e-6 <= x <= x1 + 1e-6, (
            f"{type(shape).__name__}: point ({x}, {y}) outside bounds x=[{x0}, {x1}]"
        )
        assert y0 - 1e-6 <= y <= y1 + 1e-6, (
            f"{type(shape).__name__}: point ({x}, {y}) outside bounds y=[{y0}, {y1}]"
        )
