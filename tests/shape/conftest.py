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


# ---------------------------------------------------------------------------
# dtype / device / gradient helpers
# ---------------------------------------------------------------------------
#
# NOTE on scope: `Shape` construction always defaults to float32/CPU --
# getting a different dtype/device is the caller's job via `.to(...)`,
# same as any other nn.Module. These helpers verify THAT story works
# correctly, not that construction accepts an arbitrary dtype.
#
# NOTE on mutation: `nn.Module.to()` converts parameters/buffers IN PLACE
# and returns `self` (confirmed elsewhere in this codebase) -- these
# helpers mutate the `shape` object passed in. Callers should pass a
# freshly-constructed shape they don't need afterward in its original dtype.

def assert_dtype_device_flow(shape, x=0.1, y=0.1):
    """A constructed shape defaults to float32/CPU; `.to(dtype=...)` (and
    `.to('cuda')`, when available) must convert *every* registered
    parameter and buffer -- not just the ones a caller happens to check --
    and `sdf()` must then accept matching-dtype/device query points and
    return output of that dtype/device."""
    name = type(shape).__name__

    out = shape.sdf(torch.tensor(x), torch.tensor(y))
    assert out.dtype == torch.float32, f"{name}: default sdf() output not float32"
    assert out.device.type == "cpu", f"{name}: default sdf() output not cpu"
    for pname, t in (*shape.named_parameters(), *shape.named_buffers()):
        assert t.dtype == torch.float32, f"{name}.{pname} not float32 by default"
        assert t.device.type == "cpu", f"{name}.{pname} not cpu by default"

    shape64 = shape.to(torch.float64)
    for pname, t in (*shape64.named_parameters(), *shape64.named_buffers()):
        assert t.dtype == torch.float64, f"{name}.{pname} not converted by .to(torch.float64)"
    out64 = shape64.sdf(
        torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    )
    assert out64.dtype == torch.float64, f"{name}: sdf() output not float64 after .to(float64)"

    if torch.cuda.is_available():
        shape_cuda = shape64.to("cuda")
        for pname, t in (*shape_cuda.named_parameters(), *shape_cuda.named_buffers()):
            assert t.device.type == "cuda", f"{name}.{pname} not converted by .to('cuda')"
        out_cuda = shape_cuda.sdf(
            torch.tensor(x, dtype=torch.float64, device="cuda"),
            torch.tensor(y, dtype=torch.float64, device="cuda"),
        )
        assert out_cuda.device.type == "cuda", f"{name}: sdf() output not cuda after .to('cuda')"


def assert_direct_call_dtype_promotion(shape, x=0.1, y=0.1):
    """Calling sdf() directly on a primitive (bypassing UnitCell) with
    float64 query points against float32 parameters silently promotes via
    ordinary torch arithmetic and returns float64. This is the actual
    current behavior for every primitive (verified across all 12 before
    writing this helper) -- distinct from `UnitCell.sdf()`, which requires
    an exact dtype match because `Lattice.to_fractional` uses `einsum`
    (covered separately in tests/lattice/test_unit_cell.py)."""
    out = shape.sdf(torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64))
    assert out.dtype == torch.float64, (
        f"{type(shape).__name__}: direct sdf() call did not promote to float64 "
        f"for a float64 query against float32 parameters"
    )


def assert_gradients_finite(shape, param_names, x_range=(-1.0, 1.0), n=40):
    """Grid-evaluate sdf() over `x_range`, sum, backward(); every named
    parameter in `param_names` (each must be an nn.Parameter, i.e.
    constructed with requires_grad=True) must end up with a non-None,
    fully finite gradient."""
    xs = torch.linspace(*x_range, n)
    X, Y = torch.meshgrid(xs, xs, indexing="xy")
    d = shape.sdf(X, Y)
    d.sum().backward()
    _assert_param_grads_finite(shape, param_names, context="grid")


def assert_gradients_finite_at(shape, param_names, x, y):
    """Same as assert_gradients_finite but evaluates sdf() at a single,
    possibly suspicious/boundary point rather than a grid."""
    d = shape.sdf(torch.tensor(x), torch.tensor(y))
    d.backward()
    _assert_param_grads_finite(shape, param_names, context=f"point ({x}, {y})")


def _assert_param_grads_finite(shape, param_names, context):
    name = type(shape).__name__
    params = dict(shape.named_parameters())
    for pname in param_names:
        p = params.get(pname)
        assert p is not None, f"{name}: no such nn.Parameter '{pname}' (got {list(params)})"
        assert p.grad is not None, f"{name}.{pname} has no gradient at {context}"
        assert torch.isfinite(p.grad).all(), (
            f"{name}.{pname} gradient has NaN/Inf at {context}: {p.grad}"
        )
