# metashapes/layout/sites.py
# Placement strategies: *where* layout candidates go.

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Iterable, Iterator

import torch

from metashapes.lattice.basis import Lattice
from metashapes.shape.base import Shape

__all__ = ["Sites", "LatticeSites", "RandomSites", "ExplicitSites"]


def _inside(domain: Shape, xs: list[float], ys: list[float]) -> list[bool]:
    if not xs:
        return []
    ref = next(iter(domain.buffers()), None)
    dtype = ref.dtype if ref is not None and ref.dtype.is_floating_point else torch.float64
    device = ref.device if ref is not None else None
    with torch.no_grad():
        d = domain.sdf(torch.tensor(xs, dtype=dtype, device=device),
                       torch.tensor(ys, dtype=dtype, device=device))
    return (d <= 0).cpu().tolist()


def _domain_box(domain: Shape) -> tuple[float, float, float, float]:
    (x0, y0), (x1, y1) = domain.bounds()
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)) or x1 < x0 or y1 < y0:
        raise ValueError("domain must have finite, non-empty bounds")
    return x0, y0, x1, y1


class Sites(ABC):
    """Source of candidate positions for a layout generator.

    ``exhaustible`` sites (lattices, explicit lists) end on their own;
    non-exhaustible ones (random) yield forever and generation stops on its
    targets or on saturation (``max_consecutive_failures``).
    """
    exhaustible: bool = True

    @abstractmethod
    def candidates(self, domain: Shape, rng) -> Iterator[tuple[float, float]]:
        raise NotImplementedError


class LatticeSites(Sites):
    """Lattice points ``origin + i·a1 + j·a2`` inside the domain.

    Reuses ``Lattice``, so rectangular, hexagonal and oblique grids all
    work. Sites run row by row (``j`` outer, ``i`` inner). ``jitter``
    displaces each site by a uniform offset in ``[-jitter, jitter]`` per
    axis, drawn from the generator's rng.
    """
    exhaustible = True

    def __init__(self, lattice: Lattice, origin: tuple[float, float] = (0.0, 0.0),
                 jitter: float = 0.0):
        if jitter < 0:
            raise ValueError("jitter must be >= 0")
        self.lattice = lattice
        self.origin = (float(origin[0]), float(origin[1]))
        self.jitter = float(jitter)

    def points(self, domain: Shape) -> list[tuple[float, float]]:
        """All un-jittered lattice sites inside the domain, in order."""
        x0, y0, x1, y1 = _domain_box(domain)
        ox, oy = self.origin
        A = self.lattice.matrix.detach().to(torch.float64).cpu()
        inv = torch.linalg.inv(A)
        corners = torch.tensor([[x0 - ox, x1 - ox, x0 - ox, x1 - ox],
                                [y0 - oy, y0 - oy, y1 - oy, y1 - oy]], dtype=torch.float64)
        f = inv @ corners
        i_lo, i_hi = math.floor(f[0].min().item()), math.ceil(f[0].max().item())
        j_lo, j_hi = math.floor(f[1].min().item()), math.ceil(f[1].max().item())
        jj, ii = torch.meshgrid(torch.arange(j_lo, j_hi + 1, dtype=torch.float64),
                                torch.arange(i_lo, i_hi + 1, dtype=torch.float64), indexing="ij")
        X = ox + A[0, 0] * ii + A[0, 1] * jj
        Y = oy + A[1, 0] * ii + A[1, 1] * jj
        X, Y = X.reshape(-1), Y.reshape(-1)
        box = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        X, Y = X[box].tolist(), Y[box].tolist()
        keep = _inside(domain, X, Y)
        return [(x, y) for x, y, k in zip(X, Y, keep) if k]

    def candidates(self, domain, rng):
        for x, y in self.points(domain):
            if self.jitter:
                x += rng.uniform(-self.jitter, self.jitter)
                y += rng.uniform(-self.jitter, self.jitter)
            yield x, y


class RandomSites(Sites):
    """Uniform random positions inside the domain (rejection sampling).

    Used for random sequential addition. Points are drawn in batches of
    ``batch`` from the generator's rng and filtered by ``domain.sdf <= 0``
    in one vectorised call.
    """
    exhaustible = False

    def __init__(self, batch: int = 256):
        if batch < 1:
            raise ValueError("batch must be >= 1")
        self.batch = int(batch)

    def candidates(self, domain, rng):
        x0, y0, x1, y1 = _domain_box(domain)
        while True:
            xs = [rng.uniform(x0, x1) for _ in range(self.batch)]
            ys = [rng.uniform(y0, y1) for _ in range(self.batch)]
            for x, y, k in zip(xs, ys, _inside(domain, xs, ys)):
                if k:
                    yield x, y


class ExplicitSites(Sites):
    """User-supplied positions, used in the given order."""
    exhaustible = True

    def __init__(self, points: Iterable[tuple[float, float]]):
        self.points = [(float(x), float(y)) for x, y in points]

    def candidates(self, domain, rng):
        yield from self.points
