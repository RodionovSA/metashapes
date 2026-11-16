# metashapes/shape/shapeset.py
# ShapeSet: a flat, spatially indexed union of many shapes.

from __future__ import annotations

import math
from typing import Iterator, Sequence

import torch
import torch.nn as nn

from .base import EMPTY_BOUNDS, Shape
from .registry import register_shape
from .spatial import SpatialHash

__all__ = ["ShapeSet"]


def _atom_box(shape: Shape) -> tuple[float, float, float, float]:
    (x0, y0), (x1, y1) = shape.bounds()
    return (x0, y0, x1, y1)


@register_shape("ShapeSet")
class ShapeSet(Shape):
    """Flat union of many shapes ("atoms"), evaluated through a spatial index.

    A ``Union`` chain of N shapes costs O(N) per query point and recurses N
    levels deep. ``ShapeSet`` holds the atoms in one flat ``nn.ModuleList``
    and evaluates each atom only at the points near it, so the cost is
    proportional to the area each atom covers rather than N x points.

    Atoms combine with a hard ``min`` (no smooth blending between atoms):
    overlapping atoms merge, exactly like ``Union``. Smooth or boolean
    composition *inside* an atom works as usual.

    Halo
    ----
    "Near" an atom means inside its bounding box padded by ``halo``. A point
    outside every padded box is farther than ``halo`` from every atom, so
    with a finite halo the SDF is ``min(true_sdf, halo)``: exact up to
    ``halo`` and clamped beyond it. With ``halo=math.inf`` (the default) the
    SDF is exact everywhere, computed by an expanding-radius search that
    stays cheap in populated regions (see ``sdf``).

    Atoms whose own SDF is only a lower bound far from the shape (``Star``,
    ``TShape``, ``Difference``, ...) keep that property. Skipping an atom
    relies on "its SDF is at least the distance to its box", so such an
    atom far from a query point may be skipped even though its
    (underestimated) value is smaller. The result is then larger than a
    ``Union`` chain would give, but still a lower bound on the true
    distance with the correct sign, and hard masks are unaffected.

    Every query method takes a per-call ``halo=`` override; the stored
    ``halo`` is only the default for ``sdf(x, y)`` calls that don't pass one
    (e.g. when the set is nested inside a ``Union`` or used as a
    ``UnitCell`` scene).

    Index staleness
    ---------------
    The index stores each atom's box at build time, padded by ``skin``. When
    any atom has a learnable parameter, each query recomputes the current
    boxes and rebuilds the index once the largest drift exceeds ``skin / 2``
    (Verlet-list scheme). With only buffers, the index is built once; call
    ``rebuild_index(force=True)`` after editing atom buffers in place.

    Parameters
    ----------
    shapes : sequence of Shape
        The atoms, usually ``Translate(primitive, dx, dy)``. Atoms must have
        finite bounds (``Bar`` is rejected); empty atoms are allowed and
        contribute nothing.
    halo : float
        Default halo for ``sdf(x, y)`` when none is passed. ``math.inf``
        (default) means exact.
    cell_size : float, optional
        Spatial-hash cell size. Defaults to the median atom box size.
    skin : float
        Extra box padding that lets atoms drift before the index must be
        rebuilt. Only matters for learnable atoms.
    """

    def __init__(self,
                 shapes: Sequence[Shape] = (),
                 *,
                 halo: float = math.inf,
                 cell_size: float | None = None,
                 skin: float = 0.0):
        super().__init__()
        shapes = list(shapes)
        for i, s in enumerate(shapes):
            if not isinstance(s, Shape):
                raise TypeError(f"ShapeSet atom {i} is {type(s).__name__}, not a Shape")
        halo = float(halo)
        if not halo > 0:
            raise ValueError(f"halo must be positive, got {halo}")
        if skin < 0:
            raise ValueError(f"skin must be >= 0, got {skin}")
        if cell_size is not None and not cell_size > 0:
            raise ValueError(f"cell_size must be positive, got {cell_size}")
        self.shapes = nn.ModuleList(shapes)
        self.halo = halo
        self.cell_size = None if cell_size is None else float(cell_size)
        self.skin = float(skin)
        self._index: SpatialHash | None = None
        self._built_boxes: torch.Tensor | None = None
        self._check_finite()

    def _check_finite(self) -> None:
        for i, s in enumerate(self.shapes):
            x0, y0, x1, y1 = _atom_box(s)
            if x1 < x0 or y1 < y0:
                continue  # empty atom
            if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
                raise ValueError(
                    f"ShapeSet atom {i} ({type(s).__name__}) has infinite bounds; "
                    "atoms must be finite (e.g. Bar is not allowed)")

    # --- container protocol -------------------------------------------
    # Note: defining __len__ makes an empty ShapeSet falsy; never test a
    # ShapeSet with `if shapeset:`.
    def __len__(self) -> int:
        return len(self.shapes)

    def __iter__(self) -> Iterator[Shape]:
        return iter(self.shapes)

    def __getitem__(self, i):
        return self.shapes[i]

    # --- index ----------------------------------------------------------
    def atom_boxes(self) -> torch.Tensor:
        """Current ``(x0, y0, x1, y1)`` box of every atom, [N, 4] float64 CPU."""
        if len(self.shapes) == 0:
            return torch.zeros((0, 4), dtype=torch.float64)
        return torch.tensor([_atom_box(s) for s in self.shapes], dtype=torch.float64)

    def _has_learnable(self) -> bool:
        return any(p.requires_grad for p in self.parameters())

    def rebuild_index(self, force: bool = False) -> bool:
        """Make sure the spatial index is current. Returns True if rebuilt.

        Without ``force`` the index is rebuilt only if it doesn't exist yet,
        the atom count changed, or (for learnable atoms) some atom box
        drifted by more than ``skin / 2`` since the last build.
        """
        stale = force or self._index is None or self._built_boxes is None \
            or self._built_boxes.shape[0] != len(self.shapes)
        current = None
        if not stale and self._has_learnable():
            current = self.atom_boxes()
            built = self._built_boxes
            both = torch.isfinite(current) & torch.isfinite(built)
            drift = (current - built).abs()[both]
            moved = drift.numel() > 0 and drift.max().item() > 0.5 * self.skin
            # an atom switching between empty and non-empty also invalidates
            changed = not torch.equal(torch.isfinite(current), torch.isfinite(built))
            stale = moved or changed
        if not stale:
            return False
        boxes = self.atom_boxes() if current is None else current
        self._built_boxes = boxes
        padded = boxes.clone()
        padded[:, :2] -= self.skin
        padded[:, 2:] += self.skin
        self._index = SpatialHash(padded, cell_size=self.cell_size)
        return True

    def _ready_index(self) -> SpatialHash:
        self.rebuild_index()
        return self._index

    # --- evaluation ---------------------------------------------------------
    def _resolve_halo(self, halo: float | None) -> float:
        h = self.halo if halo is None else float(halo)
        if not h > 0:
            raise ValueError(f"halo must be positive, got {h}")
        return h

    def _eval_pairs(self, atom_ids: torch.Tensor, pt_ids: torch.Tensor,
                    px: torch.Tensor, py: torch.Tensor):
        """Evaluate each atom at its own points. Returns (values, point_ids).

        This is the generic per-atom path. Grouped, vectorised evaluation of
        same-type atoms can replace it without changing any caller.
        """
        counts = torch.bincount(atom_ids, minlength=len(self.shapes)).tolist()
        chunks = torch.split(pt_ids, counts)
        vals, idxs = [], []
        for i, pts in enumerate(chunks):
            if pts.numel() == 0:
                continue
            vals.append(self.shapes[i].sdf(px[pts], py[pts]))
            idxs.append(pts)
        return vals, idxs

    def _capped_points(self, px: torch.Tensor, py: torch.Tensor, r: float,
                       index: SpatialHash) -> torch.Tensor:
        """``min(true_sdf, r)`` at flat points (px, py)."""
        atom_ids, pt_ids = index.pairs(px, py, pad=r)
        dtype = torch.promote_types(px.dtype, torch.get_default_dtype()) \
            if not px.dtype.is_floating_point else px.dtype
        vals, idxs = self._eval_pairs(atom_ids, pt_ids, px, py)
        if not vals:
            return torch.full(px.shape, r, dtype=dtype, device=px.device)
        v = torch.cat([t.reshape(-1) for t in vals])
        ix = torch.cat(idxs)
        out = torch.full(px.shape, r, dtype=v.dtype, device=px.device)
        return out.scatter_reduce(0, ix, v, reduce="amin", include_self=True)

    def _brute_points(self, px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
        """Exact min over every non-empty atom (no index)."""
        index = self._ready_index()
        best = None
        for i in index.ids.tolist():
            d = self.shapes[i].sdf(px, py)
            best = d if best is None else torch.minimum(best, d)
        if best is None:
            return torch.full(px.shape, math.inf, dtype=px.dtype, device=px.device)
        return best

    def _exact_points(self, px: torch.Tensor, py: torch.Tensor,
                      r0: float | None = None) -> torch.Tensor:
        """Exact SDF at flat points by expanding-radius search.

        Round k evaluates ``min(sdf, r_k)`` with ``r_k = r0 * 2**k`` only at
        the points still unresolved. A value ``< r_k`` is exact: every atom
        left out has a box farther than ``r_k`` away, so its SDF is larger.
        Points in populated regions resolve in the first round; only points
        in empty regions see the radius grow. Once the radius covers the
        whole index extent, the leftover points are resolved by brute force.
        """
        index = self._ready_index()
        P = px.numel()
        if P == 0:
            return px.new_zeros(px.shape)
        if len(index) == 0:
            return torch.full(px.shape, math.inf, dtype=px.dtype, device=px.device)

        ext = index.extent()
        gx0 = min(ext[0], px.min().item()); gx1 = max(ext[2], px.max().item())
        gy0 = min(ext[1], py.min().item()); gy1 = max(ext[3], py.max().item())
        r_max = math.hypot(gx1 - gx0, gy1 - gy0)
        r = index.cell_size if r0 is None else r0

        remaining = torch.arange(P, device=px.device)
        out_idx, out_val = [], []
        while remaining.numel() and r < r_max:
            sx, sy = px[remaining], py[remaining]
            v = self._capped_points(sx, sy, r, index)
            done = v < r
            if done.any():
                out_idx.append(remaining[done])
                out_val.append(v[done])
            remaining = remaining[~done]
            r *= 2.0
        if remaining.numel():
            out_idx.append(remaining)
            out_val.append(self._brute_points(px[remaining], py[remaining]))

        v = torch.cat(out_val)
        out = torch.zeros(P, dtype=v.dtype, device=px.device)
        return out.scatter(0, torch.cat(out_idx), v)

    def sdf(self, x: torch.Tensor, y: torch.Tensor, *,
            halo: float | None = None) -> torch.Tensor:
        """Signed distance at arbitrary (broadcastable) points.

        ``halo=None`` uses the stored default (``math.inf`` unless set):
        exact everywhere. A finite ``halo`` returns ``min(sdf, halo)``,
        which is much cheaper far from atoms.
        """
        h = self._resolve_halo(halo)
        x, y = torch.broadcast_tensors(torch.as_tensor(x), torch.as_tensor(y))
        shape = x.shape
        px, py = x.reshape(-1), y.reshape(-1)
        if len(self.shapes) == 0:
            return torch.full(shape, h, dtype=px.dtype, device=px.device)
        if math.isinf(h):
            out = self._exact_points(px, py)
        else:
            out = self._capped_points(px, py, h, self._ready_index())
        return out.reshape(shape)

    def sdf_grid(self, xs: torch.Tensor, ys: torch.Tensor, *,
                 halo: float | None = None) -> torch.Tensor:
        """SDF on the axis-aligned grid ``meshgrid(xs, ys, indexing="xy")``.

        Returns [len(ys), len(xs)]. Fast path ("stamping"): each atom is
        evaluated only on the sub-grid under its padded box, and all
        windows are merged with one ``scatter_reduce(amin)``, so the cost is
        O(sum of window sizes) and the autograd graph has one merge node.

        ``xs`` and ``ys`` must be 1D and ascending. ``halo=None`` uses the
        stored default; ``math.inf`` stamps with a radius of one index cell
        and then resolves the remaining pixels with the exact search.
        """
        h = self._resolve_halo(halo)
        xs = torch.as_tensor(xs)
        ys = torch.as_tensor(ys)
        if xs.ndim != 1 or ys.ndim != 1:
            raise ValueError("sdf_grid expects 1D xs and ys")
        nx, ny = xs.numel(), ys.numel()
        dtype = xs.dtype if xs.dtype.is_floating_point else torch.get_default_dtype()
        if len(self.shapes) == 0:
            return torch.full((ny, nx), h, dtype=dtype, device=xs.device)
        index = self._ready_index()
        if len(index) == 0:
            return torch.full((ny, nx), h, dtype=dtype, device=xs.device)

        r = index.cell_size if math.isinf(h) else h
        vals, idxs = [], []
        for atom, i0, i1, j0, j1 in index.grid_windows(xs, ys, pad=r):
            sx = xs[i0:i1]
            sy = ys[j0:j1]
            X, Y = torch.broadcast_tensors(sx[None, :], sy[:, None])
            vals.append(self.shapes[atom].sdf(X, Y).reshape(-1))
            cols = torch.arange(i0, i1, device=xs.device)
            rows = torch.arange(j0, j1, device=xs.device)
            idxs.append((rows[:, None] * nx + cols[None, :]).reshape(-1))
        if vals:
            v = torch.cat(vals)
            out = torch.full((ny * nx,), r, dtype=v.dtype, device=xs.device)
            out = out.scatter_reduce(0, torch.cat(idxs), v, reduce="amin", include_self=True)
        else:
            out = torch.full((ny * nx,), r, dtype=dtype, device=xs.device)

        if math.isinf(h):
            unresolved = torch.nonzero(out >= r).flatten()
            if unresolved.numel():
                px = xs[unresolved % nx]
                py = ys[unresolved // nx]
                v = self._exact_points(px, py, r0=2.0 * r)
                out = out.scatter(0, unresolved, v.to(out.dtype))
        return out.reshape(ny, nx)

    # --- geometry -------------------------------------------------------
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        x0 = y0 = math.inf
        x1 = y1 = -math.inf
        for s in self.shapes:
            (a, b), (c, d) = s.bounds()
            if c < a or d < b:
                continue
            x0, y0, x1, y1 = min(x0, a), min(y0, b), max(x1, c), max(y1, d)
        if x1 < x0 or y1 < y0:
            return EMPTY_BOUNDS
        return (x0, y0), (x1, y1)

    @property
    def min_feature_size(self) -> float | None:
        """Smallest atom feature size; None if empty or any atom is unknown.

        Like ``Union``, merged (overlapping) atoms can form a thinner seam
        than any single atom; this value only describes the atoms.
        """
        if len(self.shapes) == 0:
            return None
        sizes = [s.min_feature_size for s in self.shapes]
        if any(v is None for v in sizes):
            return None
        return min(sizes)

    def _project(self) -> None:
        """Project every shape in every atom's tree that defines a projection.

        Walks the whole tree, not just the atoms, because wrappers like
        ``Translate`` have no projection of their own. Consumers that read
        raw parameters without going through ``sdf()`` (e.g. the Shapely
        adapter) need this after an optimizer step.
        """
        for s in self.shapes:
            s.project_tree()

    # project_tree() reaches the atoms through modules() itself, so it must
    # not also call this set's _project() (which would walk them twice).
    _projects_subtree = True

    # --- serialization ------------------------------------------------
    def to_parametric(self) -> dict:
        return {
            "type": "ShapeSet",
            "halo": self.halo,
            "cell_size": self.cell_size,
            "skin": self.skin,
            "shapes": [s.to_parametric() for s in self.shapes],
        }

    @classmethod
    def from_parametric(cls, data: dict) -> "ShapeSet":
        return cls(
            [Shape.from_parametric(d) for d in data.get("shapes", [])],
            halo=float(data.get("halo", math.inf)),
            cell_size=data.get("cell_size"),
            skin=float(data.get("skin", 0.0)),
        )
