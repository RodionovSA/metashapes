# metashapes/layout/layout.py
# Layout: a finite, non-periodic area (the domain) filled with many shapes.

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn

from metashapes.shape.base import Shape, is_empty_bounds
from metashapes.shape.shapeset import ShapeSet

__all__ = ["Layout", "Tile"]

Bounds = tuple[tuple[float, float], tuple[float, float]]

# Sigmoid tail cut-off, in units of softness: sigma(-8) ~ 3e-4.
_SOFT_HALO_FACTOR = 8.0


def _check_bounds(bounds) -> Bounds:
    (x0, y0), (x1, y1) = bounds
    x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        raise ValueError(f"bounds must be finite, got {bounds}")
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"bounds must have positive area, got {bounds}")
    return (x0, y0), (x1, y1)


def _resolve_counts(bounds: Bounds, nx, ny, pixel_size) -> tuple[int, int]:
    (x0, y0), (x1, y1) = bounds
    w, h = x1 - x0, y1 - y0
    if pixel_size is not None:
        if nx is not None or ny is not None:
            raise ValueError("pass either nx/ny or pixel_size, not both")
        if not pixel_size > 0:
            raise ValueError(f"pixel_size must be positive, got {pixel_size}")
        return max(1, round(w / pixel_size)), max(1, round(h / pixel_size))
    if nx is None and ny is None:
        raise ValueError("pass nx/ny or pixel_size")
    if ny is None:
        ny = max(1, round(nx * h / w))
    if nx is None:
        nx = max(1, round(ny * w / h))
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1")
    return int(nx), int(ny)


@dataclass(frozen=True)
class Tile:
    """One rectangular window of a tiled layout raster.

    ``bounds`` is the window to render (core plus overlap), ``core`` the
    part this tile owns. ``rows``/``cols`` locate the core in the
    full-domain raster; ``crop`` locates it inside this tile's raster, so
    ``full[t.rows, t.cols] = tile_raster[t.crop]`` stitches the tiles.
    """
    index: tuple[int, int]
    bounds: Bounds
    core: Bounds
    nx: int
    ny: int
    rows: slice
    cols: slice
    crop: tuple[slice, slice]
    dtype: torch.dtype = torch.float32
    device: torch.device | None = None

    @property
    def xs(self) -> torch.Tensor:
        (x0, _), (x1, _) = self.bounds
        return x0 + torch.arange(self.nx, dtype=self.dtype, device=self.device) * ((x1 - x0) / self.nx)

    @property
    def ys(self) -> torch.Tensor:
        (_, y0), (_, y1) = self.bounds
        return y0 + torch.arange(self.ny, dtype=self.dtype, device=self.device) * ((y1 - y0) / self.ny)

    @property
    def X(self) -> torch.Tensor:
        return torch.meshgrid(self.xs, self.ys, indexing="xy")[0]

    @property
    def Y(self) -> torch.Tensor:
        return torch.meshgrid(self.xs, self.ys, indexing="xy")[1]


class Layout(nn.Module):
    """A finite, non-periodic area (``domain``) filled with shapes (``scene``).

    The non-periodic counterpart of ``UnitCell``: no lattice, no seam, no
    copies. The rendered result is ``scene ∩ domain`` (``clip=True``), so
    output can never exceed the domain even mid-optimization.

    Parameters
    ----------
    domain : Shape
        The area to fill. Must have finite, non-empty bounds (Rectangle,
        Ellipse, a Difference for exclusion zones, ...).
    scene : ShapeSet or Shape
        The atoms. A ``ShapeSet`` uses the spatially indexed fast paths;
        any other ``Shape`` is evaluated directly.
    clip : bool
        Render ``max(scene_sdf, domain_sdf)`` (default) instead of the raw
        scene.
    strict : bool
        Raise ``ValueError`` at construction if any atom is not fully inside
        the domain. Off by default; use ``check()``/``violations()`` instead.
    """

    def __init__(self, domain: Shape, scene: ShapeSet | Shape, *,
                 clip: bool = True, strict: bool = False):
        super().__init__()
        if not isinstance(domain, Shape):
            raise TypeError("domain must be a Shape")
        if not isinstance(scene, Shape):
            raise TypeError("scene must be a Shape (usually a ShapeSet)")
        b = domain.bounds()
        if is_empty_bounds(b):
            raise ValueError("domain is empty")
        _check_bounds(b)
        self.domain = domain
        self.scene = scene
        self.clip = bool(clip)
        if strict:
            bad = self.violations()
            if bad:
                raise ValueError(
                    f"{len(bad)} atom(s) not fully inside the domain (first: {bad[:10]})")

    # --- basics ----------------------------------------------------------
    @property
    def shapes(self) -> list[Shape]:
        """The atoms: the ShapeSet's children, or ``[scene]`` otherwise."""
        if isinstance(self.scene, ShapeSet):
            return list(self.scene.shapes)
        return [self.scene]

    def __len__(self) -> int:
        return len(self.scene) if isinstance(self.scene, ShapeSet) else 1

    def _ref(self) -> tuple[torch.dtype, torch.device]:
        for t in itertools.chain(self.parameters(), self.buffers()):
            if t.dtype.is_floating_point:
                return t.dtype, t.device
        return torch.get_default_dtype(), torch.device("cpu")

    def domain_bounds(self) -> Bounds:
        return _check_bounds(self.domain.bounds())

    def extent(self) -> tuple[float, float, float, float]:
        """``(xmin, xmax, ymin, ymax)`` of the domain bounds (imshow format)."""
        (x0, y0), (x1, y1) = self.domain_bounds()
        return x0, x1, y0, y1

    # --- grids -------------------------------------------------------------
    def _axes(self, nx=None, ny=None, *, pixel_size=None, bounds=None):
        bounds = self.domain_bounds() if bounds is None else _check_bounds(bounds)
        nx, ny = _resolve_counts(bounds, nx, ny, pixel_size)
        dtype, device = self._ref()
        (x0, y0), (x1, y1) = bounds
        xs = x0 + torch.arange(nx, dtype=dtype, device=device) * ((x1 - x0) / nx)
        ys = y0 + torch.arange(ny, dtype=dtype, device=device) * ((y1 - y0) / ny)
        return xs, ys, (x1 - x0) / nx, (y1 - y0) / ny

    def grid(self, nx=None, ny=None, *, pixel_size=None, bounds=None):
        """``(X, Y)`` sample grid used by ``rasterize``, each [ny, nx].

        Axis-aligned over the domain bounds (or ``bounds``),
        endpoint-excluded like ``UnitCell``: samples start at the lower-left
        corner and stop one pixel short of the upper/right edge, so tiles
        stitch with no duplicated row or column.
        """
        xs, ys, _, _ = self._axes(nx, ny, pixel_size=pixel_size, bounds=bounds)
        return torch.meshgrid(xs, ys, indexing="xy")

    # --- SDF / rendering ----------------------------------------------
    def _scene_sdf(self, x, y, halo):
        if isinstance(self.scene, ShapeSet):
            return self.scene.sdf(x, y, halo=halo)
        return self.scene.sdf(x, y)

    def sdf(self, x: torch.Tensor, y: torch.Tensor, *,
            halo: float | None = None) -> torch.Tensor:
        """Signed distance of the layout at arbitrary points.

        ``max(scene, domain)`` when ``clip`` is set. ``halo=None`` uses the
        scene's stored default (exact unless set); a finite halo caps the
        scene part at ``halo``.
        """
        d = self._scene_sdf(x, y, halo)
        if self.clip:
            d = torch.maximum(d, self.domain.sdf(x, y))
        return d

    def rasterize(self, nx=None, ny=None, *, pixel_size=None, bounds=None,
                  halo: float = math.inf) -> torch.Tensor:
        """SDF sampled on ``grid(...)``, shape [ny, nx].

        Exact by default (``halo=math.inf``). Pass a finite ``halo`` when
        only values up to that distance matter; it is much cheaper.
        """
        xs, ys, _, _ = self._axes(nx, ny, pixel_size=pixel_size, bounds=bounds)
        if isinstance(self.scene, ShapeSet):
            d = self.scene.sdf_grid(xs, ys, halo=halo)
            if self.clip:
                X, Y = torch.meshgrid(xs, ys, indexing="xy")
                d = torch.maximum(d, self.domain.sdf(X, Y))
            return d
        X, Y = torch.meshgrid(xs, ys, indexing="xy")
        return self.sdf(X, Y, halo=halo)

    def mask(self, nx=None, ny=None, *, soft: bool = False, softness=None,
             pixel_size=None, bounds=None, halo: float | None = None) -> torch.Tensor:
        """Rasterize into a mask, shape [ny, nx].

        soft=False -> hard {0, 1} mask. Only the sign matters, so the halo
                      defaults to one pixel.
        soft=True  -> differentiable ``sigmoid(-d / softness)``; ``softness``
                      defaults to one pixel and the halo to 8 x softness,
                      beyond which the sigmoid is below 3e-4.
        """
        _, _, dx, dy = self._axes(nx, ny, pixel_size=pixel_size, bounds=bounds)
        if halo is None:
            if soft:
                s = min(dx, dy) if softness is None else float(softness)
                halo = _SOFT_HALO_FACTOR * s
            else:
                halo = max(dx, dy)
        d = self.rasterize(nx, ny, pixel_size=pixel_size, bounds=bounds, halo=halo)
        if not soft:
            return (d <= 0).to(d.dtype)
        if softness is None:
            softness = min(dx, dy)
        softness = torch.as_tensor(softness, dtype=d.dtype, device=d.device)
        return torch.sigmoid(-d / softness)

    def tiles(self, tile_size: float | tuple[float, float], *, pixel_size: float,
              overlap: float = 0.0) -> Iterator[Tile]:
        """Split the domain raster into tiles of about ``tile_size``.

        Tiles align with the full-domain pixel grid of ``pixel_size``, so
        rendering each with ``rasterize``/``mask(bounds=t.bounds,
        pixel_size=...)`` and pasting ``tile[t.crop]`` into ``full[t.rows,
        t.cols]`` reproduces the full raster. ``overlap`` (world units,
        rounded to whole pixels) extends each tile on every side.

        For differentiable work at full scale, accumulate gradients per
        tile::

            for t in layout.tiles(10_000, pixel_size=5):
                loss(layout.mask(bounds=t.bounds, pixel_size=5, soft=True)).backward()
        """
        (x0, y0), (x1, y1) = self.domain_bounds()
        NX, NY = _resolve_counts(((x0, y0), (x1, y1)), None, None, pixel_size)
        dx, dy = (x1 - x0) / NX, (y1 - y0) / NY
        tw, th = (tile_size, tile_size) if not isinstance(tile_size, (tuple, list)) else tile_size
        tpx, tpy = max(1, round(tw / dx)), max(1, round(th / dy))
        ox, oy = max(0, round(overlap / dx)), max(0, round(overlap / dy))
        dtype, device = self._ref()
        for j, r0 in enumerate(range(0, NY, tpy)):
            r1 = min(r0 + tpy, NY)
            for i, c0 in enumerate(range(0, NX, tpx)):
                c1 = min(c0 + tpx, NX)
                e0, e1, f0, f1 = c0 - ox, c1 + ox, r0 - oy, r1 + oy
                yield Tile(
                    index=(i, j),
                    bounds=((x0 + e0 * dx, y0 + f0 * dy), (x0 + e1 * dx, y0 + f1 * dy)),
                    core=((x0 + c0 * dx, y0 + r0 * dy), (x0 + c1 * dx, y0 + r1 * dy)),
                    nx=e1 - e0, ny=f1 - f0,
                    rows=slice(r0, r1), cols=slice(c0, c1),
                    crop=(slice(oy, oy + (r1 - r0)), slice(ox, ox + (c1 - c0))),
                    dtype=dtype, device=device,
                )

    # --- geometry / checks ------------------------------------------------
    def project(self) -> None:
        """Clamp drifted atom parameters back into their valid ranges.

        ``sdf()`` does this on its own; the Shapely-based methods
        (``to_shapely``, ``violations``, ``check``, ``LayoutAnalyzer``) call
        it first, since an optimizer step may leave e.g. a size slightly
        negative until the next ``sdf()``.
        """
        if isinstance(self.scene, ShapeSet):
            self.scene._project()
        else:
            ShapeSet([self.scene])._project()

    def _atom_geoms(self):
        from metashapes.adapters.shapely import shape_to_shapely
        self.project()
        return [shape_to_shapely(s) for s in self.shapes]

    def _domain_geom(self):
        from metashapes.adapters.shapely import shape_to_shapely
        return shape_to_shapely(self.domain)

    def to_shapely(self):
        """``unary_union(atoms) ∩ domain`` (just the union if ``clip=False``)."""
        from shapely.ops import unary_union
        scene = unary_union(self._atom_geoms())
        return scene.intersection(self._domain_geom()) if self.clip else scene

    def violations(self, *, margin: float = 0.0) -> list[int]:
        """Indices of atoms not fully inside ``domain`` shrunk by ``margin``."""
        import shapely
        inner = self._domain_geom()
        if margin:
            inner = inner.buffer(-margin)
        shapely.prepare(inner)
        geoms = self._atom_geoms()
        if not geoms:
            return []
        inside = shapely.contains(inner, geoms)
        return [i for i, ok in enumerate(inside.tolist()) if not ok]

    def check(self, *, margin: float = 0.0, min_gap: float | None = None,
              min_feature_size: float | None = None) -> list[str]:
        """Constraint violations as readable strings (empty list = OK).

        Reports atoms outside the (margin-shrunk) domain, atoms whose
        clipped remainder is narrower than ``min_feature_size``, and gap /
        feature-size violations. See ``LayoutAnalyzer`` for the metrics.
        """
        from .analysis import LayoutAnalyzer
        return LayoutAnalyzer(min_gap=min_gap, min_feature_size=min_feature_size,
                              edge_margin=margin).check(self)

    def containment_penalty(self, *, margin: float = 0.0, softness: float | None = None,
                            pixel_size: float | None = None) -> torch.Tensor:
        """Differentiable penalty for material outside the domain.

        ``∫ sigmoid(-d_atom / s) · relu(d_domain + margin) dA``, summed over
        atoms and integrated on a pixel grid aligned with the domain raster.
        Only atoms whose box comes within ``margin + 8·s`` of the domain
        border are evaluated, each on its own padded window, so the cost
        scales with the border, not the whole layout. Atoms are penalised
        individually, so overlapping atoms each receive their own gradient.

        ``pixel_size`` defaults to ``softness`` and vice versa; at least one
        must be given.
        """
        import shapely
        if pixel_size is None and softness is None:
            raise ValueError("pass pixel_size and/or softness")
        ps = float(pixel_size if pixel_size is not None else softness)
        s = float(softness if softness is not None else ps)
        band = margin + _SOFT_HALO_FACTOR * s
        dtype, device = self._ref()

        atoms = self.shapes
        if not atoms:
            return torch.zeros((), dtype=dtype, device=device)
        boxes = []
        for a in atoms:
            (a0, b0), (a1, b1) = a.bounds()
            boxes.append((a0, b0, a1, b1))
        inner = self._domain_geom().buffer(-band)
        shapely.prepare(inner)
        finite = [all(math.isfinite(v) for v in b) and b[2] >= b[0] and b[3] >= b[1] for b in boxes]
        polys = shapely.box(*zip(*boxes)) if boxes else []
        safe = shapely.contains(inner, polys) if boxes else []

        (gx0, gy0), _ = self.domain_bounds()
        pad = _SOFT_HALO_FACTOR * s
        s_t = torch.as_tensor(s, dtype=dtype, device=device)
        total = torch.zeros((), dtype=dtype, device=device)
        for atom, box, ok, fin in zip(atoms, boxes, safe, finite):
            if ok or not fin:
                continue
            i0 = math.floor((box[0] - pad - gx0) / ps)
            i1 = math.ceil((box[2] + pad - gx0) / ps)
            j0 = math.floor((box[1] - pad - gy0) / ps)
            j1 = math.ceil((box[3] + pad - gy0) / ps)
            xs = gx0 + torch.arange(i0, i1 + 1, dtype=dtype, device=device) * ps
            ys = gy0 + torch.arange(j0, j1 + 1, dtype=dtype, device=device) * ps
            X, Y = torch.meshgrid(xs, ys, indexing="xy")
            outside = torch.relu(self.domain.sdf(X, Y) + margin)
            total = total + (torch.sigmoid(-atom.sdf(X, Y) / s_t) * outside).sum()
        return total * (ps * ps)

    # --- serialization ------------------------------------------------
    def to_parametric(self) -> dict:
        return {
            "type": "Layout",
            "domain": self.domain.to_parametric(),
            "scene": self.scene.to_parametric(),
            "clip": self.clip,
        }

    @classmethod
    def from_parametric(cls, data: dict) -> "Layout":
        return cls(
            domain=Shape.from_parametric(data["domain"]),
            scene=Shape.from_parametric(data["scene"]),
            clip=bool(data.get("clip", True)),
        )
