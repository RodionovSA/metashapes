# metashapes/lattice/unit_cell.py
# 

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import math

from metashapes.shape import Shape, Translate, Union
from metashapes.shape.base import is_empty_bounds
from .basis import Lattice
from .grid import cartesian_grid

__all__ = ["UnitCell"]


def _decompose_for_offsets(shape: Shape):
    """Yield sub-shapes to measure separately when sizing the periodic
    copy search, each with `bounds()` already reflecting its true world
    position.

    `Union` is split into its two operands: `Union.bounds()`'s
    componentwise min/max lets one child's infinite extent (e.g. a `Bar`)
    swallow another, finite child's true -- and possibly much larger --
    span, so the two must be measured independently and their resulting
    offset ranges combined, not their bounds. `Translate` is passed
    through onto each decomposed piece (same idiom as `analysis.py`'s
    `_leaf_shapes`) so decomposition still reaches into a scene like
    `Translate(Union(...))`, which is exactly what `center_scene()`
    produces. Everything else (single primitives, `Rotate`, `Scale`,
    `Intersection`, `Difference`) is measured as one atomic piece via its
    own `bounds()`.
    """
    if isinstance(shape, Union):
        yield from _decompose_for_offsets(shape.left)
        yield from _decompose_for_offsets(shape.right)
    elif isinstance(shape, Translate):
        for piece in _decompose_for_offsets(shape.shape):
            yield piece.translate(shape.dx, shape.dy)
    else:
        yield shape


class UnitCell(nn.Module):
    """
    A periodic structure: one Lattice + one Shape (the scene).

    The lattice owns the in-plane periodicity; the shape lives in
    infinite, continuous space and knows nothing about the cell. The
    UnitCell turns the shape's SDF into a *periodic* SDF by taking the
    minimum over lattice copies, and samples it on a one-cell grid.

    The cell is only a sampling viewport: shapes are never clipped to
    it, so shapes touching or crossing the boundary stay continuous --
    and a shape positioned anywhere outside the cell, even far outside,
    wraps back in rather than reading as empty.

    Parameters
    ----------
    lattice : Lattice
        Fixed in-plane periodicity.
    scene : Shape
        The shape (usually a Union of primitives) to make periodic.
    """

    def __init__(self, lattice: Lattice, scene: Shape):
        super().__init__()
        self.lattice = lattice          # frozen dataclass, not a submodule
        self.scene = scene              # nn.Module -> auto-registered

    # --- periodic copy search ------------------------------------------
    def _offsets_for(self, shape: Shape) -> tuple[tuple[int, int], tuple[int, int]]:
        """Integer lattice-copy offset range needed to cover `shape`, per
        direction, for query points folded into the unit cell.

        Position-aware: unlike a symmetric ring sized only from the
        shape's own extent, this measures the shape's actual placement
        relative to the cell, so a scene far from the cell still gets
        exactly the copies it needs to wrap back in, rather than a ring
        centered on the wrong place. See `_decompose_for_offsets` for why
        `Union`/`Translate` are handled by decomposing first and unioning
        the resulting ranges, rather than measuring `shape.bounds()`
        directly.
        """
        i_range = j_range = None
        for piece in _decompose_for_offsets(shape):
            i, j = self._offsets_for_bounds(piece.bounds())
            i_range = i if i_range is None else (min(i_range[0], i[0]), max(i_range[1], i[1]))
            j_range = j if j_range is None else (min(j_range[0], j[0]), max(j_range[1], j[1]))
        return i_range, j_range

    def _offsets_for_bounds(
        self, bounds: tuple[tuple[float, float], tuple[float, float]]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Integer offset range (per lattice direction) covering one bbox.

        For lattice direction k, the fractional coordinate is the linear
        functional f_k(x, y) = b_k . (x, y), b_k the k-th row of the
        inverse lattice matrix (matching `Lattice.to_fractional`). Its
        extremes over an axis-aligned box are separable per world axis,
        computed here directly rather than by mapping the box's four
        corners through `to_fractional` and filtering non-finite results:
        that approach hits `0 * inf -> NaN` whenever any bbox coordinate
        is infinite, corrupting even a genuinely finite direction (L-02).
        Here a zero coefficient against an infinite side simply
        contributes nothing, which is exact, not a fallback.

        A copy whose fractional interval falls entirely more than one
        full cell outside [0, 1) cannot be the nearest copy to any query
        point already folded into [0, 1), so the search is widened by
        exactly one cell on each side of that interval.
        """
        if is_empty_bounds(bounds):
            return (0, 0), (0, 0)
        (x0, y0), (x1, y1) = bounds
        inv = torch.linalg.inv(self.lattice.matrix).tolist()  # rows = b_0, b_1

        ranges = []
        for b0, b1 in inv:
            lo = hi = 0.0
            infinite = False
            for coef, (d0, d1) in ((b0, (x0, x1)), (b1, (y0, y1))):
                if coef == 0.0:
                    continue  # infinite side is irrelevant on this axis
                u, v = coef * d0, coef * d1
                if math.isinf(u) or math.isinf(v):
                    infinite = True
                    break
                lo += min(u, v)
                hi += max(u, v)
            if infinite:
                # every copy along this direction is identical -> one is enough
                ranges.append((-1, 1))
            else:
                ranges.append((math.ceil(-1.0 - hi), math.floor(2.0 - lo)))
        return ranges[0], ranges[1]

    # --- periodic SDF ------------------------------------------------
    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Periodic signed distance of the scene at Cartesian (x, y).

        Query points are first folded into the unit cell via fractional
        modulo, then the minimum is taken over the lattice copies needed
        to reach the scene from anywhere in the cell (sized and
        positioned by `_offsets_for`). Folding first means the result
        only depends on a point's cell-relative position -- matching
        `rasterize`/`mask` -- and that the scene wraps back into the cell
        correctly however far from it the scene is actually defined.
        """
        f1, f2 = self.lattice.to_fractional(x, y)
        xf, yf = self.lattice.to_cartesian(f1 % 1.0, f2 % 1.0)

        (i0, i1), (j0, j1) = self._offsets_for(self.scene)
        best = None
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                ox, oy = self.lattice.offset(i, j)
                d = self.scene.sdf(xf - ox, yf - oy)
                best = d if best is None else torch.minimum(best, d)
        return best

    # --- rasterization ----------------------------------------------
    def rasterize(self, nx: int, ny: int, *,
                  repeat: tuple[int, int] = (1, 1),
                  cartesian: bool = False):
        """Periodic SDF sampled over a supercell.

        repeat=(n1, n2) — tile n1 cells along a1 and n2 along a2.

        cartesian=False (default) — samples on the fractional parallelogram
            grid; output shape [ny·n2, nx·n1]. Rows run along a2, so for
            oblique lattices the image appears sheared when displayed with
            imshow.

        cartesian=True — samples on an axis-aligned Cartesian bounding box
            of the supercell; output shape [ny·n2, nx·n1]. Rows are
            horizontal in world space — correct for imshow display of any
            lattice geometry, including hexagonal.
            `sdf()` itself folds query points back into the unit cell via
            fractional modulo, so the copy search stays valid regardless
            of repeat size.
        """
        n1, n2 = repeat

        if not cartesian:
            X, Y = cartesian_grid(self.lattice, nx, ny,
                                  dtype=self.lattice.dtype, device=self.lattice.device)
            d = self.sdf(X, Y)
            return d if (n1 == 1 and n2 == 1) else d.tile(n2, n1)

        # Cartesian bounding box of the n1×n2 supercell
        xmin, xmax, ymin, ymax = self.extent(repeat=repeat)

        xs = torch.linspace(xmin, xmax, nx * n1,
                            dtype=self.lattice.dtype, device=self.lattice.device)
        ys = torch.linspace(ymin, ymax, ny * n2,
                            dtype=self.lattice.dtype, device=self.lattice.device)
        X, Y = torch.meshgrid(xs, ys, indexing="xy")
        return self.sdf(X, Y)

    def mask(self, nx, ny, *, soft=False, softness=None,
             repeat: tuple[int, int] = (1, 1),
             cartesian: bool = False):
        """Rasterize the periodic structure into a mask. Shape [ny·n2, nx·n1].

        soft=False      -> hard binary mask (non-differentiable, for inference).
        soft=True       -> differentiable sigmoid mask; `softness` is the edge
                           scale in world units, defaulting to one pixel.
        repeat=(n1, n2) -> tile n1 cells along a1 and n2 cells along a2.
        cartesian=True  -> sample on an axis-aligned Cartesian grid so the
                           result displays correctly with imshow for any
                           lattice geometry (see rasterize for details).
        """
        d = self.rasterize(nx, ny, repeat=repeat, cartesian=cartesian)

        if not soft:
            return (d <= 0).to(d.dtype)

        if softness is None:
            if cartesian:
                n1, n2 = repeat
                xmin, xmax, ymin, ymax = self.extent(repeat=repeat)
                softness = min(
                    (xmax - xmin) / (nx * n1),
                    (ymax - ymin) / (ny * n2),
                )
            else:
                dx = (self.lattice.a1.norm() / nx).item()
                dy = (self.lattice.a2.norm() / ny).item()
                softness = min(dx, dy)

        softness = torch.as_tensor(softness, dtype=d.dtype, device=d.device)
        return torch.sigmoid(-d / softness)

    # --- boundary continuity check -----------------------------------
    def boundary_points(self, *, resolution: int = 512) -> np.ndarray:
        """
        Return world-coordinate points on the material boundary (zero-level-set).

        Evaluates the *periodic* SDF on a ``resolution × resolution`` grid over
        one unit cell, detects sign changes between adjacent samples — including
        across the cell seam, since opposite edges are identified under
        periodicity — and linearly interpolates each crossing to sub-pixel
        accuracy.

        Non-differentiable: intended for visualization, export, and Shapely
        interop, not for use inside a loss.

        Parameters
        ----------
        resolution : int
            Grid resolution per dimension (default 512). Must be fine enough
            to resolve the smallest feature; thinner features may be missed.

        Returns
        -------
        np.ndarray, shape (M, 2)
            ``(x, y)`` world coordinates on the boundary. Empty ``(0, 2)`` if
            the cell is uniform (no zero-crossing).
        """
        n = resolution

        # --- sample the periodic SDF on one cell -------------------------
        # fractional grid, endpoint-excluded so the seam is not duplicated
        f = torch.arange(n, dtype=self.lattice.dtype, device = self.lattice.device) / n
        F1, F2 = torch.meshgrid(f, f, indexing="xy")          # [n, n]
        X, Y = self.lattice.to_cartesian(F1, F2)
        with torch.no_grad():
            D = self.sdf(X, Y)                                # periodic SDF, [n, n]

        D = D.detach().cpu().numpy()
        X = X.detach().cpu().numpy()
        Y = Y.detach().cpu().numpy()

        # uniform cell -> no boundary
        if np.all(D > 0) or np.all(D < 0):
            return np.empty((0, 2), dtype=np.float64)

        eps = 1e-12
        pts = []

        def _crossings(d0, d1, x0, y0, x1, y1):
            """Sub-pixel boundary points where d0 and d1 straddle zero.

            d* are SDF arrays, (x*, y*) the corresponding world coords of
            the two endpoints. Returns a list of (x, y).
            """
            # a sign change: one side <= 0, the other > 0 (treat exact 0 as inside)
            sign_change = (d0 <= 0) != (d1 <= 0)
            if not np.any(sign_change):
                return []
            i = np.where(sign_change)
            a = d0[i]
            b = d1[i]
            denom = a - b
            # linear interp; guard the degenerate equal-value case
            t = np.where(np.abs(denom) > eps, a / denom, 0.5)
            px = x0[i] + t * (x1[i] - x0[i])
            py = y0[i] + t * (y1[i] - y0[i])
            return list(zip(px, py))

        # --- interior crossings, horizontal neighbours (column j -> j+1) -
        pts += _crossings(D[:, :-1], D[:, 1:],
                        X[:, :-1], Y[:, :-1],
                        X[:, 1:],  Y[:, 1:])

        # --- interior crossings, vertical neighbours (row i -> i+1) ------
        pts += _crossings(D[:-1, :], D[1:, :],
                        X[:-1, :], Y[:-1, :],
                        X[1:, :],  Y[1:, :])

        # --- seam crossings: last column <-> first column ----------------
        # opposite edges are identified under periodicity, so the first
        # column's true world position for interpolation is shifted by a1.
        a1 = self.lattice.a1.detach().cpu().numpy()
        pts += _crossings(D[:, -1], D[:, 0],
                        X[:, -1],        Y[:, -1],
                        X[:, 0] + a1[0], Y[:, 0] + a1[1])

        # --- seam crossings: last row <-> first row ----------------------
        a2 = self.lattice.a2.detach().cpu().numpy()
        pts += _crossings(D[-1, :], D[0, :],
                        X[-1, :],        Y[-1, :],
                        X[0, :] + a2[0], Y[0, :] + a2[1])

        if not pts:
            return np.empty((0, 2), dtype=np.float64)
        return np.asarray(pts, dtype=np.float64)

    # --- extent ------------------------------------------------------
    def extent(self, *, repeat: tuple[int, int] = (1, 1)) -> tuple[float, float, float, float]:
        """Axis-aligned Cartesian bounding box of the supercell.

        Returns ``(xmin, xmax, ymin, ymax)`` — the format expected by
        matplotlib's ``imshow(extent=...)``.

        Works correctly for any lattice geometry (rectangular, hexagonal,
        oblique). For a 1×1 cell the four parallelogram corners are mapped
        to Cartesian space; for a repeat supercell the full corner set is used.

        Parameters
        ----------
        repeat : tuple[int, int]
            Supercell tiling ``(n1, n2)`` along ``a1`` and ``a2``.
            Defaults to ``(1, 1)`` for a single unit cell.

        Example
        -------
        ::

            cell = UnitCell(Lattice.hexagonal(1.0), scene)
            mask = cell.mask(128, 128, cartesian=True).numpy()
            xmin, xmax, ymin, ymax = cell.extent()
            plt.imshow(mask, extent=[xmin, xmax, ymin, ymax], origin='lower')
        """
        n1, n2 = repeat
        fc = torch.tensor(
            [[0.0, 0.0], [float(n1), 0.0], [0.0, float(n2)], [float(n1), float(n2)]],
            dtype=self.lattice.dtype, device=self.lattice.device,
        )
        cx, cy = self.lattice.to_cartesian(fc[:, 0], fc[:, 1])
        return cx.min().item(), cx.max().item(), cy.min().item(), cy.max().item()

    # --- centering ---------------------------------------------------
    def center_scene(self, method: str = "bbox") -> "UnitCell":
        """Return a new UnitCell with the scene translated to the cell centre.

        The scene is shifted so its centre aligns with the midpoint of the
        unit-cell parallelogram, ``(a1 + a2) / 2``.

        Parameters
        ----------
        method : {"bbox", "centroid"}
            How to compute the scene centre:

            ``"bbox"`` (default)
                Midpoint of the scene's axis-aligned bounding box (AABB).
                Fast and analytical; raises ``ValueError`` for shapes with
                infinite extent (e.g. ``Bar``).

            ``"centroid"``
                Area-weighted geometric centroid via the Shapely adapter.
                More meaningful for irregular or asymmetric scenes (L-shapes,
                crosses, cutouts).

        Returns
        -------
        UnitCell
            New unit cell with the scene translated; the original is unchanged.

        Notes
        -----
        **Gradient flow:** the centering offset is stored as a non-learnable
        buffer in the ``Translate`` wrapper.  Shape parameters that are
        ``nn.Parameter`` (e.g. a learnable centre position) retain full
        gradient flow through the SDF — centering only sets the initial
        world-space position.  If you need the offset itself to be
        differentiable, wrap it explicitly::

            dx = nn.Parameter(torch.tensor(computed_dx))
            UnitCell(lattice, scene.translate(dx, dy_param))

        Raises
        ------
        ValueError
            If ``method="bbox"`` and the scene has infinite bounds, or if
            *method* is not recognised.
        """
        # Unit cell centre: midpoint of the parallelogram defined by a1 and a2
        c = (self.lattice.a1 + self.lattice.a2) * 0.5
        cell_cx, cell_cy = c[0].item(), c[1].item()

        if method == "bbox":
            scene_bounds = self.scene.bounds()
            if is_empty_bounds(scene_bounds):
                raise ValueError(
                    "Scene is empty (e.g. the intersection of disjoint shapes); "
                    "there is no bounding box to center."
                )
            (x0, y0), (x1, y1) = scene_bounds
            if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
                raise ValueError(
                    "Scene has infinite bounds; method='bbox' requires finite bounds. "
                    "Use method='centroid' instead."
                )
            scene_cx = (x0 + x1) / 2
            scene_cy = (y0 + y1) / 2

        elif method == "centroid":
            from metashapes.adapters.shapely import shape_to_shapely
            geom = shape_to_shapely(self.scene)
            scene_cx = geom.centroid.x
            scene_cy = geom.centroid.y

        else:
            raise ValueError(
                f"Unknown centering method {method!r}. "
                "Choose 'bbox' or 'centroid'."
            )

        dx = cell_cx - scene_cx
        dy = cell_cy - scene_cy
        return UnitCell(self.lattice, self.scene.translate(dx, dy))

    # --- shapely adapter ---------------------------------------------
    def to_shapely(self):
        """Shapely geometry of the scene, wrapped periodically into the
        unit cell.

        Periodic, matching `sdf()`: shapes touching or crossing the cell
        boundary are unioned with their neighbouring lattice copies before
        clipping, so the wrapped-around part is included rather than lost
        -- and a scene positioned entirely outside the cell (even far
        outside) still contributes its wrapped-in copy rather than reading
        as empty. Uses the same copy search as `sdf()` (`_offsets_for`),
        so a shape fully inside the cell gives the same result as clipping
        it alone -- its other copies fall entirely outside `cell_poly` and
        are discarded by the intersection.
        """
        from shapely.affinity import translate
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        from metashapes.adapters.shapely import shape_to_shapely

        base = shape_to_shapely(self.scene)
        (i0, i1), (j0, j1) = self._offsets_for(self.scene)
        copies = []
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                if i == 0 and j == 0:
                    copies.append(base)
                    continue
                ox, oy = self.lattice.offset(i, j)
                copies.append(translate(base, xoff=ox.item(), yoff=oy.item()))
        unioned = unary_union(copies)

        a1 = self.lattice.a1.detach().cpu().tolist()
        a2 = self.lattice.a2.detach().cpu().tolist()
        cell_poly = Polygon([
            (0.0, 0.0),
            (a1[0], a1[1]),
            (a1[0] + a2[0], a1[1] + a2[1]),
            (a2[0], a2[1]),
        ])
        return unioned.intersection(cell_poly)

    # --- serialization -----------------------------------------------
    def to_parametric(self) -> dict:
        return {
            "type": "UnitCell",
            "lattice": {
                "a1": self.lattice.a1.detach().cpu().tolist(),
                "a2": self.lattice.a2.detach().cpu().tolist(),
            },
            "scene": self.scene.to_parametric(),
        }

    @classmethod
    def from_parametric(cls, data: dict) -> "UnitCell":
        lat = data["lattice"]
        lattice = Lattice(
            a1=torch.tensor(lat["a1"]),
            a2=torch.tensor(lat["a2"]),
        )
        scene = Shape.from_parametric(data["scene"])
        return cls(lattice=lattice, scene=scene)
