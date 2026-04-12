# metashapes/unit_cell.py
# UnitCell and UniformCell — the two concrete unit-cell types.

from __future__ import annotations

import numpy as np
import torch
from shapely.geometry.base import BaseGeometry

from metashapes.canvas import Canvas
from metashapes.shape.base import Shape
from metashapes.shape.primitives import Rectangle

__all__ = ["UnitCell", "UniformCell"]


# ---------------------------------------------------------------------------
# Shared rasterisation helper
# ---------------------------------------------------------------------------

def _mask(sdf_fn, canvas: Canvas, *, dtype, device, soft, softness):
    x, y = canvas.grid(dtype=dtype, device=device)
    d = sdf_fn(x, y)

    if not soft:
        return (d <= 0).to(dtype)

    if softness is None:
        softness = min(canvas.dx, canvas.dy)

    softness = torch.as_tensor(softness, dtype=d.dtype, device=d.device)
    return torch.sigmoid(-d / softness)


# ---------------------------------------------------------------------------
# UnitCell
# ---------------------------------------------------------------------------

class UnitCell:
    """
    Unit cell of a periodic structure.

    Parameters
    ----------
    shape : Shape
        Symbolic pattern.
    canvas : Canvas
        Defines the unit-cell physical size and raster grid.
    inverted : bool
        False (default) — shape is the material region.
        True           — shape is the void region (material is the complement).
    periodic_shifts : set[tuple[int,int]] | None
        Periodic directions the shape spans.  In those directions the shape is
        **not** clipped to the unit-cell boundary so the SDF is continuous
        across it.  When *None*, read from ``shape.allowed_self_periodic_shifts``.
    """

    def __init__(
        self,
        shape: Shape,
        canvas: Canvas,
        inverted: bool = False,
        periodic_shifts: set[tuple[int, int]] | None = None,
    ) -> None:
        if not isinstance(shape, Shape):
            raise TypeError("shape must be a Shape")
        if not isinstance(canvas, Canvas):
            raise TypeError("canvas must be a Canvas")
        self.shape = shape
        self.canvas = canvas
        self.inverted = inverted
        if periodic_shifts is None:
            periodic_shifts = shape.allowed_self_periodic_shifts
        self.periodic_shifts: set[tuple[int, int]] = periodic_shifts

    # ------------------------------------------------------------------
    # Internal geometry helpers
    # ------------------------------------------------------------------

    @property
    def unit_cell_shape(self) -> Shape:
        """Rectangle corresponding to the inner canvas region (excluding spacing)."""
        x0, y0, x1, y1 = self.canvas.inner_bounds
        return Rectangle(
            center=((x0 + x1) / 2, (y0 + y1) / 2),
            size=(x1 - x0, y1 - y0),
        )

    def _make_clip_region(self) -> Shape | None:
        """
        Clip region for the shape, skipping boundaries in periodic directions.

        Returns
        -------
        Shape
            ``unit_cell_shape`` — no periodic directions
            Wide strip         — periodic in one axis
        None
            Periodic in both axes (no clipping needed).
        """
        if not self.periodic_shifts:
            return self.unit_cell_shape

        periodic_x = any(ix != 0 for ix, iy in self.periodic_shifts)
        periodic_y = any(iy != 0 for ix, iy in self.periodic_shifts)

        # Skip if periodic in both directions
        if periodic_x and periodic_y:
            return None

        x0, y0, x1, y1 = self.canvas.inner_bounds
        xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
        LARGE = max(self.canvas.Lx, self.canvas.Ly) * 1000 # Supports any big number (default=x1000)

        if periodic_x:
            return Rectangle(center=(xc, yc), size=(LARGE, y1 - y0)) # extends along x
        else:
            return Rectangle(center=(xc, yc), size=(x1 - x0, LARGE)) # extends along y

    def _periodic_sdf(self, region: Shape, x, y):
        Lx, Ly = self.canvas.Lx, self.canvas.Ly
        xc, yc = self.canvas.xc, self.canvas.yc

        ix = torch.round((x - xc) / Lx).detach()
        iy = torch.round((y - yc) / Ly).detach()

        ox = torch.where(x - (xc + ix * Lx) >= 0, 1.0, -1.0).detach()
        oy = torch.where(y - (yc + iy * Ly) >= 0, 1.0, -1.0).detach()

        # Takes 4 cells - (0,0), (0, 1), (1, 0), (1, 1)
        ds = []
        for j in (0, 1):
            for i in (0, 1):
                ds.append(region.sdf(x - (ix + i * ox) * Lx, y - (iy + j * oy) * Ly))

        return torch.stack(ds, dim=0).amin(dim=0)

    # ------------------------------------------------------------------
    # Material / void regions
    # ------------------------------------------------------------------

    @property
    def filled_region(self) -> Shape:
        """Symbolic material region (periodic directions not clipped)."""
        clip = self._make_clip_region() # handles 1D periodic cases
        if clip is None:
            return self.shape
        return (clip - self.shape) if self.inverted else (clip & self.shape)

    @property
    def void_region(self) -> Shape:
        """Symbolic void region (periodic directions not clipped)."""
        clip = self._make_clip_region() # handles 1D periodic cases
        if clip is None:
            return self.shape
        return (clip & self.shape) if self.inverted else (clip - self.shape)

    # ------------------------------------------------------------------
    # SDF
    # ------------------------------------------------------------------

    def sdf(self, x, y):
        """
        Signed distance to the periodic material region.
        Negative inside material, positive outside.
        """
        if self.inverted:
            return -self._periodic_sdf(self.void_region, x, y)
        return self._periodic_sdf(self.filled_region, x, y)

    # ------------------------------------------------------------------
    # Rasterisation
    # ------------------------------------------------------------------

    def mask(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        soft: bool = False,
        softness: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rasterise UnitCell into a torch mask of shape (H, W)."""
        return _mask(self.sdf, self.canvas, dtype=dtype, device=device,
                     soft=soft, softness=softness)

    def mask_numpy(
        self,
        *,
        dtype=np.float32,
        soft: bool = False,
        softness: float | None = None,
    ) -> np.ndarray:
        """Rasterise UnitCell into a NumPy array."""
        return self.mask(dtype=torch.float32, device="cpu",
                         soft=soft, softness=softness
                         ).detach().cpu().numpy().astype(dtype, copy=False)

    def boundary_points(self, *, resolution: int = 512) -> np.ndarray:
        """
        Return world-coordinate points lying on the material boundary.

        Evaluates the SDF on a ``resolution × resolution`` grid, detects
        sign changes between adjacent pixels (zero-crossings), and linearly
        interpolates each crossing to sub-pixel accuracy.

        Parameters
        ----------
        resolution : int
            Grid resolution in each dimension (default 512).

        Returns
        -------
        np.ndarray, shape (M, 2)
            Array of ``(x, y)`` world coordinates on the boundary.
            May be empty (shape ``(0, 2)``) if the cell is uniform.
        """
        from metashapes.canvas import Canvas as _Canvas

        canvas_hr = _Canvas(
            Lx=self.canvas.Lx,
            Ly=self.canvas.Ly,
            xc=self.canvas.xc,
            yc=self.canvas.yc,
            W=resolution,
            H=resolution,
            spacing=self.canvas.spacing,
        )
        x, y = canvas_hr.grid()
        d = self.sdf(x, y).detach().cpu().numpy()
        x_np = x.numpy()
        y_np = y.numpy()

        sign = np.sign(d)

        # --- horizontal crossings (between column j and j+1) ---
        hm = sign[:, :-1] * sign[:, 1:] < 0
        rh, ch = np.where(hm)
        if len(rh):
            denom_h = d[rh, ch] - d[rh, ch + 1]
            t_h = np.where(np.abs(denom_h) > 1e-12, d[rh, ch] / denom_h, 0.5)
            x_h = x_np[rh, ch] + t_h * (x_np[rh, ch + 1] - x_np[rh, ch])
            y_h = y_np[rh, ch]
        else:
            x_h = y_h = np.empty(0)

        # --- vertical crossings (between row i and i+1) ---
        vm = sign[:-1, :] * sign[1:, :] < 0
        rv, cv = np.where(vm)
        if len(rv):
            denom_v = d[rv, cv] - d[rv + 1, cv]
            t_v = np.where(np.abs(denom_v) > 1e-12, d[rv, cv] / denom_v, 0.5)
            x_v = x_np[rv, cv]
            y_v = y_np[rv, cv] + t_v * (y_np[rv + 1, cv] - y_np[rv, cv])
        else:
            x_v = y_v = np.empty(0)

        if not (len(x_h) or len(x_v)):
            return np.empty((0, 2))

        return np.column_stack([
            np.concatenate([x_h, x_v]),
            np.concatenate([y_h, y_v]),
        ])

    def __repr__(self) -> str:
        inv = ", inverted=True" if self.inverted else ""
        ps = f", periodic_shifts={self.periodic_shifts!r}" if self.periodic_shifts else ""
        return f"UnitCell({self.shape!r}, {self.canvas!r}{inv}{ps})"

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_parametric(self) -> dict:
        return {
            "type": "UnitCell",
            "shape": self.shape.to_parametric(),
            "canvas": self.canvas.to_parametric(),
            "inverted": self.inverted,
            "periodic_shifts": [list(s) for s in self.periodic_shifts],
        }

    @classmethod
    def from_parametric(cls, data: dict) -> UnitCell:
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")
        if data.get("type") != "UnitCell":
            raise ValueError("data is not a UnitCell parametric dictionary")
        try:
            shape = Shape.from_parametric(data["shape"])
            canvas = Canvas.from_parametric(data["canvas"])
            inverted = bool(data["inverted"])
        except KeyError as e:
            raise ValueError(f"Missing required field: {e.args[0]}") from e
        periodic_shifts = {tuple(s) for s in data.get("periodic_shifts", [])}
        return cls(shape=shape, canvas=canvas, inverted=inverted,
                   periodic_shifts=periodic_shifts)

    # ------------------------------------------------------------------
    # Adapters
    # ------------------------------------------------------------------

    def to_shapely(self) -> BaseGeometry:
        """Convert material region to a Shapely geometry."""
        from metashapes.adapters import shape_to_shapely
        shape_plain = Shape.from_parametric(self.filled_region.to_parametric())
        return shape_to_shapely(shape_plain)


# ---------------------------------------------------------------------------
# UniformCell
# ---------------------------------------------------------------------------

class UniformCell:
    """
    A uniformly filled (or empty) unit cell.

    The SDF is ``-inf`` everywhere for a filled cell and ``+inf`` for an
    empty one, so soft masks are exact (1.0 or 0.0) with no boundary artefact.
    """

    def __init__(self, canvas: Canvas, filled: bool = True) -> None:
        if not isinstance(canvas, Canvas):
            raise TypeError("canvas must be a Canvas")
        self.canvas = canvas
        self.filled = bool(filled)

    def sdf(self, x, y):
        if self.filled:
            return torch.full_like(x, -torch.inf)
        return torch.full_like(x, torch.inf)

    def mask(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        soft: bool = False,
        softness: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _mask(self.sdf, self.canvas, dtype=dtype, device=device,
                     soft=soft, softness=softness)

    def mask_numpy(
        self,
        *,
        dtype=np.float32,
        soft: bool = False,
        softness: float | None = None,
    ) -> np.ndarray:
        return self.mask(dtype=torch.float32, device="cpu",
                         soft=soft, softness=softness
                         ).detach().cpu().numpy().astype(dtype, copy=False)

    def __repr__(self) -> str:
        return f"UniformCell({self.canvas!r}, filled={self.filled!r})"

    def to_parametric(self) -> dict:
        return {
            "type": "UniformCell",
            "canvas": self.canvas.to_parametric(),
            "filled": self.filled,
        }

    @classmethod
    def from_parametric(cls, data: dict) -> UniformCell:
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")
        if data.get("type") != "UniformCell":
            raise ValueError("data is not a UniformCell parametric dictionary")
        return cls(
            canvas=Canvas.from_parametric(data["canvas"]),
            filled=bool(data["filled"]),
        )

    def to_shapely(self) -> BaseGeometry:
        from metashapes.adapters import shape_to_shapely
        if self.filled:
            x0, y0, x1, y1 = self.canvas.inner_bounds
            cell_rect = Rectangle(
                center=((x0 + x1) / 2, (y0 + y1) / 2),
                size=(x1 - x0, y1 - y0),
            )
            return shape_to_shapely(cell_rect)
        from shapely.geometry import GeometryCollection
        return GeometryCollection()
