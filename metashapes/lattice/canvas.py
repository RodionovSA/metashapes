#metashapes/canvas.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


RoundMode = Literal["round", "floor", "ceil"]

@dataclass(frozen=True, slots=True)
class Canvas:
    """
    Defines unit cell size and grid size, and do resterization. 
    """
    # world window and raster size
    Lx: float; Ly: float   # width/height
    H: int; W: int;        # rows, cols
    xc: float = 0.0; yc: float = 0.0;  # center of the canvas
    spacing: float = 0.0   # optional spacing from edges (in world units)

    def __post_init__(self) -> None:
        if self.Lx <= 0 or self.Ly <= 0:
            raise ValueError("Lx and Ly must be positive")

        if not isinstance(self.H, int) or not isinstance(self.W, int):
            raise TypeError("H and W must be integers")

        if self.H <= 0 or self.W <= 0:
            raise ValueError("H and W must be positive")

        if self.spacing < 0:
            raise ValueError("spacing must be non-negative")

        if 2 * self.spacing >= self.Lx or 2 * self.spacing >= self.Ly:
            raise ValueError("spacing is too large for the given canvas size")
        
    def grid(self, *, dtype=torch.float32, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return torch grids of pixel-center coordinates.

        Returns:
            x, y: tensors of shape (H, W), where
                x varies along columns,
                y varies along rows.
        """
        x = torch.from_numpy(self.x_centers).to(dtype=dtype, device=device)
        y = torch.from_numpy(self.y_centers).to(dtype=dtype, device=device)
    
        y, x = torch.meshgrid(y, x, indexing="ij")
        return x, y

    @property
    def dx(self) -> float:
        " Pixel size in x direction (world units per pixel)"
        return self.Lx / self.W

    @property
    def dy(self) -> float:
        " Pixel size in y direction (world units per pixel)"
        return self.Ly / self.H

    @property
    def sx(self) -> float:
        " Scale factor from world units to pixel units in x direction (pixels per world unit)"
        return self.W / self.Lx

    @property
    def sy(self) -> float:
        " Scale factor from world units to pixel units in y direction (pixels per world unit)"
        return self.H / self.Ly

    @property
    def x0(self) -> float:
        " Left edge of the canvas in world coordinates"
        return self.xc - 0.5 * self.Lx

    @property
    def x1(self) -> float:
        " Right edge of the canvas in world coordinates"
        return self.xc + 0.5 * self.Lx

    @property
    def y0(self) -> float:
        " Bottom edge of the canvas in world coordinates"
        return self.yc - 0.5 * self.Ly

    @property
    def y1(self) -> float:
        " Top edge of the canvas in world coordinates"
        return self.yc + 0.5 * self.Ly
    
    @property
    def x_centers(self) -> np.ndarray:
        """
        X coordinates of pixel centers, shape (W,).
        """
        return self.x0 + (np.arange(self.W) + 0.5) * self.dx

    @property
    def y_centers(self) -> np.ndarray:
        """
        Y coordinates of pixel centers, shape (H,).

        Ordered from bottom to top, consistent with Cartesian coordinates.
        """
        return self.y0 + (np.arange(self.H) + 0.5) * self.dy

    @property
    def inner_bounds(self) -> tuple[float, float, float, float]:
        return (
            self.x0 + self.spacing,
            self.y0 + self.spacing,
            self.x1 - self.spacing,
            self.y1 - self.spacing,
        )

    def world_to_rc(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x)
        y = np.asarray(y)
        cols = (x - self.x0) * self.sx
        rows = (self.y1 - y) * self.sy
        return rows, cols

    def rc_to_world(self, r, c) -> tuple[np.ndarray, np.ndarray]:
        r = np.asarray(r)
        c = np.asarray(c)
        x = self.x0 + c * self.dx
        y = self.y1 - r * self.dy
        return x, y

    def with_grid(self, H: int, W: int) -> Canvas:
        """
        Return a new Canvas with the same physical size and position,
        but a different raster resolution.
        """
        return Canvas(
            Lx=self.Lx,
            Ly=self.Ly,
            H=H,
            W=W,
            xc=self.xc,
            yc=self.yc,
            spacing=self.spacing,
        )

    def with_pixel_size(
        self,
        pixel_size: float,
        rounding: RoundMode = "round",
    ) -> Canvas:
        """
        Return a new Canvas whose grid size (H, W) is chosen from a target
        physical pixel size.

        The physical canvas size stays the same, but the number of rows and
        columns is recomputed so that:

            dx ≈ pixel_size
            dy ≈ pixel_size
        """
        if pixel_size <= 0:
            raise ValueError("pixel_size must be positive")

        f = {"round": np.round, "floor": np.floor, "ceil": np.ceil}[rounding]
        H = max(int(f(self.Ly / pixel_size)), 1)
        W = max(int(f(self.Lx / pixel_size)), 1)
        return self.with_grid(H=H, W=W)

    def to_parametric(self) -> str:
        return (
            f"Canvas(Lx={self.Lx}, Ly={self.Ly}, H={self.H}, W={self.W}, "
            f"xc={self.xc}, yc={self.yc}, spacing={self.spacing})"
        )

    @classmethod
    def from_parametric(cls, param_str: str) -> Canvas:
        s = param_str.strip()

        if s.startswith("Canvas(") and s.endswith(")"):
            s = s[len("Canvas("):-1].strip()

        params: dict[str, float] = {}
        for item in s.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"Invalid parameter fragment: {item!r}")
            key, value = item.split("=", 1)
            key = key.strip()
            value = value.strip()
            try:
                params[key] = float(value)
            except ValueError as e:
                raise ValueError(f"Invalid numeric value for {key!r}: {value!r}") from e

        required = {"Lx", "Ly", "H", "W", "xc", "yc", "spacing"}
        missing = required - params.keys()
        if missing:
            raise ValueError(f"Missing parameters: {missing}")

        return cls(
            Lx=params["Lx"],
            Ly=params["Ly"],
            H=int(params["H"]),
            W=int(params["W"]),
            xc=params["xc"],
            yc=params["yc"],
            spacing=params["spacing"],
        )
       
            
    