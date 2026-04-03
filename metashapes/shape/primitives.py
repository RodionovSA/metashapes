# metashapes/shape/primitives.py
# This module defines shape primitives

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch
import numpy as np

from .base import Shape, to_plain_data, to_plain_scalar
from .registry import register_shape

__all__ = [
    "Rectangle",
    "Ellipse",
    "RegularPolygon",
]

@register_shape("Rectangle")
@dataclass(slots=True)
class Rectangle(Shape):
    """
    Symbolic rectangle.

    Parameters:
        center: (cx, cy)
        size: (width, height)
        angle: counter-clockwise rotation angle in degrees
        corner_radius: radius for soft corners (ignored if soft_corners=False)
    """
    center: tuple[Any, Any]
    size: tuple[Any, Any]
    angle: Any = 0.0
    corner_radius: Any = 0.0

    def __post_init__(self) -> None:
        if len(self.center) != 2:
            raise ValueError("center must have length 2")
        if len(self.size) != 2:
            raise ValueError("size must have length 2")
        
    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cx = torch.as_tensor(self.center[0], dtype=x.dtype, device=x.device)
        cy = torch.as_tensor(self.center[1], dtype=y.dtype, device=y.device)
        w = torch.as_tensor(self.size[0], dtype=x.dtype, device=x.device)
        h = torch.as_tensor(self.size[1], dtype=y.dtype, device=y.device)
        a = torch.as_tensor(self.angle, dtype=x.dtype, device=x.device)
        r = torch.as_tensor(self.corner_radius, dtype=x.dtype, device=x.device)

        if torch.any(w <= 0) or torch.any(h <= 0):
            raise ValueError("Rectangle size components must be positive")
        if torch.any(r < 0):
            raise ValueError("corner_radius must be non-negative")

        r = torch.minimum(r, 0.5 * torch.minimum(w, h))

        # shift to local center and rotate
        x_local, y_local = _to_local_coords(x, y, cx, cy, a)

        hx = w * 0.5 - r
        hy = h * 0.5 - r

        qx = torch.abs(x_local) - hx
        qy = torch.abs(y_local) - hy

        outside = torch.sqrt(torch.clamp(qx, min=0)**2 + torch.clamp(qy, min=0)**2)
        inside = torch.minimum(torch.maximum(qx, qy), torch.zeros((), dtype=x.dtype, device=x.device))

        return outside + inside - r
        

    def to_parametric(self) -> dict:
        return {
            "type": "Rectangle",
            "center": to_plain_data(self.center),
            "size": to_plain_data(self.size),
            "angle": to_plain_scalar(self.angle),
            "corner_radius": to_plain_scalar(self.corner_radius),
        }
        
    @classmethod
    def from_parametric(cls, data: dict) -> "Rectangle":
        return cls(
            center=tuple(data["center"]),
            size=tuple(data["size"]),
            angle=data.get("angle", 0.0),
            corner_radius=data.get("corner_radius", 0.0),
        )


@register_shape("Ellipse")
@dataclass(slots=True)
class Ellipse(Shape):
    """
    Symbolic ellipse.

    Parameters:
        center: (cx, cy)
        axes: full size axes (a, b)
        angle: counter-clockwise rotation angle in degrees
        resolution: optional hint for evaluators that approximate the ellipse
    """
    center: tuple[Any, Any]
    axes: tuple[Any, Any]
    angle: Any = 0.0

    def __post_init__(self) -> None:
        if len(self.center) != 2:
            raise ValueError("center must have length 2")
        if len(self.axes) != 2:
            raise ValueError("axes must have length 2")
        
    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cx = torch.as_tensor(self.center[0], dtype=x.dtype, device=x.device)
        cy = torch.as_tensor(self.center[1], dtype=y.dtype, device=y.device)
        a = torch.as_tensor(self.axes[0], dtype=x.dtype, device=x.device)
        b = torch.as_tensor(self.axes[1], dtype=y.dtype, device=y.device)
        angle = torch.as_tensor(self.angle, dtype=x.dtype, device=x.device)

        if torch.any(a <= 0) or torch.any(b <= 0):
            raise ValueError("Ellipse axes must be positive")
        
        a2 = a * 0.5
        b2 = b * 0.5
        
        # shift to local center and rotate
        x_local, y_local = _to_local_coords(x, y, cx, cy, angle)

        # work in first quadrant by symmetry
        px = torch.abs(x_local)
        py = torch.abs(y_local)

        eps = torch.finfo(x.dtype).eps

        # initial guess for closest-point parameter
        t = torch.atan2(a2 * py, b2 * px + eps)

        # Newton iterations
        for _ in range(3):
            st = torch.sin(t)
            ct = torch.cos(t)

            f = (a2 * a2 - b2 * b2) * st * ct - a2 * px * st + b2 * py * ct
            df = (a2 * a2 - b2 * b2) * (ct * ct - st * st) - a2 * px * ct - b2 * py * st

            t = t - f / (df + eps)
            t = torch.clamp(t, 0.0, 0.5 * torch.pi)

        # closest point on ellipse
        ex = a2 * torch.cos(t)
        ey = b2 * torch.sin(t)

        dist = torch.sqrt((px - ex) ** 2 + (py - ey) ** 2 + eps)

        # standard SDF sign: negative inside, positive outside
        inside = (x_local / a2) ** 2 + (y_local / b2) ** 2 <= 1.0
        return torch.where(inside, -dist, dist)

    def to_parametric(self) -> dict:
        return {
            "type": "Ellipse",
            "center": to_plain_data(self.center),
            "axes": to_plain_data(self.axes),
            "angle": to_plain_scalar(self.angle),
        }
        
    @classmethod
    def from_parametric(cls, data: dict) -> "Ellipse":
        return cls(
            center=tuple(data["center"]),
            axes=tuple(data["axes"]),
            angle=data.get("angle", 0.0),
        )

@register_shape("RegularPolygon")    
@dataclass(slots=True)
class RegularPolygon(Shape):
    """
    Symbolic regular polygon.

    Parameters:
        center: (cx, cy)
        n: Number of sides.
        side_length: Length of each side.
        angle: Counter-clockwise rotation angle in degrees.
    """
    center: tuple[Any, Any]
    n: int
    side_length: Any
    angle: Any = 0.0
    corner_radius: Any = 0.0

    def __post_init__(self) -> None:
        if len(self.center) != 2:
            raise ValueError("center must have length 2")
        if self.n < 3:
            raise ValueError("n must be at least 3")
        
    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cx = torch.as_tensor(self.center[0], dtype=x.dtype, device=x.device)
        cy = torch.as_tensor(self.center[1], dtype=y.dtype, device=y.device)
        s = torch.as_tensor(self.side_length, dtype=x.dtype, device=x.device)
        rr = torch.as_tensor(self.corner_radius, dtype=x.dtype, device=x.device)
        n = int(self.n)

        if n < 3:
            raise ValueError("n must be at least 3")
        if torch.any(s <= 0):
            raise ValueError("side_length must be positive")
        if torch.any(rr < 0):
            raise ValueError("corner_radius must be non-negative")

        pi = torch.as_tensor(np.pi, dtype=x.dtype, device=x.device)
        angle = torch.as_tensor(self.angle, dtype=x.dtype, device=x.device)  # degrees if _to_local_coords expects degrees

        # original polygon radii
        n_t = torch.as_tensor(float(n), dtype=x.dtype, device=x.device)
        an = pi / n_t
        R = s / (2.0 * torch.sin(an))     # circumradius
        rho = R * torch.cos(an)           # apothem

        if torch.any(rr >= rho):
            raise ValueError("corner_radius is too large")

        # inset polygon: preserve outer support lines after rounding
        rho_in = rho - rr
        R_in = rho_in / torch.cos(an)

        # local coords
        x_local, y_local = _to_local_coords(x, y, cx, cy, angle)

        # inset polygon vertices, same convention as shapely
        phi0 = pi / 2.0
        verts = []
        for k in range(n):
            th = 2.0 * pi * k / n_t + phi0
            vx = R_in * torch.cos(th)
            vy = R_in * torch.sin(th)
            verts.append((vx, vy))

        min_d2 = None
        inside = torch.ones_like(x_local, dtype=torch.bool)

        for k in range(n):
            ax, ay = verts[k]
            bx, by = verts[(k + 1) % n]

            ex = bx - ax
            ey = by - ay
            wx = x_local - ax
            wy = y_local - ay

            ee = ex * ex + ey * ey
            t = torch.clamp((wx * ex + wy * ey) / ee, 0.0, 1.0)

            px = ax + t * ex
            py = ay + t * ey

            d2 = (x_local - px) ** 2 + (y_local - py) ** 2
            min_d2 = d2 if min_d2 is None else torch.minimum(min_d2, d2)

            cross = ex * (y_local - ay) - ey * (x_local - ax)
            inside = inside & (cross >= 0)

        d_in = torch.sqrt(torch.clamp(min_d2, min=0.0))
        d_in = torch.where(inside, -d_in, d_in)

        return d_in - rr

    def to_parametric(self) -> dict:
        return {
            "type": "RegularPolygon",
            "center": to_plain_data(self.center),
            "n": int(self.n),
            "side_length": to_plain_data(self.side_length),
            "angle": to_plain_scalar(self.angle),
            "corner_radius": to_plain_scalar(self.corner_radius),
        }
        
    @classmethod
    def from_parametric(cls, data: dict) -> "RegularPolygon":
        return cls(
            center=tuple(data["center"]),
            n=data["n"],
            side_length=data["side_length"],
            angle=data.get("angle", 0.0),
            corner_radius=data.get("corner_radius", 0.0),
        )


@dataclass(slots=True)
class Cross(Shape):
    """
    Symbolic cross.

    Parameters:
        center: (cx, cy)
        size: Total arm length.
        width: Arm width.
        angle: Counter-clockwise rotation angle in degrees.
    """
    center: tuple[Any, Any]
    size: Any
    width: Any
    angle: Any = 0.0

    def __post_init__(self) -> None:
        if len(self.center) != 2:
            raise ValueError("center must have length 2")

    def to_parametric(self) -> dict:
        return {
            "type": "Cross",
            "center": to_plain_data(self.center),
            "size": to_plain_scalar(self.size),
            "width": to_plain_scalar(self.width),
            "angle": to_plain_scalar(self.angle),
        }
        
    @classmethod
    def from_parametric(cls, data: dict) -> "Cross":
        return cls(
            center=tuple(data["center"]),
            size=data["size"],
            width=data["width"],
            angle=data.get("angle", 0.0),
        )


@dataclass(slots=True)
class Ring(Shape):
    """
    Symbolic elliptical ring.

    Parameters:
        center: (cx, cy)
        outer_axes: Outer semi-axes (a_out, b_out)
        inner_axes: Inner semi-axes (a_in, b_in)
        angle: Counter-clockwise rotation angle in degrees.
        resolution: Optional approximation hint for evaluators.
    """
    center: tuple[Any, Any]
    outer_axes: tuple[Any, Any]
    inner_axes: tuple[Any, Any]
    angle: Any = 0.0
    resolution: int = 64

    def __post_init__(self) -> None:
        if len(self.center) != 2:
            raise ValueError("center must have length 2")
        if len(self.outer_axes) != 2:
            raise ValueError("outer_axes must have length 2")
        if len(self.inner_axes) != 2:
            raise ValueError("inner_axes must have length 2")
        if self.resolution < 8:
            raise ValueError("resolution must be at least 8")

    def to_parametric(self) -> dict:
        return {
            "type": "Ring",
            "center": to_plain_data(self.center),
            "outer_axes": to_plain_data(self.outer_axes),
            "inner_axes": to_plain_data(self.inner_axes),
            "angle": to_plain_scalar(self.angle),
            "resolution": int(self.resolution),
        }
        
    @classmethod
    def from_parametric(cls, data: dict) -> "Ring":
        return cls(
            center=tuple(data["center"]),
            outer_axes=tuple(data["outer_axes"]),
            inner_axes=tuple(data["inner_axes"]),
            angle=data.get("angle", 0.0),
            resolution=data.get("resolution", 64),
        )


@dataclass(slots=True)
class Moon(Shape):
    """
    Symbolic crescent / moon shape.

    Parameters:
        center: (cx, cy)
        radius: Radius of the main circle.
        cut_ratio: Controls crescent thickness.
        angle: Counter-clockwise rotation angle in degrees.
        resolution: Optional approximation hint for evaluators.
    """
    center: tuple[Any, Any]
    radius: Any
    cut_ratio: Any = 0.5
    angle: Any = 0.0
    resolution: int = 64

    def __post_init__(self) -> None:
        if len(self.center) != 2:
            raise ValueError("center must have length 2")
        if self.resolution < 8:
            raise ValueError("resolution must be at least 8")

    def to_parametric(self) -> dict:
        return {
            "type": "Moon",
            "center": to_plain_data(self.center),
            "radius": to_plain_scalar(self.radius),
            "cut_ratio": to_plain_scalar(self.cut_ratio),
            "angle": to_plain_scalar(self.angle),
            "resolution": int(self.resolution),
        }
        
    @classmethod
    def from_parametric(cls, data: dict) -> "Moon":
        return cls(
            center=tuple(data["center"]),
            radius=data["radius"],
            cut_ratio=data.get("cut_ratio", 0.5),
            angle=data.get("angle", 0.0),
            resolution=data.get("resolution", 64),
        )


@dataclass(slots=True)
class RoundedRectangle(Shape):
    """
    Symbolic rounded rectangle.

    Parameters:
        center: (cx, cy)
        size: (width, height)
        radius: Corner rounding radius.
        angle: Counter-clockwise rotation angle in degrees.
    """
    center: tuple[Any, Any]
    size: tuple[Any, Any]
    radius: Any
    angle: Any = 0.0

    def __post_init__(self) -> None:
        if len(self.center) != 2:
            raise ValueError("center must have length 2")
        if len(self.size) != 2:
            raise ValueError("size must have length 2")

    def to_parametric(self) -> dict:
        return {
            "type": "RoundedRectangle",
            "center": to_plain_data(self.center),
            "size": to_plain_data(self.size),
            "radius": to_plain_scalar(self.radius),
            "angle": to_plain_scalar(self.angle),
        }
        
    @classmethod
    def from_parametric(cls, data: dict) -> "RoundedRectangle":
        return cls(
            center=tuple(data["center"]),
            size=tuple(data["size"]),
            radius=data["radius"],
            angle=data.get("angle", 0.0),
        )


@dataclass(slots=True)
class RoundedRegularPolygon(Shape):
    """
    Symbolic rounded regular polygon.

    Parameters:
        center: (cx, cy)
        n: Number of sides.
        side_length: Length of each side.
        radius: Corner rounding radius.
        angle: Counter-clockwise rotation angle in degrees.
    """
    center: tuple[Any, Any]
    n: int
    side_length: Any
    radius: Any
    angle: Any = 0.0

    def __post_init__(self) -> None:
        if len(self.center) != 2:
            raise ValueError("center must have length 2")
        if self.n < 3:
            raise ValueError("n must be at least 3")

    def to_parametric(self) -> dict:
        return {
            "type": "RoundedRegularPolygon",
            "center": to_plain_data(self.center),
            "n": int(self.n),
            "side_length": to_plain_scalar(self.side_length),
            "radius": to_plain_scalar(self.radius),
            "angle": to_plain_scalar(self.angle),
        }
        
    @classmethod
    def from_parametric(cls, data: dict) -> "RoundedRegularPolygon":
        return cls(
            center=tuple(data["center"]),
            n=data["n"],
            side_length=data["side_length"],
            radius=data["radius"],
            angle=data.get("angle", 0.0),
        )


@dataclass(slots=True)
class RoundedCross(Shape):
    """
    Symbolic rounded cross.

    Parameters:
        center: (cx, cy)
        size: Total arm length.
        width: Arm width.
        radius: Corner rounding radius.
        angle: Counter-clockwise rotation angle in degrees.
    """
    center: tuple[Any, Any]
    size: Any
    width: Any
    radius: Any
    angle: Any = 0.0

    def __post_init__(self) -> None:
        if len(self.center) != 2:
            raise ValueError("center must have length 2")

    def to_parametric(self) -> dict:
        return {
            "type": "RoundedCross",
            "center": to_plain_data(self.center),
            "size": to_plain_scalar(self.size),
            "width": to_plain_scalar(self.width),
            "radius": to_plain_scalar(self.radius),
            "angle": to_plain_scalar(self.angle),
        }
        
    @classmethod
    def from_parametric(cls, data: dict) -> "RoundedCross":
        return cls(
            center=tuple(data["center"]),
            size=data["size"],
            width=data["width"],
            radius=data["radius"],
            angle=data.get("angle", 0.0),
        )


@dataclass(slots=True)
class RoundedMoon(Shape):
    """
    Symbolic rounded moon shape.

    Parameters:
        center: (cx, cy)
        radius: Radius of the main circle.
        cut_ratio: Controls crescent thickness.
        rounding_radius: Corner rounding radius.
        angle: Counter-clockwise rotation angle in degrees.
        resolution: Optional approximation hint for evaluators.
    """
    center: tuple[Any, Any]
    radius: Any
    cut_ratio: Any = 0.5
    rounding_radius: Any = 0.0
    angle: Any = 0.0
    resolution: int = 64

    def __post_init__(self) -> None:
        if len(self.center) != 2:
            raise ValueError("center must have length 2")
        if self.resolution < 8:
            raise ValueError("resolution must be at least 8")

    def to_parametric(self) -> dict:
        return {
            "type": "RoundedMoon",
            "center": to_plain_data(self.center),
            "radius": to_plain_scalar(self.radius),
            "cut_ratio": to_plain_scalar(self.cut_ratio),
            "rounding_radius": to_plain_scalar(self.rounding_radius),
            "angle": to_plain_scalar(self.angle),
            "resolution": int(self.resolution),
        }
        
    @classmethod
    def from_parametric(cls, data: dict) -> "RoundedMoon":
        return cls(
            center=tuple(data["center"]),
            radius=data["radius"],
            cut_ratio=data.get("cut_ratio", 0.5),
            rounding_radius=data.get("rounding_radius", 0.0),
            angle=data.get("angle", 0.0),
            resolution=data.get("resolution", 64),
        )
        
""" Helper functions """
def _to_local_coords(
    x: torch.Tensor,
    y: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    angle_deg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xr = x - cx
    yr = y - cy

    angle = torch.deg2rad(angle_deg)
    c = torch.cos(angle)
    s = torch.sin(angle)

    x_local =  c * xr + s * yr
    y_local = -s * xr + c * yr
    return x_local, y_local