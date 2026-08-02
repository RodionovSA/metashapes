# metashapes/shape/primitives/polygons.py
# This module defines shape primitives for general polygons

from __future__ import annotations

import math
import torch
import numpy as np

from metashapes.shape.base import Shape
from metashapes.shape.registry import register_shape
from metashapes.shape.utils import _to_local_coords, register
from sdflib.polygons import RegularPolygonSDF, TriangleSDF, StarSDF

__all__ = [
    "RegularPolygon",
    "Triangle",
    "Star",
]

@register_shape("RegularPolygon")
class RegularPolygon(Shape):
    """
    Symbolic regular polygon.

    Parameters:
        center: (cx, cy)
        n: Number of sides.
        side_length: Length of each side.
        angle: Counter-clockwise rotation angle in degrees.
        corner_radius: rounding radius for corners.
    """
    def __init__(self,
                 center: torch.Tensor,
                 n: int,
                 side_length: torch.Tensor,
                 angle: torch.Tensor = 0.0,
                 corner_radius: torch.Tensor = 0.0):
        super().__init__()
        if n < 3:
            raise ValueError("n must be at least 3")
        self.n = n
        register(self, "center", center)
        register(self, "side_length", side_length)
        register(self, "angle", angle)
        register(self, "corner_radius", corner_radius)

        if torch.any(self.side_length <= 0):
            raise ValueError("side_length must be positive")
        if torch.any(self.corner_radius < 0):
            raise ValueError("corner_radius must be non-negative")

        an = torch.tensor(np.pi / n)
        rho = self.side_length / (2.0 * torch.tan(an))
        if torch.any(self.corner_radius >= rho):
            raise ValueError("corner_radius is too large")

    @torch.no_grad()
    def _project(self) -> None:
        """Snap side_length/corner_radius back into their valid ranges."""
        self.side_length.clamp_(min=self._MIN_SIZE)
        rho = self.side_length / (2.0 * math.tan(math.pi / self.n))  # apothem
        self.corner_radius.clamp_(min=0.0, max=rho * self._MAX_RADIUS_FRACTION)

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._project()
        cx, cy = self.center[0], self.center[1]
        R = self.side_length / (2.0 * math.sin(math.pi / self.n))  # circumradius

        x_local, y_local = _to_local_coords(x, y, cx, cy, self.angle)
        return RegularPolygonSDF(self.n, R, self.corner_radius)(x_local, y_local)

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        self._project()
        cx, cy = self.center.detach().tolist()
        s = self.side_length.detach().item()
        angle = self.angle.detach().item()
        n = self.n

        an = np.pi / n
        R = s / (2.0 * np.sin(an))
        phi0 = np.pi / 2.0 + np.radians(angle)
        angles = [2.0 * np.pi * k / n + phi0 for k in range(n)]
        xs = [cx + R * np.cos(a) for a in angles]
        ys = [cy + R * np.sin(a) for a in angles]
        return (float(min(xs)), float(min(ys))), (float(max(xs)), float(max(ys)))

    def to_parametric(self) -> dict:
        d = super().to_parametric()
        d["n"] = self.n
        return d

    @property
    def min_feature_size(self) -> float:
        self._project()
        pi = np.pi
        R = self.side_length.detach().item() / (2.0 * np.sin(pi / self.n))
        a = R * np.cos(pi / self.n)

        if self.n % 2 == 0:
            return 2.0 * a

        return a * (1.0 + np.cos(pi / self.n))


@register_shape("Triangle")
class Triangle(Shape):
    """
    General triangle defined by two base angles and the base length (ASA).

    Parameters:
        center: (cx, cy) — centroid of the triangle
        base: length of the bottom side (between alpha and beta vertices)
        alpha: interior angle at the left base vertex in degrees
        beta: interior angle at the right base vertex in degrees
        angle: counter-clockwise rotation in degrees
        corner_radius: optional corner smoothing; must be < inradius
    """
    # Floor kept alpha/beta land on during _project(): small enough to be
    # geometrically negligible, large enough that 180 - _MIN_ANGLE_DEG - 2 *
    # _MIN_ANGLE_DEG margin used by the sum cap stays comfortably positive.
    _MIN_ANGLE_DEG = 1e-3

    def __init__(self,
                 center: torch.Tensor,
                 base: torch.Tensor,
                 alpha: torch.Tensor,
                 beta: torch.Tensor,
                 angle: torch.Tensor = 0.0,
                 corner_radius: torch.Tensor = 0.0):
        super().__init__()
        register(self, "center", center)
        register(self, "base", base)
        register(self, "alpha", alpha)
        register(self, "beta", beta)
        register(self, "angle", angle)
        register(self, "corner_radius", corner_radius)

        if torch.any(self.base <= 0):
            raise ValueError("Triangle base must be positive")
        if torch.any(self.alpha <= 0) or torch.any(self.beta <= 0):
            raise ValueError("Triangle angles must be positive")
        if torch.any(self.alpha + self.beta >= 180.0):
            raise ValueError("Triangle alpha + beta must be less than 180°")
        if torch.any(self.corner_radius < 0):
            raise ValueError("corner_radius must be non-negative")
        if torch.any((self.corner_radius > 0) & (self.corner_radius >= self._inradius())):
            raise ValueError("corner_radius must be less than the triangle inradius")

    def _inradius(self) -> torch.Tensor:
        a_rad = torch.deg2rad(self.alpha)
        b_rad = torch.deg2rad(self.beta)
        sin_ab = torch.sin(a_rad + b_rad)
        return self.base * torch.sin(a_rad) * torch.sin(b_rad) / (
            sin_ab + torch.sin(a_rad) + torch.sin(b_rad)
        )

    def _vertices(self):
        """(A, B, C) as (x, y) tensor pairs, CCW, centroid at origin."""
        a_rad = torch.deg2rad(self.alpha)
        b_rad = torch.deg2rad(self.beta)
        sin_ab = torch.sin(a_rad + b_rad)

        # Apex in base-midpoint frame: base goes from -base/2 to +base/2
        cx_apex = self.base * 0.5 - self.base * torch.sin(a_rad) * torch.cos(b_rad) / sin_ab
        cy_apex = self.base * torch.sin(a_rad) * torch.sin(b_rad) / sin_ab

        # Centroid offset: (cx_apex/3, cy_apex/3)
        gcx = cx_apex / 3.0
        gcy = cy_apex / 3.0

        Ax = -self.base * 0.5 - gcx
        Ay = -gcy
        Bx =  self.base * 0.5 - gcx
        By = -gcy
        Cx = cx_apex - gcx
        Cy = cy_apex - gcy
        return (Ax, Ay), (Bx, By), (Cx, Cy)

    @torch.no_grad()
    def _project(self) -> None:
        """Snap base/alpha/beta/corner_radius back into their valid ranges."""
        self.base.clamp_(min=self._MIN_SIZE)
        self.alpha.clamp_(min=self._MIN_ANGLE_DEG)
        self.beta.clamp_(min=self._MIN_ANGLE_DEG)
        # Scale alpha/beta down together (preserving their ratio) if their
        # sum leaves no room for gamma -- clamp(max=1.0) keeps this a no-op
        # whenever the sum is already safely under 180.
        cap = 180.0 - self._MIN_ANGLE_DEG
        scale = (cap / (self.alpha + self.beta)).clamp(max=1.0)
        self.alpha.mul_(scale)
        self.beta.mul_(scale)
        self.corner_radius.clamp_(min=0.0, max=self._inradius() * self._MAX_RADIUS_FRACTION)

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._project()
        cx, cy = self.center[0], self.center[1]
        rr = self.corner_radius

        x_local, y_local = _to_local_coords(x, y, cx, cy, self.angle)
        A, B, C = self._vertices()
        return TriangleSDF(A, B, C, rr)(x_local, y_local)

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        self._project()
        cx, cy = self.center.detach().tolist()
        theta = math.radians(self.angle.detach().item())
        c_t, s_t = math.cos(theta), math.sin(theta)

        (Ax, Ay), (Bx, By), (Cx, Cy) = self._vertices()
        local_pts = [
            (Ax.item(), Ay.item()),
            (Bx.item(), By.item()),
            (Cx.item(), Cy.item()),
        ]

        xs = [cx + p[0] * c_t - p[1] * s_t for p in local_pts]
        ys = [cy + p[0] * s_t + p[1] * c_t for p in local_pts]
        return (min(xs), min(ys)), (max(xs), max(ys))

    @property
    def min_feature_size(self) -> float:
        self._project()
        return 2.0 * self._inradius().item()


@register_shape("Star")
class Star(Shape):
    """
    Regular n-pointed star.

    Parameters:
        center: (cx, cy)
        n: number of points (≥ 3)
        outer_radius: distance from center to tips
        inner_radius: distance from center to valleys; must be in (0, outer_radius)
        angle: counter-clockwise rotation in degrees (default 0; first tip points up)
        outer_corner_radius: rounds convex tips; must be < outer_radius − inner_radius
        inner_corner_radius: rounds concave valleys; must be < inner_radius * sin(π/n)
    """
    def __init__(self,
                 center,
                 n: int,
                 outer_radius,
                 inner_radius,
                 angle=0.0,
                 outer_corner_radius=0.0,
                 inner_corner_radius=0.0):
        super().__init__()
        if n < 3:
            raise ValueError("n must be at least 3")
        self.n = n
        register(self, "center", center)
        register(self, "outer_radius", outer_radius)
        register(self, "inner_radius", inner_radius)
        register(self, "angle", angle)
        register(self, "outer_corner_radius", outer_corner_radius)
        register(self, "inner_corner_radius", inner_corner_radius)

        R = self.outer_radius.item()
        r = self.inner_radius.item()
        ocr = self.outer_corner_radius.item()
        icr = self.inner_corner_radius.item()
        an = math.pi / n

        if R <= 0:
            raise ValueError("outer_radius must be positive")
        if r <= 0 or r >= R:
            raise ValueError("inner_radius must be in (0, outer_radius)")
        if ocr < 0:
            raise ValueError("outer_corner_radius must be non-negative")
        if icr < 0:
            raise ValueError("inner_corner_radius must be non-negative")

        # Geometric quantities for corner-radius bounds.
        L = math.sqrt(R * R + r * r - 2.0 * R * r * math.cos(an))
        sin_alpha = (r * math.sin(an)) / L         # half-angle at outer tip
        sin_beta  = (R * math.sin(an)) / L         # half-angle at inner valley
        tan_alpha = sin_alpha / math.sqrt(max(0.0, 1.0 - sin_alpha * sin_alpha))
        tan_beta  = sin_beta  / math.sqrt(max(0.0, 1.0 - sin_beta  * sin_beta))
        ocr_max = L * tan_alpha
        icr_max = L * tan_beta

        if ocr >= ocr_max:
            raise ValueError(
                f"outer_corner_radius must be < {ocr_max:.6g} for R={R}, r={r}, n={n}"
            )
        if icr >= icr_max:
            raise ValueError(
                f"inner_corner_radius must be < {icr_max:.6g} for R={R}, r={r}, n={n}"
            )
            
        if ocr / tan_alpha + icr / tan_beta >= L:
            raise ValueError(
                "outer_corner_radius and inner_corner_radius together exceed edge length; "
                "their tangent points would overlap"
            )

    @torch.no_grad()
    def _project(self) -> None:
        """Snap outer_radius/inner_radius/outer_corner_radius/
        inner_corner_radius back into their valid ranges, using the same
        geometry as __init__'s bound checks (as tensor ops, so it tracks
        the current parameter values rather than the ones at construction)."""
        self.outer_radius.clamp_(min=self._MIN_SIZE)
        self.inner_radius.clamp_(min=self._MIN_SIZE, max=self.outer_radius * self._MAX_RADIUS_FRACTION)

        an = math.pi / self.n
        R, r = self.outer_radius, self.inner_radius
        L = torch.sqrt((R * R + r * r - 2.0 * R * r * math.cos(an)).clamp_min(torch.finfo(R.dtype).tiny))
        sin_alpha = (r * math.sin(an) / L).clamp(min=1e-7, max=1.0 - 1e-7)
        sin_beta = (R * math.sin(an) / L).clamp(min=1e-7, max=1.0 - 1e-7)
        tan_alpha = sin_alpha / torch.sqrt((1.0 - sin_alpha * sin_alpha).clamp_min(torch.finfo(R.dtype).tiny))
        tan_beta = sin_beta / torch.sqrt((1.0 - sin_beta * sin_beta).clamp_min(torch.finfo(R.dtype).tiny))

        self.outer_corner_radius.clamp_(min=0.0, max=L * tan_alpha * self._MAX_RADIUS_FRACTION)
        # icr_max already accounts for the joint ocr/icr edge-length bound
        # (__init__'s third check) once ocr is clamped above.
        icr_max = ((L - self.outer_corner_radius / tan_alpha) * tan_beta).clamp_min(0.0)
        self.inner_corner_radius.clamp_(min=0.0, max=icr_max * self._MAX_RADIUS_FRACTION)

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._project()
        cx, cy = self.center[0], self.center[1]
        x_local, y_local = _to_local_coords(x, y, cx, cy, self.angle)
        return StarSDF(
            self.n, self.outer_radius, self.inner_radius,
            self.outer_corner_radius, self.inner_corner_radius,
        )(x_local, y_local)

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        self._project()
        cx, cy = self.center.detach().tolist()
        R = self.outer_radius.item()
        r = self.inner_radius.item()
        angle_deg = self.angle.item()
        n = self.n
        an = math.pi / n

        pts = []
        for k in range(n):
            # outer tip at angle (π/2 + angle + 2πk/n)
            tip_ang = math.radians(angle_deg) + math.pi / 2 + 2 * math.pi * k / n
            pts.append((R * math.cos(tip_ang), R * math.sin(tip_ang)))
            # inner valley offset by π/n
            val_ang = tip_ang + an
            pts.append((r * math.cos(val_ang), r * math.sin(val_ang)))

        xs = [cx + p[0] for p in pts]
        ys = [cy + p[1] for p in pts]
        return (min(xs), min(ys)), (max(xs), max(ys))

    def to_parametric(self) -> dict:
        d = super().to_parametric()
        d["n"] = self.n
        return d

    @property
    def min_feature_size(self) -> float:
        """Narrowest place the star gets: the width of a spike's tip."""
        self._project()
        return 2.0 * self.outer_corner_radius.detach().item()
