# metashapes/shape/primitives/polygons.py
# This module defines shape primitives for general polygons

from __future__ import annotations

import math
import torch
import numpy as np

from metashapes.shape.base import Shape
from metashapes.shape.registry import register_shape
from metashapes.shape.utils import _to_local_coords, register

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

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cx, cy = self.center[0], self.center[1]
        s  = self.side_length
        rr = self.corner_radius
        n  = self.n

        pi = torch.as_tensor(np.pi, dtype=x.dtype, device=x.device)
        angle = self.angle

        # original polygon radii
        n_t = torch.as_tensor(float(n), dtype=x.dtype, device=x.device)
        an = pi / n_t
        R = s / (2.0 * torch.sin(an))     # circumradius
        rho = R * torch.cos(an)           # apothem

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

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
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

        if self.base <= 0:
            raise ValueError("Triangle base must be positive")
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("Triangle angles must be positive")
        if self.alpha + self.beta >= 180.0:
            raise ValueError("Triangle alpha + beta must be less than 180°")
        if self.corner_radius < 0:
            raise ValueError("corner_radius must be non-negative")
        if self.corner_radius > 0 and self.corner_radius >= self._inradius():
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

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cx, cy = self.center[0], self.center[1]
        rr = self.corner_radius

        x_local, y_local = _to_local_coords(x, y, cx, cy, self.angle)

        (Ax, Ay), (Bx, By), (Cx, Cy) = self._vertices()

        if rr > 0:
            alpha = self.alpha
            beta = self.beta
            gamma = torch.as_tensor(180.0, dtype=alpha.dtype, device=alpha.device) - alpha - beta

            def _inset(Vx, Vy, Nx, Ny, Px, Py, theta_deg):
                dnx, dny = Nx - Vx, Ny - Vy
                dpx, dpy = Px - Vx, Py - Vy
                dn_len = torch.sqrt(dnx * dnx + dny * dny).clamp(min=1e-8)
                dp_len = torch.sqrt(dpx * dpx + dpy * dpy).clamp(min=1e-8)
                dnx, dny = dnx / dn_len, dny / dn_len
                dpx, dpy = dpx / dp_len, dpy / dp_len
                sin_t = torch.sin(torch.deg2rad(theta_deg)).clamp(min=1e-8)
                return Vx + rr / sin_t * (dnx + dpx), Vy + rr / sin_t * (dny + dpy)

            # Inset all three vertices using original (un-inset) positions
            oAx, oAy, oBx, oBy, oCx, oCy = Ax, Ay, Bx, By, Cx, Cy
            Ax, Ay = _inset(oAx, oAy, oBx, oBy, oCx, oCy, alpha)
            Bx, By = _inset(oBx, oBy, oCx, oCy, oAx, oAy, beta)
            Cx, Cy = _inset(oCx, oCy, oAx, oAy, oBx, oBy, gamma)

        verts = [(Ax, Ay), (Bx, By), (Cx, Cy)]
        min_d2 = None
        inside = torch.ones_like(x_local, dtype=torch.bool)

        for k in range(3):
            vax, vay = verts[k]
            vbx, vby = verts[(k + 1) % 3]

            ex = vbx - vax
            ey = vby - vay
            wx = x_local - vax
            wy = y_local - vay

            ee = ex * ex + ey * ey
            t = torch.clamp((wx * ex + wy * ey) / ee, 0.0, 1.0)

            cpx = vax + t * ex
            cpy = vay + t * ey

            d2 = (x_local - cpx) ** 2 + (y_local - cpy) ** 2
            min_d2 = d2 if min_d2 is None else torch.minimum(min_d2, d2)

            cross = ex * (y_local - vay) - ey * (x_local - vax)
            inside = inside & (cross >= 0)

        d_in = torch.sqrt(torch.clamp(min_d2, min=0.0))
        d_in = torch.where(inside, -d_in, d_in)
        return d_in - rr

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
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

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cx, cy = self.center[0], self.center[1]
        R = self.outer_radius
        r = self.inner_radius
        ocr = self.outer_corner_radius
        icr = self.inner_corner_radius

        pi = torch.as_tensor(math.pi, dtype=x.dtype, device=x.device)
        n_t = torch.as_tensor(float(self.n), dtype=x.dtype, device=x.device)
        an = pi / n_t  # half-sector angle at the origin

        x_local, y_local = _to_local_coords(x, y, cx, cy, self.angle)

        # Fold into canonical half-sector with outer tip A on +x axis,
        # valley vertex B at angle an. Reflect across the bisector so p_y >= 0.
        r_point = torch.sqrt(x_local ** 2 + y_local ** 2).clamp(min=1e-9)
        theta = torch.atan2(x_local, y_local)  # first tip at +y in local frame
        bn = torch.remainder(theta + an, 2.0 * an) - an
        p_x = r_point * torch.cos(bn)
        p_y = r_point * torch.abs(torch.sin(bn))

        Bx = r * torch.cos(an)
        By = r * torch.sin(an)
        L = torch.sqrt(R ** 2 + r ** 2 - 2.0 * R * r * torch.cos(an))

        # ---- Outer tip rounding ----
        denom_outer = (r * torch.sin(an)).clamp(min=1e-9)
        shift_A = ocr * L / denom_outer
        A_eff_x = R - shift_A
        A_eff_y = torch.zeros_like(shift_A)

        # B_eff: B shifted outward along the bisector through B by ocr/sin(beta).
        sin_beta = (R * torch.sin(an) / L).clamp(min=1e-7, max=1.0 - 1e-7)
        bis_x_B = torch.cos(an)
        bis_y_B = torch.sin(an)
        OB = torch.sqrt(Bx ** 2 + By ** 2)
        OB_eff = OB - ocr / sin_beta
        B_eff_x = OB_eff * bis_x_B
        B_eff_y = OB_eff * bis_y_B

        # Signed distance to segment A_eff -> B_eff in the half-sector.
        ex = B_eff_x - A_eff_x
        ey = B_eff_y - A_eff_y
        wx = p_x - A_eff_x
        wy = p_y - A_eff_y
        ee = (ex * ex + ey * ey).clamp(min=1e-18)
        t = torch.clamp((wx * ex + wy * ey) / ee, 0.0, 1.0)
        nearest_x = A_eff_x + t * ex
        nearest_y = A_eff_y + t * ey
        dist_seg = torch.sqrt((p_x - nearest_x) ** 2 + (p_y - nearest_y) ** 2)
        cross_z = ex * wy - ey * wx
        d_sector = torch.where(cross_z >= 0, -dist_seg, dist_seg) - ocr

        # ---- Inner valley rounding ----
        edge_dx = Bx - A_eff_x   # NOTE: edge direction from A_eff to B
        edge_dy = By - A_eff_y   # (matches ex, ey from d_sector computation)
        edge_len = torch.sqrt(edge_dx ** 2 + edge_dy ** 2).clamp(min=1e-9)
        edge_ux = edge_dx / edge_len
        edge_uy = edge_dy / edge_len

        sin_beta = (R * torch.sin(an) / L).clamp(min=1e-7, max=1.0 - 1e-7)
        cos_beta = torch.sqrt(1.0 - sin_beta ** 2).clamp(min=1e-9)
        tan_beta = sin_beta / cos_beta

        # Clamp icr so tangent points stay on the edges.
        icr_max = edge_len * tan_beta
        icr_eff = torch.minimum(icr, icr_max)

        # Disk center on outward bisector, past B.
        bis_x = torch.cos(an)
        bis_y = torch.sin(an)
        OB = torch.sqrt(Bx ** 2 + By ** 2)
        OC = OB + icr_eff / sin_beta
        C_x = OC * bis_x
        C_y = OC * bis_y

        d_disk = torch.sqrt((p_x - C_x) ** 2 + (p_y - C_y) ** 2) - icr_eff

        # Half-kite constraint 1: valley-exterior side of edge A_eff -> B.
        # cross_z > 0 is polygon interior; we want cross_z < 0 (valley side) to
        # be inside the half-kite, so the signed distance is -cross_z / edge_len.
        wx = p_x - A_eff_x
        wy = p_y - A_eff_y
        cross_edge = edge_dx * wy - edge_dy * wx
        d_half1 = cross_edge / edge_len

        # Half-kite constraint 2: B side of the perpendicular through T1.
        # T1 is on the edge at distance (edge_len - icr_eff/tan(beta)) from A_eff.
        # Projection of (p - A_eff) onto edge direction: proj = w . edge_unit.
        # B side is where proj > T1_proj, i.e. (T1_proj - proj) < 0 is inside.
        proj = wx * edge_ux + wy * edge_uy
        T1_proj = edge_len - icr_eff / tan_beta
        d_half2 = T1_proj - proj

        # Half-kite (in folded coords, the part of the kite with p_y >= 0).
        d_half_kite = torch.maximum(d_half1, d_half2)

        # Patch: half-kite minus disk.
        d_patch = torch.maximum(d_half_kite, -d_disk)

        # Union with sharp star.
        return torch.minimum(d_sector, d_patch)

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
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
        return 2.0 * self.inner_radius.item()
