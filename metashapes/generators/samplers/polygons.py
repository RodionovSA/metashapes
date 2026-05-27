# metashapes/generators/samplers/polygons.py
# This module defines samplers for polygon shapes

from __future__ import annotations

import math

from metashapes.lattice.basis import Lattice
from metashapes.generators.registry import register_shape_sampler
from metashapes.generators.samplers.base import ShapeSampler
from metashapes.generators.samplers.utils import (
    get_all_fixed_param,
    get_all_param_range,
    intersect_ranges,
    lattice_inner_bounds,
    resolve_param,
    sample_center_in_bounds,
)
from metashapes.shape.primitives import RegularPolygon, Triangle, Star


@register_shape_sampler
class RegularPolygonSampler(ShapeSampler):
    shape_class = RegularPolygon

    def sample(self, rng, lattice: Lattice, config) -> RegularPolygon:
        min_size = config.min_shape_size or 0.1
        min_feature = config.min_feature_size or 0.0

        fixed = get_all_fixed_param(config, self.shape_class)
        ranges = get_all_param_range(config, self.shape_class)

        x0, y0, x1, y1 = lattice_inner_bounds(lattice)
        iw, ih = x1 - x0, y1 - y0
        max_r = min(iw, ih) / 2.0

        for _ in range(config.max_tries_per_shape):
            # sample n (rng is Python's random.Random — randint is inclusive on both ends)
            if fixed['n'] is not None:
                n = int(fixed['n'])
            elif ranges['n'] is not None:
                n = rng.randint(int(ranges['n'][0]), int(ranges['n'][1]))
            else:
                n = rng.randint(3, 8)  # triangles … octagons

            sin_an = math.sin(math.pi / n)
            cos_an = math.cos(math.pi / n)

            s_max = 2.0 * sin_an * max_r
            s_min_feature = (2.0 * sin_an * min_feature / cos_an) if cos_an > 0 else 0.0
            s_min = max(min_size, s_min_feature, 1e-6)

            if s_min > s_max:
                continue

            side_length = resolve_param(
                rng,
                fixed_value=fixed['side_length'],
                user_range=ranges['side_length'],
                default_range=(s_min, s_max),
            )

            angle = resolve_param(
                rng,
                fixed_value=fixed['angle'],
                user_range=ranges['angle'],
                default_range=(0.0, 360.0),
            )

            R = side_length / (2.0 * sin_an)
            apothem = R * cos_an

            corner_radius = resolve_param(
                rng,
                fixed_value=fixed['corner_radius'],
                user_range=ranges['corner_radius'],
                default_range=(0.0, apothem),
            )

            if R > iw / 2 or R > ih / 2:
                continue

            cx, cy = sample_center_in_bounds(
                rng,
                fixed_center=fixed['center'],
                center_range=ranges['center'],
                x_bounds=(x0 + R, x1 - R),
                y_bounds=(y0 + R, y1 - R),
            )

            return RegularPolygon(
                center=(cx, cy),
                n=n,
                side_length=side_length,
                angle=angle,
                corner_radius=corner_radius,
            )

        raise RuntimeError("regular_polygon_too_large_for_canvas")


@register_shape_sampler
class TriangleSampler(ShapeSampler):
    shape_class = Triangle

    def sample(self, rng, lattice: Lattice, config) -> Triangle:
        min_size = config.min_shape_size or 0.1
        min_feature = config.min_feature_size or 0.0

        fixed = get_all_fixed_param(config, self.shape_class)
        ranges = get_all_param_range(config, self.shape_class)

        x0, y0, x1, y1 = lattice_inner_bounds(lattice)
        iw, ih = x1 - x0, y1 - y0

        for _ in range(config.max_tries_per_shape):
            base = resolve_param(
                rng,
                fixed_value=fixed['base'],
                user_range=ranges['base'],
                default_range=intersect_ranges((min_size, min(iw, ih)), (min_feature, min(iw, ih))),
            )

            alpha = resolve_param(
                rng,
                fixed_value=fixed['alpha'],
                user_range=ranges['alpha'],
                default_range=(20.0, 140.0),
            )

            beta_max = min(140.0, 160.0 - alpha)
            if beta_max <= 20.0:
                continue

            beta = resolve_param(
                rng,
                fixed_value=fixed['beta'],
                user_range=ranges['beta'],
                default_range=(20.0, beta_max),
            )

            if alpha + beta >= 180.0:
                continue

            angle = resolve_param(
                rng,
                fixed_value=fixed['angle'],
                user_range=ranges['angle'],
                default_range=(0.0, 360.0),
            )

            a_rad = math.radians(alpha)
            b_rad = math.radians(beta)
            sin_ab = math.sin(a_rad + b_rad)
            inradius = base * math.sin(a_rad) * math.sin(b_rad) / (
                sin_ab + math.sin(a_rad) + math.sin(b_rad)
            )

            corner_radius = resolve_param(
                rng,
                fixed_value=fixed['corner_radius'],
                user_range=ranges['corner_radius'],
                default_range=(0.0, inradius * 0.8),
            )
            corner_radius = min(corner_radius, inradius * 0.99)

            # Outer vertices in local centroid frame
            cx_apex = base / 2.0 - base * math.sin(a_rad) * math.cos(b_rad) / sin_ab
            cy_apex = base * math.sin(a_rad) * math.sin(b_rad) / sin_ab
            gcx = cx_apex / 3.0
            gcy = cy_apex / 3.0

            local_pts = [
                (-base / 2.0 - gcx, -gcy),
                (base / 2.0 - gcx, -gcy),
                (cx_apex - gcx, cy_apex - gcy),
            ]

            theta = math.radians(angle)
            c_t, s_t = math.cos(theta), math.sin(theta)
            rotated_xs = [p[0] * c_t - p[1] * s_t for p in local_pts]
            rotated_ys = [p[0] * s_t + p[1] * c_t for p in local_pts]
            dx = max(abs(rx) for rx in rotated_xs)
            dy = max(abs(ry) for ry in rotated_ys)

            if dx > iw / 2 or dy > ih / 2:
                continue

            cxc, cyc = sample_center_in_bounds(
                rng,
                fixed_center=fixed['center'],
                center_range=ranges['center'],
                x_bounds=(x0 + dx, x1 - dx),
                y_bounds=(y0 + dy, y1 - dy),
            )

            return Triangle(
                center=(cxc, cyc),
                base=base,
                alpha=alpha,
                beta=beta,
                angle=angle,
                corner_radius=corner_radius,
            )

        raise RuntimeError("triangle_too_large_for_canvas")


@register_shape_sampler
class StarSampler(ShapeSampler):
    shape_class = Star

    def sample(self, rng, lattice: Lattice, config) -> Star:
        min_size = config.min_shape_size or 0.1
        min_feature = config.min_feature_size or 0.0

        fixed = get_all_fixed_param(config, self.shape_class)
        ranges = get_all_param_range(config, self.shape_class)

        x0, y0, x1, y1 = lattice_inner_bounds(lattice)
        iw, ih = x1 - x0, y1 - y0
        max_R = min(iw, ih) / 2.0

        for _ in range(config.max_tries_per_shape):
            if fixed['n'] is not None:
                n = int(fixed['n'])
            elif ranges['n'] is not None:
                n = rng.randint(int(ranges['n'][0]), int(ranges['n'][1]))
            else:
                n = rng.randint(3, 7)

            an = math.pi / n

            outer_radius = resolve_param(
                rng,
                fixed_value=fixed['outer_radius'],
                user_range=ranges['outer_radius'],
                default_range=(min_size / 2.0, max_R),
            )

            if outer_radius > max_R:
                continue

            r_min = max(min_feature, outer_radius * 0.1)
            r_max = outer_radius * 0.85
            if r_min >= r_max:
                continue

            inner_radius = resolve_param(
                rng,
                fixed_value=fixed['inner_radius'],
                user_range=ranges['inner_radius'],
                default_range=(r_min, r_max),
            )

            if inner_radius <= 0 or inner_radius >= outer_radius:
                continue

            angle = resolve_param(
                rng,
                fixed_value=fixed['angle'],
                user_range=ranges['angle'],
                default_range=(0.0, 360.0 / n),
            )

            ocr_max = (outer_radius - inner_radius) * 0.4
            outer_corner_radius = resolve_param(
                rng,
                fixed_value=fixed['outer_corner_radius'],
                user_range=ranges['outer_corner_radius'],
                default_range=(0.0, ocr_max),
            )
            outer_corner_radius = min(outer_corner_radius, outer_radius - inner_radius - 1e-6)

            icr_max = inner_radius * math.sin(an) * 0.8
            inner_corner_radius = resolve_param(
                rng,
                fixed_value=fixed['inner_corner_radius'],
                user_range=ranges['inner_corner_radius'],
                default_range=(0.0, icr_max),
            )
            inner_corner_radius = min(inner_corner_radius, inner_radius * math.sin(an) - 1e-6)

            cx, cy = sample_center_in_bounds(
                rng,
                fixed_center=fixed['center'],
                center_range=ranges['center'],
                x_bounds=(x0 + outer_radius, x1 - outer_radius),
                y_bounds=(y0 + outer_radius, y1 - outer_radius),
            )

            return Star(
                center=(cx, cy),
                n=n,
                outer_radius=outer_radius,
                inner_radius=inner_radius,
                angle=angle,
                outer_corner_radius=outer_corner_radius,
                inner_corner_radius=inner_corner_radius,
            )

        raise RuntimeError("star_too_large_for_canvas")
