# metashapes/generators/samplers/polygons.py
# This module defines samplers for polygon shapes

from __future__ import annotations

import math

from metashapes.lattice.canvas import Canvas
from metashapes.generators.registry import register_shape_sampler
from metashapes.generators.samplers.base import ShapeSampler
from metashapes.generators.samplers.utils import (
    get_all_fixed_param,
    get_all_param_range,
    intersect_ranges,
    resolve_param,
    sample_center_in_bounds,
)
from metashapes.shape.primitives import RegularPolygon


@register_shape_sampler
class RegularPolygonSampler(ShapeSampler):
    shape_class = RegularPolygon

    def sample(self, rng, canvas: Canvas, config) -> RegularPolygon:
        min_size = config.min_shape_size or 0.1
        min_feature = config.min_feature_size or 0.0

        fixed = get_all_fixed_param(config, self.shape_class)
        ranges = get_all_param_range(config, self.shape_class)

        x0, y0, x1, y1 = canvas.inner_bounds
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
