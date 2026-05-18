# metashapes/generators/samplers/conics.py
# This module defines samplers for conic shapes

from __future__ import annotations

import numpy as np

from metashapes.lattice.canvas import Canvas
from metashapes.generators.registry import register_shape_sampler
from metashapes.generators.samplers.base import ShapeSampler
from metashapes.generators.samplers.utils import (
    get_all_fixed_param,
    get_all_param_range,
    intersect_ranges,
    resolve_param,
    resolve_pair_param,
    sample_center_in_bounds,
)
from metashapes.shape.primitives import Ellipse


@register_shape_sampler
class EllipseSampler(ShapeSampler):
    shape_class = Ellipse

    def sample(self, rng, canvas: Canvas, config) -> Ellipse:
        min_size = config.min_shape_size or 0.1
        min_feature = config.min_feature_size or 0.0

        fixed = get_all_fixed_param(config, self.shape_class)
        ranges = get_all_param_range(config, self.shape_class)

        x0, y0, x1, y1 = canvas.inner_bounds
        iw, ih = x1 - x0, y1 - y0

        for _ in range(config.max_tries_per_shape):
            ax, ay = resolve_pair_param(
                rng,
                fixed_value=fixed['axes'],
                user_range=ranges['axes'],
                default_x_range=intersect_ranges((min_size, iw), (min_feature, iw)),
                default_y_range=intersect_ranges((min_size, ih), (min_feature, ih)),
            )

            angle = resolve_param(
                rng,
                fixed_value=fixed['angle'],
                user_range=ranges['angle'],
                default_range=(0.0, 360.0),
            )

            hx = 0.5 * ax
            hy = 0.5 * ay
            theta = np.deg2rad(angle)

            dx = abs(hx * np.cos(theta)) + abs(hy * np.sin(theta))
            dy = abs(hx * np.sin(theta)) + abs(hy * np.cos(theta))

            if dx > iw / 2 or dy > ih / 2:
                continue

            cx, cy = sample_center_in_bounds(
                rng,
                fixed_center=fixed['center'],
                center_range=ranges['center'],
                x_bounds=(x0 + dx, x1 - dx),
                y_bounds=(y0 + dy, y1 - dy),
            )

            return Ellipse(center=(cx, cy), axes=(ax, ay), angle=angle)

        raise RuntimeError("ellipse_too_large_for_canvas")
