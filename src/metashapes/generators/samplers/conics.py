# metashapes/generators/samplers/conics.py
# This module defines samplers for conic shapes

from __future__ import annotations

import numpy as np

from src.metashapes.lattice.basis import Lattice
from src.metashapes.generators.registry import register_shape_sampler
from src.metashapes.generators.samplers.base import ShapeSampler
from src.metashapes.generators.samplers.utils import (
    get_all_fixed_param,
    get_all_param_range,
    intersect_ranges,
    lattice_inner_bounds,
    resolve_param,
    resolve_pair_param,
    sample_center_in_bounds,
)
from src.metashapes.shape.primitives import Ellipse, Egg, Stadium


@register_shape_sampler
class EllipseSampler(ShapeSampler):
    shape_class = Ellipse

    def sample(self, rng, lattice: Lattice, config) -> Ellipse:
        min_size = config.min_shape_size or 0.1
        min_feature = config.min_feature_size or 0.0

        fixed = get_all_fixed_param(config, self.shape_class)
        ranges = get_all_param_range(config, self.shape_class)

        x0, y0, x1, y1 = lattice_inner_bounds(lattice)
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


@register_shape_sampler
class EggSampler(ShapeSampler):
    shape_class = Egg

    def sample(self, rng, lattice: Lattice, config) -> Egg:
        min_size = config.min_shape_size or 0.1
        min_feature = config.min_feature_size or 0.0

        fixed = get_all_fixed_param(config, self.shape_class)
        ranges = get_all_param_range(config, self.shape_class)

        x0, y0, x1, y1 = lattice_inner_bounds(lattice)
        iw, ih = x1 - x0, y1 - y0

        for _ in range(config.max_tries_per_shape):
            width = resolve_param(
                rng,
                fixed_value=fixed['width'],
                user_range=ranges['width'],
                default_range=intersect_ranges((min_size, iw), (min_feature, iw)),
            )

            height = resolve_param(
                rng,
                fixed_value=fixed['height'],
                user_range=ranges['height'],
                default_range=intersect_ranges((min_size, ih), (min_feature, ih)),
            )

            skew = resolve_param(
                rng,
                fixed_value=fixed['skew'],
                user_range=ranges['skew'],
                default_range=(-0.5, 0.5),
            )

            angle = resolve_param(
                rng,
                fixed_value=fixed['angle'],
                user_range=ranges['angle'],
                default_range=(0.0, 360.0),
            )

            a = width / 2.0
            b_top = height / 2.0 * (1.0 + skew)
            b_bot = height / 2.0 * (1.0 - skew)
            theta = np.deg2rad(angle)
            c_a, s_a = np.cos(theta), np.sin(theta)

            b_max = max(b_top, b_bot)
            corners = [(a, b_top), (-a, b_top), (a, -b_bot), (-a, -b_bot)]
            rotated_xs = [p[0] * c_a - p[1] * s_a for p in corners]
            rotated_ys = [p[0] * s_a + p[1] * c_a for p in corners]
            dx = max(abs(rx) for rx in rotated_xs)
            dy = max(abs(ry) for ry in rotated_ys)

            if dx > iw / 2 or dy > ih / 2:
                continue

            cx, cy = sample_center_in_bounds(
                rng,
                fixed_center=fixed['center'],
                center_range=ranges['center'],
                x_bounds=(x0 + dx, x1 - dx),
                y_bounds=(y0 + dy, y1 - dy),
            )

            return Egg(center=(cx, cy), width=width, height=height, skew=skew, angle=angle)

        raise RuntimeError("egg_too_large_for_canvas")


@register_shape_sampler
class StadiumSampler(ShapeSampler):
    shape_class = Stadium

    def sample(self, rng, lattice: Lattice, config) -> Stadium:
        min_size = config.min_shape_size or 0.1
        min_feature = config.min_feature_size or 0.0

        fixed = get_all_fixed_param(config, self.shape_class)
        ranges = get_all_param_range(config, self.shape_class)

        x0, y0, x1, y1 = lattice_inner_bounds(lattice)
        iw, ih = x1 - x0, y1 - y0

        for _ in range(config.max_tries_per_shape):
            length = resolve_param(
                rng,
                fixed_value=fixed['length'],
                user_range=ranges['length'],
                default_range=intersect_ranges((min_size, iw), (min_feature, iw)),
            )

            width_max = min(length, ih)
            width = resolve_param(
                rng,
                fixed_value=fixed['width'],
                user_range=ranges['width'],
                default_range=intersect_ranges((min_feature, width_max), (min_size, width_max)),
            )

            if width > length:
                continue

            angle = resolve_param(
                rng,
                fixed_value=fixed['angle'],
                user_range=ranges['angle'],
                default_range=(0.0, 360.0),
            )

            half_len = length / 2.0
            radius = width / 2.0
            theta = np.deg2rad(angle)
            c_a, s_a = np.cos(theta), np.sin(theta)

            hw = abs(half_len * c_a) + abs(radius * s_a)
            hh = abs(half_len * s_a) + abs(radius * c_a)

            if hw > iw / 2 or hh > ih / 2:
                continue

            cx, cy = sample_center_in_bounds(
                rng,
                fixed_center=fixed['center'],
                center_range=ranges['center'],
                x_bounds=(x0 + hw, x1 - hw),
                y_bounds=(y0 + hh, y1 - hh),
            )

            return Stadium(center=(cx, cy), length=length, width=width, angle=angle)

        raise RuntimeError("stadium_too_large_for_canvas")
