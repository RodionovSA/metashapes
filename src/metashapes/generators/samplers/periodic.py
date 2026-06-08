# metashapes/generators/samplers/periodic.py
# Samplers for periodic primitive shapes.

from __future__ import annotations

from src.metashapes.lattice.basis import Lattice
from src.metashapes.generators.registry import register_shape_sampler
from src.metashapes.generators.samplers.base import ShapeSampler
from src.metashapes.generators.samplers.utils import (
    get_all_fixed_param,
    get_all_param_range,
    intersect_ranges,
    lattice_inner_bounds,
    resolve_param,
)
from src.metashapes.shape.primitives import Stripe


@register_shape_sampler
class StripeSampler(ShapeSampler):
    shape_class = Stripe

    def sample(self, rng, lattice: Lattice, config) -> Stripe:
        min_size = config.min_shape_size or 0.1
        min_feature = config.min_feature_size or 0.0

        fixed = get_all_fixed_param(config, self.shape_class)
        ranges = get_all_param_range(config, self.shape_class)

        axis = fixed['axis'] if fixed['axis'] is not None else rng.choice(["x", "y"])

        x0, y0, x1, y1 = lattice_inner_bounds(lattice)
        perp_lo, perp_hi = (y0, y1) if axis == "x" else (x0, x1)
        perp_len = perp_hi - perp_lo

        width = resolve_param(
            rng,
            fixed_value=fixed['width'],
            user_range=ranges['width'],
            default_range=intersect_ranges((min_size, perp_len), (min_feature, perp_len)),
        )

        offset = resolve_param(
            rng,
            fixed_value=fixed['offset'],
            user_range=ranges['offset'],
            default_range=(perp_lo + width / 2.0, perp_hi - width / 2.0),
        )

        return Stripe(offset=offset, width=width, axis=axis)
