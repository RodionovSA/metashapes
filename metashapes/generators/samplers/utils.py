# metashapes/generators/samplers/utils.py
# This module defines utility functions for shape samplers

from __future__ import annotations


def get_fixed_param(config, shape_name: str, param_name: str, default=None):
    shape_params = config.fixed_shape_params.get(shape_name, {})
    return shape_params.get(param_name, default)


def get_param_range(config, shape_name: str, param_name: str, default=None):
    shape_params = config.shape_param_ranges.get(shape_name, {})
    return shape_params.get(param_name, default)


def sample_param(rng, config, shape_name: str, param_name: str, default_range=None, default_value=None):
    fixed = get_fixed_param(config, shape_name, param_name, None)
    if fixed is not None:
        return fixed

    param_range = get_param_range(config, shape_name, param_name, default_range)
    if param_range is not None:
        lo, hi = param_range
        return rng.uniform(lo, hi)

    return default_value

def intersect_ranges(*ranges):
    lo = None
    hi = None

    for r in ranges:
        if r is None:
            continue
        a, b = r
        if a > b:
            raise ValueError(f"invalid range: {r}")
        lo = a if lo is None else max(lo, a)
        hi = b if hi is None else min(hi, b)

    if lo is None or hi is None:
        return None

    if lo > hi:
        raise ValueError(f"incompatible ranges: {ranges}")

    return (lo, hi)

def sample_from_effective_range(rng, *, default_range=None, user_range=None, constraint_range=None):
    r = intersect_ranges(default_range, user_range, constraint_range)
    if r is None:
        raise ValueError("no sampling range available")
    return rng.uniform(*r)

def resolve_param(
    rng,
    *,
    fixed_value,
    user_range,
    default_range,
):
    if fixed_value is not None:
        return fixed_value
    return rng.uniform(*intersect_ranges(default_range, user_range))

def resolve_pair_param(
    rng,
    *,
    fixed_value,
    user_range,
    default_x_range,
    default_y_range,
):
    if fixed_value is not None:
        return fixed_value

    if user_range is None:
        xr = default_x_range
        yr = default_y_range
    else:
        if len(user_range) == 2 and not isinstance(user_range[0], (list, tuple)):
            lo, hi = user_range
            xr = intersect_ranges(default_x_range, (lo, hi))
            yr = intersect_ranges(default_y_range, (lo, hi))
        else:
            xr = intersect_ranges(default_x_range, user_range[0])
            yr = intersect_ranges(default_y_range, user_range[1])

    return rng.uniform(*xr), rng.uniform(*yr)

def sample_center_in_bounds(
    rng,
    *,
    fixed_center,
    center_range,
    x_bounds,
    y_bounds,
):
    if fixed_center is not None:
        return fixed_center

    if center_range is None:
        xr = x_bounds
        yr = y_bounds
    else:
        if len(center_range) != 2:
            raise ValueError("center range must have length 2")
        xr = intersect_ranges(x_bounds, center_range[0])
        yr = intersect_ranges(y_bounds, center_range[1])

    return rng.uniform(*xr), rng.uniform(*yr)
