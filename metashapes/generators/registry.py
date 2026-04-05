# metashapes/generators/registry.py
# This module defines registry of shapes samplers

from __future__ import annotations

SHAPE_SAMPLER_REGISTRY: dict[str, object] = {}


def register_shape_sampler(shape_name: str):
    if not shape_name:
        raise ValueError("shape_name must be non-empty")

    def decorator(cls):
        if shape_name in SHAPE_SAMPLER_REGISTRY:
            raise ValueError(f"Shape sampler '{shape_name}' is already registered")

        sampler = cls()
        SHAPE_SAMPLER_REGISTRY[shape_name] = sampler
        return cls

    return decorator