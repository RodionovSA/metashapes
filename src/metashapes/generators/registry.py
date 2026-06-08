# metashapes/generators/registry.py
# This module defines registry of shapes samplers

from __future__ import annotations

SHAPE_SAMPLER_REGISTRY: dict[str, object] = {}

def register_shape_sampler(cls):
    name = cls.shape_class.__name__
    if name in SHAPE_SAMPLER_REGISTRY:
        raise ValueError(f"Shape sampler '{name}' is already registered")
    SHAPE_SAMPLER_REGISTRY[name] = cls()
    return cls