#metashapes/shape/registry.py
# This module defines the shape registry for symbolic 2D shapes.

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Shape

SHAPE_REGISTRY: dict[str, type["Shape"]] = {}

def register_shape(name: str):
    def deco(cls: type["Shape"]) -> type["Shape"]:
        if name in SHAPE_REGISTRY:
            raise ValueError(f"Shape '{name}' already registered")
        SHAPE_REGISTRY[name] = cls
        return cls
    return deco