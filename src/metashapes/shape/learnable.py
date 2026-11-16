# metashapes/shape/learnable.py
# Post-hoc conversion of shape buffers into optimizable nn.Parameters.

from __future__ import annotations

from typing import Callable, Iterable

import torch
import torch.nn as nn

__all__ = ["make_learnable"]

Selector = Iterable[int] | Callable[[int, nn.Module], bool]


def _promote(module: nn.Module, names: set[str] | None) -> list[str]:
    """Turn this module's own (non-recursive) float buffers into Parameters."""
    done = []
    for name, buf in list(module.named_buffers(recurse=False)):
        if buf is None or not buf.dtype.is_floating_point:
            continue
        if names is not None and name not in names:
            continue
        del module._buffers[name]
        module.register_parameter(name, nn.Parameter(buf.detach().clone()))
        done.append(name)
    return done


def _walk(module: nn.Module, names, select, recurse: bool, prefix: str, out: list[str]):
    from .shapeset import ShapeSet

    for n in _promote(module, names):
        out.append(prefix + n)
    if not recurse:
        return
    if isinstance(module, ShapeSet) and select is not None:
        if callable(select):
            chosen = [i for i, a in enumerate(module.shapes) if select(i, a)]
        else:
            chosen = sorted({int(i) for i in select})
            n = len(module.shapes)
            bad = [i for i in chosen if not -n <= i < n]
            if bad:
                raise IndexError(f"select indices out of range for {n} atoms: {bad[:10]}")
            chosen = sorted({i % n for i in chosen})
        for i in chosen:
            _walk(module.shapes[i], names, None, True, f"{prefix}shapes.{i}.", out)
        return
    for cname, child in module.named_children():
        _walk(child, names, select, True, f"{prefix}{cname}.", out)


def make_learnable(module: nn.Module, names: Iterable[str] | None = None,
                   select: Selector | None = None, recurse: bool = True) -> list[str]:
    """Replace matching buffers with ``nn.Parameter`` s holding the same values.

    Generators produce plain shapes (buffers only); this assigns
    learnability afterwards. It works on any shape tree, a ``ShapeSet``, a
    ``Layout`` or a ``UnitCell``.

    Parameters
    ----------
    module : nn.Module
        Root of the tree to walk.
    names : iterable of str, optional
        Buffer names to promote (e.g. ``("dx", "dy")`` or ``("axes",)``).
        ``None`` promotes every floating-point buffer.
    select : iterable of int, or callable(index, atom) -> bool, optional
        Restrict the conversion to these atoms of every ``ShapeSet`` met
        during the walk. Unselected atoms keep their buffers and so receive
        no gradient. Without a ``ShapeSet`` in the tree it has no effect.
    recurse : bool
        Walk into child modules (default). ``False`` converts only the
        root's own buffers.

    Returns
    -------
    list of str
        Dotted names of the converted tensors, relative to ``module``.

    Example
    -------
    ::

        make_learnable(layout.scene, names=("dx", "dy"))            # all positions
        make_learnable(layout.scene, names=("axes",), select=[3, 17])
    """
    names = None if names is None else set(names)
    if select is not None and not callable(select):
        select = list(select)
    out: list[str] = []
    with torch.no_grad():
        _walk(module, names, select, recurse, "", out)
    return out
