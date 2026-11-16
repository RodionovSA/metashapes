# metashapes/adapters/npz.py
# Columnar .npz serialization for large layouts (10^5-10^6 atoms).

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["save_layouts_npz", "load_layouts_npz"]

_VERSION = "1"


def _flatten(d: Any, prefix: str, nums: dict, meta: dict) -> None:
    """Split a parametric dict into numeric leaves (``nums``, path -> array)
    and everything else (``meta``, path -> JSON value)."""
    if isinstance(d, dict):
        for k in sorted(d):
            _flatten(d[k], f"{prefix}/{k}" if prefix else k, nums, meta)
        return
    if isinstance(d, bool) or d is None or isinstance(d, str):
        meta[prefix] = d
        return
    if isinstance(d, (int, float)):
        nums[prefix] = np.asarray(d, dtype=np.float64)
        meta[prefix] = "__num__:int" if isinstance(d, int) else "__num__"
        return
    if isinstance(d, (list, tuple)):
        if all(isinstance(v, dict) for v in d) and d:
            raise TypeError("lists of child shapes are not supported inside an atom")
        arr = np.asarray(d, dtype=np.float64)
        nums[prefix] = arr
        meta[prefix] = "__arr__"
        return
    raise TypeError(f"cannot serialise {type(d).__name__} at {prefix!r}")


def _unflatten(meta: dict, nums: dict, row: int) -> dict:
    out: dict = {}
    for path, m in meta.items():
        if m == "__arr__":
            v = nums[path][row].tolist()
        elif m == "__num__":
            v = float(nums[path][row])
        elif m == "__num__:int":
            v = int(nums[path][row])
        else:
            v = m
        node = out
        keys = path.split("/")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = v
    return out


def _encode_layout(lay, prefix: str, arrays: dict) -> dict:
    from metashapes.shape.shapeset import ShapeSet

    scene = lay.scene
    atoms = list(scene.shapes) if isinstance(scene, ShapeSet) else [scene]
    groups: dict[str, dict] = {}
    for i, atom in enumerate(atoms):
        nums: dict = {}
        meta: dict = {}
        _flatten(atom.to_parametric(), "", nums, meta)
        sig = json.dumps({"meta": meta, "shapes": {k: list(v.shape) for k, v in nums.items()}},
                         sort_keys=True)
        g = groups.setdefault(sig, {"meta": meta, "rows": [], "nums": {k: [] for k in nums}})
        g["rows"].append(i)
        for k, v in nums.items():
            g["nums"][k].append(v)

    header_groups = []
    for gi, g in enumerate(groups.values()):
        gp = f"{prefix}/g{gi}"
        arrays[f"{gp}/rows"] = np.asarray(g["rows"], dtype=np.int64)
        cols = {}
        for ci, (k, vals) in enumerate(g["nums"].items()):
            arrays[f"{gp}/c{ci}"] = np.stack(vals)
            cols[k] = f"{gp}/c{ci}"
        header_groups.append({"meta": g["meta"], "columns": cols, "rows": f"{gp}/rows"})

    scene_info = {"type": type(scene).__name__}
    if isinstance(scene, ShapeSet):
        scene_info.update(halo=None if math.isinf(scene.halo) else scene.halo,
                          cell_size=scene.cell_size, skin=scene.skin)
    return {
        "domain": lay.domain.to_parametric(),
        "clip": lay.clip,
        "scene": scene_info,
        "n_atoms": len(atoms),
        "groups": header_groups,
    }


def save_layouts_npz(path: str | Path, layouts) -> None:
    """Save layouts in a columnar ``.npz``: atoms with the same parametric
    structure (type tree and array shapes) form a group, and each numeric
    parameter of a group is stored as one stacked array.

    Much more compact and faster than YAML for 10^5+ atoms. The domain and
    group structure live in a small JSON header inside the archive.
    """
    from metashapes.layout import Layout
    if isinstance(layouts, Layout):
        layouts = [layouts]
    arrays: dict[str, np.ndarray] = {}
    header = {"version": _VERSION,
              "layouts": [_encode_layout(lay, f"L{k}", arrays) for k, lay in enumerate(layouts)]}
    arrays["__header__"] = np.frombuffer(json.dumps(header).encode(), dtype=np.uint8)
    np.savez_compressed(Path(path), **arrays)


def load_layouts_npz(path: str | Path) -> list:
    """Load layouts written by :func:`save_layouts_npz`."""
    from metashapes.layout import Layout
    from metashapes.shape.base import Shape
    from metashapes.shape.shapeset import ShapeSet

    with np.load(Path(path), allow_pickle=False) as z:
        header = json.loads(bytes(z["__header__"]).decode())
        if str(header.get("version")) != _VERSION:
            raise ValueError(f"Unsupported npz layout version {header.get('version')!r}")
        out = []
        for info in header["layouts"]:
            atoms: list = [None] * info["n_atoms"]
            for g in info["groups"]:
                nums = {k: z[col] for k, col in g["columns"].items()}
                rows = z[g["rows"]].tolist()
                for r, i in enumerate(rows):
                    atoms[i] = Shape.from_parametric(_unflatten(g["meta"], nums, r))
            sc = info["scene"]
            if sc["type"] == "ShapeSet":
                scene = ShapeSet(atoms,
                                 halo=math.inf if sc.get("halo") is None else sc["halo"],
                                 cell_size=sc.get("cell_size"), skin=sc.get("skin", 0.0))
            else:
                scene = atoms[0]
            out.append(Layout(Shape.from_parametric(info["domain"]), scene, clip=info["clip"]))
        return out
