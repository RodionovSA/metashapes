"""Layout rendering benchmark: 10^4 atoms on a 4096^2 raster.

    python benchmarks/layout_bench.py [--atoms 10000] [--res 4096] [--device cpu|cuda]

Atoms are ~50-unit ellipses/rectangles on a jittered square grid filling the
domain (about 50 % fill), roughly the meta-atom density of a real layout.
"""

from __future__ import annotations

import argparse
import math
import random
import time

import torch

from metashapes import Layout
from metashapes.shape import Ellipse, Rectangle, ShapeSet, Translate, make_learnable


def build(n_atoms: int, seed: int = 0) -> Layout:
    rng = random.Random(seed)
    side = math.ceil(math.sqrt(n_atoms))
    pitch = 100.0
    atoms = []
    for k in range(n_atoms):
        i, j = divmod(k, side)
        x = (i + 0.5) * pitch + rng.uniform(-10, 10)
        y = (j + 0.5) * pitch + rng.uniform(-10, 10)
        a, b = rng.uniform(40, 70), rng.uniform(40, 70)
        if k % 2:
            atoms.append(Translate(Ellipse(center=(0.0, 0.0), axes=(a, b), angle=rng.uniform(0, 180)), x, y))
        else:
            atoms.append(Rectangle(center=(x, y), size=(a, b), angle=rng.uniform(0, 90)))
    L = side * pitch
    domain = Rectangle(center=(L / 2, L / 2), size=(L, L))
    return Layout(domain, ShapeSet(atoms))


def timed(label, fn, device, repeat=1):
    fn()  # warm-up (also builds the index)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(repeat):
        out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / repeat
    print(f"  {label:<38s} {dt:8.2f} s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atoms", type=int, default=10_000)
    ap.add_argument("--res", type=int, default=4096)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    devices = [args.device] if args.device else (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])

    for dev in devices:
        device = torch.device(dev)
        lay = build(args.atoms).to(device)
        n = args.res
        print(f"{args.atoms} atoms, {n}x{n} raster, device={dev}, "
              f"pixel={lay.extent()[1] / n:.2f}")
        m = timed("mask(hard)  halo=1px", lambda: lay.mask(n, n), device)
        print(f"    fill fraction {m.mean().item():.3f}")
        timed("mask(soft)  halo=8px", lambda: lay.mask(n, n, soft=True), device)
        timed("rasterize   halo=inf (exact)", lambda: lay.rasterize(n, n), device)

        make_learnable(lay.scene, names=("dx", "dy", "center"))
        def fwd_bwd():
            lay.zero_grad()
            lay.mask(n, n, soft=True).sum().backward()
        timed("mask(soft) forward + backward", fwd_bwd, device)
        del lay


if __name__ == "__main__":
    main()
