# benchmarks/gds_bench.py
#
# GDS export timings. Run:  python benchmarks/gds_bench.py [--unique 10000 100000]
#
#   unique atoms  : every atom different (SamplerSource) -> plain polygons
#                   (serial and with --workers processes)
#   deduplicated  : N atoms drawn from 20 distinct shapes -> sub-cells + references
#   array         : 1000 x 1000 periodic UnitCell array -> AREFs

from __future__ import annotations

import argparse
import math
import os
import random
import tempfile
import time

from metashapes import Lattice
from metashapes.generators import GeneratorConfig
from metashapes.gds import GDSAssembly
from metashapes.lattice.unit_cell import UnitCell
from metashapes.layout import Layout, SamplerSource
from metashapes.shape import Ellipse, Rectangle, ShapeSet, Translate

SHAPES = ("Rectangle", "Ellipse", "Cross", "Egg", "Star", "Stadium", "RegularPolygon",
          "Triangle", "TShape", "ConvexQuad", "IsoscelesTrapezoid")


def _grid_sites(n, pitch):
    side = math.ceil(math.sqrt(n))
    half = 0.5 * side * pitch
    return [(-half + (k % side + 0.5) * pitch, -half + (k // side + 0.5) * pitch)
            for k in range(n)], half


def unique_layout(n, pitch=400.0, seed=0):
    src = SamplerSource(GeneratorConfig(allowed_shapes=SHAPES, min_feature_size=30,
                                        min_shape_size=50), atom_box=(300.0, 300.0))
    rng = random.Random(seed)
    sites, half = _grid_sites(n, pitch)
    atoms = []
    for x, y in sites:
        a = None
        while a is None:
            a = src.atom(x, y, rng)
        atoms.append(Translate(a, x, y))
    domain = Rectangle(center=(0.0, 0.0), size=(2 * half, 2 * half))
    return Layout(domain, ShapeSet(atoms, halo=1.0))


def dedup_layout(n, kinds=20, pitch=400.0, seed=0):
    src = SamplerSource(GeneratorConfig(allowed_shapes=SHAPES, min_feature_size=30,
                                        min_shape_size=50), atom_box=(300.0, 300.0))
    rng = random.Random(seed)
    lib = []
    while len(lib) < kinds:
        a = src.atom(0.0, 0.0, rng)
        if a is not None:
            lib.append(a)
    sites, half = _grid_sites(n, pitch)
    atoms = [Translate(lib[rng.randrange(kinds)], x, y) for x, y in sites]
    domain = Rectangle(center=(0.0, 0.0), size=(2 * half, 2 * half))
    return Layout(domain, ShapeSet(atoms, halo=1.0))


def timed(label, chip, path):
    t0 = time.perf_counter()
    lib = chip.to_library()
    t1 = time.perf_counter()
    lib.write_gds(path, max_points=chip.max_points)
    t2 = time.perf_counter()
    size = os.path.getsize(path) / 1e6
    print(f"{label:<34} build {t1 - t0:7.2f} s   write {t2 - t1:6.2f} s   file {size:8.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unique", type=int, nargs="*", default=[10_000, 100_000])
    ap.add_argument("--dedup", type=int, default=100_000)
    ap.add_argument("--array", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = os.path.join(tempfile.mkdtemp(), "bench.gds")

    for n in args.unique:
        lay = unique_layout(n)
        chip = GDSAssembly(coord_unit=1e-9)
        chip.place(lay, name="L")
        timed(f"unique atoms      n={n:>7,}", chip, out)
        chip.workers = args.workers
        timed(f"  ... workers={args.workers}", chip, out)

    lay = dedup_layout(args.dedup)
    chip = GDSAssembly(coord_unit=1e-9)
    chip.place(lay, name="L")
    timed(f"20 kinds (dedupe) n={args.dedup:>7,}", chip, out)

    for name, lat in (("rectangular", Lattice.rectangular(400.0, 400.0)),
                      ("hexagonal", Lattice.hexagonal(400.0))):
        uc = UnitCell(lat, Ellipse(center=(200.0, 200.0), axes=(250.0, 150.0)))
        chip = GDSAssembly(coord_unit=1e-9)
        chip.place_array(uc, repeats=(args.array, args.array), name="A")
        timed(f"{name} array {args.array}x{args.array}", chip, out)


if __name__ == "__main__":
    main()
